#!/usr/bin/env python3
"""Habitat inference server for the two small action heads on frozen FutureNav features.

Same /health, /reset and /step protocol as deploy/server.py, so eval/run.py drives it unchanged.

  --arm discrete        ~/jetracer_paired_20261104/checkpoints/head_expert_final.pt
                        4 actions per query from the base-FutureNav final-norm last-prompt-token state.
  --arm lora_waypoint   ~/lora_waypoint_20261104/waypoint_lora_v2.pt (+ anchors.pt, LoRA adapter)
                        8 robot-relative poses + stop flags; poses are turned back into the simulator's
                        primitives (forward 0.25 m / turn 15 deg / stop) interval by interval.

Execution follows the paired JetRacer protocol: at most --max-exec (4) actions per query, re-query once
the plan has turned --turn-budget (30) degrees, STOP ends a plan, no forward fallback on an invalid answer
(it returns STOP and is counted). Steps served from the queue still advance VGGT. If the pose change does
not match the action returned last step (collision), the queue is dropped and the head is re-queried.
The previous-action embedding is the last action returned (4 = none, at the first step).

    python deploy/head_eval_server.py --arm discrete --port 8651
    python deploy/head_eval_server.py --arm lora_waypoint --port 8652
"""
import argparse
import base64
import importlib.util
import io
import math
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT), str(HERE)]

import server as base  # noqa: E402
from chunk_server import executed  # noqa: E402
from futurenav_core import ACTION_ID_TO_NAME, FutureNavPolicy  # noqa: E402

HOME = Path.home()
DISCRETE_CKPT = HOME / "jetracer_paired_20261104/checkpoints/head_expert_final.pt"
LORA_WP_DIR = HOME / "lora_waypoint_20261104"
STATE = {"policy": None, "cfg": {}, "session": None, "instruction": None, "expected_step": 0, "queue": [],
         "turned": 0, "last": None, "prev": 4, "lock": threading.Lock(), "stats": {}}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DiscretePolicy:
    """Frozen base FutureNav + Colab ChunkHead -> 4 actions (mirrors jetracer_paired policy_server.Policy)."""
    arm = "discrete"

    def __init__(self, checkpoint, head_path):
        import torch
        self.torch = torch
        heads = load_module("paired_heads", HOME / "jetracer_paired_20261104/heads.py")
        saved = torch.load(head_path, map_location="cpu", weights_only=True)
        if saved["keys"] != ["final"] or saved["version"] != "expert":
            raise ValueError("requires the Colab expert final-token checkpoint")
        self.fn = FutureNavPolicy(checkpoint, attn_implementation="sdpa")
        self.fn.enable_hidden_capture()
        self.fn.model.requires_grad_(False)
        self.head = heads.ChunkHead()
        self.head.load_state_dict(saved["model"], strict=True)
        self.head.cuda().eval().requires_grad_(False)

    def reset(self):
        self.fn.reset()

    def observe(self, image):
        with self.torch.inference_mode():
            self.fn.advance_vggt(image)

    def query(self, image, instruction, prev):
        torch = self.torch
        result = self.fn.act(image, instruction, want_hidden=True, capture=False)
        if self.fn._hidden is None:
            raise RuntimeError("hidden-state capture failed")
        h = torch.as_tensor(self.fn._hidden, device="cuda").float()[None]
        with torch.inference_mode():
            logits = self.head([h], torch.tensor([prev], device="cuda"))
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite head output")
        return {"raw": result["raw"], "actions": logits.argmax(-1)[0].tolist()}


class LoRAWaypoint:
    """LoRAWaypointPolicy from ~/lora_waypoint_20261104 -> 8 poses -> simulator primitives."""
    arm = "lora_waypoint"

    def __init__(self, checkpoint, wp_dir, log_probs=False):
        sys.path.insert(0, str(wp_dir))
        mod = load_module("lora_waypoint_evaluation", wp_dir / "evaluation.py")
        self.p = mod.LoRAWaypointPolicy(project=PROJECT, adapter=PROJECT / "outputs/s2_lora/lora_v1/adapter_final",
                                        checkpoint=wp_dir / "waypoint_lora_v2.pt", anchors=wp_dir / "anchors.pt")
        self.fn, self.log_probs = self.p.fn, log_probs

    def reset(self):
        self.p.reset()

    def observe(self, image):
        self.p.observe(image)

    def query(self, image, instruction, prev):
        if self.log_probs:
            return self._query_logged(image, instruction, prev)
        out = self.p.query(image, instruction, prev)
        return {"raw": out["raw"], "anchor": out["anchor"], "poses": out["poses"], "stopped": out["stopped"],
                "actions": poses_to_actions(out["poses"], out["stopped"])}

    def _query_logged(self, image, instruction, prev):
        """LoRAWaypointPolicy.query step for step (same min-risk decode), plus the 256 anchor probabilities."""
        import torch
        p = self.p
        with torch.inference_mode():
            result = p.fn.act(image, instruction, want_hidden=True, capture=False)
            if p.fn._hidden is None:
                raise RuntimeError("missing LoRA final-token state")
            feature = torch.as_tensor(np.asarray(p.fn._hidden), device="cuda").float()[None]
            logits = p.head([feature], torch.tensor([prev], device="cuda"))[0, 0]
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite waypoint logits")
            probs = logits.softmax(-1).cpu()
        p.cache_status()
        anchor = int((probs @ p.distances).argmin())
        row = p.anchors[anchor]
        xy, yaw = row[:16].reshape(8, 2), row[16:24] / .5
        poses, stopped = torch.cat([xy, yaw[:, None]], -1).tolist(), (row[24:32] / 2 > .5).tolist()
        return {"raw": result["raw"], "anchor": anchor, "poses": poses, "stopped": stopped,
                "actions": poses_to_actions(poses, stopped), "probs": [round(float(x), 5) for x in probs]}


PRIMITIVES = {1: (0.25, 0.0, 0.0), 2: (0.0, 0.0, math.radians(15)), 3: (0.0, 0.0, -math.radians(15))}
YAW_W = 0.5       # metres per radian, the weight the anchors were built with


def poses_to_actions(poses, stopped):
    """Pose chain (x, y, yaw) -> 0 STOP / 1 FORWARD / 2 LEFT / 3 RIGHT, one action per interval.
    Anchors are cluster centres, not exact sums of primitives, so each increment (in the previous pose's
    frame) is matched to the nearest primitive under the anchors' own metric. Always returns a primitive
    until the stop flag."""
    prev, acts = (0.0, 0.0, 0.0), []
    for p, stop in zip(poses, stopped):
        if stop:
            acts.append(0)
            break
        dx, dy = p[0] - prev[0], p[1] - prev[1]
        c, s = math.cos(prev[2]), math.sin(prev[2])
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        dyaw = (p[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi
        acts.append(min(PRIMITIVES, key=lambda a: (lx - PRIMITIVES[a][0]) ** 2 + ly ** 2
                        + (YAW_W * (dyaw - PRIMITIVES[a][2])) ** 2))
        prev = tuple(p)
    return acts


def increments(poses, stopped):
    """Pose chain -> per-interval motion (forward m, left m, yaw deg) in the previous pose's frame; None = STOP."""
    prev, out = (0.0, 0.0, 0.0), []
    for p, stop in zip(poses, stopped):
        if stop:
            out.append(None)
            break
        dx, dy = p[0] - prev[0], p[1] - prev[1]
        c, s = math.cos(prev[2]), math.sin(prev[2])
        dyaw = (p[2] - prev[2] + math.pi) % (2 * math.pi) - math.pi
        out.append((c * dx + s * dy, -s * dx + c * dy, math.degrees(dyaw)))
        prev = tuple(p)
    return out


def nearest_primitive(m):
    return min(PRIMITIVES, key=lambda a: (m[0] - PRIMITIVES[a][0]) ** 2 + m[1] ** 2
               + (YAW_W * (math.radians(m[2]) - PRIMITIVES[a][2])) ** 2)


def step_continuous(img, instruction, pose):
    """--exec continuous: each step executes one predicted pose interval (eval/run.py --waypoint-exec) instead
    of the nearest primitive. Same protocol otherwise: <= max_exec intervals per query, re-query once |yaw|
    summed over the plan reaches turn_budget, STOP flag ends the plan. Collision = the agent moved less than
    half the commanded translation (commands >= 5 cm); the queue is then dropped and the head re-queried.
    The previous-action embedding gets the primitive nearest to the executed interval."""
    p, cfg, S = STATE["policy"], STATE["cfg"], STATE["stats"]
    dropped, last = None, STATE["last"]
    if last is not None and pose is not None and last.get("motion"):
        want = math.hypot(*last["motion"][:2])
        got = math.hypot(pose[0] - last["pose"][0], pose[2] - last["pose"][2])
        if want >= 0.05 and got < 0.5 * want:
            S["blocked"] = S.get("blocked", 0) + 1
            if STATE["queue"]:
                dropped = {"commanded_m": round(want, 3), "moved_m": round(got, 3), "left": len(STATE["queue"])}
                STATE["queue"] = []
                S["dropped"] = S.get("dropped", 0) + 1
    q = None
    if STATE["queue"]:
        p.observe(img)
    else:
        q = p.query(img, instruction, STATE["prev"])
        plan = increments(q["poses"], q["stopped"])
        if not plan:
            plan = [None]
            S["invalid"] = S.get("invalid", 0) + 1
        STATE["queue"], STATE["turned"] = plan[:cfg["max_exec"]], 0
        S["queries"] = S.get("queries", 0) + 1
    m = STATE["queue"].pop(0)
    if m is None:
        action, STATE["queue"] = 0, []
    else:
        action = 1
        STATE["turned"] += abs(m[2])
        if STATE["turned"] >= cfg["turn_budget"]:
            STATE["queue"] = []
    S["steps"] = S.get("steps", 0) + 1
    STATE["last"] = {"pose": pose, "action": action, "motion": m}
    STATE["prev"] = 0 if m is None else nearest_primitive(m)
    frames = p.fn.vggt_cache_frames()
    head = {"queried": q is not None, "left": len(STATE["queue"]), "turned": round(STATE["turned"], 1),
            "motion": None if m is None else [round(v, 4) for v in m]}
    if q is not None:
        head.update({k: v for k, v in q.items() if k in ("raw", "actions", "anchor", "probs")})
    if dropped:
        head["dropped"] = dropped
    out = {"action": action, "action_name": ACTION_ID_TO_NAME[action], "fallback": False, "step": p.fn.step,
           "vggt_cache_frames": frames,
           "vggt_cache_ok": frames == min(p.fn.step, p.fn.model.kv_cache_vggt.cache_size), "head": head}
    if m is not None:
        out["motion"] = {"dx": m[0], "dy": m[1], "dyaw": m[2]}
    return out


def step_plan(img, instruction, pose):
    """--exec plan: System 2 only. Every step advances the model's frame history; the head is queried only when
    the client asks (request "query": true) and the whole pose plan is returned. The client's local planner
    (eval/s1_local.py) drives and decides when to ask. The previous-action embedding is the primitive nearest
    to the agent's actual motion over the last step (from the poses the client sends)."""
    p, S = STATE["policy"], STATE["stats"]
    last = STATE["last"]
    if last is not None and pose is not None:
        c, s = math.cos(last[3]), math.sin(last[3])
        # Habitat: yaw about +y, forward = (-sin, -cos) in (x, z); express the step in the previous pose's frame
        dx, dz = pose[0] - last[0], pose[2] - last[2]
        fwd, left = -s * dx - c * dz, -c * dx + s * dz
        dyaw = math.degrees((pose[3] - last[3] + math.pi) % (2 * math.pi) - math.pi)
        STATE["prev"] = nearest_primitive((fwd, left, dyaw)) if math.hypot(fwd, left) > 0.03 or abs(dyaw) > 3 else 0
    STATE["last"] = pose
    S["steps"] = S.get("steps", 0) + 1
    head = {"queried": False}
    out = {"action": 1, "action_name": "MOVE_FORWARD", "fallback": False}
    if STATE["req"].get("query"):
        q = p.query(img, instruction, 4 if last is None else STATE["prev"])
        S["queries"] = S.get("queries", 0) + 1
        out["plan"] = {"poses": q["poses"], "stopped": q["stopped"]}
        head = {"queried": True, "anchor": q.get("anchor"), "raw": q.get("raw"), "prev": STATE["prev"]}
    else:
        p.observe(img)
    frames = p.fn.vggt_cache_frames()
    out.update(step=p.fn.step, vggt_cache_frames=frames, head=head,
               vggt_cache_ok=frames == min(p.fn.step, p.fn.model.kv_cache_vggt.cache_size))
    return out


def step(img, instruction, pose):
    if STATE["cfg"].get("exec") == "continuous":
        return step_continuous(img, instruction, pose)
    if STATE["cfg"].get("exec") == "plan":
        return step_plan(img, instruction, pose)
    p, cfg, S = STATE["policy"], STATE["cfg"], STATE["stats"]
    dropped, last = None, STATE["last"]
    if last is not None and pose is not None and STATE["queue"]:
        done = executed(last["pose"], pose)
        if done != last["action"]:
            dropped = {"returned": ACTION_ID_TO_NAME[last["action"]], "executed": done, "left": len(STATE["queue"])}
            STATE["queue"] = []
            S["dropped"] = S.get("dropped", 0) + 1
    q = None
    if STATE["queue"]:
        p.observe(img)
    else:
        q = p.query(img, instruction, STATE["prev"])
        plan = q["actions"]
        if 0 in plan:
            plan = plan[:plan.index(0) + 1]
        if not plan:
            plan = [0]
            S["invalid"] = S.get("invalid", 0) + 1
        STATE["queue"], STATE["turned"] = plan[:cfg["max_exec"]], 0
        S["queries"] = S.get("queries", 0) + 1
    action = STATE["queue"].pop(0)
    if action in (2, 3):
        STATE["turned"] += 1
        if STATE["turned"] * 15 >= cfg["turn_budget"]:
            STATE["queue"] = []
    if action == 0:
        STATE["queue"] = []
    S["steps"] = S.get("steps", 0) + 1
    STATE["last"], STATE["prev"] = {"pose": pose, "action": action}, action
    frames = p.fn.vggt_cache_frames()
    head = {"queried": q is not None, "left": len(STATE["queue"]), "turned": STATE["turned"]}
    if q is not None:
        head.update({k: v for k, v in q.items() if k in ("raw", "actions", "anchor", "probs")})
    if dropped:
        head["dropped"] = dropped
    return {"action": action, "action_name": ACTION_ID_TO_NAME[action], "fallback": False, "step": p.fn.step,
            "vggt_cache_frames": frames,
            "vggt_cache_ok": frames == min(p.fn.step, p.fn.model.kv_cache_vggt.cache_size), "head": head}


class Handler(base.Handler):
    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"ready": STATE["policy"] is not None, "session": STATE["session"],
                         "expected_step": STATE["expected_step"], "config": {"policy": "head_eval", **STATE["cfg"]},
                         "stats": STATE["stats"]})

    def do_POST(self):
        try:
            req = self._read_json()
        except Exception as exc:
            return self._send(400, {"error": "bad json: %s" % exc})
        if self.path == "/reset":
            with STATE["lock"]:
                STATE["policy"].reset()
                STATE.update(session=req.get("session"), instruction=req.get("instruction", ""), expected_step=0,
                             queue=[], turned=0, last=None, prev=4)
            print("[head_eval] reset session=%r" % STATE["session"], flush=True)
            return self._send(200, {"ok": True})
        if self.path != "/step":
            return self._send(404, {"error": "not found"})
        with STATE["lock"]:
            if req.get("session") != STATE["session"]:
                return self._send(409, {"error": "unknown session; POST /reset first"})
            n = int(req.get("step", -1))
            if n != STATE["expected_step"]:
                return self._send(409, {"error": "step out of order", "expected": STATE["expected_step"], "got": n})
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB"))
            t0 = time.time()
            STATE["req"] = req
            try:
                out = step(img, req.get("instruction") or STATE["instruction"], req.get("pose"))
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print("[head_eval] STEP %d FAILED\n%s" % (n, tb), flush=True)
                return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc), "traceback": tb.splitlines()[-12:]})
            out["latency_s"] = round(time.time() - t0, 3)
            STATE["expected_step"] += 1
        hd = out["head"]
        print("[head_eval] step=%3d -> %-12s %.2fs %s%s" % (
            n, out["action_name"], out["latency_s"], ("Q %s" % hd.get("actions", hd.get("anchor"))) if hd["queried"] else "  queue",
            "  DROP(%s)" % hd["dropped"] if "dropped" in hd else ""), flush=True)
        return self._send(200, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["discrete", "lora_waypoint"], required=True)
    ap.add_argument("--checkpoint", default=str(PROJECT / "pretrain_models/FutureNav-4B-Base"))
    ap.add_argument("--head", default=None, help="discrete head checkpoint (default: jetracer_paired expert head)")
    ap.add_argument("--wp-dir", default=str(LORA_WP_DIR))
    ap.add_argument("--max-exec", type=int, default=4)
    ap.add_argument("--turn-budget", type=float, default=30.0, help="degrees")
    ap.add_argument("--log-probs", action="store_true", help="lora_waypoint: return the 256 anchor probabilities per query")
    ap.add_argument("--exec", choices=["discrete", "continuous", "plan"], default="discrete",
                    help="lora_waypoint: continuous = execute each pose interval as is (needs eval/run.py "
                         "--waypoint-exec, WAYPOINT_EXEC=1 in eval_r2r_remote.sh)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8651)
    a = ap.parse_args()
    print("[head_eval] loading %s ..." % a.arm, flush=True)
    if a.arm == "discrete":
        STATE["policy"] = DiscretePolicy(a.checkpoint, a.head or DISCRETE_CKPT)
    else:
        STATE["policy"] = LoRAWaypoint(a.checkpoint, Path(a.wp_dir), log_probs=a.log_probs)
    if a.exec in ("continuous", "plan") and a.arm != "lora_waypoint":
        ap.error("--exec continuous/plan needs --arm lora_waypoint (the discrete head predicts no poses)")
    STATE["cfg"] = {"arm": a.arm, "max_exec": a.max_exec, "turn_budget": a.turn_budget, "log_probs": a.log_probs,
                    "exec": a.exec}
    print("[head_eval] ready on %s:%d %s" % (a.host, a.port, STATE["cfg"]), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
