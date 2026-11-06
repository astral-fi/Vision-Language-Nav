#!/usr/bin/env python3
"""Inference server for the future-steps LoRA (experiments/s2_lora).

Same /health, /reset and /step protocol as deploy/server.py, so eval/run.py and
eval/agent_remote.py drive it unchanged. The model answers with up to K=8
actions; the server executes them from a plan queue:
  --mode A    query every step, execute the first action
  --mode B    query, execute up to --max-exec actions (default 4)
  --mode C    as B, but re-query once the plan has turned --turn-budget degrees
Steps served from the queue still add the frame to the history and the VGGT
cache (advance_vggt), so the next query sees the same state as if the model had
run on every step.

Divergence: run.py overrides the action during scripted derails ("walked") and
on its own early-stop rules, and a forward step can collide. The client sends
"pose" [x, y, z, yaw] on every step; if the pose change does not match the action
the server returned last step, the queue is dropped and the model re-queried.

Each /step returns "head": {"queried", "plan", "raw", "left", "dropped"}, which
run.py writes into the per-step trace.

    python deploy/chunk_server.py --checkpoint pretrain_models/FutureNav-4B-Base \
        --adapter outputs/s2_lora/lora_v1/adapter_final --mode A --port 8644
"""
import argparse
import base64
import io
import math
import re
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

import server as base  # noqa: E402  (reuses its HTTP handler plumbing)
from futurenav_core import ACTION_ID_TO_NAME, ACTION_NAME_TO_ID, FutureNavPolicy  # noqa: E402

WORD = re.compile(r"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT")
STATE = {"policy": None, "cfg": {}, "session": None, "instruction": None, "expected_step": 0,
         "queue": [], "turned": 0, "last": None, "lock": threading.Lock(), "stats": {}}


def executed(prev_pose, pose):
    """Action implied by the pose change: 0 none, 1 forward, 2 left, 3 right."""
    dyaw = math.degrees((pose[3] - prev_pose[3] + math.pi) % (2 * math.pi) - math.pi)
    if abs(dyaw) > 7.5:
        return 2 if dyaw > 0 else 3
    if math.hypot(pose[0] - prev_pose[0], pose[2] - prev_pose[2]) > 0.1:
        return 1
    return 0


def step(img, instruction, pose):
    p, cfg = STATE["policy"], STATE["cfg"]
    dropped = None
    last = STATE["last"]
    if last is not None and pose is not None and STATE["queue"]:
        done = executed(last["pose"], pose)
        if done != last["action"]:
            dropped = {"returned": ACTION_ID_TO_NAME[last["action"]], "executed": done, "left": len(STATE["queue"])}
            STATE["queue"] = []
            STATE["stats"]["dropped"] = STATE["stats"].get("dropped", 0) + 1
    raw = None
    if STATE["queue"]:
        p.advance_vggt(img)
    else:
        p.rgb_list.append(Image.fromarray(img).convert("RGB"))
        raw = p._call_model(p._prepare_images(), instruction).strip()
        p.step += 1
        plan = [ACTION_NAME_TO_ID[w] for w in WORD.findall(raw)]
        if not plan:
            plan = [ACTION_NAME_TO_ID["MOVE_FORWARD"]]
            STATE["stats"]["unparsed"] = STATE["stats"].get("unparsed", 0) + 1
        if 0 in plan:
            plan = plan[:plan.index(0) + 1]
        STATE["queue"] = plan[:1] if cfg["mode"] == "A" else plan[:cfg["max_exec"]]
        STATE["turned"] = 0
        STATE["stats"]["queries"] = STATE["stats"].get("queries", 0) + 1
    action = STATE["queue"].pop(0)
    if action in (2, 3):
        STATE["turned"] += 1
        if cfg["mode"] == "C" and STATE["turned"] * 15 >= cfg["turn_budget"]:
            STATE["queue"] = []
    if action == 0:
        STATE["queue"] = []
    STATE["stats"]["steps"] = STATE["stats"].get("steps", 0) + 1
    STATE["last"] = {"pose": pose, "action": action}
    frames = p.vggt_cache_frames()
    head = {"queried": raw is not None, "left": len(STATE["queue"]), "turned": STATE["turned"]}
    if raw is not None:
        head["raw"] = raw
    if dropped:
        head["dropped"] = dropped
    return {"action": action, "action_name": ACTION_ID_TO_NAME[action], "fallback": False,
            "step": p.step, "vggt_cache_frames": frames,
            "vggt_cache_ok": frames == min(p.step, p.model.kv_cache_vggt.cache_size), "head": head}


class Handler(base.Handler):
    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"ready": STATE["policy"] is not None, "session": STATE["session"],
                         "expected_step": STATE["expected_step"],
                         "config": {"policy": "s2_lora_chunk", **STATE["cfg"]}, "stats": STATE["stats"]})

    def do_POST(self):
        try:
            req = self._read_json()
        except Exception as exc:
            return self._send(400, {"error": "bad json: %s" % exc})
        if self.path == "/reset":
            with STATE["lock"]:
                STATE["policy"].reset()
                STATE.update(session=req.get("session"), instruction=req.get("instruction", ""),
                             expected_step=0, queue=[], turned=0, last=None)
            print("[chunk] reset session=%r" % STATE["session"], flush=True)
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
            try:
                out = step(img, req.get("instruction") or STATE["instruction"], req.get("pose"))
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print("[chunk] STEP %d FAILED\n%s" % (n, tb), flush=True)
                return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc), "traceback": tb.splitlines()[-12:]})
            out["latency_s"] = round(time.time() - t0, 3)
            STATE["expected_step"] += 1
        hd = out["head"]
        print("[chunk] step=%3d -> %-12s %.2fs %s%s" % (
            n, out["action_name"], out["latency_s"], ("Q " + hd["raw"]) if hd["queried"] else "  queue",
            "  DROP(%s)" % hd["dropped"] if "dropped" in hd else ""), flush=True)
        return self._send(200, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--mode", choices=["A", "B", "C"], default="A")
    ap.add_argument("--max-exec", type=int, default=4)
    ap.add_argument("--turn-budget", type=float, default=30.0, help="degrees, mode C")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8644)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    a = ap.parse_args()
    from peft import PeftModel
    print("[chunk] loading FutureNav + adapter ...", flush=True)
    p = FutureNavPolicy(a.checkpoint, attn_implementation=a.attn)
    PeftModel.from_pretrained(p.model, a.adapter).eval()     # LoRA layers are injected into p.model in place
    p.max_new_tokens = 48
    STATE["cfg"] = {"adapter": a.adapter, "mode": a.mode, "max_exec": a.max_exec, "turn_budget": a.turn_budget}
    STATE["policy"] = p
    print("[chunk] ready on %s:%d %s" % (a.host, a.port, STATE["cfg"]), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
