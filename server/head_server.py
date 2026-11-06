#!/usr/bin/env python3
"""Inference server for the motion-head policy (experiments/motion_head/policy.py).

Same /health, /reset and /step protocol as deploy/server.py, so eval/run.py and
eval/agent_remote.py drive it unchanged. /step additionally needs "pose":
[x, y, z, yaw_rad] of the agent at the frame (run.py sends it on every step),
and returns the head's diagnostics under "head".

    python deploy/head_server.py --checkpoint pretrain_models/FutureNav-4B-Base \
        --head outputs/motion_head/flow_v1/best.pt --attn sdpa --port 8643
"""
import argparse
import base64
import io
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments/motion_head"))

import server as base  # noqa: E402  (reuses its HTTP handler plumbing)
from futurenav_core import FutureNavPolicy  # noqa: E402
from policy import HeadPolicy  # noqa: E402

STATE = {"head": None, "session": None, "instruction": None, "expected_step": 0, "lock": threading.Lock()}


class Handler(base.Handler):
    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        hp = STATE["head"]
        self._send(200, {"ready": hp is not None, "session": STATE["session"],
                         "expected_step": STATE["expected_step"],
                         "config": {"policy": "motion_head", **(hp.ckpt_info if hp else {})}})

    def do_POST(self):
        try:
            req = self._read_json()
        except Exception as exc:
            return self._send(400, {"error": "bad json: %s" % exc})
        if self.path == "/reset":
            with STATE["lock"]:
                STATE["head"].reset()
                STATE["session"] = req.get("session")
                STATE["instruction"] = req.get("instruction", "")
                STATE["expected_step"] = 0
            print("[head] reset session=%r" % STATE["session"], flush=True)
            return self._send(200, {"ok": True})
        if self.path != "/step":
            return self._send(404, {"error": "not found"})
        with STATE["lock"]:
            if req.get("session") != STATE["session"]:
                return self._send(409, {"error": "unknown session; POST /reset first"})
            step = int(req.get("step", -1))
            if step != STATE["expected_step"]:
                return self._send(409, {"error": "step out of order", "expected": STATE["expected_step"], "got": step})
            if "pose" not in req:
                return self._send(400, {"error": "the motion head needs 'pose' [x, y, z, yaw] on every step; "
                                                 "is eval/run.py up to date?"})
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB"))
            t0 = time.time()
            try:
                out = STATE["head"].step(img, req.get("instruction") or STATE["instruction"], req["pose"])
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print("[head] STEP %d FAILED\n%s" % (step, tb), flush=True)
                return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc), "traceback": tb.splitlines()[-12:]})
            out["latency_s"] = round(time.time() - t0, 3)
            STATE["expected_step"] += 1
        hd = out["head"]
        print("[head] step=%3d -> %-12s %.2fs age=%d fn=%-12s p_stop=%.2f rem=%.1f off=%.1f spread=%.2f%s%s"
              % (step, out["action_name"], out["latency_s"], hd["age"], hd["fn_action"], hd["p_stop"],
                 hd["remaining_m"], hd["off_route_m"], hd["spread_m"], " SWITCH" if hd["switched"] else "",
                 " stop_by=%s" % hd["stop_by"] if hd["stop_by"] else ""), flush=True)
        return self._send(200, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8643)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("[head] loading FutureNav + head ...", flush=True)
    STATE["head"] = HeadPolicy(a.head, FutureNavPolicy(a.checkpoint, attn_implementation=a.attn), seed=a.seed)
    print("[head] ready on %s:%d (%s)" % (a.host, a.port, STATE["head"].ckpt_info), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
