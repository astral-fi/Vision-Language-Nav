"""
Off-board FutureNav inference server.

Run this on a workstation with >= 24 GB of VRAM (RTX 3090/4090/A5000 or better);
see deploy/README.md for the memory arithmetic. The JetRacer talks to it over
WiFi with deploy/jetracer_client.py.

  export PYTHONPATH=$PWD/src:$PWD
  python deploy/server.py --checkpoint /path/to/FutureNav-4B-Base --port 8642

Protocol (HTTP, JSON):
  POST /reset  {"session": str, "instruction": str}          -> {"ok": true}
  POST /step   {"session": str, "step": int, "image": b64}   -> {"action_name": ...}
  GET  /health                                               -> {"ready": bool, ...}
  POST /capture_flush {"session", "episode_id", "path"}      -> {"path", "steps", "bytes", ...}
       (experiments/capture500: /step with "capture": true records the model's
        internals for that step; flush writes the episode to <path>/<id>.npz)

The `step` field is a monotonically increasing counter owned by the client. The
server rejects out-of-order or duplicate steps instead of silently corrupting
the VGGT KV cache, which is append-only spatial memory: a replayed frame is
inserted twice and a dropped frame leaves a hole, and either one degrades every
subsequent decision in the episode.

Only one episode is served at a time (the KV cache lives on the model object).
"""

from __future__ import annotations

import os
# The VGGT cache grows by ~153 MB every step for the first 56 steps. That
# monotonic growth is the worst case for the caching allocator: reserved memory
# runs well ahead of allocated. Expandable segments keep the gap small, which is
# the difference between fitting and not fitting on a 24 GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from futurenav_core import FutureNavPolicy

STATE = {"policy": None, "session": None, "instruction": None,
         "expected_step": 0, "lock": threading.Lock()}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # quieter than the default access log
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        p = STATE["policy"]
        # Report the settings that change results, so a run can record what
        # produced its numbers rather than relying on anyone's memory.
        cfg = {}
        if p is not None:
            cache = p.model.kv_cache_vggt
            cfg = {"lam": float(p.model.lam),
                   "vggt_cache_start": cache.start_size,
                   "vggt_cache_recent": cache.recent_size,
                   "max_history_frames": p.max_history_frames,
                   "attn": getattr(p.model.config, "_attn_implementation", "?"),
                   "dtype": str(next(p.model.parameters()).dtype),
                   "history_mode": p.history_mode, "vggt_reset": p.vggt_reset}
        # probes_supported lets the habitat-side client abort at construction
        # rather than run for hours against a server that silently ignores the
        # "probes" key and returns probe-free results. See eval/agent_remote.py.
        self._send(200, {"ready": p is not None, "session": STATE["session"],
                         "expected_step": STATE["expected_step"], "config": cfg,
                         "probes_supported": True,
                         "hidden_supported": True,
                         "capture_supported": True,
                         "ablation_supported": True})

    def do_POST(self):
        try:
            req = self._read_json()
        except Exception as exc:
            return self._send(400, {"error": f"bad json: {exc}"})

        if self.path == "/reset":
            with STATE["lock"]:
                STATE["policy"].reset()
                # experiments/failure_detection/history_ablation: optional, default off.
                # A /reset without these keys restores the normal policy.
                STATE["policy"].set_ablation(
                    history_mode=req.get("history_mode") or "full",
                    vggt_reset=bool(req.get("vggt_reset")),
                    seed=int(req.get("history_seed") or 0),
                    log=req.get("history_log"),
                    episode=req.get("episode_id") or "")
                STATE["session"] = req.get("session")
                STATE["instruction"] = req.get("instruction", "")
                STATE["expected_step"] = 0
            print(f"[server] reset session={STATE['session']!r} "
                  f"instruction={STATE['instruction']!r}")
            return self._send(200, {"ok": True})

        if self.path == "/capture_flush":
            with STATE["lock"]:
                if req.get("session") != STATE["session"]:
                    return self._send(409, {"error": "unknown session"})
                try:
                    out = STATE["policy"].capture_flush(req["path"], str(req["episode_id"]))
                except Exception as exc:
                    return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            print(f"[server] capture flushed {out['steps']} steps -> {out['path']} "
                  f"({out['bytes'] / 1e6:.1f} MB, {out['bytes_per_step'] / 1e3:.0f} KB/step)")
            return self._send(200, out)

        if self.path == "/step":
            session = req.get("session")
            step = int(req.get("step", -1))
            with STATE["lock"]:
                if session != STATE["session"]:
                    return self._send(409, {"error": "unknown session; POST /reset first"})
                if step != STATE["expected_step"]:
                    # Never silently accept: a wrong step means the client
                    # retried or dropped a frame and the cache is now suspect.
                    return self._send(409, {"error": "step out of order",
                                            "expected": STATE["expected_step"],
                                            "got": step})
                try:
                    raw = base64.b64decode(req["image"])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:
                    return self._send(400, {"error": f"bad image: {exc}"})

                instruction = req.get("instruction") or STATE["instruction"]
                # Probes run inside this same lock and this same request, so
                # they cannot interleave with a real step or touch
                # expected_step. That is why they extend /step instead of
                # getting an endpoint of their own.
                probes = req.get("probes")
                probe_min_step = int(req.get("probe_min_step", 9))
                probe_seed = int(req.get("probe_seed", 0))
                want_hidden = bool(req.get("want_hidden"))
                capture = bool(req.get("capture"))
                capture_validate = bool(req.get("capture_validate"))
                capture_profile = str(req.get("capture_profile", "full"))
                # experiments/steer_plan: {"layer", "mode", "vec", "decode", "final"};
                # "final" also returns the final-norm state, as want_hidden does.
                steer = req.get("steer")
                if steer and steer.get("final"):
                    want_hidden = True
                if want_hidden:
                    # Idempotent; registers the forward hook on first use.
                    STATE["policy"].enable_hidden_capture()
                t0 = time.time()
                try:
                    result = STATE["policy"].act(np.asarray(img), instruction,
                                                 probes=probes,
                                                 probe_min_step=probe_min_step,
                                                 probe_seed=probe_seed,
                                                 want_hidden=want_hidden,
                                                 capture=capture,
                                                 capture_validate=capture_validate,
                                                 capture_profile=capture_profile,
                                                 steer=steer)
                except Exception as exc:
                    # Without this, an exception here kills the handler thread and
                    # the client sees only "Remote end closed connection without
                    # response" -- which hides the actual cause. The common one is
                    # CUDA OOM around step 56, where the VGGT cache saturates at
                    # start_size + recent_size frames and VRAM peaks.
                    import traceback
                    tb = traceback.format_exc()
                    oom = "out of memory" in str(exc).lower()
                    print(f"[server] STEP {step} FAILED: {type(exc).__name__}: {exc}")
                    print(tb)
                    if oom:
                        print("[server] ^ CUDA OOM. The VGGT cache grows ~159 MB/step "
                              "to 56 frames (~8.9 GB). Restart with --vggt-recent-frames 24 "
                              "(~3.8 GB) and close other GPU users.")
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    return self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                                            "oom": oom, "step": step,
                                            "traceback": tb.splitlines()[-12:]})
                result["latency_s"] = round(time.time() - t0, 3)
                STATE["expected_step"] += 1
            vram = result.get("vram") or {}
            warn = ""
            if not result.get("vggt_cache_ok", True):
                warn = f"  !! VGGT cache = {result['vggt_cache_frames']} frames, expected "\
                       f"{min(result['step'], STATE['policy'].model.kv_cache_vggt.cache_size)}"
            hid_txt = "  h=%d" % result["hidden_dim"] if result.get("hidden") else ""
            sr = result.get("steer") or {}
            if sr.get("mode", "none") != "none":
                hid_txt += "  steer=%s|d|=%.1f/|h|=%.1f" % (sr["mode"], sr["norm_delta"], sr["norm_pre"])
            cap = result.get("capture") or {}
            if cap:
                hid_txt += "  cap" + ("" if cap.get("layout_ok") else "[LAYOUT!]")
                if cap.get("attn_check_max") is not None:
                    hid_txt += " attn_err=%.1e" % cap["attn_check_max"]
                if cap.get("turn_check") is not None:
                    hid_txt += " turn_err=%.1e" % cap["turn_check"]
            if result.get("hidden_error"):
                print("[server] !! %s" % result["hidden_error"])
            pr = result.get("probe") or {}
            if pr.get("ran") and not pr.get("cache_ok", True):
                print(f"[server] !! VGGT cache advanced during probes at step {step}: "
                      f"{pr.get('cache_frames_before')} -> {pr.get('cache_frames_after')}. "
                      f"Feature reuse did not engage; this episode is invalid.")
            probe_txt = ""
            if pr.get("ran"):
                probe_txt = "  probe=[" + " ".join(
                    f"{a['name'].split('_')[-1][:4]}:{a.get('action_name', a.get('error','ERR'))[:4]}"
                    for a in pr.get("arms", [])) + "]"
            print(f"[server] step={step:3d} -> {result['action_name']:<13} "
                  f"{result['latency_s']:.2f}s  cache={result.get('vggt_cache_frames')}f  "
                  f"vram={vram.get('alloc_gb')}/{vram.get('peak_gb')}GB"
                  + (" [FALLBACK]" if result["fallback"] else "") + warn + probe_txt + hid_txt)
            return self._send(200, result)

        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-history", type=int, default=8,
                    help="history frames shown to the LLM (eval used 8)")
    ap.add_argument("--vggt-recent-frames", type=int, default=None,
                    help="shrink the VGGT KV cache; each cached frame is ~159 MB, so "
                         "the default 8+48 frames is ~8.9 GB and 8+16 is ~3.8 GB, at "
                         "some loss of long-range spatial memory")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--lam", type=float, default=None,
                    help="override the geometry-fusion weight (checkpoint default 0.2). "
                         "--lam 0 ablates the VGGT contribution without changing anything "
                         "else, which measures what the geometry tower is worth.")
    args = ap.parse_args()

    print("[server] loading FutureNav ...")
    STATE["policy"] = FutureNavPolicy(
        args.checkpoint, device=args.device, max_history_frames=args.num_history,
        vggt_recent_frames=args.vggt_recent_frames, attn_implementation=args.attn,
        lam=args.lam,
    )
    v = STATE["policy"].vram()
    cache = STATE["policy"].model.kv_cache_vggt
    per_frame_gb = 24 * 2 * 16 * 1615 * 64 * 2 / 1e9
    print(f"[server] weights resident: {v.get('alloc_gb')} GB")
    print(f"[server] VGGT cache budget: {cache.start_size}+{cache.recent_size} frames "
          f"x {per_frame_gb * 1000:.0f} MB = {per_frame_gb * cache.cache_size:.2f} GB at steady state")
    print(f"[server] projected peak: ~{v.get('alloc_gb', 0) + per_frame_gb * cache.cache_size + 1.0:.1f} GB")
    print(f"[server] ready on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
