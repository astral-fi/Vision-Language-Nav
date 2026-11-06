"""
On-board JetRacer client for FutureNav.

Runs on the Jetson (Nano / Orin Nano). Does no neural-network work: it captures
a frame, rectifies it into a Habitat-equivalent view, asks the off-board server
for one discrete action, executes it as an Ackermann maneuver, and repeats.

  python3 deploy/jetracer_client.py \
      --server http://192.168.1.20:8642 \
      --instruction "walk out of the bedroom and stop next to the kitchen table" \
      --calib deploy/fisheye_calib.json --motion-calib deploy/motion_calib.json

Safety model, in order of precedence:
  1. hardware watchdog   -- throttle is zeroed if the loop stalls or dies
  2. forward range gate  -- MOVE_FORWARD aborts mid-motion if clearance < 0.25 m
  3. step budget         -- hard cap, mirroring EVAL.EARLY_STOP_STEPS = 400
  4. rotation-stall gate -- mirrors EVAL.EARLY_STOP_ROTATION = 50; the policy
                            gets stuck spinning and must be cut off
  5. the policy's own STOP
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import signal
import sys
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image

from camera_adapter import build_adapter
from motion import JetRacerMotion, MotionCalibration

EARLY_STOP_STEPS = 400        # EVAL.EARLY_STOP_STEPS
EARLY_STOP_ROTATION = 50      # EVAL.EARLY_STOP_ROTATION


def post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def open_camera(width: int, height: int, fps: int, device: int = 0):
    """CSI camera via jetcam if available, else any V4L2 device."""
    try:
        from jetcam.csi_camera import CSICamera
        cam = CSICamera(width=width, height=height, capture_fps=fps)
        cam.running = False
        return ("jetcam", cam)
    except Exception as exc:
        print(f"[client] jetcam unavailable ({exc}); falling back to cv2.VideoCapture")
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            raise RuntimeError("no camera")
        return ("cv2", cap)


def grab(cam_handle) -> np.ndarray:
    kind, cam = cam_handle
    if kind == "jetcam":
        return cam.read()                      # BGR
    for _ in range(3):                         # flush stale V4L2 buffers
        ok, frame = cam.read()
    if not ok:
        raise RuntimeError("camera read failed")
    return frame


def open_car(dry_run: bool):
    if dry_run:
        return None
    from jetracer.nvidia_racecar import NvidiaRacecar
    return NvidiaRacecar()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--calib", default="deploy/fisheye_calib.json")
    ap.add_argument("--motion-calib", default="deploy/motion_calib.json")
    ap.add_argument("--mount-pitch", type=float, default=0.0,
                    help="measured camera nose-down angle in degrees; corrected in the remap")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=EARLY_STOP_STEPS)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--dry-run", action="store_true", help="no motors; log actions only")
    ap.add_argument("--save-dir", default=None, help="dump the frames actually sent")
    args = ap.parse_args()

    session = args.session or f"jetracer-{int(time.time())}"
    adapter = build_adapter(args.calib, mount_pitch_deg=args.mount_pitch)
    cam = open_camera(args.cam_width, args.cam_height, args.cam_fps)
    car = open_car(args.dry_run)
    motion = JetRacerMotion(car, MotionCalibration.load(args.motion_calib),
                            dry_run=args.dry_run)

    def emergency_stop(*_):
        try:
            motion.stop()
        finally:
            print("\n[client] emergency stop")
            sys.exit(1)

    atexit.register(lambda: motion.stop())
    signal.signal(signal.SIGINT, emergency_stop)
    signal.signal(signal.SIGTERM, emergency_stop)

    post(f"{args.server}/reset", {"session": session, "instruction": args.instruction})
    print(f"[client] session={session}\n[client] instruction={args.instruction!r}")

    consecutive_turns = 0
    for step in range(args.max_steps):
        frame_bgr = adapter(grab(cam))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if args.save_dir:
            cv2.imwrite(f"{args.save_dir}/step_{step:04d}.jpg", frame_bgr)

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=args.jpeg_quality)
        payload = {"session": session, "step": step,
                   "image": base64.b64encode(buf.getvalue()).decode()}

        t0 = time.time()
        try:
            resp = post(f"{args.server}/step", payload)
        except Exception as exc:
            motion.stop()
            print(f"[client] server error at step {step}: {exc}")
            break
        rtt = time.time() - t0
        action = resp["action_name"]
        print(f"[client] step {step:3d} {action:<13} rtt={rtt:.2f}s "
              f"infer={resp.get('latency_s')}s" + ("  [parse-fallback]" if resp.get("fallback") else ""))

        if action == "STOP":
            motion.stop()
            print("[client] policy issued STOP")
            break
        if action == "MOVE_FORWARD":
            consecutive_turns = 0
            info = motion.move_forward()
            if info.get("aborted") == "obstacle":
                print("[client] forward aborted on range gate; recovering")
                motion.recover_from_collision()
        else:
            consecutive_turns += 1
            if consecutive_turns > EARLY_STOP_ROTATION:
                motion.stop()
                print("[client] rotation-stall guard tripped; ending episode")
                break
            motion.turn(+1 if action == "TURN_LEFT" else -1)
    else:
        print("[client] step budget exhausted")

    motion.stop()


if __name__ == "__main__":
    main()
