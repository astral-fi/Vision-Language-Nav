"""
Fisheye intrinsics calibration for the JetRacer CSI camera.

Print a checkerboard (a 9x6 inner-corner board on rigid A3 works), then:

  python3 deploy/calibrate_camera.py --capture --out deploy/fisheye_calib.json

Press SPACE to keep a view, q to finish. Collect 20-30 views covering the whole
frame, especially the corners -- the 160 deg lens distorts most there, and the
corners are exactly what the 79 deg virtual view crops away, so a bad corner fit
warps the useful centre too.

Then measure the mounting pitch (nose-down angle) and pass it to the client via
--mount-pitch; it is folded into the rectification rotation.
"""

from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np

from camera_adapter import FisheyeCalibration


def capture(board, square_m, device, width, height, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, board, corners, found)
        cv2.putText(view, f"kept={n}  SPACE=keep  q=quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("calib", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            cv2.imwrite(f"{out_dir}/calib_{n:03d}.png", frame)
            n += 1
    cap.release()
    cv2.destroyAllWindows()
    return n


def solve(images_glob, board, square_m, out_path):
    objp = np.zeros((1, board[0] * board[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_m

    objpoints, imgpoints, shape = [], [], None
    for path in sorted(glob.glob(images_glob)):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"  skip (no board): {path}")
            continue
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(objpoints) < 8:
        raise SystemExit(f"only {len(objpoints)} usable views; need >= 8 (20+ preferred)")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, shape, K, D,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6),
    )
    print(f"views={len(objpoints)}  rms={rms:.4f} px  (aim for < 0.6)")
    print("K =\n", K, "\nD =", D.ravel())
    FisheyeCalibration(K=K, D=D, width=shape[0], height=shape[1]).save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--images", default="deploy/calib_images/*.png")
    ap.add_argument("--out", default="deploy/fisheye_calib.json")
    ap.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    board = tuple(int(v) for v in a.board.lower().split("x"))
    if a.capture:
        n = capture(board, a.square, a.device, a.width, a.height, "deploy/calib_images")
        print(f"captured {n} views")
    solve(a.images, board, a.square, a.out)
