"""
Sensor-domain adapter: JetRacer fisheye CSI camera -> Habitat-like pinhole view.

FutureNav was trained on Habitat RGB_SENSOR frames:
    WIDTH 640, HEIGHT 480, HFOV 79 deg, POSITION [0, 1.25, 0], no pitch.

The stock JetRacer camera (IMX219-160) is a ~160 deg fisheye mounted ~0.1-0.15 m
above the floor, usually tilted down. Feeding those frames to the model raw is
the single largest avoidable source of accuracy loss, because both the Qwen3-VL
tower and the VGGT geometry tower are sensitive to projection and to apparent
scale.

This module rectifies the fisheye into a virtual 640x480 / 79 deg HFOV pinhole
camera, and can bake a mounting-pitch (and roll/yaw) correction into the same
remap so a physically tilted camera still produces a level, forward-looking
image.

Camera height cannot be fixed in software. See deploy/README.md.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

HABITAT_W = 640
HABITAT_H = 480
HABITAT_HFOV_DEG = 79.0


def habitat_intrinsics(width: int = HABITAT_W, height: int = HABITAT_H,
                       hfov_deg: float = HABITAT_HFOV_DEG) -> np.ndarray:
    """Pinhole K matching Habitat's RGB sensor (square pixels, fx == fy)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Camera-frame rotation (x right, y down, z forward). Positive pitch = nose down."""
    p, y, r = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class FisheyeCalibration:
    """OpenCV fisheye model produced by deploy/calibrate_camera.py."""
    K: np.ndarray          # 3x3 fisheye intrinsics
    D: np.ndarray          # 4x1 fisheye distortion
    width: int             # calibration image width
    height: int            # calibration image height

    @staticmethod
    def load(path: str) -> "FisheyeCalibration":
        with open(path) as fh:
            blob = json.load(fh)
        return FisheyeCalibration(
            K=np.array(blob["K"], dtype=np.float64),
            D=np.array(blob["D"], dtype=np.float64).reshape(4, 1),
            width=int(blob["width"]),
            height=int(blob["height"]),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"K": self.K.tolist(), "D": self.D.reshape(-1).tolist(),
                       "width": self.width, "height": self.height}, fh, indent=2)


class HabitatViewAdapter:
    """
    Precomputes a single remap that turns a raw fisheye frame into a
    Habitat-equivalent 640x480 / 79 deg pinhole frame.

    Cost on a Jetson Nano: ~4 ms per frame at 640x480 with INTER_LINEAR, so it
    is cheap enough to run on the car rather than on the inference server.
    """

    def __init__(self, calib: FisheyeCalibration, mount_pitch_deg: float = 0.0,
                 mount_yaw_deg: float = 0.0, mount_roll_deg: float = 0.0,
                 out_size: tuple[int, int] = (HABITAT_W, HABITAT_H),
                 hfov_deg: float = HABITAT_HFOV_DEG):
        self.out_w, self.out_h = out_size
        self.K_target = habitat_intrinsics(self.out_w, self.out_h, hfov_deg)
        # Un-rotate the mount: a nose-down camera needs the virtual view pitched up.
        R = _rotation(-mount_pitch_deg, -mount_yaw_deg, -mount_roll_deg)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            calib.K, calib.D, R, self.K_target,
            (self.out_w, self.out_h), cv2.CV_16SC2,
        )
        self._src_size = (calib.width, calib.height)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Raw BGR fisheye frame -> Habitat-like BGR frame (640x480)."""
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self._src_size:
            frame_bgr = cv2.resize(frame_bgr, self._src_size, interpolation=cv2.INTER_AREA)
        return cv2.remap(frame_bgr, self.map1, self.map2,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


class PassthroughAdapter:
    """Fallback when no calibration exists: plain letterbox-free resize to 640x480.

    This is measurably worse than rectifying. Use it only to smoke-test the
    pipeline, never to judge navigation quality.
    """

    def __init__(self, out_size: tuple[int, int] = (HABITAT_W, HABITAT_H)):
        self.out_w, self.out_h = out_size

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.resize(frame_bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)


def build_adapter(calib_path: str | None, mount_pitch_deg: float = 0.0,
                  mount_yaw_deg: float = 0.0, mount_roll_deg: float = 0.0):
    if calib_path and os.path.exists(calib_path):
        return HabitatViewAdapter(FisheyeCalibration.load(calib_path),
                                  mount_pitch_deg, mount_yaw_deg, mount_roll_deg)
    print("[camera_adapter] WARNING: no fisheye calibration found; using naive resize. "
          "Navigation quality will be degraded.")
    return PassthroughAdapter()
