"""Rectify a standard (plumb_bob) camera into FutureNav's Habitat view.

FutureNav's deploy/camera_adapter.py only reads its own fisheye JSON (4
coefficients). A ROS camera_info calibration -- what camera_calibration writes,
and what a camera driver loads -- uses the 5-coefficient plumb_bob model, so
it is undistorted here instead, into the same target FutureNav trains on: a
640x480 pinhole with a 79 deg horizontal field of view.
"""

from __future__ import annotations

import math
import json

import cv2
import numpy as np
import yaml

HABITAT_W, HABITAT_H = 640, 480     # SIMULATOR RGB_SENSOR WIDTH/HEIGHT
HABITAT_HFOV_DEG = 79.0             # SIMULATOR RGB_SENSOR HFOV


def habitat_intrinsics(width: int = HABITAT_W, height: int = HABITAT_H,
                       hfov_deg: float = HABITAT_HFOV_DEG) -> np.ndarray:
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def is_ros_calibration(path: str) -> bool:
    return path.endswith(('.yaml', '.yml'))


class PinholeViewAdapter:
    """One precomputed remap: raw frame -> Habitat-like 640x480 / 79 deg frame."""

    def __init__(self, path: str, mount_pitch_deg: float = 0.0):
        with open(path) as fh:
            calib = yaml.safe_load(fh)
        model = calib.get('distortion_model', 'plumb_bob')
        if model != 'plumb_bob':
            raise SystemExit(f'{path}: distortion_model {model!r}, expected plumb_bob')
        K = np.array(calib['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
        D = np.array(calib['distortion_coefficients']['data'], dtype=np.float64)
        self.size = (int(calib['image_width']), int(calib['image_height']))
        # Positive pitch = camera nose down; the virtual view is pitched back up.
        p = math.radians(-mount_pitch_deg)
        R = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, R, habitat_intrinsics(), (HABITAT_W, HABITAT_H), cv2.CV_16SC2)
        print(f'[view] {path}: {self.size[0]}x{self.size[1]} plumb_bob -> '
              f'{HABITAT_W}x{HABITAT_H} @ {HABITAT_HFOV_DEG:g} deg HFOV', flush=True)

    def __call__(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


class FisheyeViewAdapter:
    """Shared calibrated view for the preview and exact model observations."""
    def __init__(self, path: str, mount_pitch_deg: float = 0.0):
        with open(path) as stream:
            calibration = json.load(stream)
        self.size = (int(calibration['width']), int(calibration['height']))
        self.hfov_deg = float(calibration.get('virtual_hfov_deg', HABITAT_HFOV_DEG))
        pitch = math.radians(-mount_pitch_deg)
        rotation = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                             [0, math.sin(pitch), math.cos(pitch)]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            np.array(calibration['K'], np.float64), np.array(calibration['D'], np.float64),
            rotation, habitat_intrinsics(hfov_deg=self.hfov_deg),
            (HABITAT_W, HABITAT_H), cv2.CV_16SC2)
        print(f'[view] {path}: fisheye {self.size} -> 640x480 @ {self.hfov_deg:g} deg HFOV', flush=True)

    def __call__(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
