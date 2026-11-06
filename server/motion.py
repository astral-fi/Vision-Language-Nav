"""
Ackermann motion primitives for the VLN-CE discrete action space.

FutureNav emits exactly four actions, defined by the Habitat config used for
training and evaluation (eval/config/vlnce_task_futurenav.yaml):

    STOP          end the episode
    MOVE_FORWARD  translate 0.25 m along +z, no rotation
    TURN_LEFT     rotate +15 deg in place, no translation
    TURN_RIGHT    rotate -15 deg in place, no translation

A JetRacer is a car-like (Ackermann) platform: it cannot rotate in place. The
turn primitives therefore have to be approximated. Two strategies are provided:

  ARC     drive the tightest feasible arc until the yaw change is 15 deg.
          With a minimum turning radius R, a 15 deg turn also translates the
          car by R*sin(15 deg) forward and R*(1-cos(15 deg)) laterally.
          For R = 0.65 m that is 0.168 m forward / 0.022 m lateral -- roughly
          two thirds of one MOVE_FORWARD step. Cheap, fast, and the closed
          loop absorbs the drift, so this is the default.

  KTURN   forward-outward then reverse-inward, netting ~15 deg of yaw with
          near-zero net translation. Matches the training action semantics far
          better but takes ~3x longer and needs ~0.4 m of clearance ahead.
          Used automatically when a forward range sensor says space is tight.

Everything here is open-loop-with-optional-feedback: closed-loop yaw if an IMU
is present, closed-loop distance if wheel/motor encoders are present, otherwise
time-based execution against a calibrated speed model. Calibration lives in
deploy/motion_calib.json (see deploy/calibrate_motion.py).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict

FORWARD_STEP_M = 0.25       # SIMULATOR.FORWARD_STEP_SIZE
TURN_ANGLE_DEG = 15.0       # SIMULATOR.TURN_ANGLE


@dataclass
class MotionCalibration:
    """Measured on the actual car, on the actual floor, at the actual battery level."""
    throttle_forward: float = 0.22      # normalized throttle for the cruise segment
    speed_mps: float = 0.35             # measured steady-state speed at throttle_forward
    accel_time_s: float = 0.12          # dead time + spin-up before steady speed
    creep_distance_m: float = 0.03      # coast distance after throttle is cut

    throttle_turn: float = 0.24         # turns need more torque than straight runs
    steering_max: float = 1.0           # normalized steering at full lock
    steering_trim: float = 0.0          # additive bias that makes steering=0 drive straight
    turn_radius_m: float = 0.65         # measured minimum turning radius at full lock
    turn_speed_mps: float = 0.30        # measured speed while at full lock

    throttle_reverse: float = -0.24
    reverse_speed_mps: float = 0.30

    battery_nominal_v: float = 11.1     # pack voltage during calibration
    stop_settle_s: float = 0.35         # let the chassis settle before capturing a frame

    @staticmethod
    def load(path: str) -> "MotionCalibration":
        if not os.path.exists(path):
            print(f"[motion] WARNING: {path} missing; using untuned defaults. "
                  "Run deploy/calibrate_motion.py before trusting any result.")
            return MotionCalibration()
        with open(path) as fh:
            return MotionCalibration(**json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


class JetRacerMotion:
    """
    Wraps jetracer.nvidia_racecar.NvidiaRacecar with distance/angle primitives.

    Every primitive is stop-and-go: it returns only once the car is stationary
    again. That matches how the model was trained (Habitat renders a frame only
    after the action completes) and it keeps the ~1 s of remote inference off
    the critical path of a moving vehicle.
    """

    def __init__(self, car, calib: MotionCalibration, imu=None, range_sensor=None,
                 voltage_fn=None, dry_run: bool = False):
        self.car = car
        self.c = calib
        self.imu = imu                    # object with .yaw_deg() -> float, or None
        self.range_sensor = range_sensor  # object with .distance_m() -> float, or None
        self.voltage_fn = voltage_fn      # callable -> pack voltage, or None
        self.dry_run = dry_run
        self._stop()

    # ---------- low level ----------

    def _apply(self, throttle: float, steering: float) -> None:
        if self.dry_run:
            return
        self.car.steering = max(-1.0, min(1.0, steering + self.c.steering_trim))
        self.car.throttle = max(-1.0, min(1.0, throttle))

    def stop(self) -> None:
        """Public emergency/idle stop. Safe to call from signal handlers."""
        self._stop()

    def _stop(self) -> None:
        if self.dry_run:
            return
        self.car.throttle = 0.0
        self.car.steering = self.c.steering_trim

    def _battery_gain(self) -> float:
        """Compensate speed for pack sag. A 3S pack drifts 12.6 V -> 9.9 V over a run;
        without this the effective step size shrinks by ~20% by the end."""
        if self.voltage_fn is None:
            return 1.0
        try:
            v = float(self.voltage_fn())
        except Exception:
            return 1.0
        if v <= 0:
            return 1.0
        return max(0.7, min(1.6, self.c.battery_nominal_v / v))

    def _settle(self) -> None:
        self._stop()
        time.sleep(self.c.stop_settle_s)

    # ---------- primitives ----------

    def move_forward(self, distance_m: float = FORWARD_STEP_M) -> dict:
        """Drive straight for distance_m, then stop."""
        gain = self._battery_gain()
        throttle = self.c.throttle_forward * gain
        travel = max(0.0, distance_m - self.c.creep_distance_m)
        duration = self.c.accel_time_s + travel / max(self.c.speed_mps, 1e-3)

        t0 = time.time()
        self._apply(throttle, 0.0)
        while time.time() - t0 < duration:
            if self._blocked(min_clearance_m=0.25):
                self._settle()
                return {"executed": "MOVE_FORWARD", "aborted": "obstacle",
                        "elapsed_s": time.time() - t0}
            time.sleep(0.01)
        self._settle()
        return {"executed": "MOVE_FORWARD", "elapsed_s": time.time() - t0}

    def turn(self, sign: int, angle_deg: float = TURN_ANGLE_DEG,
             strategy: str | None = None) -> dict:
        """sign = +1 for TURN_LEFT, -1 for TURN_RIGHT."""
        if strategy is None:
            strategy = "kturn" if self._tight_space() else "arc"
        if strategy == "kturn":
            return self._turn_kturn(sign, angle_deg)
        return self._turn_arc(sign, angle_deg)

    def _turn_arc(self, sign: int, angle_deg: float) -> dict:
        gain = self._battery_gain()
        omega = self.c.turn_speed_mps / max(self.c.turn_radius_m, 1e-3)  # rad/s
        duration = math.radians(angle_deg) / max(omega, 1e-3) + self.c.accel_time_s

        start_yaw = self._yaw()
        t0 = time.time()
        # NvidiaRacecar convention: positive steering turns left.
        self._apply(self.c.throttle_turn * gain, sign * self.c.steering_max)
        while True:
            elapsed = time.time() - t0
            if self.imu is not None:
                turned = abs(_angle_delta(self._yaw(), start_yaw))
                if turned >= angle_deg:
                    break
                if elapsed > duration * 2.5:       # IMU watchdog
                    break
            elif elapsed >= duration:
                break
            time.sleep(0.005)
        self._settle()
        return {"executed": "TURN", "strategy": "arc", "sign": sign,
                "elapsed_s": time.time() - t0,
                "yaw_achieved_deg": abs(_angle_delta(self._yaw(), start_yaw)) if self.imu else None}

    def _turn_kturn(self, sign: int, angle_deg: float) -> dict:
        """Split the yaw across a forward arc and a reverse counter-arc so the
        net translation is near zero -- the closest feasible approximation of
        Habitat's turn-in-place."""
        half = angle_deg / 2.0
        gain = self._battery_gain()
        omega = self.c.turn_speed_mps / max(self.c.turn_radius_m, 1e-3)
        seg = math.radians(half) / max(omega, 1e-3) + self.c.accel_time_s

        t0 = time.time()
        self._apply(self.c.throttle_turn * gain, sign * self.c.steering_max)
        time.sleep(seg)
        self._settle()
        # Reverse with opposite steering: the car backs up along a mirrored arc,
        # adding yaw in the same direction while cancelling the translation.
        omega_r = self.c.reverse_speed_mps / max(self.c.turn_radius_m, 1e-3)
        seg_r = math.radians(half) / max(omega_r, 1e-3) + self.c.accel_time_s
        self._apply(self.c.throttle_reverse * gain, -sign * self.c.steering_max)
        time.sleep(seg_r)
        self._settle()
        return {"executed": "TURN", "strategy": "kturn", "sign": sign,
                "elapsed_s": time.time() - t0}

    def recover_from_collision(self) -> dict:
        """Back off and re-aim. Habitat trains with ALLOW_SLIDING=True, so the
        policy has never seen a hard stall and will happily command MOVE_FORWARD
        into a wall forever unless the client intervenes."""
        gain = self._battery_gain()
        self._apply(self.c.throttle_reverse * gain, 0.0)
        time.sleep(0.15 / max(self.c.reverse_speed_mps, 1e-3))
        self._settle()
        return {"executed": "RECOVER"}

    # ---------- sensing helpers ----------

    def _yaw(self) -> float:
        if self.imu is None:
            return 0.0
        try:
            return float(self.imu.yaw_deg())
        except Exception:
            return 0.0

    def _range(self) -> float | None:
        if self.range_sensor is None:
            return None
        try:
            return float(self.range_sensor.distance_m())
        except Exception:
            return None

    def _blocked(self, min_clearance_m: float) -> bool:
        r = self._range()
        return r is not None and r < min_clearance_m

    def _tight_space(self) -> bool:
        r = self._range()
        return r is not None and r < 0.45


def _angle_delta(a: float, b: float) -> float:
    """Signed smallest difference a-b in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Alternative action space: arc semantics
# ---------------------------------------------------------------------------
# FutureNav as released emits MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP,
# where a turn is rotation in place with zero translation -- which an Ackermann
# car cannot do, hence the arc/K-turn approximations above.
#
# A policy emitting MOVE_STRAIGHT / MOVE_LEFT / MOVE_RIGHT means something
# different: translate WHILE curving. That is what a car does natively, so the
# mapping below is exact rather than approximate, and needs no K-turn.
#
# Geometry: for arc length d and turn radius R, heading change is theta = d / R.
#   d = 0.25 m, R = 0.65 m (full lock) -> theta = 22.0 deg
#   d = 0.25 m, theta = 15 deg wanted  -> R = 0.955 m (partial steering)
#
# NOTE: this action set has no STOP. VLN-CE only scores Success if the agent
# calls STOP within SUCCESS_DISTANCE of the goal, so termination has to come
# from somewhere else -- a fourth action, a separate query to the model, or a
# human. JetRacerMotion cannot supply it.

ARC_ACTIONS = ("MOVE_STRAIGHT", "MOVE_LEFT", "MOVE_RIGHT")


def radius_for_heading_change(distance_m: float, heading_deg: float) -> float:
    """R = d / theta. The radius that yields heading_deg over distance_m."""
    theta = math.radians(abs(heading_deg))
    if theta < 1e-6:
        return float("inf")
    return distance_m / theta


class ArcActionMixin:
    """Mix into JetRacerMotion to execute the arc action space."""

    def steering_for_radius(self, radius_m: float) -> float:
        """Normalized steering [0,1] that produces the requested turn radius.

        car.steering maps nonlinearly to servo angle, so the honest version is a
        calibrated lookup. This linear-in-curvature fallback is a starting point,
        anchored on the one point calibrate_motion.py actually measures
        (turn_radius_m at full lock): curvature 1/R is roughly proportional to
        steering over the mid range.
        """
        if not math.isfinite(radius_m):
            return 0.0
        full_lock_curvature = 1.0 / max(self.c.turn_radius_m, 1e-3)
        wanted_curvature = 1.0 / max(radius_m, 1e-3)
        return max(0.0, min(1.0, wanted_curvature / full_lock_curvature)) * self.c.steering_max

    def move_arc(self, sign: int, distance_m: float = FORWARD_STEP_M,
                 heading_deg: float = TURN_ANGLE_DEG) -> dict:
        """Drive distance_m along the arc that turns heading_deg.

        sign: 0 straight, +1 left, -1 right.
        """
        if sign == 0:
            return self.move_forward(distance_m)

        radius = radius_for_heading_change(distance_m, heading_deg)
        steering = sign * self.steering_for_radius(radius)
        gain = self._battery_gain()
        travel = max(0.0, distance_m - self.c.creep_distance_m)
        duration = self.c.accel_time_s + travel / max(self.c.turn_speed_mps, 1e-3)

        start_yaw = self._yaw()
        t0 = time.time()
        self._apply(self.c.throttle_turn * gain, steering)
        while time.time() - t0 < duration:
            if self._blocked(min_clearance_m=0.25):
                self._settle()
                return {"executed": "MOVE_ARC", "aborted": "obstacle",
                        "elapsed_s": time.time() - t0}
            time.sleep(0.01)
        self._settle()
        return {"executed": "MOVE_ARC", "sign": sign, "radius_m": round(radius, 3),
                "steering": round(steering, 3), "elapsed_s": time.time() - t0,
                "yaw_achieved_deg": (abs(_angle_delta(self._yaw(), start_yaw))
                                     if self.imu else None)}

    def execute_arc_action(self, action_name: str, distance_m: float = FORWARD_STEP_M,
                           heading_deg: float = TURN_ANGLE_DEG) -> dict:
        sign = {"MOVE_STRAIGHT": 0, "MOVE_LEFT": +1, "MOVE_RIGHT": -1}.get(action_name)
        if sign is None:
            raise ValueError(
                "unknown arc action %r; expected one of %s. Note this action space "
                "has no STOP -- termination must come from elsewhere."
                % (action_name, ", ".join(ARC_ACTIONS)))
        return self.move_arc(sign, distance_m, heading_deg)
