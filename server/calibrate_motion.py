"""
Motion calibration for the JetRacer, run on the car.

The policy assumes each MOVE_FORWARD advances exactly 0.25 m and each turn is
exactly 15 deg. Nothing on a stock JetRacer measures either, so both are
open-loop and must be tuned against a tape measure on the surface you will
actually drive on. Carpet vs hardwood changes the answer by 30-40%, and so does
battery charge.

  python3 deploy/calibrate_motion.py --stage speed
  python3 deploy/calibrate_motion.py --stage radius
  python3 deploy/calibrate_motion.py --stage verify

Redo this whenever you change surface, tyres, battery, or payload.
"""

from __future__ import annotations

import argparse
import math
import time

from motion import JetRacerMotion, MotionCalibration, FORWARD_STEP_M, TURN_ANGLE_DEG

CALIB_PATH = "deploy/motion_calib.json"


def car():
    from jetracer.nvidia_racecar import NvidiaRacecar
    return NvidiaRacecar()


def stage_speed(m: JetRacerMotion, c: MotionCalibration):
    """Drive a fixed duration at the cruise throttle, measure the distance."""
    duration = 2.0
    input(f"Clear 2 m of straight floor. Throttle={c.throttle_forward}, "
          f"{duration}s run. ENTER to go: ")
    t0 = time.time()
    m._apply(c.throttle_forward, 0.0)
    time.sleep(duration)
    m.stop()
    dist = float(input("Measured distance travelled (m): "))
    creep = float(input("Coast distance after throttle cut (m), 0 if unsure: ") or 0)
    c.speed_mps = max(dist - creep, 1e-3) / max(duration - c.accel_time_s, 1e-3)
    c.creep_distance_m = creep
    print(f"  speed_mps = {c.speed_mps:.3f}")
    if c.speed_mps > 0.6:
        print("  NOTE: > 0.6 m/s makes a 0.25 m step under 0.5 s, which is hard to "
              "time accurately. Lower throttle_forward and rerun.")


def stage_radius(m: JetRacerMotion, c: MotionCalibration):
    """Drive a full-lock circle, measure its diameter and lap time."""
    input("Clear a 2 m circle. Full left lock, one full lap. ENTER to go: ")
    t0 = time.time()
    m._apply(c.throttle_turn, c.steering_max)
    input("Press ENTER the instant the car completes one full circle: ")
    lap = time.time() - t0
    m.stop()
    diameter = float(input("Measured circle diameter (m): "))
    c.turn_radius_m = diameter / 2.0
    c.turn_speed_mps = (2 * math.pi * c.turn_radius_m) / max(lap, 1e-3)
    print(f"  turn_radius_m = {c.turn_radius_m:.3f}  turn_speed_mps = {c.turn_speed_mps:.3f}")
    print(f"  one {TURN_ANGLE_DEG:.0f} deg arc turn will also translate "
          f"{c.turn_radius_m * math.sin(math.radians(TURN_ANGLE_DEG)):.3f} m forward and "
          f"{c.turn_radius_m * (1 - math.cos(math.radians(TURN_ANGLE_DEG))):.3f} m sideways")


def stage_verify(m: JetRacerMotion, c: MotionCalibration):
    """Ten forward steps should cover 2.5 m; twenty-four turns should be a circle."""
    input(f"Verify: 10 x MOVE_FORWARD should travel {10 * FORWARD_STEP_M:.2f} m. ENTER: ")
    for _ in range(10):
        m.move_forward()
    err = float(input("Measured distance (m): ")) - 10 * FORWARD_STEP_M
    print(f"  forward error = {err:+.3f} m over 10 steps ({err / 10 * 100:+.1f} cm/step)")

    n = int(round(360 / TURN_ANGLE_DEG))
    input(f"Verify: {n} x TURN_LEFT should end facing the original heading. ENTER: ")
    for _ in range(n):
        m.turn(+1)
    print("  compare final heading with the start mark; adjust turn_speed_mps to correct.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["speed", "radius", "verify"])
    ap.add_argument("--calib", default=CALIB_PATH)
    a = ap.parse_args()

    c = MotionCalibration.load(a.calib)
    m = JetRacerMotion(car(), c)
    {"speed": stage_speed, "radius": stage_radius, "verify": stage_verify}[a.stage](m, c)
    m.stop()
    c.save(a.calib)
    print(f"saved -> {a.calib}")
