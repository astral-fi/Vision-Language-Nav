"""The four VLN-CE primitives as Twist commands on a real robot.

FutureNav emits STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT, defined by the
Habitat config it was trained with: forward translates 0.25 m, each turn
rotates 15 deg *in place*. How a turn is driven depends on the drive:

  ackermann     A JetRacer cannot rotate in place, so a turn is driven as the
                tightest feasible arc -- the same ARC strategy, radius and
                reasoning as FutureNav's own deploy/motion.py, which this
                mirrors without importing or modifying it.
  differential  A TurtleBot3 pivots in place (linear 0, angular w), which is
                the Habitat turn exactly.

What changes from deploy/motion.py is only how the command reaches the wheels
and how the result is measured: a Twist on cmd_vel instead of the NvidiaRacecar
throttle API, and either odometry or the clock instead of a time-and-throttle
speed model. Odometry closes the loop on distance and yaw; without it (an
open-loop profile, or a missing topic) each primitive runs for the time its
commanded speed says it needs.

Every primitive is stop-and-go: it returns only once the car is stationary,
which is what the policy was trained on and what keeps ~1 s of remote
inference off the critical path of a moving vehicle.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from transport import forward_clearance

FORWARD_STEP_M = 0.25        # SIMULATOR.FORWARD_STEP_SIZE
TURN_ANGLE_DEG = 15.0        # SIMULATOR.TURN_ANGLE


@dataclass
class _Progress:
    """How a primitive knows it is done, and in what unit it reports.

    Odometry measures metres or degrees; the clock fallback measures only
    seconds, and saying so keeps a duration from being read as a distance.
    """
    reached: object                  # () -> (done: bool, value: float)
    feedback: str                    # 'odom' or 'clock'
    unit: str                        # 'm', 'deg' or 's'
    remaining: object = None         # () -> s left, clock only: lets the loop
                                     # stop on time instead of up to a tick late


@dataclass
class MotionParams:
    """Defaults are deliberately slow and untuned. Measure forward_speed on the
    actual floor before trusting a step size (drive 10 steps, divide)."""
    drive: str = 'ackermann'         # or 'differential'; set by the robot profile
    forward_speed: float = 0.15      # m/s commanded on cmd_vel.linear.x
    turn_speed: float = 0.15         # m/s while arcing (ackermann)
    turn_radius: float = 0.65        # m, minimum turning radius at full lock (ackermann)
    turn_rate: float = 0.5           # rad/s while pivoting in place (differential)
    forward_step: float = FORWARD_STEP_M
    turn_angle_deg: float = TURN_ANGLE_DEG
    rate_hz: float = 20.0            # cmd_vel republish rate
    obstacle_stop: float = 0.25      # m of forward clearance; 0 disables the gate
    scan_max_age: float = 1.0        # s before a scan counts as missing
    odom_max_age: float = 0.5        # s before odometry counts as missing
    yaw_feedback: bool = True        # False: close turns on arc length instead
    coast_m: float = 0.03            # carried after the stop command; measured
                                     # 2026-10-15 as 0.11 m against an 0.081 m
                                     # target, and it is a fixed distance, not a
                                     # fraction, so it matters most on turns
    settle_s: float = 0.35           # let the chassis settle before the next frame
    timeout_margin: float = 2.5      # allowed overrun on the predicted duration
    recover_distance: float = 0.15   # m of reverse after a blocked forward


class ZenohMotion:
    """Mirrors deploy/motion.py's surface -- move_forward / turn / stop /
    recover_from_collision -- so the episode loop keeps the same shape."""

    def __init__(self, drive, odom=None, scan=None, params: MotionParams | None = None,
                 dry_run: bool = False):
        self.drive = drive            # TwistPublisher
        self.odom = odom              # Latest of (x, y, yaw), or None
        self.scan = scan              # Latest of LaserScan, or None
        self.p = params or MotionParams()
        self.dry_run = dry_run
        self._warned_open_loop = False
        self.guard = None

    # ---------- feedback ----------

    def _pose(self):
        return None if self.odom is None else self.odom.get(self.p.odom_max_age)

    def _blocked(self, backward: bool = False) -> str | None:
        """None when clear. A configured-but-silent lidar blocks motion on
        purpose: a safety gate that cannot see is not a gate."""
        if self.p.obstacle_stop <= 0 or self.scan is None:
            return None
        scan = self.scan.get(self.p.scan_max_age)
        if scan is None:
            return 'no_scan'
        if forward_clearance(scan, backward=backward) < self.p.obstacle_stop:
            return 'obstacle'
        return None

    # ---------- execution ----------

    def _run(self, linear: float, angular: float, progress: _Progress,
             timeout_s: float, gate: bool, backward: bool = False) -> dict:
        """Hold a velocity until the primitive is done.

        cmd_vel is republished every tick because the robot's own watchdog
        expects a stream, not a single command.
        """
        blocked = self._blocked(backward) if gate else None
        if blocked:
            self.drive.stop()
            return {'aborted': blocked, 'measured': 0.0, 'unit': progress.unit,
                    'feedback': progress.feedback, 'elapsed': 0.0}
        start, period = time.monotonic(), 1.0 / self.p.rate_hz
        result = {'aborted': None}
        while True:
            reason = self.guard() if self.guard is not None else None
            if reason:
                result['aborted'] = reason
                break
            elapsed = time.monotonic() - start
            done, _ = progress.reached()
            if done:
                break
            if elapsed > timeout_s:
                result['aborted'] = 'timeout'
                break
            if gate:
                blocked = self._blocked(backward)
                if blocked:
                    result['aborted'] = blocked
                    break
            if not self.dry_run:
                self.drive.send(linear, angular)
            left = progress.remaining() if progress.remaining else period
            time.sleep(max(0.0, min(period, left)))
        # The clock's figure is how long the command was held, so read it now;
        # read after the stop and settle it would count time spent standing still.
        held = progress.reached()[1] if progress.feedback == 'clock' else None
        self.drive.stop()
        time.sleep(self.p.settle_s)
        # Odometry is read after settling on purpose: it includes the coast.
        measured = held if held is not None else progress.reached()[1]
        result.update(measured=measured, unit=progress.unit,
                      feedback=progress.feedback, elapsed=time.monotonic() - start)
        return result

    def _distance_reached(self, target_m: float):
        """Closed loop on odometry when it is there, a clock when it is not.

        In a dry run the wheels never turn, so odometry could never report the
        target met; the clock stands in and each primitive takes the time it
        would have taken.
        """
        # Cut the command short by the coast distance rather than overshooting
        # every step; the floor is 0.02 m of travel either way.
        target_m = max(target_m - self.p.coast_m, 0.02)
        pose = None if self.dry_run else self._pose()
        if pose is None:
            return self._clock_reached(target_m / max(self.p.forward_speed, 1e-3))
        x0, y0, _ = pose
        def reached():
            now = self._pose()
            if now is None:                      # odometry died mid-primitive
                return True, math.nan
            travelled = math.hypot(now[0] - x0, now[1] - y0)
            return travelled >= target_m, travelled
        return _Progress(reached, 'odom', 'm')

    def _turn_seconds(self, target_rad: float) -> float:
        """How long a turn takes at the commanded speed: angle over rate for a
        pivot, arc length over speed for an arc."""
        if self.p.drive == 'differential':
            return abs(target_rad) / max(self.p.turn_rate, 1e-3)
        return abs(target_rad) * self.p.turn_radius / max(self.p.turn_speed, 1e-3)

    def _yaw_reached(self, target_rad: float):
        pose = None if self.dry_run else self._pose()
        if pose is None:
            return self._clock_reached(self._turn_seconds(target_rad))
        yaw0 = pose[2]
        def reached():
            now = self._pose()
            if now is None:
                return True, math.nan
            turned = abs(math.atan2(math.sin(now[2] - yaw0), math.cos(now[2] - yaw0)))
            return turned >= abs(target_rad), math.degrees(turned)
        return _Progress(reached, 'odom', 'deg')

    def _clock_reached(self, duration_s: float):
        if not self._warned_open_loop:
            reason = ('dry run' if self.dry_run
                      else 'no odometry' if self.odom is None
                      else 'yaw feedback off')
            print(f'[motion] {reason}; primitives run open-loop on the clock',
                  flush=True)
            self._warned_open_loop = True
        start = time.monotonic()
        def reached():
            elapsed = time.monotonic() - start
            return elapsed >= duration_s, elapsed
        def remaining():
            return duration_s - (time.monotonic() - start)
        return _Progress(reached, 'clock', 's', remaining)

    # ---------- primitives ----------

    def move_forward(self, distance_m: float | None = None) -> dict:
        distance = self.p.forward_step if distance_m is None else distance_m
        timeout = distance / max(self.p.forward_speed, 1e-3) * self.p.timeout_margin + 1.0
        info = self._run(self.p.forward_speed, 0.0,
                         self._distance_reached(distance), timeout, gate=True)
        info['action'] = 'MOVE_FORWARD'
        return info

    def turn(self, sign: int, angle_deg: float | None = None) -> dict:
        """sign +1 left, -1 right."""
        angle = math.radians(self.p.turn_angle_deg if angle_deg is None else angle_deg)
        if self.p.drive == 'differential':
            return self._pivot(sign, angle)
        return self._arc(sign, angle)

    def _pivot(self, sign: int, angle: float) -> dict:
        """Rotate in place, the Habitat turn exactly. Not range-gated: a pivot
        sweeps the robot's own footprint and travels nowhere."""
        rate = self.p.turn_rate
        progress = (self._yaw_reached(angle) if self.p.yaw_feedback
                    else self._clock_reached(self._turn_seconds(angle)))
        info = self._run(0.0, sign * rate, progress,
                         self._turn_seconds(angle) * self.p.timeout_margin + 1.0, gate=False)
        info['action'] = 'TURN_LEFT' if sign > 0 else 'TURN_RIGHT'
        return info

    def _arc(self, sign: int, angle: float) -> dict:
        """The arc also carries the car about R*sin(angle) forward -- 0.17 m at
        15 deg -- which the policy cannot sense; that is the known sim-to-real
        penalty, not a bug here."""
        speed, arc = self.p.turn_speed, angle * self.p.turn_radius
        # Without a gyro, watch the arc length instead of the heading: wheel
        # odometry is sound on this robot even though its yaw is not, and
        # distance self-corrects when the car runs slower than commanded.
        # Measured 2026-10-15: bot1/imu streams at 49 Hz with every field frozen
        # (acc_z = 0, so gravity is missing) and odom yaw is an EKF over that
        # constant, so a turn watched for yaw would never end.
        progress = (self._yaw_reached(angle) if self.p.yaw_feedback
                    else self._distance_reached(arc))
        info = self._run(speed, sign * speed / self.p.turn_radius, progress,
                         arc / max(speed, 1e-3) * self.p.timeout_margin + 1.0, gate=True)
        info['action'] = 'TURN_LEFT' if sign > 0 else 'TURN_RIGHT'
        return info

    def recover_from_collision(self) -> dict:
        """Back away from whatever the range gate found, so the next forward
        has somewhere to go."""
        distance = self.p.recover_distance
        timeout = distance / max(self.p.forward_speed, 1e-3) * self.p.timeout_margin + 1.0
        info = self._run(-self.p.forward_speed, 0.0,
                         self._distance_reached(distance), timeout,
                         gate=True, backward=True)
        info['action'] = 'RECOVER'
        return info

    def stop(self) -> None:
        """Safe to call from a signal handler or atexit."""
        self.drive.stop()
