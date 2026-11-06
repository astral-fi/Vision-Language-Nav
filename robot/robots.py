"""Robot profiles: what differs between the robots this client can drive.

A profile fixes three things -- the topic layout under its namespace, how a
turn is driven (an Ackermann arc or a differential pivot), and whether the
primitives may close the loop on odometry and gate on the lidar at all. Motion
defaults come with it; any of them can still be overridden on the command line.

    jetracer  Ackermann car behind /bot1. Closed loop on wheel odometry, lidar
              obstacle gate. Unchanged from before profiles existed.
    waffle    TurtleBot3 Waffle Pi behind /tb3. Strictly open loop: odometry
              and lidar are never subscribed, every primitive runs on the clock,
              and a turn is an in-place pivot -- which is exactly the Habitat
              turn FutureNav was trained on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    name: str
    namespace: str
    camera: str                       # topic suffix under the namespace
    odom: str | None                  # None: never subscribed (open loop)
    scan: str | None                  # None: never subscribed (no obstacle gate)
    drive: str                        # 'ackermann' or 'differential'
    motion: dict = field(default_factory=dict)   # MotionParams defaults
    calib: str | None = None          # camera calibration, relative to this folder
    color: str | None = None          # optional devbox-side color correction
    # Differential-drive limits, from the robot's own configuration. None for
    # robots whose limits this client does not check.
    wheel_separation: float | None = None        # m
    max_wheel_speed: float | None = None         # m/s at the rim

    @property
    def calib_path(self) -> str | None:
        if self.calib is None:
            return None
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), self.calib)

    @property
    def color_path(self) -> str | None:
        if self.color is None:
            return None
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), self.color)

    @property
    def open_loop(self) -> bool:
        return self.odom is None and self.scan is None

    @property
    def max_turn_rate(self) -> float | None:
        """Pivoting in place drives each wheel at w * separation / 2, so the
        wheel speed limit sets the turn rate limit."""
        if self.wheel_separation is None or self.max_wheel_speed is None:
            return None
        return self.max_wheel_speed / (self.wheel_separation / 2.0)

    def check(self, params) -> None:
        """Refuse a command the robot would clip. In open loop a clipped speed
        is not an error anyone sees -- the step simply comes out short -- so it
        has to be caught before driving, not after."""
        if self.max_wheel_speed is not None and params.forward_speed > self.max_wheel_speed:
            raise ValueError(f'{self.name}: forward speed {params.forward_speed} m/s exceeds '
                             f'the {self.max_wheel_speed} m/s wheel limit')
        limit = self.max_turn_rate
        if self.drive == 'differential' and limit is not None and params.turn_rate > limit:
            raise ValueError(f'{self.name}: turn rate {params.turn_rate} rad/s exceeds '
                             f'{limit:.2f} rad/s (wheel limit / half the wheel separation)')


PROFILES = {
    'jetracer': Profile(
        name='jetracer',
        namespace='bot1',
        camera='csi_cam_0/image_raw/compressed',
        odom='odom',
        scan='scan_filtered',
        drive='ackermann',
        calib='calib/jetracer_fisheye.json',
        color='calib/jetracer_color.json',
    ),
    'waffle': Profile(
        name='waffle',
        namespace='tb3',
        camera='flir_camera/image_raw/compressed',
        odom=None,
        scan=None,
        drive='differential',
        motion={
            'forward_speed': 0.15,    # m/s, well inside the 0.26 m/s limit
            'turn_rate': 0.5,         # rad/s; 15 deg takes 0.52 s
            # Dynamixel profile_acceleration is 0 in waffle_pi.yaml, i.e.
            # unlimited: the wheels step to speed and stop without a ramp, so
            # there is no measured coast to subtract (the JetRacer's 0.03 m was
            # measured on the JetRacer and does not transfer).
            'coast_m': 0.0,
        },
        # The FLIR's own calibration (FutureNav/calib_camera.yaml on pi0, the
        # same file the camera driver loads), so frames are undistorted into
        # FutureNav's 640x480 / 79 deg view rather than resized raw.
        calib='calib/flir_waffle.yaml',
        # turtlebot3_bringup/param/humble/waffle_pi.yaml (wheels.separation)
        # and turtlebot3_teleop WAFFLE_MAX_LIN_VEL, as installed on the robot.
        wheel_separation=0.287,
        max_wheel_speed=0.26,
    ),
}


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(f'unknown robot {name!r}; choose from {", ".join(PROFILES)}')
