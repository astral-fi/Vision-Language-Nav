"""Every address and key expression this client touches, in one place.

Nothing is hard-wired to one robot: the router, the four topics and the model
server are settings, each with a CLI flag and an environment variable. --robot
picks a profile (robots.py) that supplies the namespace and topic layout; the
topic defaults are derived from the namespace, so a second robot of the same
kind needs only `--namespace bot2`, and an unusual layout can override any
topic outright.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

import robots


@dataclass
class Settings:
    # Robot profile: namespace, topic layout, drive kind, open or closed loop
    robot: str = 'jetracer'             # see robots.PROFILES

    # Transport
    router: str = '127.0.0.1:7447'      # 'ip', 'ip:port' or 'tcp/ip:port'
    namespace: str = ''                 # default: the profile's (bot1, tb3)
    camera_topic: str = ''              # default: <ns>/<profile camera>
    cmd_vel_topic: str = ''             # default: <ns>/cmd_vel
    odom_topic: str = ''                # default: <ns>/odom; '-' forces open loop
    scan_topic: str = ''                # default: <ns>/scan_filtered; '-' disables the gate
    connect_timeout: float = 5.0

    # Model server (FutureNav deploy/server.py, left exactly as it is)
    server: str = 'http://127.0.0.1:8642'
    request_timeout: float = 60.0

    # FutureNav checkout, imported read-only for its camera rectification
    futurenav_root: str = ''
    calib: str = ''                     # fisheye calibration json; empty = resize only
    mount_pitch: float = 0.0

    ENV = {'robot': 'ROBOT', 'router': 'ZENOH_ROUTER', 'namespace': 'ROBOT_NAMESPACE',
           'camera_topic': 'CAMERA_TOPIC', 'cmd_vel_topic': 'CMD_VEL_TOPIC',
           'odom_topic': 'ODOM_TOPIC', 'scan_topic': 'SCAN_TOPIC',
           'server': 'FUTURENAV_SERVER', 'futurenav_root': 'FUTURENAV_ROOT'}

    @property
    def profile(self) -> robots.Profile:
        return robots.get(self.robot)

    def resolve(self) -> 'Settings':
        """Fill the namespace and topic defaults from the profile. '-' means
        'no such topic'.

        An open-loop profile has no odometry or lidar topic, and naming one
        explicitly is refused rather than quietly honoured: on that robot they
        are meant to play no part in driving at all.
        """
        profile = self.profile
        if not self.namespace:
            self.namespace = profile.namespace
        prefix = f'{self.namespace.strip("/")}/' if self.namespace.strip('/') else ''
        for name, suffix in (('camera_topic', profile.camera),
                             ('cmd_vel_topic', 'cmd_vel'),
                             ('odom_topic', profile.odom),
                             ('scan_topic', profile.scan)):
            value = getattr(self, name)
            if suffix is None:
                if value.strip() not in ('', '-'):
                    raise SystemExit(f'--robot {profile.name} is open loop: '
                                     f'{name} must not be set (got {value!r})')
                setattr(self, name, '-')
            elif not value:
                setattr(self, name, f'{prefix}{suffix}')
        return self

    def topic(self, name: str) -> str | None:
        value = getattr(self, name)
        return None if value.strip() in ('-', '') else value

    def describe(self) -> str:
        return (f'robot={self.robot} router={self.router} camera={self.camera_topic} '
                f'cmd_vel={self.cmd_vel_topic} odom={self.topic("odom_topic")} '
                f'scan={self.topic("scan_topic")} server={self.server}')


def add_arguments(parser):
    """CLI flags for every setting; the default shown is the environment's."""
    defaults = Settings()
    for field in fields(Settings):
        env = Settings.ENV.get(field.name)
        default = os.environ.get(env) if env else None
        if default is None:
            default = getattr(defaults, field.name)
        elif field.type == 'float':
            default = float(default)
        flag = '--' + field.name.replace('_', '-')
        kind = float if field.type == 'float' else str
        parser.add_argument(flag, type=kind, default=default,
                            help=f'(default: {default!r})')
    return parser


def from_args(args) -> Settings:
    return Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)}).resolve()
