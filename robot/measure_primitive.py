"""Drive ONE primitive and report what the robot actually did.

The policy is told a turn is 15 deg and a forward step is 0.25 m. If the car
does something else, every later decision rests on a false premise -- so
measure one primitive on the real floor before reading anything into a run.

    python measure_primitive.py --router 127.0.0.1:17447 --primitive turn_left
    python measure_primitive.py --router 127.0.0.1:17447 --primitive forward \
        --forward-speed 0.2

Reports intended vs measured yaw and distance, from odometry (verified against
the IMU) and the IMU's own heading. It moves the robot: clear floor first.

On an open-loop profile (--robot waffle) the primitive is driven exactly as
client.py drives it -- on the clock, with no odometry in the loop. Odometry and
IMU are still read here, but only to report afterwards what the robot did:
they are observers, never feedback. --observe-odom-topic - turns that off too,
and then a tape measure is the only judge.

    python measure_primitive.py --robot waffle --router 127.0.0.1:7447 \
        --primitive turn_left --turn-angle-deg 90
"""

from __future__ import annotations

import argparse
import math

from rosbags.typesys import Stores, get_typestore

import config
from client import MOTION_FLAGS, motion_params
from motion import ZenohMotion
from transport import ZenohLink, decode_odom

TYPES = get_typestore(Stores.ROS2_HUMBLE)


def decode_imu_yaw(payload: bytes) -> float:
    msg = TYPES.deserialize_cdr(payload, 'sensor_msgs/msg/Imu')
    q = msg.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def delta_deg(a: float, b: float) -> float:
    return math.degrees(math.atan2(math.sin(a - b), math.cos(a - b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_arguments(parser)
    parser.add_argument('--primitive', required=True,
                        choices=['forward', 'turn_left', 'turn_right'])
    parser.add_argument('--imu-topic', default='',
                        help='observer only (default: <ns>/imu; - disables)')
    parser.add_argument('--observe-odom-topic', default='',
                        help='odometry read only to report the result, never fed to '
                             'the motion on an open-loop profile (default: <ns>/odom; '
                             '- disables)')
    parser.add_argument('--no-yaw-feedback', action='store_true',
                        help='close the turn on arc length instead of heading')
    for name in MOTION_FLAGS:
        parser.add_argument(f'--{name.replace("_", "-")}', type=float, default=None)
    args = parser.parse_args()
    settings = config.from_args(args)
    params = motion_params(settings.profile, args, yaw_feedback=not args.no_yaw_feedback,
                           obstacle_stop=0.0)   # measured by hand, with eyes on it

    def observed(value, suffix):
        value = value or f'{settings.namespace.strip("/")}/{suffix}'
        return None if value.strip() == '-' else value

    link = ZenohLink(settings.router, settings.connect_timeout)
    odom_key, imu_key = observed(args.observe_odom_topic, 'odom'), observed(args.imu_topic, 'imu')
    odom = link.subscribe(odom_key, decode_odom) if odom_key else None
    imu = link.subscribe(imu_key, decode_imu_yaw) if imu_key else None
    drive = link.publisher(settings.topic('cmd_vel_topic'))
    # The motion only gets odometry when the profile closes the loop on it.
    feedback = None if settings.profile.open_loop else odom
    motion = ZenohMotion(drive, odom=feedback, scan=None, params=params)

    start_pose = odom.wait(5.0, 2.0) if odom else None
    start_imu = imu.wait(5.0, 2.0) if imu else None
    if start_pose is None and not settings.profile.open_loop:
        raise SystemExit('no odometry; nothing to measure against')
    speed = (f'{params.forward_speed} m/s' if args.primitive == 'forward'
             else f'{params.turn_rate} rad/s' if params.drive == 'differential'
             else f'{params.turn_speed} m/s')
    print(f'{settings.describe()}\nprimitive={args.primitive} speed={speed} '
          f'control={"open loop (clock)" if settings.profile.open_loop else "odometry"}')
    try:
        if args.primitive == 'forward':
            intended = f'{params.forward_step:.2f} m forward, 0 deg'
            info = motion.move_forward()
        else:
            carried = (0.0 if params.drive == 'differential'
                       else params.turn_radius * math.sin(math.radians(params.turn_angle_deg)))
            intended = (f'{params.turn_angle_deg:.0f} deg '
                        f'{"left" if args.primitive == "turn_left" else "right"}, '
                        f'{carried:.2f} m of arc translation')
            info = motion.turn(+1 if args.primitive == 'turn_left' else -1)
        end_pose = odom.wait(2.0, 1.0) if odom else None
        end_imu = imu.wait(2.0, 1.0) if imu else None
    finally:
        motion.stop()

    print(f'\nintended: {intended}')
    if odom is None:
        print('measured: no odometry observer -- measure the floor by hand')
    elif start_pose is None or end_pose is None:
        print('measured: odometry silent -- measure the floor by hand')
    else:
        moved = math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
        print(f'measured: {delta_deg(end_pose[2], start_pose[2]):+.1f} deg yaw (odom), '
              f'{moved:.3f} m travelled')
    if start_imu is not None and end_imu is not None:
        print(f'          {delta_deg(end_imu, start_imu):+.1f} deg yaw (imu, independent)')
    print(f'          {info["elapsed"]:.2f}s'
          + (f', aborted={info["aborted"]}' if info['aborted'] else ', completed'))
    link.close()


if __name__ == '__main__':
    main()
