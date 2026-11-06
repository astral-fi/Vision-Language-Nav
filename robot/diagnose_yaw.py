"""Which yaw signal survives driving? Log all three while the motors run.

Rotated by hand, `bot1/odom` yaw, the IMU's fused heading and the integrated
gyro all agree. Under power the first two have been seen to read ~1 deg through
a turn the robot visibly made, which makes every closed-loop primitive blind.
This logs the three independently, with message rates, so the liar is named
rather than guessed at.

    python diagnose_yaw.py --router 127.0.0.1:17447 --seconds 2 --idle
    python diagnose_yaw.py --router 127.0.0.1:17447 --seconds 2 --turn-radius 0.65

--idle publishes nothing (rotate by hand); otherwise it drives an arc for
--seconds at --speed, so clear the floor.
"""

from __future__ import annotations

import argparse
import math
import time

from rosbags.typesys import Stores, get_typestore

import config
from transport import ZenohLink, decode_odom

TYPES = get_typestore(Stores.ROS2_HUMBLE)


def decode_imu(payload: bytes):
    msg = TYPES.deserialize_cdr(payload, 'sensor_msgs/msg/Imu')
    q = msg.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return yaw, float(msg.angular_velocity.z), time.monotonic()


def delta_deg(a: float, b: float) -> float:
    return math.degrees(math.atan2(math.sin(a - b), math.cos(a - b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_arguments(parser)
    parser.add_argument('--imu-topic', default='bot1/imu')
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--speed', type=float, default=0.2)
    parser.add_argument('--turn-radius', type=float, default=0.65)
    parser.add_argument('--idle', action='store_true', help='publish nothing')
    args = parser.parse_args()
    settings = config.from_args(args)

    link = ZenohLink(settings.router, settings.connect_timeout)
    odom = link.subscribe(settings.topic('odom_topic'), decode_odom)
    imu = link.subscribe(args.imu_topic, decode_imu)
    drive = link.publisher(settings.topic('cmd_vel_topic'))
    time.sleep(1.5)

    first_pose, first_imu = odom.get(2.0), imu.get(2.0)
    if first_pose is None or first_imu is None:
        raise SystemExit(f'missing odom={first_pose is None} imu={first_imu is None}')
    angular = 0.0 if args.idle else args.speed / args.turn_radius
    print(f'{"idle (rotate by hand)" if args.idle else f"driving v={args.speed} w={angular:.3f}"}'
          f' for {args.seconds:g}s\n')
    print(f'{"t":>5} {"gyro z":>8} {"gyro int":>9} {"imu yaw":>8} {"odom yaw":>9} {"moved":>7}')

    # Integrate the gyro ourselves: it is the one signal nothing downstream has
    # filtered, fused or reset.
    integral, imu_seen, odom_seen = 0.0, set(), set()
    last_sample = time.monotonic()
    start = last_sample
    end = start + args.seconds
    while time.monotonic() < end:
        if not args.idle:
            drive.send(args.speed, angular)
        time.sleep(0.05)
        now = time.monotonic()
        pose, reading = odom.get(1.0), imu.get(1.0)
        if pose is None or reading is None:
            continue
        yaw, rate, stamp = reading
        integral += rate * (now - last_sample)
        last_sample = now
        imu_seen.add(stamp)
        odom_seen.add(pose)
        print(f'{now - start:5.2f} {rate:8.3f} {math.degrees(integral):9.1f} '
              f'{delta_deg(yaw, first_imu[0]):8.1f} {delta_deg(pose[2], first_pose[2]):9.1f} '
              f'{math.hypot(pose[0] - first_pose[0], pose[1] - first_pose[1]):7.3f}')
    drive.stop()
    time.sleep(0.5)

    pose, reading = odom.get(2.0), imu.get(2.0)
    elapsed = time.monotonic() - start
    print(f'\nover {elapsed:.1f}s, {len(imu_seen) / elapsed:.0f} imu/s, '
          f'{len(odom_seen) / elapsed:.0f} distinct odom/s')
    print(f'  integrated gyro : {math.degrees(integral):+.1f} deg')
    if reading is not None:
        print(f'  imu fused yaw   : {delta_deg(reading[0], first_imu[0]):+.1f} deg')
    if pose is not None:
        print(f'  odom yaw        : {delta_deg(pose[2], first_pose[2]):+.1f} deg')
        print(f'  odom distance   : {math.hypot(pose[0] - first_pose[0], pose[1] - first_pose[1]):.3f} m')
    print('\nA gyro that reads zero through a real turn is a sensor or firmware '
          'fault;\na live gyro with a flat fused yaw is a fusion fault, and the '
          'gyro can be\nintegrated instead.')
    link.close()


if __name__ == '__main__':
    main()
