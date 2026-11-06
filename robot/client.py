"""Run one FutureNav episode on a real robot, over zenoh.

--robot picks the profile (robots.py): `jetracer` (default) or `waffle`, the
TurtleBot3 Waffle Pi, which runs strictly open loop.

This replaces only the transport of FutureNav's deploy/jetracer_client.py: the
camera comes from a zenoh topic instead of jetcam, and actions leave as Twist
messages on cmd_vel instead of NvidiaRacecar throttle writes. The episode
protocol, the step budget, the rotation-stall guard and the model server are
unchanged, and the FutureNav checkout is imported read-only for its camera
rectification -- nothing in it is written to.

    python client.py --preflight --router 127.0.0.1:17447
    python client.py --router 127.0.0.1:17447 --server http://127.0.0.1:8642 \
        --instruction "walk out of the room and stop next to the table" --dry-run

Drop --dry-run only when the floor is clear and someone can reach the robot.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import math
import os
import signal
import sys
import time
import urllib.request

import cv2

import config
from motion import MotionParams, ZenohMotion
from transport import ZenohLink, decode_image, decode_odom, decode_scan, forward_clearance
from view import PinholeViewAdapter, FisheyeViewAdapter, is_ros_calibration
from recording import RunRecording
from pathlib import Path
from camera_color import ColorCorrection

EARLY_STOP_STEPS = 400        # EVAL.EARLY_STOP_STEPS
EARLY_STOP_ROTATION = 50      # EVAL.EARLY_STOP_ROTATION
HABITAT_SIZE = (640, 480)     # SIMULATOR RGB_SENSOR WIDTH/HEIGHT
MOTION_FLAGS = ('forward_speed', 'turn_speed', 'turn_radius', 'turn_rate', 'forward_step',
                'turn_angle_deg', 'obstacle_stop', 'coast_m')


def motion_params(profile, args, **extra) -> MotionParams:
    """The profile's motion defaults, then any flag given on the command line,
    then a check that the robot can actually execute the result."""
    values = dict(profile.motion, drive=profile.drive, **extra)
    values.update({n: getattr(args, n) for n in MOTION_FLAGS
                   if getattr(args, n, None) is not None})
    params = MotionParams(**values)
    try:
        profile.check(params)
    except ValueError as exc:
        raise SystemExit(str(exc))
    return params


def post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def build_view_adapter(settings):
    """A ROS camera_info calibration (--calib x.yaml, or the profile's own) is
    undistorted here by view.py; otherwise FutureNav's own fisheye
    rectification when its checkout is given -- imported read-only -- and
    otherwise a plain resize to the Habitat frame."""
    calib = settings.calib or settings.profile.calib_path
    if calib and is_ros_calibration(calib):
        return PinholeViewAdapter(os.path.expanduser(calib), settings.mount_pitch)
    if calib and Path(calib).suffix.lower() == '.json':
        return FisheyeViewAdapter(os.path.expanduser(calib), settings.mount_pitch)
    if settings.futurenav_root:
        # camera_adapter.py lives in this repo's server/ directory.
        deploy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server')
        if not os.path.isdir(deploy):
            raise SystemExit(f'server/ directory not found: {deploy}')
        sys.path.insert(0, deploy)
        # Read-only means read-only: no .pyc left behind in their checkout.
        sys.dont_write_bytecode = True
        from camera_adapter import build_adapter
        return build_adapter(calib or None, mount_pitch_deg=settings.mount_pitch)
    return lambda frame: cv2.resize(frame, HABITAT_SIZE, interpolation=cv2.INTER_AREA)


def connect(settings, dry_run: bool):
    """Open the session and declare everything the episode needs."""
    link = ZenohLink(settings.router, settings.connect_timeout)
    camera_key = settings.topic('camera_topic')
    compressed = camera_key.endswith('/compressed')
    color = ColorCorrection(settings.profile.color_path)
    frames = link.subscribe(camera_key, lambda raw: color(decode_image(raw, compressed)))
    frames.color_profile = settings.profile.color_path
    odom_key, scan_key = settings.topic('odom_topic'), settings.topic('scan_topic')
    odom = link.subscribe(odom_key, decode_odom) if odom_key else None
    scan = link.subscribe(scan_key, decode_scan) if scan_key else None
    drive = link.publisher(settings.topic('cmd_vel_topic'), dry_run=dry_run)
    return link, frames, odom, scan, drive


def preflight(settings, frames, odom, scan, wait: float) -> int:
    """Report what the robot is actually giving us, and publish nothing."""
    print(f'[preflight] {settings.describe()}')
    ok = True
    frame = frames.wait(wait, max_age=wait)
    if frame is None:
        print(f'[preflight] camera   {settings.camera_topic}: NO FRAME in {wait:g}s')
        ok = False
    else:
        print(f'[preflight] camera   {settings.camera_topic}: {frame.shape[1]}x{frame.shape[0]} BGR')
    if odom is not None:
        pose = odom.wait(wait, max_age=wait)
        if pose is None:
            print(f'[preflight] odom     {settings.odom_topic}: silent -- primitives would run open-loop')
        else:
            print(f'[preflight] odom     {settings.odom_topic}: x={pose[0]:.2f} y={pose[1]:.2f} '
                  f'yaw={math.degrees(pose[2]):.1f} deg')
    elif settings.profile.open_loop:
        print(f'[preflight] odom     not used -- {settings.robot} is open loop, '
              'primitives run on the clock')
    else:
        print('[preflight] odom     disabled -- primitives would run open-loop')
    if scan is not None:
        reading = scan.wait(wait, max_age=wait)
        if reading is None:
            print(f'[preflight] scan     {settings.scan_topic}: silent -- the obstacle gate '
                  'will refuse to move (set --scan-topic - to disable it)')
            ok = False
        else:
            ahead = forward_clearance(reading)
            print(f'[preflight] scan     {settings.scan_topic}: {len(reading.ranges)} ranges, '
                  f'forward clearance {ahead:.2f} m')
    elif settings.profile.open_loop:
        print(f'[preflight] scan     not used -- {settings.robot} is open loop, no obstacle gate')
    else:
        print('[preflight] scan     disabled -- no obstacle gate')
    print(f'[preflight] cmd_vel  {settings.cmd_vel_topic}: declared, nothing published')
    print('[preflight] ' + ('ready' if ok else 'NOT ready'))
    return 0 if ok else 1


def run_episode(args, settings, frames, motion) -> None:
    with RunRecording(frames, args, settings) as recording:
        recording.metadata['motion_parameters'] = vars(motion.p).copy()
        if not args.save_dir:
            args.save_dir = str(recording.path / 'observations')
        deadline = time.monotonic() + args.max_duration
        def guard():
            if recording.error:
                return 'recording_error'
            if frames.age > args.frame_max_age:
                return 'stale_camera'
            if time.monotonic() >= deadline:
                return 'duration_limit'
            return None
        motion.guard = guard
        try:
            reason = _run_episode(args, settings, frames, motion, recording, guard)
            recording.metadata['stop_reason'] = reason
        except BaseException as error:
            recording.metadata['stop_reason'] = ('interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit))
                                                 else 'client_exception')
            raise
        finally:
            motion.stop()
            motion.guard = None
            if recording.metadata.get('activation_session'):
                try:
                    captured = post(f'{settings.server}/finish_run',
                                    {'session': recording.metadata['activation_session'],
                                     'stop_reason': recording.metadata.get('stop_reason', 'client_exit')},
                                    settings.request_timeout)
                    recording.metadata['activations'] = captured
                    print(f'[activations] saved {captured.get("steps", 0)} decisions: {captured.get("directory")}', flush=True)
                except Exception as error:
                    recording.metadata['activation_finalize_error'] = str(error)
                    print(f'[activations] finalization error: {error}; committed per-step files remain on disk', flush=True)


def _run_episode(args, settings, frames, motion, recording, guard):
    adapter = build_view_adapter(settings)
    session = args.session or recording.path.name
    recording.metadata['session'] = session
    recording.odom = motion.odom
    reset = {'session': session, 'instruction': args.instruction}
    if not args.no_activations:
        with urllib.request.urlopen(f'{settings.server}/health', timeout=settings.request_timeout) as reply:
            health = json.load(reply)
        recording.metadata['server_health'] = health
        if not health.get('run_capture_supported'):
            raise RuntimeError('Server lacks automatic run capture; start recording_server.py before driving')
        recording.metadata['activation_session'] = session
        reset.update(run_dir=str(recording.path.resolve()), recording_start_unix_s=recording.started_unix_s)
    response = post(f'{settings.server}/reset', reset, settings.request_timeout)
    recording.metadata['activation_reset'] = response.get('run_capture')
    print(f'[client] session={session}\n[client] instruction={args.instruction!r}')
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    consecutive_turns = failed_primitives = 0
    for step in range(args.max_steps):
        reason = guard()
        if reason:
            print(f'[client] {reason}; stopping')
            return reason
        frame = frames.wait(args.frame_timeout, max_age=args.frame_max_age)
        if frame is None:
            print(f'[client] no fresh frame within {args.frame_timeout:g}s; stopping')
            return 'no_fresh_frame'
        view = adapter(frame)
        ok, buffer = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            print('[client] JPEG encode failed; stopping')
            return 'jpeg_error'
        if args.save_dir:
            Path(args.save_dir, f'step_{step:04d}.jpg').write_bytes(buffer.tobytes())

        timing = {'observation_elapsed_s': time.monotonic() - recording.started,
                  'request_unix_s': time.time(), 'camera_receive_age_s': frames.age}
        recording.event(step=step, phase='inference_request', timing=timing)

        started = time.time()
        try:
            response = post(f'{settings.server}/step',
                            {'session': session, 'step': step,
                             'image': base64.b64encode(buffer.tobytes()).decode(),
                             'recording_timing': timing},
                            settings.request_timeout)
        except Exception as exc:
            motion.stop()
            print(f'[client] server error at step {step}: {exc}')
            recording.event(step=step, error=str(exc))
            return 'server_error'
        action = response['action_name']
        recording.event(step=step, response=response)
        capture_status = response.get('run_capture', {})
        if not args.no_activations and not (capture_status.get('accepted') or capture_status.get('complete')):
            print('[client] activation capture was not accepted; stopping before motion')
            return 'activation_recording_error'
        print(f'[client] step {step:3d} {action:<13} rtt={time.time() - started:.2f}s '
              f'infer={response.get("latency_s")}s'
              + ('  [parse-fallback]' if response.get('fallback') else ''), flush=True)

        if action == 'STOP':
            motion.stop()
            print('[client] policy issued STOP')
            return 'policy_stop'
        reason = guard()
        if reason:
            print(f'[client] {reason}; stopping before motion')
            return reason
        if action not in ('MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT') or response.get('fallback'):
            print(f'[client] invalid or fallback action {action!r}; stopping')
            return 'invalid_action'
        if args.allowed_actions and action not in args.allowed_actions.split(','):
            print(f'[client] action {action} is outside this run action limit; stopping')
            return 'action_limit'
        if action == 'MOVE_FORWARD':
            consecutive_turns = 0
            info = motion.move_forward()
        else:
            consecutive_turns += 1
            if consecutive_turns > EARLY_STOP_ROTATION:
                motion.stop()
                print('[client] rotation-stall guard tripped; ending episode')
                return 'rotation_limit'
            info = motion.turn(+1 if action == 'TURN_LEFT' else -1)
        print(f'[client]          {info["action"]} {info["measured"]:.3f}{info["unit"]} '
              f'({info["feedback"]}) elapsed={info["elapsed"]:.2f}s'
              + (f' aborted={info["aborted"]}' if info['aborted'] else ''), flush=True)
        recording.event(step=step, motion={key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                                           for key, value in info.items()})
        if info['aborted'] in ('stale_camera', 'recording_error', 'duration_limit'):
            return info['aborted']
        if info['aborted'] == 'obstacle':
            print('[client] forward aborted on range gate; recovering')
            motion.recover_from_collision()
            failed_primitives = 0
        elif info['aborted'] == 'no_scan':
            print('[client] obstacle gate has no scan; stopping the episode')
            return 'no_scan'
        elif info['aborted'] == 'timeout':
            # A primitive that runs out of time has not moved the robot the way
            # the policy was told it would, so every later decision is made on a
            # false premise. One is a slip; three in a row is a broken robot.
            failed_primitives += 1
            if failed_primitives >= args.max_failed:
                motion.stop()
                print(f'[client] {failed_primitives} primitives in a row timed out '
                      '-- the robot is not executing commands; stopping')
                return 'motion_timeout'
        else:
            failed_primitives = 0
    else:
        print('[client] step budget exhausted')
        return 'step_limit'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_arguments(parser)
    parser.add_argument('--instruction', help='the navigation instruction (required to drive)')
    parser.add_argument('--session', default=None)
    parser.add_argument('--preflight', action='store_true',
                        help='report camera/odom/scan and exit without driving')
    parser.add_argument('--dry-run', action='store_true',
                        help='run the full loop but publish no cmd_vel')
    parser.add_argument('--max-steps', type=int, default=EARLY_STOP_STEPS)
    parser.add_argument('--frame-timeout', type=float, default=10.0)
    parser.add_argument('--frame-max-age', type=float, default=1.0,
                        help='seconds; an older frame is stale and is not acted on')
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--max-failed', type=int, default=3,
                        help='consecutive timed-out primitives before giving up')
    parser.add_argument('--save-dir', default=None, help='dump the frames actually sent')
    parser.add_argument('--runs-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
                        help='persistent per-command recordings and logs')
    parser.add_argument('--video-fps', type=float, default=10)
    parser.add_argument('--no-activations', action='store_true',
                        help='explicitly disable activation capture for legacy diagnostic servers')
    parser.add_argument('--max-duration', type=float, default=300,
                        help='wall-clock run limit, including model inference')
    parser.add_argument('--allowed-actions', default='', help='optional comma-separated motion action limit')
    parser.add_argument('--wait', type=float, default=5.0, help='preflight wait per topic')
    parser.add_argument('--no-yaw-feedback', action='store_true',
                        help='close turns on arc length instead of heading; required on '
                             'a robot whose gyro is frozen (see README)')
    for name in MOTION_FLAGS:
        parser.add_argument(f'--{name.replace("_", "-")}', type=float, default=None,
                            help=f'MotionParams.{name} (default: the robot profile\'s)')
    args = parser.parse_args()
    if not math.isfinite(args.video_fps) or args.video_fps <= 0 or not math.isfinite(args.max_duration) or args.max_duration <= 0:
        parser.error('--video-fps and --max-duration must be finite and positive')
    settings = config.from_args(args)
    if not args.preflight and not args.instruction:
        parser.error('--instruction is required unless --preflight is given')

    params = motion_params(settings.profile, args, yaw_feedback=not args.no_yaw_feedback)
    link, frames, odom, scan, drive = connect(settings, dry_run=args.dry_run or args.preflight)
    motion = ZenohMotion(drive, odom=odom, scan=scan, params=params,
                         dry_run=args.dry_run or args.preflight)

    def emergency_stop(*_):
        motion.stop()
        print('\n[client] emergency stop', flush=True)
        sys.exit(1)

    atexit.register(motion.stop)
    signal.signal(signal.SIGINT, emergency_stop)
    signal.signal(signal.SIGTERM, emergency_stop)
    try:
        if args.preflight:
            raise SystemExit(preflight(settings, frames, odom, scan, args.wait))
        print(f'[client] {settings.describe()}')
        if args.dry_run:
            print('[client] DRY RUN: no cmd_vel will be published')
        run_episode(args, settings, frames, motion)
    finally:
        motion.stop()
        atexit.unregister(motion.stop)
        link.close()


if __name__ == '__main__':
    main()
