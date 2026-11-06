"""One brief manual Zenoh motion test with video and read-only odometry."""
from argparse import Namespace, ArgumentParser
import json
import math
from pathlib import Path
import signal
import time
from config import Settings
from recording import RunRecording
from transport import ZenohLink, decode_image, decode_odom

parser = ArgumentParser(description=__doc__)
parser.add_argument('--robot', choices=['waffle', 'jetracer'], default='waffle')
selection = parser.parse_args()
settings = Settings(robot=selection.robot).resolve()
args = Namespace(runs_dir=str(Path(__file__).resolve().parent/'runs'), video_fps=10.,
                 instruction='Manual Zenoh test: forward 0.08 m/s for 1 second, then stop', dry_run=False)
with ZenohLink(settings.router) as link:
    frames = link.subscribe(settings.camera_topic, lambda raw: decode_image(raw, True))
    odom = link.subscribe(f'{settings.namespace}/odom', decode_odom)
    raw_odom = link.subscribe(f'{settings.namespace}/odom_raw', decode_odom)
    drive = link.publisher(settings.cmd_vel_topic)
    def stop(*_):
        drive.stop(repeat=10)
        raise SystemExit(1)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        drive.stop()
        with RunRecording(frames, args, settings) as recording:
            recording.odom = odom
            recording.metadata['activations'] = {'applicable': False, 'reason': 'manual command; no model inference'}
            odom.wait(2, max_age=1)
            before = {'odom': odom.get(1), 'odom_raw': raw_odom.get(1)}
            recording.event(phase='before', **before)
            started, count = time.monotonic(), 0
            while time.monotonic() - started < 1.0:
                if frames.age > 1:
                    raise RuntimeError('Camera became stale; stopping')
                drive.send(.08, 0.)
                count += 1
                time.sleep(.05)
            drive.stop(repeat=10)
            time.sleep(.5)
            after = {'odom': odom.get(1), 'odom_raw': raw_odom.get(1)}
            distances = {key: None if before[key] is None or after[key] is None else
                         math.hypot(after[key][0] - before[key][0], after[key][1] - before[key][1])
                         for key in before}
            result = {'robot': settings.robot, 'topic': settings.cmd_vel_topic,
                      'published_commands': count, 'linear_m_s': .08, 'duration_s': 1.,
                      'before': before, 'after': after, 'displacement_m': distances}
            recording.event(phase='after', **result)
            recording.metadata.update(stop_reason='manual_test_complete', manual_test=result)
            print(json.dumps(result), flush=True)
    finally:
        drive.stop(repeat=10)
