"""Offline checks: recording finalizes on interruption and guards block motion."""
from argparse import Namespace
import tempfile
import time

import cv2
import numpy as np

from config import Settings
from motion import ZenohMotion
from recording import RunRecording
from transport import Latest


class Drive:
    sent = 0
    stops = 0

    def send(self, *_):
        self.sent += 1

    def stop(self):
        self.stops += 1


drive = Drive()
motion = ZenohMotion(drive)
for reason in ('stale_camera', 'recording_error', 'duration_limit'):
    motion.guard = lambda: reason
    result = motion.turn(-1)
    assert result['aborted'] == reason and drive.sent == 0
assert drive.stops == 3

with tempfile.TemporaryDirectory() as folder:
    frames = Latest()
    frames.set(np.zeros((42, 64, 3), dtype=np.uint8))
    args = Namespace(runs_dir=folder, video_fps=10., instruction='offline test', dry_run=True)
    recording = RunRecording(frames, args, Settings().resolve())
    try:
        with recording:
            time.sleep(.25)
            raise KeyboardInterrupt('test interruption')
    except KeyboardInterrupt:
        pass
    assert (recording.path / 'run.json').exists()
    assert recording.error is None
    video = cv2.VideoCapture(str(recording.path / 'camera.mp4'))
    ok, frame = video.read()
    video.release()
    assert ok and frame.shape == (42, 64, 3)
print('PASS: all three guards publish no velocity; interrupted recording is playable')
