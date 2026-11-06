"""Continuous camera video and run evidence, independent of inference timing."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid
import hashlib

import cv2


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


class RunRecording:
    """Sample the latest camera at 10 fps while the model thinks and robot moves.

    Frame timestamps report receive age, including repeats and gaps. A missing
    camera never silently becomes a frozen successful recording. Errors are
    exposed to the motion guard, so driving stops if recording fails.
    """
    def __init__(self, frames, args, settings):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = Path(args.runs_dir).expanduser() / f'{stamp}_{settings.namespace}_{uuid.uuid4().hex[:6]}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.frames, self.fps = frames, args.video_fps
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.count = 0
        self.started = time.monotonic()
        self.started_unix_s = time.time()
        self.odom = None
        self.log = (self.path / 'client.log').open('w', buffering=1)
        self.actions = (self.path / 'actions.jsonl').open('w', buffering=1)
        self.metadata = {'started_utc': datetime.now(timezone.utc).isoformat(),
                         'instruction': args.instruction, 'dry_run': args.dry_run,
                         'settings': vars(settings), 'arguments': vars(args).copy(),
                         'video': 'camera.mp4', 'video_fps': self.fps,
                         'recording_start_unix_s': self.started_unix_s,
                         'video_source': 'unmodified BGR camera; model inputs in observations/'}
        self.metadata['camera_processing'] = {}
        for kind, source in [('color', getattr(frames, 'color_profile', None)),
                             ('lens', settings.calib or settings.profile.calib_path)]:
            if source:
                source = Path(source).expanduser()
                content = source.read_bytes()
                target = self.path / ('camera_' + kind + source.suffix)
                target.write_bytes(content)
                self.metadata['camera_processing'][kind] = {
                    'source': str(source), 'file': target.name,
                    'sha256': hashlib.sha256(content).hexdigest()}
        if getattr(frames, 'color_profile', None):
            self.metadata['video_source'] = 'BGR camera after recorded devbox color correction; lens-rectified model inputs in observations/'
        self._thread = threading.Thread(target=self._record, daemon=True, name='camera-recorder')

    def event(self, **values):
        self.actions.write(json.dumps({'elapsed_s': time.monotonic() - self.started, **values}, allow_nan=False) + '\n')

    def _record(self):
        writer = None
        try:
            frame = self.frames.wait(10, max_age=1)
            if frame is None:
                raise RuntimeError('no camera frame to begin recording')
            height, width = frame.shape[:2]
            self.metadata['video_size'] = [width, height]
            writer = cv2.VideoWriter(str(self.path / 'camera.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('could not open camera.mp4')
            cv2.imwrite(str(self.path / 'first.jpg'), frame)
            tick = time.monotonic()
            with (self.path / 'frames.jsonl').open('w', buffering=1) as times:
                while not self._stop.is_set():
                    age = self.frames.age
                    frame = self.frames.get(max_age=1)
                    if frame is None:
                        raise RuntimeError('camera stale for more than one second')
                    if self.count % int(max(1, self.fps)) == 0:
                        if shutil.disk_usage(self.path).free < 512 * 1024 * 1024:
                            raise RuntimeError('less than 512 MiB free for recording')
                        preview = self.path / 'latest.tmp.jpg'
                        if not cv2.imwrite(str(preview), frame):
                            raise RuntimeError('camera preview write failed')
                        preview.replace(self.path / 'latest.jpg')
                    writer.write(frame)
                    pose = self.odom.get(max_age=.5) if self.odom is not None else None
                    times.write(json.dumps({'frame': self.count, 'elapsed_s': time.monotonic() - self.started,
                                            'received_age_s': age, 'odom': pose}) + '\n')
                    self.count += 1
                    self._ready.set()
                    tick += 1 / self.fps
                    self._stop.wait(max(0, tick - time.monotonic()))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if writer is not None:
                writer.release()
            self._ready.set()

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(sys.stdout, self.log), _Tee(sys.stderr, self.log)
        self._thread.start()
        if not self._ready.wait(12) or self.error:
            self.close()
            raise RuntimeError(self.error or 'recording startup timeout')
        print(f'[recording] run directory: {self.path}', flush=True)
        return self

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self.error = self.error or 'recorder did not stop'
        self.metadata.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                             elapsed_s=time.monotonic() - self.started,
                             video_frames=self.count, recording_error=self.error)
        (self.path / 'run.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.actions.close()
        self.log.close()

    def __exit__(self, kind, value, traceback):
        if value is not None:
            self.metadata['exception'] = str(value)
        self.close()
