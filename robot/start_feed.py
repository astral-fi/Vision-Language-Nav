"""Start the local-only camera preview once, detached from SSH."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import urllib.request
from urllib.error import HTTPError
from config import Settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--robot', choices=['jetracer', 'waffle'], default='jetracer')
parser.add_argument('--router', default=os.environ.get('ZENOH_ROUTER', '127.0.0.1:7447'))
args = parser.parse_args()
settings = Settings(robot=args.robot).resolve()
port = 8767 if args.robot == 'waffle' else 8766
log_name = 'camera_feed_waffle' if args.robot == 'waffle' else 'camera_feed'

runtime = Path(__file__).resolve().parent
logs = runtime / 'logs'
logs.mkdir(exist_ok=True)
try:
    try:
        reply = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)
    except HTTPError as response:
        reply = response
    with reply:
        print(json.dumps(json.load(reply)))
except OSError:
    command = [str(runtime / '.venv/bin/python'), '-u', str(runtime / 'camera_feed.py'),
               '--router', args.router, '--camera-topic', settings.camera_topic, '--port', str(port)]
    if settings.profile.color_path:
        command += ['--color-profile', settings.profile.color_path]
    if args.robot == 'jetracer' and settings.profile.calib_path:
        command += ['--lens-profile', settings.profile.calib_path]
    with (logs / f'{log_name}.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command,
                                   cwd=runtime, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (logs / f'{log_name}.pid').write_text(str(process.pid) + '\n')
    print(f'Started camera feed PID {process.pid}')
