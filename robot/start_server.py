"""Start the existing FutureNav model server once, with persistent logs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

root = Path(os.environ.get('FUTURENAV_ROOT', Path.home() / 'FutureNav'))  # upstream checkout + weights
runtime = Path(__file__).resolve().parent  # this robot/ directory
repo = runtime.parent
logs = runtime / 'logs'
logs.mkdir(exist_ok=True)
try:
    with urllib.request.urlopen('http://127.0.0.1:8642/health', timeout=3) as reply:
        print(json.dumps(json.load(reply)))
except OSError:
    pidfile = logs / 'server.pid'
    if pidfile.exists():
        pid = int(pidfile.read_text())
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError:
            cmdline = b''
        if b'server/server.py' in cmdline or b'deploy/server.py' in cmdline or b'recording_server.py' in cmdline:
            print(f'Server already loading, PID {pid}')
            raise SystemExit(0)
    env = dict(os.environ, PYTHONPATH=f'{repo}/server:{root}/src:{root}', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1')
    command = [os.environ.get('FUTURENAV_PYTHON', sys.executable), '-u',
               str(runtime / 'recording_server.py'), '--checkpoint', str(root / 'pretrain_models/FutureNav-4B-Base'),
               '--attn', 'sdpa', '--host', '127.0.0.1', '--port', '8642']
    with (logs / 'server.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(process.pid) + '\n')
    print(f'Started server PID {process.pid}; log: {logs / "server.log"}')
