"""Verify video/input/activation alignment and interruption recovery, dry-run only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from inspect_capture import validate

RUNTIME = Path(__file__).resolve().parent
CHECKS = RUNTIME / 'capture_checks'
CHECKS.mkdir(exist_ok=True)


def locate(session):
    for path in (RUNTIME / 'runs').glob('*/activations/manifest.json'):
        data = json.loads(path.read_text())
        if data['session'] == session:
            return path.parent.parent


def check(interrupt):
    session = 'client-capture-check-' + uuid.uuid4().hex[:8]
    command = [str(RUNTIME / '.venv/bin/python'), '-u', 'client.py', '--dry-run',
               '--robot', 'jetracer', '--namespace', 'bot1', '--scan-topic', '-', '--no-yaw-feedback',
               '--turn-radius', '.31', '--router', '127.0.0.1:7447', '--futurenav-root', os.environ.get('FUTURENAV_ROOT', str(Path.home() / 'FutureNav')),
               '--instruction', 'Turn right toward the open doorway and continue forward through it.',
               '--session', session, '--max-steps', '60' if interrupt else '3']
    with (CHECKS / f'{session}.log').open('w') as log:
        process = subprocess.Popen(command, cwd=RUNTIME, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                   stdout=log, stderr=subprocess.STDOUT)
        before = None
        if interrupt:
            deadline = time.monotonic() + 90
            while process.poll() is None and time.monotonic() < deadline:
                run = locate(session)
                if run:
                    manifest = json.loads((run / 'activations/manifest.json').read_text())
                    if len(manifest['steps']) >= 2:
                        before = validate(run)
                        process.send_signal(signal.SIGINT)
                        break
                time.sleep(.1)
            else:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                raise RuntimeError('Dry-run did not reach interruption checkpoint')
        code = process.wait(timeout=90)
    assert code == (1 if interrupt else 0), (code, (CHECKS / f'{session}.log').read_text())
    run = locate(session)
    result = validate(run)
    recorded = json.loads((run / 'run.json').read_text())
    assert recorded['dry_run'] and recorded['recording_error'] is None
    assert recorded['activations']['complete'] and recorded['activations']['steps'] == result['steps']
    assert result['finished'] and result['status'] == 'complete'
    if interrupt:
        assert recorded['stop_reason'] == 'interrupted' and not before['finished']
    assert recorded['video_frames'] > 0
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(run / 'camera.mp4'), '-f', 'null', '-'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for path in (run / 'activations/inputs').glob('*.jpg'):
        assert path.read_bytes() == (run / 'observations' / f'step_{path.stem}.jpg').read_bytes()
    result.update(run=str(run), stop_reason=recorded['stop_reason'], dry_run=True,
                  exact_jpegs_match=True, video_frames=recorded['video_frames'],
                  committed_before_interrupt=before)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    results = [check(False), check(True)]
    (CHECKS / 'client_results.json').write_text(json.dumps(results, indent=2))
