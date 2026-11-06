"""Run the unchanged FutureNav server with durable passive robot-run capture."""
from __future__ import annotations

import os

import base64
from pathlib import Path
import sys
import threading

ROOT = Path(os.environ.get('FUTURENAV_ROOT', Path.home() / 'FutureNav'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'server'))
import server as original
from run_capture import RunCapture

BasePolicy, BaseHandler = original.FutureNavPolicy, original.Handler
original.STATE['lock'] = threading.RLock()


class RecordingPolicy(BasePolicy):
    def __init__(self, checkpoint_path, *args, **kwargs):
        super().__init__(checkpoint_path, *args, **kwargs)
        self.run_capture = RunCapture(self, checkpoint_path)


class Handler(BaseHandler):
    def _read_json(self):
        return self._cached_request

    def _send(self, code, payload):
        if getattr(self, '_handling_step', False):
            self._handling_step = False
            recorder = original.STATE['policy'].run_capture
            if code == 200:
                try:
                    payload['run_capture'] = recorder.commit(payload)
                except Exception as error:
                    recorder.fail(error)
                    return super()._send(500, {'error': f'Activation recording failed: {error}'})
            else:
                recorder.fail(payload)
        if self.path == '/health' and code == 200:
            payload['run_capture_supported'] = True
            payload['run_capture_mode'] = 'passive-v1'
            recorder = original.STATE['policy'].run_capture
            payload['activation_writer'] = {'mode': 'background', 'pending': recorder.write_queue.unfinished_tasks,
                                            'error': recorder.writer_error}
        return super()._send(code, payload)

    def do_POST(self):
        try:
            self._cached_request = BaseHandler._read_json(self)
            req = self._cached_request
            with original.STATE['lock']:
                policy = original.STATE['policy']
                recorder = policy.run_capture
                if self.path == '/reset':
                    run_dir = req.get('run_dir')
                    if not run_dir:
                        # Keep non-robot evaluation clients compatible; no silent
                        # fallback for robot clients, which check /health first.
                        if recorder.directory is not None and not recorder.manifest['finished']:
                            recorder.finish('reset_without_robot_recording')
                        recorder.directory = None
                        return super().do_POST()
                    directory = Path(run_dir).expanduser().resolve()
                    allowed = (Path(__file__).resolve().parent/'runs').resolve()
                    if not directory.is_relative_to(allowed) or not directory.is_dir():
                        return self._send(400, {'error': 'run_dir must be an existing robot run folder'})
                    if (directory / 'activations').exists():
                        return self._send(409, {'error': 'Activation folder already exists; use a unique run'})
                    policy.reset()
                    original.STATE.update(session=req['session'], instruction=req.get('instruction', ''), expected_step=0)
                    recorder.open(directory, req['session'], req.get('instruction', ''), req.get('recording_start_unix_s'))
                    recorder.validate_attention = bool(req.get('validate_attention', False))
                    return self._send(200, {'ok': True, 'run_capture': {'directory': str(recorder.directory), 'mode': 'passive-v1'}})
                if self.path == '/finish_run':
                    if req.get('session') != original.STATE['session']:
                        return self._send(409, {'error': 'Unknown session'})
                    return self._send(200, recorder.finish(req.get('stop_reason', 'client_finish')))
                if self.path == '/step' and recorder.directory is not None:
                    if req.get('session') != original.STATE['session'] or int(req.get('step', -1)) != original.STATE['expected_step']:
                        return super().do_POST()
                    if req.get('probes') or req.get('capture') or req.get('want_hidden'):
                        return self._send(400, {'error': 'Run capture is passive; request analysis separately after the run'})
                    recorder.begin(int(req['step']), base64.b64decode(req['image'], validate=True), req)
                    self._handling_step = True
                return super().do_POST()
        except Exception as error:
            policy = original.STATE.get('policy')
            if getattr(self, '_handling_step', False):
                self._handling_step = False
                policy.run_capture.fail(error)
            import traceback
            traceback.print_exc()
            return self._send(500, {'error': f'{type(error).__name__}: {error}'})
        finally:
            self._cached_request = None


if __name__ == '__main__':
    original.FutureNavPolicy = RecordingPolicy
    original.Handler = Handler
    original.main()
