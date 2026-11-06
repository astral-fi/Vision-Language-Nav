"""No-motion fixed-JPEG comparison of original and passively recorded inference."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request
import uuid
import numpy as np
from inspect_capture import decisions, validate

RUNTIME = Path(__file__).resolve().parent
FIXTURE = RUNTIME / 'runs/20261020T071947Z_bot1_c4bec0/observations'
CHECKS = RUNTIME / 'capture_checks'
CHECKS.mkdir(exist_ok=True)
INSTRUCTION = 'Turn left toward the open doorway and stop.'


def post(route, value):
    request = urllib.request.Request('http://127.0.0.1:8642' + route, data=json.dumps(value).encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['baseline', 'recorded'])
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    session = f'capture-check-{args.mode}-{uuid.uuid4().hex[:8]}'
    reset = {'session': session, 'instruction': INSTRUCTION}
    if args.mode == 'recorded':
        run = RUNTIME / 'runs' / session
        run.mkdir()
        reset.update(run_dir=str(run), recording_start_unix_s=time.time(), validate_attention=True)
    post('/reset', reset)
    responses = []
    for index in range(args.steps):
        request = {'session': session, 'step': index,
                   'image': base64.b64encode((FIXTURE / f'step_{index:04d}.jpg').read_bytes()).decode()}
        if args.mode == 'baseline':
            request['want_hidden'] = True
        responses.append(post('/step', request))
    (CHECKS / f'{args.mode}.json').write_text(json.dumps(responses, indent=2))
    result = {'mode': args.mode, 'actions': [response['action_name'] for response in responses],
              'latencies': [response['latency_s'] for response in responses]}
    if args.mode == 'recorded':
        result['finish'] = post('/finish_run', {'session': session, 'stop_reason': 'no_motion_validation'})
        baseline = json.loads((CHECKS / 'baseline.json').read_text())
        errors = []
        for index, (metadata, arrays) in enumerate(decisions(run)):
            if index < len(baseline):
                assert metadata['response']['action_name'] == baseline[index]['action_name']
                assert metadata['response']['raw'] == baseline[index]['raw']
                original_hidden = np.frombuffer(base64.b64decode(baseline[index]['hidden']), np.float16)
                error = float(np.max(np.abs(arrays['q0_final_hidden'].astype(np.float16) - original_hidden)))
                assert error == 0, error
                errors.append(error)
        result.update(validate(run), final_hidden_max_errors=errors, run=str(run))
    (CHECKS / f'{args.mode}_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
