"""Validate and load a saved episode without a model, GPU, or robot connection.

Usage: python inspect_capture.py /path/to/run --verify
Library: for metadata, arrays in decisions(run): ...
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def decisions(run):
    root = Path(run)
    if root.name != 'activations':
        root = root / 'activations'
    manifest = json.loads((root / 'manifest.json').read_text())
    for record in manifest['steps']:
        metadata = json.loads((root / record['metadata']).read_text())
        with np.load(root / record['tensors'], allow_pickle=False) as archive:
            yield metadata, {name: archive[name] for name in archive.files}


def validate(run):
    root = Path(run)
    if root.name != 'activations':
        root /= 'activations'
    manifest = json.loads((root / 'manifest.json').read_text())
    provenance = json.loads((root / manifest['provenance']).read_text())
    layers = provenance['model_config']['text_config']['num_hidden_layers']
    heads = provenance['model_config']['text_config']['num_attention_heads']
    hidden = provenance['model_config']['text_config']['hidden_size']
    total, row_error, reconstruction_error = 0, 0., 0.
    for record, (step, arrays) in zip(manifest['steps'], decisions(root)):
        assert record['step'] == step['step'] == total
        assert hashlib.sha256((root / record['tensors']).read_bytes()).hexdigest() == record['sha256']
        assert hashlib.sha256((root / step['frame']).read_bytes()).hexdigest() == step['input_sha256']
        assert step['history_indices'][-1] == total
        assert len(step['history_indices']) == len(step['image_spans']) == len(step['image_grids'])
        assert len(step['queries']) == len(arrays['generated_token_ids']) == len(arrays['logits'])
        assert step['cache_frames'] == min(total + 1, provenance['cache_start'] + provenance['cache_recent'])
        assert np.array_equal(arrays['input_ids'].shape, (step['seq_len'],))
        assert len(arrays['segment_ids']) == step['seq_len'] == sum(step['segment_counts'])
        for name, info in step['arrays'].items():
            assert list(arrays[name].shape) == info['shape'] and str(arrays[name].dtype) == info['dtype']
            assert np.isfinite(arrays[name]).all(), name
        for n, query in enumerate(step['queries']):
            attn = arrays[f'q{n}_attention']
            assert attn.shape == (layers, heads, step['seq_len'] + n)
            assert query['prefix_token_ids'] == step['generated_tokens'][:n]
            assert query['predicted_token_id'] == step['generated_tokens'][n]
            assert (attn >= 0).all() and (attn <= 1).all()
            error = float(np.max(np.abs(attn.sum(-1) - 1)))
            row_error = max(row_error, error)
            assert error < 2e-6
            for name in ('residual_in', 'residual_out', 'attention_output', 'mlp_output'):
                assert arrays[f'q{n}_{name}'].shape == (layers, hidden)
            assert arrays[f'q{n}_final_hidden'].shape == (hidden,)
            if f'q{n}_attention_validation_error' in arrays:
                error = float(arrays[f'q{n}_attention_validation_error'].max())
                reconstruction_error = max(reconstruction_error, error)
                assert error < .04
        for (a, b), (t, h, w) in zip(step['image_spans'], step['image_grids']):
            merge = step['spatial_merge_size']
            assert b - a == t * (h // merge) * (w // merge)
        total += 1
    return {'valid': True, 'steps': total, 'finished': manifest['finished'], 'status': manifest['status'],
            'bytes': sum(record['bytes'] for record in manifest['steps']),
            'max_attention_row_sum_error': row_error, 'max_attention_reconstruction_error': reconstruction_error}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(validate(args.run), indent=2))
    else:
        for metadata, arrays in decisions(args.run):
            print(json.dumps({'step': metadata['step'], 'action': metadata['response']['action_name'],
                              'history_indices': metadata['history_indices'],
                              'queries': metadata['queries'], 'array_names': list(arrays)}))
