"""Durable, passive decision capture. No forced tokens or extra model forwards.

Attention rows are reconstructed from the actual normalized queries and text KV
cache used by SDPA. Only the last query of each real generation pass is stored.
BF16 activations are losslessly represented in float32 NPZ arrays; compression
recovers their repeated low bits. Full token residuals and full KV caches are
deliberately omitted; exact JPEGs, token layout, weights and geometry permit replay.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import queue
import threading
import time
import zipfile

import numpy as np
import torch
from capture import split_instruction

FIXED_SEGMENTS = ['system'] + [f'image_{i}' for i in range(1, 10)] + ['prompt', 'action_options', 'assistant', 'special']


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prompt_layout(ids, instruction, tokenizer, config):
    """Same real-token clause/image accounting as the episode scrubber."""
    labels = np.full(len(ids), 13, np.int32)
    spans, i = [], 0
    while i < len(ids):
        if ids[i] != config.image_token_id:
            i += 1
            continue
        j = i + 1
        while j < len(ids) and ids[j] == config.image_token_id:
            j += 1
        spans.append([i, j])
        i = j
    if not 1 <= len(spans) <= 9:
        raise ValueError(f'Expected 1..9 images, found {len(spans)}')
    for slot, (a, b) in enumerate(spans):
        labels[a:b] = 1 + 9 - len(spans) + slot
    first = ids.index(config.vision_start_token_id)
    last = len(ids) - 1 - ids[::-1].index(config.vision_end_token_id)
    labels[:first] = 0
    suffix_ids = ids[last + 1:]
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(suffix, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != suffix_ids:
        raise ValueError('Prompt suffix tokenization mismatch')
    start = suffix.find('Your task is to ')
    if start < 0 or suffix.find(instruction, start) != start + len('Your task is to '):
        raise ValueError('Instruction is missing from the actual prompt')
    start += len('Your task is to ')
    end = start + len(instruction)
    assistant = suffix.rfind('<|im_start|>assistant')
    clauses = split_instruction(instruction)
    for n, (a, b) in enumerate(encoded['offset_mapping']):
        char = max(a, b - 1)
        if assistant >= 0 and a >= assistant:
            group = 12
        elif char < start:
            group = 10
        elif char >= end:
            group = 11
        else:
            relative = char - start
            group = 14 + next(k for k, (s, e) in enumerate(clauses)
                              if s <= relative < e or k == len(clauses) - 1)
        labels[last + 1 + n] = group
    return labels, spans, clauses


class RunCapture:
    def __init__(self, policy, checkpoint):
        self.policy, self.model = policy, policy.model
        self.layers = self.model.model.language_model.layers
        self.original_generate = self.model.generate
        self.model.generate = self.generate
        self.active = False
        self.directory = None
        self.step = None
        self.validate_attention = False
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.q_norm.register_forward_hook(self.query_hook(i)))
            self.handles.append(layer.self_attn.register_forward_hook(self.attention_hook(i), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self.layer_hook(i), with_kwargs=True))
            self.handles.append(layer.mlp.register_forward_hook(self.mlp_hook(i)))
        self.handles.append(self.model.model.language_model.register_forward_hook(self.final_hook))
        self.handles.append(self.model.merger_vggt.register_forward_hook(self.geometry_hook))
        self.provenance = self.make_provenance(Path(checkpoint))
        self.writer_lock = threading.RLock()
        self.write_queue = queue.Queue(maxsize=4)
        self.writer_error = None
        self.submitted_steps = 0
        self.writer = threading.Thread(target=self.write_loop, daemon=True, name='activation-writer')
        self.writer.start()

    def make_provenance(self, checkpoint):
        root = Path(__file__).resolve().parent
        paths = [Path(inspect.getfile(self.policy.__class__)), Path(inspect.getfile(self.model.__class__)),
                 Path(inspect.getfile(self.layers[0].__class__)), Path(__file__), root / 'recording_server.py',
                 root / 'client.py', root / 'recording.py', root / 'motion.py', root / 'transport.py']
        paths += [Path(__file__).resolve().parent.parent / 'server' / name for name in ('futurenav_core.py', 'server.py', 'capture.py')]
        paths += list(checkpoint.glob('*.json'))
        self.source_paths = list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
        source = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size,
                             'archived_path': f'sources/{n:02d}_{path.name}'}
                  for n, path in enumerate(self.source_paths)}
        # Hash the released weight shards once per server startup, never per step.
        weights = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size}
                   for path in sorted(checkpoint.glob('*.safetensors'))}
        packages = {}
        for name in ('torch', 'transformers', 'numpy', 'Pillow', 'safetensors', 'opencv-python-headless'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        config = self.model.config.to_dict()
        return {'created_utc': utc_now(), 'checkpoint': str(checkpoint), 'weights': weights,
                'sources': source, 'packages': packages, 'cuda': torch.version.cuda,
                'device': str(self.model.device), 'gpu': torch.cuda.get_device_name(self.model.device),
                'model_config': config, 'generation_config': self.model.generation_config.to_dict(),
                'lam': float(self.model.lam), 'history_frames': self.policy.max_history_frames,
                'cache_start': self.model.kv_cache_vggt.start_size,
                'cache_recent': self.model.kv_cache_vggt.recent_size,
                'capture_definitions': {
                    'attention': 'Float32 softmax reconstructed from actual normalized Q and RoPE text-cache K; last query of each actual generation pass, all layers and heads. Fused image positions are not pixel attribution.',
                    'residual_in_out': 'Last query before and after each decoder layer; float32 stores native BF16 values exactly.',
                    'attention_output': 'Actual projected SDPA output at the last query, before residual addition.',
                    'mlp_output': 'Actual decoder MLP output at the last query, before residual addition.',
                    'segment_residual_mean': 'Float32 mean of prefill layer outputs grouped by actual prompt token segment.',
                    'geometry': 'Actual merger_vggt output before interpolation across all Qwen image slots.',
                    'logits': 'Unprocessed full-vocabulary logits computed for actual generated tokens, including EOS.',
                    'omitted': 'Full per-token layer residuals, vision tower internals and full VGGT/text KV caches. Recompute from the saved JPEG sequence and pinned weights for deeper analysis.',
                    'deferred': 'Forced TURN/direction on non-turn decisions, canonical action probabilities, counterfactuals, plots and scrubber export run later.'}}

    def open(self, directory, session, instruction, start_unix_s):
        if self.directory is not None and not self.manifest['finished']:
            self.finish('superseded_by_reset')
        self.directory = Path(directory) / 'activations'
        self.writer_error = None
        self.submitted_steps = 0
        self.directory.mkdir(exist_ok=False)
        (self.directory / 'steps').mkdir()
        (self.directory / 'inputs').mkdir()
        sources = self.directory / 'sources'
        sources.mkdir()
        for n, path in enumerate(self.source_paths):
            shutil.copy2(path, sources / f'{n:02d}_{path.name}')
        atomic_json(self.directory / 'provenance.json', self.provenance)
        self.manifest = {'schema_version': 1, 'session': session, 'instruction': instruction,
                         'started_utc': utc_now(), 'recording_start_unix_s': start_unix_s,
                         'finished': False, 'status': 'recording', 'steps': [], 'errors': [],
                         'write_mode': 'background', 'queue_capacity': 4, 'submitted_steps': 0,
                         'provenance': 'provenance.json', 'video': '../camera.mp4',
                         'analysis_performed': False, 'extra_model_forwards': 0}
        self.save_manifest()

    def save_manifest(self):
        atomic_json(self.directory / 'manifest.json', self.manifest)

    def begin(self, index, jpeg, request):
        if self.directory is None or self.manifest['finished']:
            raise RuntimeError('No active activation recording')
        if self.writer_error:
            raise RuntimeError(f'Activation writer failed: {self.writer_error}')
        if self.submitted_steps != index:
            raise RuntimeError('Activation step counter mismatch')
        if shutil.disk_usage(self.directory).free < 512 * 1024 * 1024:
            raise RuntimeError('Less than 512 MiB free for activations')
        filename = f'inputs/{index:04d}.jpg'
        with (self.directory / filename).open('wb') as stream:
            stream.write(jpeg)
            stream.flush()
            os.fsync(stream.fileno())
        self.step = {'step': index, 'frame': filename, 'input_sha256': hashlib.sha256(jpeg).hexdigest(),
                     'received_unix_s': time.time(), 'client_timing': request.get('recording_timing'),
                     'queries': [], 'history_indices': None}
        self.arrays, self.queries, self.query_norm = {}, [], {}
        self.active = True

    @staticmethod
    def array(value):
        return value.detach().float().cpu().numpy()

    def query_hook(self, index):
        def hook(module, args, output):
            if self.active:
                self.query_norm[index] = output[:, -1:].detach()
        return hook

    def attention_hook(self, index):
        def hook(module, args, kwargs, output):
            if not self.active:
                return
            h = kwargs.get('hidden_states', args[0] if args else None)
            if index == 0:
                self.queries.append({'attention': {}, 'attention_output': {}, 'residual_in': {},
                                     'residual_out': {}, 'mlp_output': {}})
            query = self.queries[-1]
            q = self.query_norm.pop(index).transpose(1, 2)
            cos, sin = kwargs['position_embeddings']
            cq, sq = cos[:, -1:].unsqueeze(1), sin[:, -1:].unsqueeze(1)
            half = q.shape[-1] // 2
            q = q * cq + torch.cat((-q[..., half:], q[..., :half]), -1) * sq
            cache = kwargs.get('past_key_values', kwargs.get('past_key_value'))
            if hasattr(cache, 'layers'):
                keys = cache.layers[index].keys
            else:
                keys = cache.key_cache[index]
            # Grouped-query matmul avoids materializing repeated full K/V tensors.
            groups = module.num_key_value_groups
            q = q[0, :, 0].float().reshape(keys.shape[1], groups, -1)
            scores = torch.matmul(q, keys[0].float().transpose(-1, -2)) * module.scaling
            mask = kwargs.get('attention_mask')
            if mask is not None:
                if mask.ndim == 4:
                    scores = scores + mask[0, 0, -1, :keys.shape[-2]].float()
                elif mask.ndim == 2:
                    scores = scores.masked_fill(~mask[0, :keys.shape[-2]].bool(), -torch.inf)
                else:
                    raise ValueError(f'Unsupported attention mask: {mask.shape}')
            weights = torch.softmax(scores, -1).reshape(-1, keys.shape[-2])
            query['attention'][index] = weights.cpu().numpy()
            actual = output[0] if isinstance(output, (tuple, list)) else output
            query['attention_output'][index] = self.array(actual[0, -1])
            if self.validate_attention:
                values = cache.layers[index].values if hasattr(cache, 'layers') else cache.value_cache[index]
                context = torch.matmul(weights.reshape(keys.shape[1], groups, -1), values[0].float())
                rebuilt = module.o_proj(context.reshape(1, 1, -1).to(h.dtype))[0, 0].float()
                reference = actual[0, -1].float()
                query.setdefault('attention_validation_error', {})[index] = float(
                    (rebuilt - reference).norm() / reference.norm().clamp(min=1e-6))
        return hook

    def layer_hook(self, index):
        def hook(module, args, kwargs, output):
            if not self.active:
                return
            h = kwargs.get('hidden_states', args[0] if args else None)
            out = output[0] if isinstance(output, (tuple, list)) else output
            query = self.queries[-1]
            query['residual_in'][index] = self.array(h[0, -1])
            query['residual_out'][index] = self.array(out[0, -1])
            if h.shape[1] > 1:
                labels = torch.as_tensor(self.labels, device=out.device, dtype=torch.long)
                counts = torch.bincount(labels, minlength=self.segment_count)
                sums = torch.zeros(self.segment_count, out.shape[-1], device=out.device, dtype=torch.float32)
                sums.index_add_(0, labels, out[0].float())
                self.pooled[index] = (sums / counts.clamp(min=1)[:, None]).cpu().numpy()
        return hook

    def mlp_hook(self, index):
        def hook(module, args, output):
            if self.active:
                self.queries[-1]['mlp_output'][index] = self.array(output[0, -1])
        return hook

    def final_hook(self, module, args, output):
        if self.active:
            h = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
            self.queries[-1]['final_hidden'] = self.array(h[0, -1])

    def geometry_hook(self, module, args, output):
        if self.active:
            if 'geometry' in self.arrays:
                raise RuntimeError('VGGT merger ran twice in one decision')
            self.arrays['geometry'] = self.array(output)

    def generate(self, **kwargs):
        if not self.active:
            return self.original_generate(**kwargs)
        if self.step['history_indices'] is not None:
            raise RuntimeError('Extra generation pass attempted during passive capture')
        ids = kwargs['input_ids'][0].tolist()
        labels, spans, clauses = prompt_layout(ids, self.manifest['instruction'], self.policy.tokenizer, self.model.config)
        self.labels, self.segment_count = labels, len(FIXED_SEGMENTS) + len(clauses)
        self.pooled = {}
        self.step.update(history_indices=self.policy._history_indices(), seq_len=len(ids),
                         prompt_text=self.policy.tokenizer.decode(ids, skip_special_tokens=False),
                         image_spans=spans, image_grids=kwargs['image_grid_thw'].tolist(),
                         spatial_merge_size=self.policy.processor.image_processor.merge_size,
                         segments=FIXED_SEGMENTS + [f'clause_{i}' for i in range(len(clauses))],
                         clause_spans=clauses,
                         clauses=[self.manifest['instruction'][a:b] for a, b in clauses],
                         segment_counts=np.bincount(labels, minlength=self.segment_count).tolist(),
                         activation_dtype='float32', native_dtype=str(next(self.model.parameters()).dtype))
        self.arrays.update(input_ids=np.asarray(ids, np.int64), segment_ids=labels,
                           input_attention_mask=kwargs['attention_mask'].detach().cpu().numpy(),
                           image_grid_thw=kwargs['image_grid_thw'].detach().cpu().numpy())
        settings = dict(kwargs, return_dict_in_generate=True, output_logits=True)
        result = self.original_generate(**settings)
        generated = result.sequences[0, len(ids):].tolist()
        if len(self.queries) != len(generated) or len(result.logits) != len(generated):
            raise RuntimeError('Generation query/token count mismatch')
        self.arrays['generated_token_ids'] = np.asarray(generated, np.int64)
        self.arrays['logits'] = np.stack([self.array(row[0]) for row in result.logits])
        self.arrays['segment_residual_mean'] = np.stack([self.pooled[i] for i in range(len(self.layers))])
        for n, query in enumerate(self.queries):
            for name in ('attention', 'attention_output', 'residual_in', 'residual_out', 'mlp_output'):
                if len(query[name]) != len(self.layers):
                    raise RuntimeError(f'Missing {name} layers for query {n}')
                self.arrays[f'q{n}_{name}'] = np.stack([query[name][i] for i in range(len(self.layers))])
            self.arrays[f'q{n}_final_hidden'] = query['final_hidden']
            if self.validate_attention:
                self.arrays[f'q{n}_attention_validation_error'] = np.asarray(
                    [query['attention_validation_error'][i] for i in range(len(self.layers))], np.float32)
            self.step['queries'].append({'index': n, 'prefix_token_ids': generated[:n],
                                         'predicted_token_id': generated[n],
                                         'predicted_token': self.policy.tokenizer.decode([generated[n]]),
                                         'key_length': len(ids) + n})
        self.step['generated_tokens'] = generated
        return result if kwargs.get('return_dict_in_generate', False) else result.sequences

    def commit(self, response):
        self.active = False
        if self.step is None or not self.step['queries'] or 'geometry' not in self.arrays:
            raise RuntimeError('Missing decision activations or geometry')
        if self.writer_error:
            raise RuntimeError(f'Activation writer failed: {self.writer_error}')
        metadata, arrays = self.step, self.arrays
        metadata.update(response=response.copy(), cache_frames=self.policy.vggt_cache_frames())
        started = time.monotonic()
        # Ownership of detached CPU arrays moves to the writer; no GPU state is
        # read there and the next decision gets fresh containers.
        self.write_queue.put((self.directory, metadata, arrays), timeout=5)
        self.submitted_steps += 1
        with self.writer_lock:
            self.manifest['submitted_steps'] = self.submitted_steps
        self.arrays, self.queries, self.query_norm, self.pooled = {}, [], {}, {}
        self.step = None
        return {'accepted': True, 'complete': False, 'step': metadata['step'],
                'directory': str(self.directory), 'pending': self.write_queue.unfinished_tasks,
                'enqueue_seconds': time.monotonic() - started}

    def write_loop(self):
        while True:
            directory, metadata, arrays = self.write_queue.get()
            try:
                reference = self.write_step(directory, metadata, arrays)
                with self.writer_lock:
                    self.manifest['steps'].append(reference)
                    self.save_manifest()
            except Exception as error:
                self.writer_error = f'{type(error).__name__}: {error}'
                with self.writer_lock:
                    self.manifest['errors'].append({'utc': utc_now(), 'step': metadata['step'],
                                                     'error': self.writer_error})
                    try:
                        self.save_manifest()
                    except OSError:
                        pass
                print(f'[activation-writer] {self.writer_error}', flush=True)
            finally:
                self.write_queue.task_done()

    def write_step(self, directory, metadata, arrays):
        start = time.monotonic()
        index = metadata['step']
        path = directory / 'steps' / f'{index:04d}.npz'
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for name, array in arrays.items():
                    with archive.open(name + '.npy', 'w', force_zip64=True) as entry:
                        np.lib.format.write_array(entry, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        metadata.update(
                         committed_utc=utc_now(), arrays={key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                                                         for key, value in arrays.items()},
                         tensor_sha256=digest(path), tensor_bytes=path.stat().st_size,
                         write_seconds=time.monotonic() - start)
        atomic_json(path.with_suffix('.json'), metadata)
        reference = {'step': index, 'tensors': f'steps/{index:04d}.npz', 'metadata': f'steps/{index:04d}.json',
                     'sha256': metadata['tensor_sha256'], 'bytes': metadata['tensor_bytes']}
        return reference

    def fail(self, error):
        self.active = False
        if self.directory is not None:
            with self.writer_lock:
                self.manifest['errors'].append({'utc': utc_now(), 'step': None if self.step is None else self.step['step'],
                                                 'error': str(error)})
                self.save_manifest()
        self.arrays, self.queries, self.query_norm, self.pooled = {}, [], {}, {}
        self.step = None

    def finish(self, reason):
        self.active = False
        if self.directory is None:
            return {'complete': False, 'reason': 'no_recording'}
        self.write_queue.join()
        with self.writer_lock:
            complete = not self.manifest['errors'] and len(self.manifest['steps']) == self.submitted_steps
            self.manifest.update(finished=True, ended_utc=utc_now(), stop_reason=reason,
                                 status='complete' if complete else 'partial')
            self.save_manifest()
        return {'complete': self.manifest['status'] == 'complete', 'steps': len(self.manifest['steps']),
                'directory': str(self.directory), 'bytes': sum(step['bytes'] for step in self.manifest['steps'])}
