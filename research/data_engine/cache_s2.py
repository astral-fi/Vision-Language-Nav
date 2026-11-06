"""Frozen-FutureNav (System 2) features for the motion head. futurenav env.

Replays a collected episode (collect.py) through FutureNav exactly as the dual
system runs online (MOTION_HEAD_DESIGN.md): every frame enters the history and
the VGGT spatial memory (advance_vggt), and the language model is called every R
steps. That is bit-identical to calling FutureNav on every step (checked
2026-10-30: diff 0.0, same actions), at ~0.37 s/step instead of 0.75.

Per language-model call: the last prompt token's residual after decoder layers
LAYERS (index k = output of layers[k-1]) and after the final norm, plus
FutureNav's action. Per step: the current frame's VGGT tokens (aggregator layer
-2, the ones FutureNav itself reads), average-pooled from the 35 x 46 patch grid
to GRID, kept at their full 2048 dims (the head learns its own projection;
~196 KB per frame, storage is not a constraint).

Output outputs/data_engine/s2_v1/<scene>/<eid>.npz:
  t (n,)                     step of each language-model call (0, R, 2R, ...)
  h (n, len(LAYERS)+1, 2560) float16   LAYERS in order, then the final norm
  action (n,)                FutureNav's action at the call (0 STOP .. 3 RIGHT)
  v (T, GRID_H*GRID_W, 2048) float16   pooled VGGT tokens, every step

    python cache_s2.py --max 1500
"""
from __future__ import print_function

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "deploy"), os.path.join(ROOT, "src"), ROOT):
    sys.path.insert(0, p)

SRC = os.path.join(ROOT, "outputs/data_engine/expert_v1")
OUT = os.path.join(ROOT, "outputs/data_engine/s2_v1")
CKPT = os.path.join(ROOT, "pretrain_models/FutureNav-4B-Base")
R = 4
LAYERS = (18, 22, 26)
GRID_H, GRID_W = 6, 8


def chosen(max_n):
    """Deterministic subset: episodes ranked by a hash of their id."""
    files = sorted(glob.glob(os.path.join(SRC, "*", "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    files.sort(key=lambda f: hashlib.md5(os.path.basename(f).encode()).hexdigest())
    return files[:max_n] if max_n else files


class Hooks(object):
    """Prefill-only last-token residuals per decoder layer, and pooled VGGT tokens per frame."""

    def __init__(self, model):
        import torch
        self.torch = torch
        self.on, self.h, self.v = False, {}, []
        layers = model.model.language_model.layers
        for k in LAYERS:
            layers[k - 1].register_forward_hook(self._layer(k))
        model.vggt.aggregator.register_forward_hook(self._vggt)

    def _layer(self, k):
        def hook(module, args, output):
            if not self.on:
                return
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h.dim() == 3 and h.shape[1] > 1:
                self.h[k] = h[0, -1].detach().float().cpu().numpy()
        return hook

    def _vggt(self, module, args, output):
        F = self.torch.nn.functional
        toks, psi = output[0][-2], output[1]
        _, _, _, H, W = args[0].shape
        g = toks[0, -1, psi:].float().reshape(H // 14, W // 14, -1).permute(2, 0, 1)
        self.v.append(F.adaptive_avg_pool2d(g[None], (GRID_H, GRID_W))[0].flatten(1).T.cpu().numpy())


def frames(src):
    z = np.load(src)
    meta = json.loads(bytes(z["meta_json"]).decode())
    blob, off = z["jpg_blob"], z["jpg_off"]
    return meta, [np.ascontiguousarray(cv2.imdecode(blob[off[t]:off[t + 1]], cv2.IMREAD_COLOR)[:, :, ::-1])
                  for t in range(len(off) - 1)]


def run(policy, hooks, src):
    meta, fr = frames(src)
    policy.reset()
    hooks.v = []
    ts, hs, acts = [], [], []
    for t, rgb in enumerate(fr):
        if t % R:
            policy.advance_vggt(rgb)
            continue
        hooks.on, hooks.h = True, {}
        try:
            res = policy.act(rgb, meta["instruction"], want_hidden=True)
        finally:
            hooks.on = False
        if len(hooks.h) != len(LAYERS) or policy._hidden is None:
            raise RuntimeError("missing hidden states at step %d of %s" % (t, src))
        if not res["vggt_cache_ok"]:
            raise RuntimeError("VGGT cache advanced incorrectly at step %d of %s" % (t, src))
        ts.append(t)
        hs.append(np.stack([hooks.h[k] for k in LAYERS] + [policy._hidden.astype(np.float32)]))
        acts.append(res["action"])
    if len(hooks.v) != len(fr):
        raise RuntimeError("expected %d VGGT frames, got %d in %s" % (len(fr), len(hooks.v), src))
    return {"t": np.asarray(ts, np.int16), "h": np.asarray(hs, np.float16), "action": np.asarray(acts, np.int8),
            "v": np.stack(hooks.v).astype(np.float16)}                # T x 48 x 2048


def load_policy():
    from futurenav_core import FutureNavPolicy
    policy = FutureNavPolicy(CKPT, attn_implementation="sdpa")
    policy.enable_hidden_capture()
    return policy, Hooks(policy.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    files = chosen(a.max)[a.shard::a.shards]
    todo = [f for f in files if not os.path.exists(f.replace(SRC, OUT))]
    print("cache_s2: %d chosen, %d to do" % (len(files), len(todo)), flush=True)
    policy, hooks = load_policy()
    t0, steps = time.time(), 0
    for n, src in enumerate(todo):
        out = run(policy, hooks, src)
        dst = src.replace(SRC, OUT)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez(dst + ".tmp.npz", **out)
        os.replace(dst + ".tmp.npz", dst)
        steps += len(out["v"])
        if n % 20 == 0 or n == len(todo) - 1:
            el = time.time() - t0
            print("%d/%d  %.2f s/step  eta %.1f h" % (n + 1, len(todo), el / steps,
                                                     el / (n + 1) * (len(todo) - n - 1) / 3600), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
