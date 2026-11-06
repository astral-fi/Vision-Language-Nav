"""Shared pieces for the FutureNav LoRA (future-steps) experiment. See PLAN.md.

Everything that touches model inputs goes through deploy/futurenav_core.py so that
training sees exactly what the server builds at inference:
  - prompt + Qwen image inputs: FutureNavPolicy.build_inputs
  - shown frames: the policy's linspace history selection (8 history + current)
  - VGGT: the current frame's merged, pre-interpolate tensor from the streaming
    cache (StreamVGGT below mirrors advance_vggt and the forward's merge code)
"""
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)

EXPERT = os.path.join(ROOT, "outputs/data_engine/expert_v1")
OUT = os.path.join(ROOT, "outputs/s2_lora")
CKPT = os.path.join(ROOT, "pretrain_models/FutureNav-4B-Base")
GT = os.path.join(ROOT, "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz")
NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
K = 8
MAX_HISTORY = 8


def episode_path(scene, eid):
    return os.path.join(EXPERT, scene, "%s.npz" % eid)


class Episode(object):
    """One expert_v1 episode: frames decoded on demand from the JPEG blob."""

    def __init__(self, path):
        e = np.load(path)
        self.meta = json.loads(bytes(e["meta_json"]).decode())
        self.blob, self.off = e["jpg_blob"], e["jpg_off"]
        self.chunk_act = e["chunk_act"]
        self.T = len(e["label"])
        self._cache = {}

    def frame(self, t):
        if t not in self._cache:
            b = self.blob[self.off[t]:self.off[t + 1]].tobytes()
            self._cache[t] = Image.open(io.BytesIO(b)).convert("RGB")
        return self._cache[t]

    def shown(self, t):
        """Frames the policy shows at step t (its _prepare_images, history_mode full)."""
        if t <= MAX_HISTORY:
            idx = list(range(t)) + [t]
        else:
            idx = [int(i) for i in np.linspace(0, t, MAX_HISTORY + 1, dtype=int)]
        return [self.frame(i) for i in idx]

    def answer(self, t):
        """The expert's next K actions as action words, cut after STOP."""
        words = []
        for a in self.chunk_act[t, :K]:
            words.append(NAMES[int(a)])
            if a == 0:
                break
        return " ".join(words)


def load_policy(attn="sdpa"):
    from futurenav_core import FutureNavPolicy
    return FutureNavPolicy(CKPT, attn_implementation=attn)


class StreamVGGT(object):
    """Streams frames through VGGT exactly as the server does (advance_vggt / the
    forward's evaluation branch) and returns the current frame's merged,
    pre-interpolate tensor: what vggt_feature_reuse(capture=True) captures."""

    def __init__(self, model):
        self.m = model
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.reset()

    def reset(self):
        self.m.past_key_values_vggt = None

    @torch.no_grad()
    def step(self, image, want=True):
        from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
        m = self.m
        frame = load_and_preprocess_images([image])[0].to(m.device)
        if m.past_key_values_vggt is None:
            m.past_key_values_vggt = [None] * m.vggt.aggregator.depth
        with torch.amp.autocast("cuda", dtype=self.dtype):
            tokens, psi, pkv = m.vggt.aggregator(frame.unsqueeze(0).unsqueeze(0),
                                                 past_key_values=m.past_key_values_vggt,
                                                 use_cache=True, past_frame_idx=0)
        m.past_key_values_vggt = m.kv_cache_vggt(pkv)
        if not want:
            return None
        # same steps as modeling_qwen3_vl.py forward (reference_frame "first": no flips)
        h, w = frame.shape[-2:]
        ms = m.config.vision_config.spatial_merge_size
        hg, wg = h // 14, w // 14
        hm, wm = hg // ms, wg // ms
        f = tokens[-2][0, :, psi:].view(1, hg, wg, -1)
        f = f[:, :hm * ms, :wm * ms, :].contiguous().view(1, hm, ms, wm, ms, -1)
        f = f.permute(0, 1, 3, 2, 4, 5).contiguous().to(m.visual.dtype)
        return m.merger_vggt(f)
