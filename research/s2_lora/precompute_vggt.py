"""Gate 4 of PLAN.md: streaming VGGT features for the sampled steps.

For each episode in samples_<split>.json, stream frames 0..max(step) through VGGT
exactly as the server does (StreamVGGT) and save the current frame's merged,
pre-interpolate tensor at each sampled step:
  outputs/s2_lora/vggt/<eid>.npz   t (n,), x (n, 391, 2560) fp16
Resumable (skips existing files), atomic writes, stops below MIN_FREE_GB.
--shard i/n splits episodes across processes.

    python precompute_vggt.py --split train --shard 0/2
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

from common import OUT, Episode, StreamVGGT, episode_path, load_policy

MIN_FREE_GB = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    with open(os.path.join(OUT, "samples_%s.json" % a.split)) as f:
        rows = json.load(f)[i::n]
    dst = os.path.join(OUT, "vggt")
    os.makedirs(dst, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(dst, r["eid"] + ".npz"))]
    print("%s shard %s: %d episodes, %d to do" % (a.split, a.shard, len(rows), len(todo)), flush=True)
    if not todo:
        return
    from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images  # noqa: F401
    policy = load_policy()
    stream = StreamVGGT(policy.model)
    t0, frames = time.time(), 0
    for k, r in enumerate(todo):
        if shutil.disk_usage(OUT).free / 1e9 < MIN_FREE_GB:
            print("stopping: less than %d GB free" % MIN_FREE_GB, flush=True)
            return
        ep = Episode(episode_path(r["scene"], r["eid"]))
        want = set(r["steps"])
        stream.reset()
        xs = []
        for t in range(max(want) + 1):
            x = stream.step(ep.frame(t), want=t in want)
            if x is not None:
                xs.append(x.to(torch.float16).cpu().numpy())
        frames += max(want) + 1
        out = os.path.join(dst, r["eid"] + ".npz")
        np.savez(out[:-4] + ".tmp.npz", t=np.asarray(sorted(want), np.int16), x=np.stack(xs))
        os.replace(out[:-4] + ".tmp.npz", out)
        if k % 20 == 0:
            el = time.time() - t0
            print("%d/%d  %.3f s/frame  eta %.1f h" % (k + 1, len(todo), el / frames,
                                                       el / (k + 1) * (len(todo) - k - 1) / 3600), flush=True)
    print("done %s shard %s" % (a.split, a.shard), flush=True)


if __name__ == "__main__":
    main()
