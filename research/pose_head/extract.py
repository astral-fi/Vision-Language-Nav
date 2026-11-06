"""Features for the future-pose head: one prefill per sampled step, frozen FutureNav
and lora_v1 (same weights, adapter off / on), same inputs as experiments/s2_lora.

Per step (the s2_lora samples, train + val):
  h_base, h_lora  (4, 2560) fp16  last prompt token at LLM layers LAYERS
  vg              (24, 2560) fp16 the current frame's merged VGGT tensor, pooled 4x6
  pose            (8, 3)          expert's next 8 local poses (x fwd, y left, dyaw rad)
  act             (8,)            expert's next 8 actions
Writes outputs/pose_head/feats/<eid>.npz (t, h_base, h_lora, vg, pose, act). Resumable.

    python extract.py --split val
    python extract.py --split train --max-episodes 1200
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "s2_lora"))
from common import K, OUT, Episode, episode_path, load_policy  # noqa: E402

DST = os.path.join(os.path.dirname(OUT), "pose_head", "feats")
LAYERS = (18, 22, 26, 36)     # output of decoder layer d (layers[d-1]); 36 = after the final norm
ADAPTER = os.path.join(OUT, "lora_v1", "adapter_final")


def pool(x, hw=(17, 23), out=(4, 6)):
    g = x.view(1, hw[0], hw[1], -1).permute(0, 3, 1, 2).float()
    return F.adaptive_avg_pool2d(g, out).flatten(2)[0].T          # (24, 2560)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-episodes", type=int, default=0, help="first N episodes of the split")
    a = ap.parse_args()
    from peft import PeftModel
    with open(os.path.join(OUT, "samples_%s.json" % a.split)) as f:
        rows = json.load(f)
    rows = rows[:a.max_episodes] if a.max_episodes else rows
    os.makedirs(DST, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(DST, r["eid"] + ".npz"))]
    print("%s: %d episodes, %d to do" % (a.split, len(rows), len(todo)), flush=True)
    policy = load_policy()
    base = policy.model
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    lm = base.model.language_model
    grab = {}

    def hook(d):
        def f(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            grab[d] = h[0, -1].detach()
        return f
    for d in LAYERS[:-1]:
        lm.layers[d - 1].register_forward_hook(hook(d))
    lm.norm.register_forward_hook(hook(LAYERS[-1]))
    t0, n = time.time(), 0
    for k, r in enumerate(todo):
        ep = Episode(episode_path(r["scene"], r["eid"]))
        e = np.load(episode_path(r["scene"], r["eid"]))
        v = np.load(os.path.join(OUT, "vggt", r["eid"] + ".npz"))
        vg = dict(zip(v["t"].tolist(), v["x"]))
        out = {"t": [], "h_base": [], "h_lora": [], "vg": [], "pose": [], "act": []}
        for t in r["steps"]:
            x = torch.from_numpy(vg[t]).to(base.device, torch.bfloat16)
            if x.shape[0] != 17 * 23:
                raise RuntimeError("unexpected VGGT grid %s" % (tuple(x.shape),))
            inputs, _, _ = policy.build_inputs(ep.shown(t), ep.meta["instruction"])
            inputs["images_vggt"] = None
            for arm in ("base", "lora"):
                ctx = model.disable_adapter() if arm == "base" else torch.autocast("cuda", enabled=False)
                with torch.no_grad(), ctx, base.vggt_feature_reuse(inject=x):
                    grab.clear()
                    model(**inputs, use_cache=False)
                out["h_" + arm].append(torch.stack([grab[d] for d in LAYERS]).half().cpu().numpy())
            out["t"].append(t)
            out["vg"].append(pool(x).half().cpu().numpy())
            out["pose"].append(e["chunk_pose"][t, :K])
            out["act"].append(e["chunk_act"][t, :K])
            n += 1
        dst = os.path.join(DST, r["eid"] + ".npz")
        np.savez(dst[:-4] + ".tmp.npz", **{kk: np.stack(vv) for kk, vv in out.items()})
        os.replace(dst[:-4] + ".tmp.npz", dst)
        if k % 20 == 0:
            el = time.time() - t0
            print("%d/%d  %.2f s/step  eta %.1f h" % (k + 1, len(todo), el / n, el / (k + 1) * (len(todo) - k - 1) / 3600),
                  flush=True)
    print("done %s" % a.split, flush=True)


if __name__ == "__main__":
    main()
