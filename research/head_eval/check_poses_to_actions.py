"""Sanity check of deploy/head_eval_server.poses_to_actions on the 256 waypoint anchors: every anchor must give
a plan, and re-integrating the first 4 actions should stay close to the anchor's poses."""
import math
import sys
import numpy as np
import torch

src = open(sys.argv[1]).read()
ns = {"math": math}
exec(src[src.index("PRIMITIVES ="):src.index("def step(")], ns)
f = ns["poses_to_actions"]
C = torch.load(sys.argv[2], weights_only=True).numpy()
errs, lens = [], []
for r in C:
    xy, yaw, st = r[:16].reshape(8, 2), r[16:24] / .5, (r[24:32] / 2) > .5
    poses = np.concatenate([xy, yaw[:, None]], 1).tolist()
    acts = f(poses, st.tolist())
    lens.append(len(acts))
    x = y = th = 0.
    e = 0.
    for h in range(4):
        a = acts[h] if h < len(acts) else 0
        if a == 1:
            x += .25 * math.cos(th); y += .25 * math.sin(th)
        elif a == 2:
            th += math.radians(15)
        elif a == 3:
            th -= math.radians(15)
        e = max(e, math.hypot(x - poses[h][0], y - poses[h][1]))
    errs.append(e)
print("anchors", len(C), "plan len min/mean", min(lens), round(float(np.mean(lens)), 2),
      "| max pos err over first 4 intervals: mean %.3f p95 %.3f max %.3f" % (np.mean(errs), np.percentile(errs, 95), max(errs)))
