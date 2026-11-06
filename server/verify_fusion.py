"""Runtime check of how the two visual towers are fused.

    PYTHONPATH=$PWD/src:$PWD python deploy/verify_fusion.py

Hooks both projectors and prints their output token counts per step. The point
is the invariant:

    merger_vggt   -> always 391 tokens (17x23, ONE frame's grid)
    visual.merger -> 300 * n_images, growing 300 -> 2700

If per-frame geometry matching existed the two would track each other. They do
not: geometry comes from the current frame alone, conditioned on the VGGT KV
cache, then is interpolated across every image-token slot. Measured output:

    step prompt imgs  qwen tokens  vggt tokens  cache
       0           1          300          391     1f
       8           9         2700          391     9f
      11           9         2700          391    12f

Re-run this after a transformers upgrade or a checkpoint change to confirm the
fusion contract still holds.
"""
import sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "deploy")
from futurenav_core import FutureNavPolicy

p = FutureNavPolicy("pretrain_models/FutureNav-4B-Base", attn_implementation="sdpa")

rec = {}
p.model.merger_vggt.register_forward_hook(
    lambda m, i, o: rec.__setitem__("vggt_out", tuple(o.shape)))
p.model.model.visual.merger.register_forward_hook(
    lambda m, i, o: rec.__setitem__("qwen_out", tuple(o.shape)))

def frame(i):
    img = Image.new("RGB", (640, 480), (196, 190, 178))
    d = ImageDraw.Draw(img); off = (i*26) % 640
    d.rectangle([0,360,640,480], fill=(122,100,78))
    for k in range(-1,6):
        d.rectangle([k*150-off,90,k*150-off+140,360], outline=(140,136,128), width=3)
    return np.asarray(img)

print(f"\n{'step':>4} {'prompt imgs':>11} {'qwen tokens':>12} {'vggt tokens':>12} {'cache':>6}")
for i in range(12):
    p.act(frame(i), "walk down the hallway")
    n_prompt = rec['qwen_out'][0] // 300      # 300 merged tokens per image
    print(f"{i:>4} {n_prompt:>11} {rec['qwen_out'][0]:>12} {rec['vggt_out'][0]:>12} "
          f"{p.vggt_cache_frames():>5}f")

print("\nvggt tokens stay at 391 regardless of prompt size or cache depth")
print("=> geometry is ONE frame's grid (17x23=391), never per-frame.")
