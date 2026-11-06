"""How good is depth from FutureNav's own VGGT, run exactly as the policy runs it?

FutureNav streams VGGT one frame at a time (past_frame_idx=0, KV cache trimmed to the first 8 + last 48 frames,
bf16 autocast, 644-wide input from load_and_preprocess_images) and uses only aggregator layer -2. The checkpoint
also carries VGGT's depth_head (and point_head), never called. This script replays collect.py's expert frames
through that same streaming aggregator, runs depth_head on the current frame, and compares with Habitat's
ground-truth depth.

Per frame (GT pixels with 0.1 m < d < 10 m):
  shape    AbsRel / delta<1.25 after per-frame median scaling (scale-free quality)
  floor    metric depth from a RANSAC floor-plane fit in the predicted point cloud, scaled so the camera sits at
           its known 1.25 m: AbsRel, and the floor scale vs the median-matched scale (ratio 1 = perfect)
  once     the episode's first-frame floor scale reused for every frame (does VGGT's scale stay constant?)
  obstacle top-down occupancy (5 cm cells, points 0.1-1.5 m above the floor, out to 4 m) from GT and from
           floor-scaled VGGT; every waypoint-head anchor's first 4 intervals swept with a 0.18 m radius;
           collision agreement (precision / recall of "anchor collides") and free distance straight ahead.

    PYTHONPATH=src:. ~/miniforge3/envs/futurenav/bin/python experiments/vggt_depth/eval_depth.py
Output: outputs/vggt_depth/results.json (+ per-frame rows), printed summary.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from qwen_vl.model.modeling_qwen3_vl import StartRecentKVCache  # noqa: E402
from qwen_vl.model.vggt.models.vggt import VGGT  # noqa: E402
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=str(ROOT / "outputs/vggt_depth/frames"))
ap.add_argument("--ckpt", default=str(ROOT / "pretrain_models/FutureNav-4B-Base"))
ap.add_argument("--recent", type=int, default=48, help="KV cache recent frames (FutureNav: 48)")
ap.add_argument("--out", default=str(ROOT / "outputs/vggt_depth/results.json"))
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-frames", type=int, default=0, help="per episode, 0 = all (for dry runs)")
ap.add_argument("--cam-height", type=float, default=1.25, help="camera height the frames were collected at (m)")
ap.add_argument("--dump-dir", default=None, help="also save raw VGGT depth/conf per episode here (scale_study.py)")
a = ap.parse_args()

CAM_H, HFOV, W0, H0 = a.cam_height, 79.0, 640, 480
CELL, RANGE, R_ROBOT = 0.05, 4.0, 0.18
H_LO, H_HI = 0.1, 1.5


def load_vggt():
    m = VGGT()
    m.camera_head = None
    m.track_head = None
    idx = json.load(open(Path(a.ckpt) / "model.safetensors.index.json"))["weight_map"]
    sd = {}
    for shard in sorted({f for k, f in idx.items() if k.startswith("vggt.")}):
        with safe_open(str(Path(a.ckpt) / shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("vggt."):
                    sd[k[5:]] = f.get_tensor(k)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VGGT weights: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]}")
    return m.to(a.device).eval().requires_grad_(False)


def intrinsics(h, w):
    f0 = (W0 / 2) / math.tan(math.radians(HFOV / 2))     # Habitat pinhole, square pixels at 640x480
    return f0 * w / W0, f0 * h / H0, w / 2, h / 2


def backproject(d, K):
    fx, fy, cx, cy = K
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
    return np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], -1)   # camera frame: x right, y down, z forward


def fit_floor(P, mask, rng, iters=200, tol=0.03):
    """RANSAC plane on candidate floor points (lower 45% of the image). Returns (n, c) with n.p + c = 0,
    n pointing up (towards the camera), or None."""
    Q = P[mask]
    Q = Q[np.isfinite(Q).all(1) & (Q[:, 2] > 0)]
    if len(Q) < 200:
        return None
    Q = Q[rng.choice(len(Q), min(len(Q), 20000), replace=False)]
    scale = np.median(Q[:, 2])
    best, best_in = None, 0
    for _ in range(iters):
        p = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.8:          # floor normal is close to camera -y for a level camera
            continue
        c = -n @ p[0]
        inl = np.abs(Q @ n + c) < tol * scale
        if inl.sum() > best_in:
            best, best_in = (n, c), inl.sum()
    if best is None:
        return None
    n, c = best
    inl = Q[np.abs(Q @ n + c) < tol * scale]
    cen = inl.mean(0)
    n = np.linalg.svd(inl - cen)[2][-1]          # least-squares refit on inliers
    c = -n @ cen
    if n @ np.array([0, -1, 0]) < 0:              # make n point up, so height = n.p + c > 0 above the floor
        n, c = -n, -c
    return n, c


def occupancy(P, height):
    """Top-down grid (forward x lateral-left) of points 0.1-1.5 m above the floor within RANGE."""
    fwd, left = P[..., 2], -P[..., 0]
    m = (height > H_LO) & (height < H_HI) & (fwd > 0.05) & (fwd < RANGE) & (np.abs(left) < RANGE)
    g = np.zeros((int(RANGE / CELL), int(2 * RANGE / CELL)), bool)
    i = (fwd[m] / CELL).astype(int)
    j = ((left[m] + RANGE) / CELL).astype(int)
    ok = (i < g.shape[0]) & (j < g.shape[1])
    g[i[ok], j[ok]] = True
    return g


def anchor_paths():
    C = torch.load(Path.home() / "lora_waypoint_20261104/anchors.pt", weights_only=True).float().numpy()
    xy = C[:, :16].reshape(-1, 8, 2)[:, :4]       # first 4 intervals = what the server executes
    paths = []
    for p in xy:
        pts = np.concatenate([[[0, 0]], p])
        seg = [pts[k] + (pts[k + 1] - pts[k]) * t for k in range(4) for t in np.linspace(0, 1, 6)]
        paths.append(np.array(seg))
    return paths


def collides(g, path):
    r = int(math.ceil(R_ROBOT / CELL))
    for fx, fl in path:
        i0, j0 = int(fx / CELL), int((fl + RANGE) / CELL)
        sub = g[max(i0 - r, 0):i0 + r + 1, max(j0 - r, 0):j0 + r + 1]
        if sub.size and sub.any():
            ii, jj = np.nonzero(sub)
            ii = ii + max(i0 - r, 0)
            jj = jj + max(j0 - r, 0)
            if (np.hypot((ii + .5) * CELL - fx, (jj + .5) * CELL - RANGE - fl) <= R_ROBOT).any():
                return True
    return False


def free_ahead(g, half=R_ROBOT):
    j0, j1 = int((RANGE - half) / CELL), int(math.ceil((RANGE + half) / CELL))
    rows = np.nonzero(g[:, j0:j1].any(1))[0]
    return RANGE if len(rows) == 0 else rows[0] * CELL


def absrel(pred, gt, m):
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


def delta(pred, gt, m):
    r = np.maximum(pred[m] / gt[m], gt[m] / pred[m])
    return float(np.mean(r < 1.25))


@torch.no_grad()
def main():
    model = load_vggt()
    cache = StartRecentKVCache(start_size=8, recent_size=a.recent, k_seq_dim=2, v_seq_dim=2)
    paths = anchor_paths()
    rng = np.random.default_rng(0)
    rows = []
    files = sorted(Path(a.frames).glob("*.npz"))
    print(f"{len(files)} episodes, cache 8+{a.recent}", flush=True)
    for f in files:
        z = np.load(f)
        pkv = [None] * model.aggregator.depth
        s_once = None
        dump = {}
        for t in range(len(z["rgb"]) if not a.max_frames else min(a.max_frames, len(z["rgb"]))):
            img = load_and_preprocess_images([Image.fromarray(z["rgb"][t])])[0]     # 3 x h x 644
            h, w = img.shape[1:]
            images = img[None, None].to(a.device)
            with torch.amp.autocast(a.device, dtype=torch.bfloat16):
                tokens, psi, pkv = model.aggregator(images, past_key_values=pkv, use_cache=True, past_frame_idx=0)
            pkv = cache(pkv)
            with torch.amp.autocast(a.device, enabled=False):
                used = set(model.depth_head.intermediate_layer_idx)
                d, conf = model.depth_head([x.float() if i in used else None for i, x in enumerate(tokens)],
                                           images=images, patch_start_idx=psi)
            d = d[0, 0, ..., 0].cpu().numpy()
            conf = conf[0, 0].cpu().numpy()
            del tokens
            if a.dump_dir:
                dump.setdefault("depth", []).append(d.astype(np.float16))
                dump.setdefault("conf", []).append(conf.astype(np.float16))
            gt = np.array(Image.fromarray(z["depth"][t].astype(np.float32)).resize((w, h), Image.NEAREST))
            m = (gt > 0.1) & (gt < 10) & np.isfinite(d) & (d > 0)
            if m.sum() < 1000:
                continue
            K = intrinsics(h, w)
            s_med = np.median(gt[m]) / np.median(d[m])
            row = {"ep": f.stem, "t": t, "s_med": float(s_med), "shape_absrel": absrel(d * s_med, gt, m),
                   "shape_d125": delta(d * s_med, gt, m)}
            P = backproject(d, K)
            floor_mask = np.zeros_like(m)
            floor_mask[int(h * 0.55):] = True
            fl = fit_floor(P, floor_mask & (conf > np.percentile(conf, 20)), rng)
            Pg = backproject(gt, K)
            g_gt = occupancy(Pg, CAM_H - Pg[..., 1])
            row["gt_free_ahead"] = free_ahead(g_gt)
            cg = np.array([collides(g_gt, p) for p in paths])
            # upper bound: oracle (median-matched) scale and the known level camera pose
            Po = P * s_med
            co = np.array([collides(occupancy(Po, CAM_H - Po[..., 1]), p) for p in paths])
            row.update(o_tp=int((cg & co).sum()), o_fp=int((~cg & co).sum()), o_fn=int((cg & ~co).sum()))
            if fl is not None:
                n, c = fl
                s_floor = CAM_H / c                      # c = camera height above the fitted floor (pred units)
                if s_once is None:
                    s_once = s_floor
                row.update(s_floor=float(s_floor), floor_ratio=float(s_floor / s_med),
                           floor_absrel=absrel(d * s_floor, gt, m), floor_d125=delta(d * s_floor, gt, m),
                           once_absrel=absrel(d * s_once, gt, m), once_ratio=float(s_once / s_med))
                Pp = P * s_floor
                g_p = occupancy(Pp, Pp @ n + c * s_floor)        # height above the fitted floor, metres
                row["pred_free_ahead"] = free_ahead(g_p)
                cp = np.array([collides(g_p, p) for p in paths])
                row.update(n_coll_gt=int(cg.sum()), n_coll_pred=int(cp.sum()), tp=int((cg & cp).sum()),
                           fp=int((~cg & cp).sum()), fn=int((cg & ~cp).sum()))
            rows.append(row)
        if a.dump_dir:      # raw VGGT depth + confidence per frame (pred units), for offline scale studies
            Path(a.dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(a.dump_dir) / f.name, **{k: np.stack(v) for k, v in dump.items()})
        print(f"  {f.stem}: {len(z['rgb'])} frames, last row {rows[-1] if rows else None}", flush=True)

    def agg(k, R=rows):
        v = [r[k] for r in R if k in r]
        return (float(np.median(v)), float(np.mean(v)), len(v)) if v else None

    res = {"cache": [8, a.recent], "cam_height": CAM_H, "frames": len(rows), "episodes": len(files)}
    otp, ofp, ofn = (sum(r[k] for r in rows) for k in ("o_tp", "o_fp", "o_fn"))
    res["anchor_collision_oracle_scale"] = {"precision": otp / max(otp + ofp, 1), "recall": otp / max(otp + ofn, 1)}
    for k in ("shape_absrel", "shape_d125", "floor_absrel", "floor_d125", "once_absrel", "floor_ratio", "once_ratio"):
        res[k] = agg(k)
    res["floor_fit_rate"] = float(np.mean(["s_floor" in r for r in rows]))
    # raw VGGT scale stability within an episode: coefficient of variation of the median-matched scale
    cv = []
    for e in {r["ep"] for r in rows}:
        s = np.array([r["s_med"] for r in rows if r["ep"] == e])
        cv.append(float(s.std() / s.mean()))
    res["raw_scale_cv_per_episode"] = (float(np.median(cv)), float(np.max(cv)))
    tp, fp, fn = (sum(r.get(k, 0) for r in rows) for k in ("tp", "fp", "fn"))
    res["anchor_collision"] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                               "gt_colliding_frac": float(np.mean([r["n_coll_gt"] / len(paths) for r in rows if "n_coll_gt" in r]))}
    fa = [(r["gt_free_ahead"], r["pred_free_ahead"]) for r in rows if "pred_free_ahead" in r]
    fa = np.array(fa)
    near = fa[:, 0] < 1.5
    res["free_ahead"] = {"median_abs_err_m": float(np.median(np.abs(fa[:, 0] - fa[:, 1]))),
                         "median_abs_err_m_when_gt_lt_1.5m": float(np.median(np.abs(fa[near, 0] - fa[near, 1]))) if near.any() else None,
                         "blocked_within_1m": {"gt": int((fa[:, 0] < 1).sum()), "pred": int((fa[:, 1] < 1).sum()),
                                               "both": int(((fa[:, 0] < 1) & (fa[:, 1] < 1)).sum())}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
