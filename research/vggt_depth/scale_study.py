"""Offline study: robust metric scale for FutureNav's VGGT depth (raw depth dumped by eval_depth.py --dump-dir).

Scale s = metres per VGGT depth unit, per frame. Methods (none uses ground truth):
  ransac    RANSAC floor plane with the most inliers in the lower image; s = camera height / plane offset
            (what eval_depth.py does)
  lowest    up to 3 horizontal planes found in turn; take the LOWEST one with enough support (tables and seats
            are horizontal too, but higher than the floor)
  rowprior  known geometry of a level camera at height h: a floor pixel in row v has depth h*fy/(v-cy), so for any
            pixel below the horizon r = h*fy/(v-cy) / d_pred >= s (nothing is seen through the floor), with
            equality on floor pixels. s = the r value with the most pixels within +-5%, among candidates that leave
            < 5% of pixels below it (a 1-D RANSAC with the physical constraint)
  +smooth   per-episode filter on log s: accept a measurement within x1.35 of the running estimate (or 3 in a row
            that agree with each other), blend 50/50; carry the estimate forward when a frame has no valid one
  +bias     divide by the median (s_method / s_true) of the OTHER episodes (leave-one-episode-out)
  oracle    s = median(gt) / median(pred) (upper bound)
Metrics: s / s_true within 0.8-1.25; over-estimate > 1.25 (obstacles look farther: the dangerous side); anchor
collision precision / recall against GT occupancy (level camera, points 0.1-1.5 m above the floor, robot radius
0.18 m, first 4 anchor intervals), overall and on frames with GT free distance ahead < 1 m.

    ~/miniforge3/envs/futurenav/bin/python experiments/vggt_depth/scale_study.py
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vggt_depth"
HFOV, W0, H0 = 79.0, 640, 480
CELL, RANGE, R_ROBOT, H_LO, H_HI = 0.05, 4.0, 0.18, 0.1, 1.5
NI, NJ = int(RANGE / CELL), int(2 * RANGE / CELL)
rr = int(math.ceil(R_ROBOT / CELL))
DISK = np.hypot(*np.mgrid[-rr:rr + 1, -rr:rr + 1]) * CELL <= R_ROBOT


def intrinsics(h, w):
    f0 = (W0 / 2) / math.tan(math.radians(HFOV / 2))
    return f0 * w / W0, f0 * h / H0, w / 2, h / 2


def backproject(d, K):
    fx, fy, cx, cy = K
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
    return np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], -1)


def occupancy(P, height):
    fwd, left = P[..., 2], -P[..., 0]
    m = (height > H_LO) & (height < H_HI) & (fwd > 0.05) & (fwd < RANGE) & (np.abs(left) < RANGE)
    g = np.zeros((NI, NJ), bool)
    i, j = (fwd[m] / CELL).astype(int), ((left[m] + RANGE) / CELL).astype(int)
    ok = (i < NI) & (j < NJ)
    g[i[ok], j[ok]] = True
    return binary_dilation(g, DISK)        # cells the robot centre cannot enter


def anchor_cells():
    C = torch.load(Path.home() / "lora_waypoint_20261104/anchors.pt", weights_only=True).float().numpy()
    xy = C[:, :16].reshape(-1, 8, 2)[:, :4]
    pts = np.concatenate([np.zeros((len(xy), 1, 2)), xy], 1)
    t = np.linspace(0, 1, 6)
    seg = (pts[:, :-1, None] + (pts[:, 1:, None] - pts[:, :-1, None]) * t[None, None, :, None]).reshape(len(xy), -1, 2)
    i = np.clip((seg[..., 0] / CELL).astype(int), 0, NI - 1)
    j = np.clip(((seg[..., 1] + RANGE) / CELL).astype(int), 0, NJ - 1)
    return i, j


AI, AJ = anchor_cells()


def collisions(g):
    return g[AI, AJ].any(1)


def free_ahead(g):
    rows = np.nonzero(g[:, NJ // 2])[0]
    return RANGE if len(rows) == 0 else rows[0] * CELL


# ---------------------------------------------------------------- scale methods
def planes(P, mask, rng, k=3, iters=150, tol=0.03):
    Q = P[mask]
    Q = Q[np.isfinite(Q).all(1) & (Q[:, 2] > 0)]
    if len(Q) < 300:
        return []
    Q = Q[rng.choice(len(Q), min(len(Q), 15000), replace=False)]
    n0, out = len(Q), []
    for _ in range(k):
        if len(Q) < 200:
            break
        scale = np.median(Q[:, 2])
        best, best_in = None, None
        for _ in range(iters):
            p = Q[rng.choice(len(Q), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            if abs(n[1]) < 0.9:
                continue
            inl = np.abs(Q @ n + (-n @ p[0])) < tol * scale
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = n, inl
        if best is None or best_in.sum() < 0.05 * n0:
            break
        R = Q[best_in]
        cen = R.mean(0)
        n = np.linalg.svd(R - cen)[2][-1]
        if n[1] > 0:                       # normal points up (camera -y), so c = camera height above the plane
            n = -n
        c = -n @ cen
        out.append((c, best_in.sum() / n0))
        Q = Q[~best_in]
    return out


def s_ransac(P, mask, rng, H):
    pl = planes(P, mask, rng, k=1)
    return H / pl[0][0] if pl and pl[0][0] > 0 else None


def s_lowest(P, mask, rng, H):
    pl = [c for c, f in planes(P, mask, rng, k=3) if c > 0]
    return H / max(pl) if pl else None


def s_rowprior(d, conf_ok, K, H):
    fx, fy, cx, cy = K
    h, w = d.shape
    v = np.arange(h)[:, None] + .5
    below = (v - cy) > 0.05 * h                     # a few rows under the horizon: depth too far / unstable
    m = below & conf_ok & (d > 0) & np.isfinite(d)
    if m.sum() < 2000:
        return None
    r = np.log((H * fy / np.maximum(v - cy, 1e-3) * np.ones((1, w)))[m] / d[m])
    r = np.sort(r)
    grid = np.linspace(r[int(.001 * len(r))], r[int(.6 * len(r))], 200)
    lo = np.searchsorted(r, grid - 0.05)
    hi = np.searchsorted(r, grid + 0.05)
    below_frac = np.searchsorted(r, grid - 0.10) / len(r)
    support = (hi - lo) / len(r)
    ok = below_frac < 0.05
    if not ok.any():
        return None
    k = np.argmax(np.where(ok, support, -1))
    if support[k] < 0.08:                          # not enough floor in view
        return None
    return float(np.exp(grid[k]))


def cam_axes(pos, yaw, H):
    """World position of the camera and its right / down / forward axes (Habitat: y up, agent forward -Z)."""
    c = np.array([pos[0], pos[1] + H, pos[2]])
    return c, np.array([math.cos(yaw), 0, -math.sin(yaw)]), np.array([0, -1.0, 0]), np.array([-math.sin(yaw), 0, -math.cos(yaw)])


def s_odom(d0, d1, conf0, pose0, pose1, K, H, min_move=0.1):
    """Scale from known motion between two frames (Habitat pose = ideal odometry): back-project frame 0 at scale s,
    move it into frame 1, and compare with frame 1's depth at scale s. Only when the camera translated."""
    c0, r0, dn0, f0 = cam_axes(*pose0, H)
    c1, r1, dn1, f1 = cam_axes(*pose1, H)
    if np.linalg.norm(c1 - c0) < min_move:
        return None
    fx, fy, cx, cy = K
    h, w = d0.shape
    m = (conf0 > np.percentile(conf0, 20)) & (d0 > 0) & np.isfinite(d0)
    m[::2] = False
    m[:, ::2] = False                                # subsample
    v, u = np.nonzero(m)
    x, y, z = (u + .5 - cx) / fx * d0[v, u], (v + .5 - cy) / fy * d0[v, u], d0[v, u]
    best, best_err = None, None
    for s in np.exp(np.linspace(math.log(0.05), math.log(50), 241)):
        W = c0 + (s * x)[:, None] * r0 + (s * y)[:, None] * dn0 + (s * z)[:, None] * f0
        Q = W - c1
        X, Y, Z = Q @ r1, Q @ dn1, Q @ f1
        ok = Z > 0.1
        uu = (fx * X[ok] / Z[ok] + cx).astype(int)
        vv = (fy * Y[ok] / Z[ok] + cy).astype(int)
        inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        if inb.sum() < 500:
            continue
        e = np.median(np.abs(np.log(Z[ok][inb]) - np.log(s * np.maximum(d1[vv[inb], uu[inb]], 1e-6))))
        if best_err is None or e < best_err:
            best, best_err = float(s), e
    return best


def smooth(seq):
    out, est, pending = [], None, []
    for m in seq:
        if m is not None and est is not None and abs(math.log(m / est)) > math.log(1.35):
            pending.append(m)
            if len(pending) >= 3 and max(pending[-3:]) / min(pending[-3:]) < 1.35:
                est, pending = float(np.median(pending[-3:])), []
            out.append(est)
            continue
        if m is not None:
            est = m if est is None else math.exp(0.5 * math.log(est) + 0.5 * math.log(m))
            pending = []
        out.append(est)
    return out


def main():
    rng = np.random.default_rng(0)
    report = {}
    for tag, H in (("1.25 m", 1.25), ("0.75 m", 0.75)):
        D = OUT if H == 1.25 else OUT / f"h{H}"
        frames = []
        for f in sorted((D / "pred").glob("*.npz")):
            z, p = np.load(D / "frames" / f.name), np.load(f)
            S = {k: [] for k in ("ransac", "lowest", "rowprior", "odom")}
            rows = []
            for t in range(len(p["depth"])):
                d, conf = p["depth"][t].astype(np.float32), p["conf"][t].astype(np.float32)
                h, w = d.shape
                gt = np.array(Image.fromarray(z["depth"][t].astype(np.float32)).resize((w, h), Image.NEAREST))
                m = (gt > 0.1) & (gt < 10) & np.isfinite(d) & (d > 0)
                K = intrinsics(h, w)
                P = backproject(d, K)
                conf_ok = conf > np.percentile(conf, 20)
                lowmask = np.zeros_like(m)
                lowmask[int(h * 0.55):] = True
                S["ransac"].append(s_ransac(P, lowmask & conf_ok, rng, H))
                S["lowest"].append(s_lowest(P, lowmask & conf_ok, rng, H))
                S["rowprior"].append(s_rowprior(d, conf_ok, K, H))
                S["odom"].append(None if t == 0 else s_odom(
                    p["depth"][t - 1].astype(np.float32), d, p["conf"][t - 1].astype(np.float32),
                    (z["pos"][t - 1], float(z["yaw"][t - 1])), (z["pos"][t], float(z["yaw"][t])), K, H))
                Pg = backproject(gt, K)
                gg = occupancy(Pg, H - Pg[..., 1])
                rows.append({"ep": f.stem, "t": t, "s_true": float(np.median(gt[m]) / np.median(d[m])) if m.sum() > 1000 else None,
                             "cg": collisions(gg), "free": free_ahead(gg), "P": P,
                             "fwd_next": int(z["action"][t]) == 1,
                             "after_turn": int(z["action"][t]) == 1 and t > 0 and int(z["action"][t - 1]) in (2, 3)})
            S["odom|rowprior"] = [o if o is not None else r for o, r in zip(S["odom"], S["rowprior"])]
            for k in list(S):
                S[k + "+smooth"] = smooth(S[k])
            for i, r in enumerate(rows):
                r["s"] = {k: v[i] for k, v in S.items()}
                r["s"]["oracle"] = r["s_true"]
                # occupancy needs the point cloud only at evaluation; compute collisions per method now, drop P
                r["cp"] = {}
                for k, s in r["s"].items():
                    if s is not None:
                        Pp = r["P"] * s
                        r["cp"][k] = collisions(occupancy(Pp, H - Pp[..., 1]))
                del r["P"]
            frames += rows
            print(f"{tag} {f.stem}: {len(rows)} frames", flush=True)
        methods = list(frames[0]["s"].keys())
        eps = sorted({r["ep"] for r in frames})
        # leave-one-episode-out bias for the smoothed methods
        for k in [m for m in methods if m.endswith("+smooth")]:
            kb = k.replace("+smooth", "+smooth+bias")
            for e in eps:
                other = [r["s"][k] / r["s_true"] for r in frames if r["ep"] != e and r["s"][k] and r["s_true"]]
                b = float(np.median(other))
                for r in frames:
                    if r["ep"] == e:
                        s = r["s"][k] / b if r["s"][k] else None
                        r["s"][kb] = s
            methods.append(kb)
        # collisions for the bias variants need the point cloud again: recompute from the stored smoothed result
        # by rescaling is not possible on a dilated grid, so reload depth for those frames
        for f in sorted((D / "pred").glob("*.npz")):
            p = np.load(f)
            for r in [r for r in frames if r["ep"] == f.stem]:
                d = p["depth"][r["t"]].astype(np.float32)
                P = backproject(d, intrinsics(*d.shape))
                for k in [m for m in methods if m.endswith("+bias")]:
                    if r["s"][k]:
                        Pp = P * r["s"][k]
                        r["cp"][k] = collisions(occupancy(Pp, H - Pp[..., 1]))
        res = {}
        near = [r for r in frames if r["free"] < 1.0]
        fwd = [r for r in frames if r["fwd_next"]]
        aft = [r for r in frames if r["after_turn"]]
        for k in methods:
            rat = np.array([r["s"][k] / r["s_true"] for r in frames if r["s"][k] and r["s_true"]])

            def pr(R):
                tp = fp = fn = 0
                for r in R:
                    cg = r["cg"]
                    cp = r["cp"].get(k, np.zeros_like(cg))      # no scale: no masking
                    tp += int((cg & cp).sum()); fp += int((~cg & cp).sum()); fn += int((cg & ~cp).sum())
                return round(tp / max(tp + fp, 1), 3), round(tp / max(tp + fn, 1), 3)
            res[k] = {"coverage": round(len(rat) / len(frames), 3),
                      "within_0.8_1.25": round(float(np.mean((rat > .8) & (rat < 1.25))) * len(rat) / len(frames), 3),
                      "over_1.25": round(float(np.mean(rat > 1.25)) * len(rat) / len(frames), 3),
                      "median_ratio": round(float(np.median(rat)), 3) if len(rat) else None,
                      "collision_PR": pr(frames), "collision_PR_wall_lt_1m": pr(near),
                      "collision_PR_next_forward": pr(fwd), "collision_PR_first_forward_after_turn": pr(aft)}
        report[tag] = {"frames": len(frames), "wall_lt_1m_frames": len(near), "next_forward_frames": len(fwd),
                       "first_forward_after_turn_frames": len(aft), "methods": res}
        print(f"\n== {tag}: {len(frames)} frames, {len(near)} with a wall < 1 m ahead, {len(fwd)} before a forward step, "
              f"{len(aft)} first forward after a turn")
        print(f"{'method':24s} {'coverage':>8s} {'0.8-1.25':>9s} {'>1.25':>6s} {'med ratio':>9s} {'P/R all':>14s} {'P/R wall<1m':>14s} {'P/R next fwd':>14s} {'P/R after turn':>14s}")
        for k, v in res.items():
            print(f"{k:24s} {v['coverage']:8.2f} {v['within_0.8_1.25']:9.2f} {v['over_1.25']:6.2f} {str(v['median_ratio']):>9s} "
                  f"{str(v['collision_PR']):>14s} {str(v['collision_PR_wall_lt_1m']):>14s} "
                  f"{str(v['collision_PR_next_forward']):>14s} {str(v['collision_PR_first_forward_after_turn']):>14s}")
    (OUT / "scale_study.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
