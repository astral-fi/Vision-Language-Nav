"""Training windows for the motion head.

Pairs each collected episode (outputs/data_engine/expert_v1, collect.py) with its
cached System 2 features (outputs/data_engine/s2_v1, cache_s2.py). A sample is a
step t with:
  context   FutureNav states and action from a call r <= t with t - r <= MAX_AGE,
            random among the valid calls, so the head learns to use stale context
  vision    VGGT tokens of frames t, t-1, ..., t-PAST (masked before the start)
  targets   expert chunk poses at t; stop = expert label is STOP; remaining route
            distance; progress fraction; off-route distance
  weight    DECISION_W for decision steps (turn or stop label, a turn in the
            next 8 expert actions, > 0.5 m off-route, < 2 m from the end,
            inside a perturbation), else 1
Validation holds out VAL_FRAC of trajectory ids, so no path appears on both sides.

The VGGT tokens (frames x 48 x 2048 fp16, ~200 KB a frame) stay in pinned CPU
memory and are gathered per batch; everything else lives on the GPU.
"""
import glob
import gzip
import hashlib
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERT = os.path.join(ROOT, "outputs/data_engine/expert_v1")
S2 = os.path.join(ROOT, "outputs/data_engine/s2_v1")
GT = os.path.join(ROOT, "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz")
MAX_AGE = 7
PAST = 3
DECISION_W = 3.0
VAL_FRAC = 0.05
STEP_M = 0.25


def is_val(traj):
    return int(hashlib.md5(str(traj).encode()).hexdigest(), 16) % 1000 < VAL_FRAC * 1000


def _cat(xs, dtype=None):
    x = np.concatenate(xs)
    return torch.from_numpy(x.astype(dtype) if dtype else x)


class Windows(object):
    def __init__(self, split, device="cuda", limit=0):
        with gzip.open(GT) as f:
            n_route = {k: len(v["locations"]) for k, v in json.load(f).items()}
        files = sorted(f for f in glob.glob(os.path.join(S2, "*", "*.npz")) if not f.endswith(".tmp.npz"))
        cols = {k: [] for k in ("h", "act", "call_t", "v", "pose", "frame", "t", "call0", "ncall",
                                "stop", "remaining", "progress", "off_route", "weight")}
        h_off = v_off = 0
        self.episodes = []
        for f in files:
            e = np.load(f.replace(S2, EXPERT))
            meta = json.loads(bytes(e["meta_json"]).decode())
            if is_val(meta["trajectory_id"]) != (split == "val"):
                continue
            s = np.load(f)
            T, nr = len(e["label"]), n_route[meta["episode_id"]]
            if len(s["v"]) != T:
                raise RuntimeError("frame count mismatch in %s" % f)
            calls = s["t"].astype(np.int64)
            t = np.arange(T)
            # calls are every R steps from 0, so the valid ones for step t are contiguous
            first = np.searchsorted(calls, t - MAX_AGE)
            last = np.searchsorted(calls, t, side="right")
            prog = e["progress"].astype(np.int64)
            remaining = (nr - 1 - prog) * STEP_M
            turn8 = np.isin(e["chunk_act"][:, :8], (2, 3)).any(1)
            decision = (e["label"] != 1) | turn8 | (e["off_route"] > 0.5) | (remaining < 2.0) | (e["perturb"] > 0)
            cols["h"].append(s["h"])
            cols["act"].append(s["action"])
            cols["call_t"].append(calls)
            cols["v"].append(s["v"])
            cols["pose"].append(e["chunk_pose"])
            cols["frame"].append(v_off + t)
            cols["t"].append(t)
            cols["call0"].append(h_off + first)
            cols["ncall"].append(last - first)
            cols["stop"].append(e["label"] == 0)
            cols["remaining"].append(remaining)
            cols["progress"].append(prog / max(nr - 1, 1))
            cols["off_route"].append(e["off_route"])
            cols["weight"].append(np.where(decision, DECISION_W, 1.0))
            self.episodes.append(meta["episode_id"])
            h_off += len(calls)
            v_off += T
            if limit and len(self.episodes) >= limit:
                break
        if not self.episodes:
            raise RuntimeError("no %s episodes in %s" % (split, S2))
        dev = self.device = torch.device(device)
        self.v = _cat(cols.pop("v")).pin_memory()                 # frames x 48 x 2048 fp16, CPU
        self.h = _cat(cols["h"]).to(dev)                          # calls x 4 x 2560 fp16
        self.act = _cat(cols["act"], np.int64).to(dev)
        self.call_t = _cat(cols["call_t"], np.int64).to(dev)
        self.pose = _cat(cols["pose"], np.float32).to(dev)        # frames x H x 3
        for k, dt in (("frame", np.int64), ("t", np.int64), ("call0", np.int64), ("ncall", np.int64),
                      ("stop", bool), ("remaining", np.float32), ("progress", np.float32),
                      ("off_route", np.float32), ("weight", np.float32)):
            setattr(self, k, _cat(cols[k], dt).to(dev))
        if int(self.ncall.min()) < 1:
            raise RuntimeError("a step without a valid FutureNav call")

    def __len__(self):
        return len(self.frame)

    def batch(self, idx, generator=None):
        """idx: LongTensor of sample indices on the data device."""
        pick = (torch.rand(len(idx), device=idx.device, generator=generator) * self.ncall[idx]).long()
        call = self.call0[idx] + pick
        fr = self.frame[idx]
        offs = torch.arange(PAST + 1, device=idx.device)
        vis_mask = self.t[idx][:, None] >= offs[None]
        vi = (fr[:, None] - offs[None]).clamp(min=0).cpu()
        vis = self.v[vi].to(self.device, non_blocking=True) * vis_mask[:, :, None, None]
        return {"h": self.h[call], "fn_action": self.act[call], "age": self.t[idx] - self.call_t[call],
                "vis": vis, "vis_mask": vis_mask, "pose": self.pose[fr], "stop": self.stop[idx],
                "remaining_m": self.remaining[idx], "progress": self.progress[idx],
                "off_route_m": self.off_route[idx], "weight": self.weight[idx]}
