"""Gate 1 of PLAN.md: choose the training and validation steps.

From expert_v1, train buildings only (split.json). Validation = the motion-head
val trajectories (motion_head/data.py is_val), so offline numbers are comparable.
Per episode ~PER_EP steps, decision steps weighted DECISION_W over cruising.
Decision = >= 45 deg net turn in the next K expert actions (the expert corrects
its heading in 15 deg steps on ~55% of steps, so "any turn" flags nearly
everything), inside a detour or wrong-start perturbation, > 0.5 m
off-route, < 2 m from the goal, or one of the first 15 steps.

Writes outputs/s2_lora/samples_{train,val}.json: list of
  {"scene", "eid", "steps": [t, ...], "decision": [0/1, ...]}

    python build_samples.py --train-episodes 2500 --val-episodes 100
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

from common import EXPERT, GT, K, OUT, ROOT

sys.path.insert(0, os.path.join(ROOT, "experiments/motion_head"))
from data import is_val  # noqa: E402

PER_EP = 10
DECISION_W = 2.0
STEP_M = 0.25
NET_TURNS = 3      # >= 45 deg net heading change in the chunk (the expert wiggles in 15 deg steps)


def pick(path, n_route, rng):
    e = np.load(path)
    T = len(e["label"])
    remaining = (n_route - 1 - e["progress"].astype(np.int64)) * STEP_M
    ch = e["chunk_act"][:, :K]
    turn = np.abs((ch == 2).sum(1) - (ch == 3).sum(1)) >= NET_TURNS
    dec = turn | np.isin(e["perturb"], (2, 3)) | (e["off_route"] > 0.5) | (remaining < 2.0) | (np.arange(T) < 15)
    w = np.where(dec, DECISION_W, 1.0)
    n = min(PER_EP, T)
    steps = np.sort(rng.choice(T, size=n, replace=False, p=w / w.sum()))
    return steps.tolist(), dec[steps].astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=2500)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(os.path.join(ROOT, "outputs/data_engine/split.json")) as f:
        train_ids = set(json.load(f)["train"])
    with gzip.open(GT) as f:
        n_route = {k: len(v["locations"]) for k, v in json.load(f).items()}
    tr, va = [], []
    for p in sorted(glob.glob(os.path.join(EXPERT, "*", "*.npz"))):
        eid = os.path.basename(p)[:-4]
        if eid not in train_ids:
            continue
        meta = json.loads(bytes(np.load(p)["meta_json"]).decode())
        (va if is_val(meta["trajectory_id"]) else tr).append(p)
    rng = np.random.default_rng(a.seed)
    os.makedirs(OUT, exist_ok=True)
    for name, pool, n in (("train", tr, a.train_episodes), ("val", va, a.val_episodes)):
        chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]
        rows = []
        for p in chosen:
            eid = os.path.basename(p)[:-4]
            steps, dec = pick(p, n_route[eid], rng)
            rows.append({"scene": p.split("/")[-2], "eid": eid, "steps": steps, "decision": dec})
        with open(os.path.join(OUT, "samples_%s.json" % name), "w") as f:
            json.dump(rows, f)
        ns = sum(len(r["steps"]) for r in rows)
        nd = sum(sum(r["decision"]) for r in rows)
        frames = sum(max(r["steps"]) + 1 for r in rows)
        print("%s: %d episodes (pool %d), %d steps (%.0f%% decision), %d frames to stream, %d buildings"
              % (name, len(rows), len(pool), ns, 100.0 * nd / ns, frames, len({r["scene"] for r in rows})))


if __name__ == "__main__":
    main()
