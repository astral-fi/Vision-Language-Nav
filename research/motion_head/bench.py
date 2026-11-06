"""Evaluation sets for the motion head (MOTION_HEAD_DESIGN.md section 12).

Writes outputs/motion_head/bench/:
  holdout_ids.txt     100 episodes from the 6 held-out train buildings
                      (split.json), distinct paths, stairs-free
  holdout_derail.json the same 100, each with one perturbation, executed in
                      run.py --derail-spec "walked" mode (the policy sees every
                      frame, the script's actions are executed):
                        wrong start  t0 = 0, 6-12 turns (90-180 deg) to one side
                        detour       t0 in [8, 0.6 x expert steps], 4-10 turns
                                     (60-150 deg) then 6-16 forward (1.5-4 m)
                      Seeded per episode, so FutureNav and the head face the
                      identical perturbation.
  val300_ids.txt      first 300 of experiments/baseline500/ids.txt (val_unseen,
                      stairs-free); FutureNav already has traces for these (base500)

Run (habitat side), e.g. the head on the perturbed holdout:
  DATASET_SPLIT=train EPISODE_IDS=.../holdout_ids.txt DERAIL_SPEC=.../holdout_derail.json \
    TAG=head_flow_v1_holdout_derail bash eval/eval_r2r_remote.sh train 0 http://127.0.0.1:8643
"""
from __future__ import print_function

import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "outputs/motion_head/bench")
R2R = os.path.join(ROOT, "data/datasets/R2R_VLNCE_v1-3_preprocessed/train")
N_HOLDOUT = 100
N_VAL = 300


def main():
    with open(os.path.join(ROOT, "outputs/data_engine/split.json")) as f:
        split = json.load(f)
    with gzip.open(os.path.join(R2R, "train.json.gz")) as f:
        eps = {str(e["episode_id"]): e for e in json.load(f)["episodes"]}
    with gzip.open(os.path.join(R2R, "train_gt.json.gz")) as f:
        gt = json.load(f)
    stairs, seen, ids = set(split["uses_stairs"]), set(), []
    for i in np.random.default_rng(7).permutation(split["holdout"]):
        if i in stairs or eps[i]["trajectory_id"] in seen:
            continue
        seen.add(eps[i]["trajectory_id"])
        ids.append(str(i))
        if len(ids) == N_HOLDOUT:
            break
    ids.sort(key=int)
    episodes = {}
    for i in ids:
        rng = np.random.default_rng(int(i) + 1000)
        side = "TURN_LEFT" if rng.random() < 0.5 else "TURN_RIGHT"
        if rng.random() < 0.5:
            episodes[i] = {"t0": 0, "kind": "wrong_start", "script": [side] * int(rng.integers(6, 13))}
        else:
            n = len(gt[i]["actions"])
            t0 = int(rng.integers(8, max(9, int(0.6 * n))))
            episodes[i] = {"t0": t0, "kind": "detour",
                           "script": [side] * int(rng.integers(4, 11)) + ["MOVE_FORWARD"] * int(rng.integers(6, 17))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "holdout_ids.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(os.path.join(OUT, "holdout_derail.json"), "w") as f:
        json.dump({"arm": "bench_v1", "mode": "walked", "episodes": episodes}, f, indent=1)
    with open(os.path.join(ROOT, "experiments/baseline500/ids.txt")) as f:
        val = f.read().split()[:N_VAL]
    with open(os.path.join(OUT, "val300_ids.txt"), "w") as f:
        f.write("\n".join(val) + "\n")
    kinds = [e["kind"] for e in episodes.values()]
    print("holdout %d episodes (%d buildings): %d wrong starts, %d detours; val %d"
          % (len(ids), len(set(eps[i]["scene_id"] for i in ids)), kinds.count("wrong_start"),
             kinds.count("detour"), len(val)))


if __name__ == "__main__":
    main()
