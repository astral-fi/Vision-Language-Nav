"""Split R2R-CE train by building into train / train-holdout (STAGE_A.md A0).

train-holdout is 6 of the 61 train buildings, picked with a fixed seed among
buildings of middling size so it is neither trivial nor dominated by one house.
Later stages check models on it, so val_unseen is only touched for final numbers.

Every episode also gets `uses_stairs`: the reference route changes height by
more than STAIR_DY, the same rule as experiments/baseline500/select.py. Stairs
episodes are kept and flagged, not dropped.

Writes outputs/data_engine/split.json: {"train": [ids], "holdout": [ids],
"holdout_scenes": [...], "uses_stairs": [ids]}. Deterministic.
"""
from __future__ import print_function

import gzip
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
OUT = os.path.join(ROOT, "outputs/data_engine/split.json")
N_HOLDOUT = 6
STAIR_DY = 0.5
SEED = 42


def main():
    with gzip.open(DATA) as f:
        eps = json.load(f)["episodes"]
    count = Counter(e["scene_id"] for e in eps)
    sizes = np.array(sorted(count.values()))
    lo, hi = np.percentile(sizes, 25), np.percentile(sizes, 75)
    middling = sorted(s for s, n in count.items() if lo <= n <= hi)
    holdout = sorted(np.random.default_rng(SEED).choice(middling, N_HOLDOUT, replace=False).tolist())
    split = {"train": [], "holdout": [], "holdout_scenes": holdout, "uses_stairs": []}
    for e in eps:
        ys = [e["start_position"][1]] + [p[1] for p in e["reference_path"]] + [e["goals"][0]["position"][1]]
        if max(ys) - min(ys) > STAIR_DY:
            split["uses_stairs"].append(str(e["episode_id"]))
        split["holdout" if e["scene_id"] in holdout else "train"].append(str(e["episode_id"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(split, f)
    print("buildings %d, holdout %s" % (len(count), [s.split("/")[1] for s in holdout]))
    print("episodes train %d  holdout %d  uses_stairs %d"
          % (len(split["train"]), len(split["holdout"]), len(split["uses_stairs"])))


if __name__ == "__main__":
    main()
