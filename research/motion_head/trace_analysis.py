"""Where and why closed-loop episodes go wrong, from traces (EXP_SAVE=trace).

Per episode, against the dense reference route (train_gt locations, 0.25 m):
  off_m[t]    distance from the agent to the route
  prog[t]     index of the nearest route point ahead-of-best (monotone max)
  departure   first step with off_m > DEPART_M (and the route point there)
  at_turn     whether the departure is within TURN_WIN route points of a bend
              (> 30 deg heading change of the route over +-4 points)
Outcome classes for failures:
  wrong_turn    departed the route and never came back within REJOIN_M
  stopped_short stayed on route but stopped with > 3 m of route left
  overshoot     reached within 3 m of the goal (oracle success) then stopped elsewhere
  wander        hit the step limit

    python trace_analysis.py outputs/eval/r2r/flow_v1_trace100/train/trace [other trace dir]
"""
import glob
import gzip
import json
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT = os.path.join(ROOT, "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz")
DEPART_M, REJOIN_M, TURN_WIN = 1.5, 1.0, 6


def bends(loc):
    xz = loc[:, [0, 2]]
    d = np.diff(xz, axis=0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    n = len(loc)
    out = np.zeros(n, bool)
    for i in range(n):
        a, b = max(0, i - 4), min(len(hd) - 1, i + 4)
        if b > a:
            dh = (hd[b] - hd[a] + math.pi) % (2 * math.pi) - math.pi
            out[i] = abs(dh) > math.radians(30)
    return out


def analyse(tr, loc, bend):
    steps = tr["steps"]
    P = np.array([s["pos"] for s in steps])
    D = np.linalg.norm(P[:, None, [0, 2]] - loc[None, :, [0, 2]], axis=-1)
    off = D.min(1)
    near = D.argmin(1)
    prog = np.maximum.accumulate(near)
    dep = next((t for t in range(len(off)) if off[t] > DEPART_M), None)
    r = {"n": len(steps), "off_max": float(off.max()), "final_prog": float(prog[-1] / (len(loc) - 1)),
         "departure": dep, "dtg": steps[-1]["dtg"]}
    if dep is not None:
        k = int(prog[dep])
        r["dep_route_frac"] = k / (len(loc) - 1)
        r["dep_at_turn"] = bool(bend[max(0, k - TURN_WIN):k + TURN_WIN + 1].any())
        r["dep_start"] = k <= max(4, 0.1 * (len(loc) - 1))
        r["rejoined"] = bool((off[dep:] < REJOIN_M).any() and prog[-1] > k + 8)
        hs = [s.get("head") for s in steps[max(0, dep - 8):dep + 1]]
        acts = [s["action"] for s in steps[max(0, dep - 8):dep + 1]]
        r["pre_dep"] = list(zip(acts, [h["fn_action"] if h else None for h in hs]))
    return r


def outcome(st, r):
    if st["success"]:
        return "success"
    if r["n"] >= 399:
        return "wander"
    if st.get("oracle_success"):
        return "overshoot"
    if r["departure"] is not None and not r.get("rejoined"):
        return "wrong_turn"
    if r["final_prog"] < 0.9:
        return "stopped_short" if r["off_max"] < DEPART_M else "lost_then_stopped"
    return "near_miss"


def main():
    with gzip.open(GT) as f:
        gt = json.load(f)
    for d in sys.argv[1:]:
        res = {}
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            tr = json.load(open(f))
            eid = os.path.basename(f)[:-5]
            st = tr["final"]
            loc = np.array(gt[eid]["locations"], float)
            r = analyse(tr, loc, bends(loc))
            r["outcome"] = outcome(st, r)
            res[eid] = r
        print("==", d, len(res), "episodes")
        oc = Counter(r["outcome"] for r in res.values())
        print("outcomes:", dict(oc))
        deps = [r for r in res.values() if r["departure"] is not None and r["outcome"] != "success"]
        if deps:
            print("failed episodes with a departure: %d; at start %d, at a route bend %d, on a straight %d; median route frac %.2f"
                  % (len(deps), sum(r["dep_start"] for r in deps), sum(r["dep_at_turn"] and not r["dep_start"] for r in deps),
                     sum(not r["dep_at_turn"] and not r["dep_start"] for r in deps), np.median([r["dep_route_frac"] for r in deps])))
        sd = [r for r in res.values() if r["departure"] is not None and r["outcome"] == "success"]
        print("successes that departed and recovered: %d" % len(sd))
        json.dump(res, open(os.path.join(os.path.dirname(d), "trace_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
