"""Validate the path-following expert (STAGE_A.md A2). Habitat env.

Checks, each a free run of the expert from a start state (step cap MAX_STEPS):
  clean     native start; also teacher-forced agreement with the native expert's
            actions on its own states
  heading   native start position, uniform random heading (the wrong-start case)
  teleport  expert runs to a random 30-70% of the route, is moved 1-2 m off-route
            (navigable, same floor, <= 4 m geodesic) with a random heading, and
            continues
  failure   val_unseen: the first state where a recorded FutureNav run (base500,
            capture_pool) is > 2 m off the reference route, in failed episodes
Per run: success (< 3 m geodesic to goal), nDTW against the dense route,
`rejoin` (path comes within 0.5 m of the route after the perturbation),
`coverage` (fraction of route points from the perturbation's progress index on
that the path passes within 1 m of, so a shortcut that skips the described
route scores low), steps, and how often the expert hit a fallback or error.

    python validate.py --check clean --n 300 --shard 0 --shards 4
    python validate.py --summary
"""
from __future__ import print_function

import argparse
import glob
import gzip
import json
import math
import os

import numpy as np

import expert as X
from simenv import (R2R, ROOT, STOP, agent_pose, env_episode, load_gt, make_env, ndtw,
                    set_pose)

OUT = os.path.join(ROOT, "outputs/data_engine/validate")
MAX_STEPS = 500
TRACES = ("base500", "capture_pool")


def episodes(split):
    with gzip.open(os.path.join(R2R, split, "%s.json.gz" % split)) as f:
        return {str(e["episode_id"]): e for e in json.load(f)["episodes"]}


def pick_train(n):
    """n episodes with distinct paths, stairs-free, from the train side of the split."""
    with open(os.path.join(ROOT, "outputs/data_engine/split.json")) as f:
        split = json.load(f)
    eps, stairs, seen, ids = episodes("train"), set(split["uses_stairs"]), set(), []
    for i in np.random.default_rng(1).permutation(split["train"]):
        t = eps[i]["trajectory_id"]
        if t not in seen and i not in stairs:
            seen.add(t)
            ids.append(i)
        if len(ids) == n:
            break
    return ids, eps


def failure_starts(n):
    """(eid, pos, yaw) of the first > 2 m off-route state in failed val_unseen runs."""
    gt, out = load_gt("val_unseen"), []
    files = sorted(set(p for t in TRACES for p in glob.glob(
        os.path.join(ROOT, "outputs/eval/r2r", t, "val_unseen/trace/*.json"))))
    done = set()
    for p in files:
        eid = os.path.basename(p)[:-5]
        if eid in done or eid not in gt:
            continue
        with open(p) as f:
            tr = json.load(f)
        if tr["final"]["success"] >= 1.0:
            continue
        L = np.asarray(gt[eid]["locations"])
        for s in tr["steps"]:
            if np.linalg.norm(L - np.asarray(s["pos"]), axis=1).min() > 2.0:
                out.append((eid, s["pos"], math.radians(s["yaw_deg"])))
                done.add(eid)
                break
    rng = np.random.default_rng(2)
    return [out[i] for i in sorted(rng.choice(len(out), min(n, len(out)), replace=False))]


def offroute_point(sim, pos, L, rng):
    import habitat_sim
    for _ in range(200):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 2.0)
        q = sim.pathfinder.snap_point(np.array([pos[0] + r * math.cos(a), pos[1], pos[2] + r * math.sin(a)],
                                               dtype=np.float32))
        q = np.asarray(q, dtype=np.float64)
        if not np.all(np.isfinite(q)) or abs(q[1] - pos[1]) > 0.3:
            continue
        if np.linalg.norm(L - q, axis=1).min() < 1.0:
            continue
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = np.asarray(pos, dtype=np.float32), q.astype(np.float32)
        if sim.pathfinder.find_path(sp) and sp.geodesic_distance <= 4.0:
            return q
    return None


def free_run(env, ex, path, from_idx):
    stats = {"fallback": 0, "error": 0}
    for t in range(MAX_STEPS):
        a, info = ex.label()
        stats["fallback"] += info["fallback"]
        stats["error"] += info["error"]
        if a == STOP:
            break
        env.step(a)
        path.append(agent_pose(env.sim)[0])
    L, P = ex.L, np.asarray(path[-(t + 1):])
    d = np.linalg.norm(P[:, None] - L[None, from_idx:], axis=2)
    m = env.get_metrics()
    return dict(stats, steps=t, stopped=a == STOP, dtg=float(m["distance_to_goal"]),
                success=bool(m["distance_to_goal"] < 3.0), ndtw=ndtw(path, L),
                rejoin=bool(d.min() < 0.5), coverage=float((d.min(0) < 1.0).mean()))


def run(check, n, shard, shards):
    rng = np.random.default_rng(100 + shard)
    if check == "failure":
        starts = failure_starts(n)
        split, eps = "val_unseen", episodes("val_unseen")
    else:
        ids, eps = pick_train(n)
        starts = [(i, None, None) for i in ids]
        split = "train"
    starts = sorted(starts, key=lambda s: (eps[s[0]]["scene_id"], s[0]))[shard::shards]
    gt = load_gt(split)
    env = make_env(split, [s[0] for s in starts])
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for eid, pos, yaw in starts:
        env_episode(env, eid)
        g = gt[eid]
        ex = X.Expert(env.sim, g["locations"])
        row = {"eid": eid, "check": check}
        if check == "clean":
            agree = []
            for a_gt in g["actions"]:
                a, _ = ex.label()
                agree.append(a == a_gt)
                if a_gt == STOP:
                    break
                env.step(a_gt)
            row["agree"] = float(np.mean(agree))
            env_episode(env, eid)
            ex = X.Expert(env.sim, g["locations"])
        path, from_idx = [agent_pose(env.sim)[0]], 0
        if check == "heading":
            p, _ = agent_pose(env.sim)
            set_pose(env.sim, p, rng.uniform(-math.pi, math.pi))
        elif check == "teleport":
            stop_at = int(rng.uniform(0.3, 0.7) * (len(ex.L) - 1))
            for _ in range(MAX_STEPS):
                if ex.tracker.i >= stop_at:
                    break
                a, _ = ex.label()
                if a == STOP:
                    break
                env.step(a)
                path.append(agent_pose(env.sim)[0])
            p, _ = agent_pose(env.sim)
            q = offroute_point(env.sim, p, ex.L, rng)
            if q is None:
                row["skipped"] = "no off-route point"
                rows.append(row)
                continue
            set_pose(env.sim, q, rng.uniform(-math.pi, math.pi))
            from_idx = ex.tracker.i
            path.append(agent_pose(env.sim)[0])
        elif check == "failure":
            set_pose(env.sim, pos, yaw)
            ex.tracker.update(pos)
            # The tracker starts at 0; snap it to where the run was, searching the whole route.
            ex.tracker.i = int(np.argmin(np.linalg.norm(ex.L - np.asarray(pos), axis=1)))
            from_idx = ex.tracker.i
            path = [agent_pose(env.sim)[0]]
        row.update(free_run(env, ex, path, from_idx))
        rows.append(row)
    with open(os.path.join(OUT, "%s_%d.jsonl" % (check, shard)), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("done", check, shard, len(rows), flush=True)


def summary():
    out = {}
    for check in ("clean", "heading", "teleport", "failure"):
        rows = []
        for p in sorted(glob.glob(os.path.join(OUT, "%s_*.jsonl" % check))):
            with open(p) as f:
                rows += [json.loads(l) for l in f]
        rows_ok = [r for r in rows if "skipped" not in r]
        if not rows_ok:
            continue
        s = {"n": len(rows_ok), "skipped": len(rows) - len(rows_ok)}
        for k in ("success", "ndtw", "rejoin", "coverage", "steps", "dtg", "stopped", "agree"):
            v = [float(r[k]) for r in rows_ok if k in r]
            if v:
                s[k] = round(float(np.mean(v)), 3)
        s["ndtw_p10"] = round(float(np.percentile([r["ndtw"] for r in rows_ok], 10)), 3)
        s["runs_with_fallback"] = round(float(np.mean([r["fallback"] > 0 for r in rows_ok])), 3)
        s["runs_with_error"] = round(float(np.mean([r["error"] > 0 for r in rows_ok])), 3)
        s["fail_ids"] = [r["eid"] for r in rows_ok if not r["success"]][:20]
        out[check] = s
        print(check, json.dumps(s))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=("clean", "heading", "teleport", "failure"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run(a.check, a.n, a.shard, a.shards)
