"""Failure analysis for one evaluation run, joined with R2R episode metadata.

    python deploy/analyze_failures.py outputs/eval/r2r/cam012_p20_150/val_unseen \
        --baseline outputs/eval/r2r/base_v2_150/val_unseen

Without traces (run.py --exp-save data,trace) only end-of-episode stats exist, so
the outcome classes below are inferred from DTG / OS / steps. With traces the
script additionally separates model STOPs from harness-forced STOPs.
"""

import argparse
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict

SUCCESS_RADIUS = 3.0
STEP_CAP = 395

OUTCOMES = [
    "success",
    "stopped_near",       # OS=0 but ended within 3-5 m: short of or past the goal
    "reached_no_stop",    # OS=1, passed within 3 m, ended elsewhere
    "step_cap",           # ran out of steps without ever reaching the goal
    "wrong_region",       # OS=0, ended far away
]


def clean(v):
    v = float(v or 0)
    return 0.0 if (math.isnan(v) or math.isinf(v)) else v


def load_stats(run_dir):
    rows = {}
    log = os.path.join(run_dir, "log")
    for f in os.listdir(log):
        if f.startswith("stats_") and f.endswith(".json"):
            with open(os.path.join(log, f)) as fh:
                r = json.load(fh)
            rows[str(r.get("id", f[6:-5]))] = r
    return rows


def load_traces(run_dir):
    traces = {}
    tdir = os.path.join(run_dir, "trace")
    if not os.path.isdir(tdir):
        return traces
    for f in os.listdir(tdir):
        if f.endswith(".json"):
            with open(os.path.join(tdir, f)) as fh:
                t = json.load(fh)
            traces[str(t["episode_id"])] = t
    return traces


def load_episodes(split):
    path = f"data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
    with gzip.open(path) as fh:
        eps = json.load(fh)["episodes"]
    return {str(e["episode_id"]): e for e in eps}


def classify(r, trace=None):
    if clean(r.get("success")) >= 0.5:
        return "success"
    if clean(r.get("oracle_success")) >= 0.5:
        return "reached_no_stop"
    if clean(r.get("steps")) >= STEP_CAP:
        return "step_cap"
    if clean(r.get("distance_to_goal")) <= 5.0:
        return "stopped_near"
    return "wrong_region"


STAIR_WORDS = re.compile(r"\b(stair|stairs|staircase|steps|upstairs|downstairs|up the|down the)\b", re.I)


def features(ep):
    text = ep["instruction"]["instruction_text"]
    dy = ep["goals"][0]["position"][1] - ep["start_position"][1]
    return {
        "geodesic_m": ep["info"]["geodesic_distance"],
        "words": len(text.split()),
        "waypoints": len(ep["reference_path"]),
        "floor_change": abs(dy) > 1.0,
        "stairs_mentioned": bool(STAIR_WORDS.search(text)),
        "scene": ep["scene_id"].split("/")[1],
        "text": text.strip(),
    }


def bin_table(title, ids, rows, feats, key, edges, label_fmt):
    print(f"\n{title}")
    print(f"  {'bin':<14}{'n':>5}{'SR':>8}{'OS':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [i for i in ids if lo <= feats[i][key] < hi]
        if not sel:
            continue
        sr = sum(clean(rows[i].get("success")) for i in sel) / len(sel)
        os_ = sum(clean(rows[i].get("oracle_success")) for i in sel) / len(sel)
        print(f"  {label_fmt(lo, hi):<14}{len(sel):>5}{sr*100:>7.1f}%{os_*100:>7.1f}%")


def bool_table(title, ids, rows, feats, key, base=None):
    print(f"\n{title}")
    hdr = f"  {'':<10}{'n':>5}{'SR':>8}{'OS':>8}"
    if base:
        hdr += f"{'base SR':>10}{'delta':>8}"
    print(hdr)
    for val in (True, False):
        sel = [i for i in ids if feats[i][key] == val]
        if not sel:
            continue
        sr = sum(clean(rows[i].get("success")) for i in sel) / len(sel)
        os_ = sum(clean(rows[i].get("oracle_success")) for i in sel) / len(sel)
        line = f"  {str(val):<10}{len(sel):>5}{sr*100:>7.1f}%{os_*100:>7.1f}%"
        if base:
            bsr = sum(clean(base[i].get("success")) for i in sel) / len(sel)
            line += f"{bsr*100:>9.1f}%{(sr-bsr)*100:>+8.1f}"
        print(line)


def trace_summary(t):
    """Behavioural facts that end-of-episode stats cannot show."""
    steps = t["steps"]
    acts = [s["action"] for s in steps]
    dtg = [s["dtg"] for s in steps]
    min_i = min(range(len(dtg)), key=dtg.__getitem__) if dtg else 0
    # Longest run of MOVE_FORWARD commands that did not change position: bumping a wall.
    blocked = cur = 0
    for s in steps:
        if s["action"] == "MOVE_FORWARD" and s.get("moved", 1.0) < 0.05:
            cur += 1
            blocked = max(blocked, cur)
        else:
            cur = 0
    # Longest same-direction turn streak: 24 turns of 15 deg is a full spin.
    spin = cur = 0
    prev = None
    for a in acts:
        cur = cur + 1 if a == prev and a in ("TURN_LEFT", "TURN_RIGHT") else (1 if a in ("TURN_LEFT", "TURN_RIGHT") else 0)
        spin = max(spin, cur)
        prev = a
    return {
        "stop_by": t.get("stop_by", "unknown"),
        "min_dtg": min(dtg) if dtg else None,
        "min_dtg_step": min_i,
        "final_dtg": dtg[-1] if dtg else None,
        "max_blocked_fwd": blocked,
        "max_turn_streak": spin,
        "n": len(steps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--json-out", default=None, help="write per-episode table for further analysis")
    a = ap.parse_args()

    rows = load_stats(a.run)
    base = load_stats(a.baseline) if a.baseline else None
    eps = load_episodes(a.split)
    traces = load_traces(a.run)

    ids = sorted(rows, key=int)
    if base:
        ids = [i for i in ids if i in base]
    feats = {i: features(eps[i]) for i in ids}
    outcome = {i: classify(rows[i], traces.get(i)) for i in ids}
    b_outcome = {i: classify(base[i]) for i in ids} if base else None

    n = len(ids)
    print(f"run: {a.run}   episodes: {n}" + (f"   baseline: {a.baseline}" if base else "")
          + f"   traces: {len(traces)}")

    print(f"\noutcome breakdown")
    c = Counter(outcome.values())
    bc = Counter(b_outcome.values()) if base else None
    print(f"  {'class':<18}{'run':>6}" + (f"{'base':>7}{'delta':>7}" if base else ""))
    for k in OUTCOMES:
        line = f"  {k:<18}{c[k]:>6}"
        if base:
            line += f"{bc[k]:>7}{c[k]-bc[k]:>+7}"
        print(line)

    if base:
        print("\ntransition matrix (rows = baseline outcome, cols = this run)")
        short = {"success": "succ", "stopped_near": "near", "reached_no_stop": "noStop",
                 "step_cap": "cap", "wrong_region": "wrong"}
        print("  " + " " * 16 + "".join(f"{short[k]:>8}" for k in OUTCOMES))
        for bk in OUTCOMES:
            print(f"  {bk:<16}" + "".join(
                f"{sum(1 for i in ids if b_outcome[i]==bk and outcome[i]==k):>8}" for k in OUTCOMES))

    bin_table("by route length (geodesic)", ids, rows, feats, "geodesic_m",
              [0, 6, 9, 12, 100], lambda lo, hi: f"{lo:.0f}-{hi:.0f} m" if hi < 100 else f">{lo:.0f} m")
    bin_table("by instruction length", ids, rows, feats, "words",
              [0, 20, 30, 45, 1000], lambda lo, hi: f"{lo}-{hi} w" if hi < 1000 else f">{lo} w")
    bin_table("by reference waypoints", ids, rows, feats, "waypoints",
              [0, 5, 6, 7, 100], lambda lo, hi: f"{lo}-{hi-1}" if hi < 100 else f">={lo}")
    bool_table("multi-floor route (|dy| > 1 m)", ids, rows, feats, "floor_change", base)
    bool_table("instruction mentions stairs", ids, rows, feats, "stairs_mentioned", base)

    if base:
        print("\nscenes by SR change (min 4 episodes)")
        by_scene = defaultdict(list)
        for i in ids:
            by_scene[feats[i]["scene"]].append(i)
        scen = []
        for s, sel in by_scene.items():
            if len(sel) < 4:
                continue
            sr = sum(clean(rows[i].get("success")) for i in sel) / len(sel)
            bsr = sum(clean(base[i].get("success")) for i in sel) / len(sel)
            scen.append((sr - bsr, s, len(sel), bsr, sr))
        for d, s, k, bsr, sr in sorted(scen):
            print(f"  {s:<14} n={k:<3} base {bsr*100:5.1f}%  run {sr*100:5.1f}%  {d*100:+6.1f}")

    if traces:
        print(f"\ntrace diagnostics ({len(traces)} episodes)")
        stop_by = Counter()
        for i in ids:
            if i in traces and outcome[i] != "success":
                stop_by[(outcome[i], trace_summary(traces[i])["stop_by"])] += 1
        print("  failure class x who ended the episode")
        for (k, who), m in sorted(stop_by.items()):
            print(f"    {k:<18}{who:<18}{m:>4}")

    print("\nexamples per failure class")
    for k in OUTCOMES[1:]:
        sel = [i for i in ids if outcome[i] == k]
        if base:
            sel.sort(key=lambda i: b_outcome[i] != "success")  # regressions first
        if not sel:
            continue
        print(f"\n  [{k}]  {len(sel)} episodes")
        for i in sel[:a.examples]:
            r, f = rows[i], feats[i]
            extra = f"  base={b_outcome[i]}" if base else ""
            print(f"    ep {i:>5}  {f['scene']}  geo {f['geodesic_m']:.1f} m  DTG {clean(r['distance_to_goal']):.2f} m"
                  f"  steps {int(clean(r['steps']))}  floor_change={f['floor_change']}{extra}")
            if i in traces:
                ts = trace_summary(traces[i])
                print(f"           stop_by={ts['stop_by']}  closest {ts['min_dtg']:.2f} m @ step {ts['min_dtg_step']}"
                      f"  blocked_fwd={ts['max_blocked_fwd']}  turn_streak={ts['max_turn_streak']}")
            print(f"           \"{f['text'][:160]}{'...' if len(f['text']) > 160 else ''}\"")

    if a.json_out:
        out = []
        for i in ids:
            d = {"id": i, "outcome": outcome[i], **{k: v for k, v in feats[i].items()},
                 **{k: rows[i].get(k) for k in ("success", "oracle_success", "spl", "distance_to_goal", "steps", "path_length")}}
            if base:
                d["base_outcome"] = b_outcome[i]
            if i in traces:
                d.update(trace_summary(traces[i]))
            out.append(d)
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
