"""Thorough comparison across several evaluation runs.

    python3 deploy/compare_multi.py \
        outputs/eval/r2r/base_v2_150/val_unseen \
        outputs/eval/r2r/cam012_p20_150/val_unseen \
        outputs/eval/r2r/jetracer_full_150/val_unseen

Designed for a NESTED LADDER, where each run adds one factor to the previous.
Adjacent pairs then attribute the damage to a single change, and the first-to-last
pair gives the total.

What this adds over compare_runs.py:
  * Wilson 95% intervals, so a point estimate is never read as exact
  * paired McNemar per adjacent step AND first-to-last, with a Bonferroni note
  * the full 2^N success pattern -- which episodes are robust vs fragile
  * failure-mode decomposition per run
  * turn fraction with the estimator chosen per run (see turn_fraction below)
  * per-scene breakdown, since 11 scenes over 150 episodes hides variation
  * a stairs split: R2R instructions routinely say "go up the stairs", which a
    1/10-scale car cannot do at all. Those episodes are unwinnable on the real
    robot, so the interesting number is performance on the ones it could attempt.
"""

import argparse
import glob
import gzip
import json
import math
import os
from collections import Counter, defaultdict

R2R = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
FORWARD_STEP = 0.25
Z = 1.959964


# ---------- loading ----------

def load_run(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "log", "stats_*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        rows[str(r.get("id"))] = r
    cfg = {}
    p = os.path.join(d, "run_config.json")
    if os.path.exists(p):
        with open(p) as fh:
            cfg = json.load(fh)
    return rows, cfg


def load_episodes():
    """episode_id -> (scene, instruction). Used for the scene and stairs splits."""
    if not os.path.exists(R2R):
        return {}
    with gzip.open(R2R, "rt") as fh:
        eps = json.load(fh)["episodes"]
    return {str(e["episode_id"]): (e["scene_id"].split("/")[-2],
                                   e["instruction"]["instruction_text"].strip())
            for e in eps}


def clean(v):
    v = float(v or 0)
    return 0.0 if (math.isnan(v) or math.isinf(v)) else v


# ---------- statistics ----------

def wilson(k, n):
    """Wilson score interval for a binomial proportion. Better than Wald at the
    small n and extreme p you get from 150-episode subsets."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z / d * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def paired_diff_ci(b, c, n):
    """CI for the difference of paired proportions (b = A-only wins, c = B-only)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    d = (c - b) / n
    var = (b + c - (c - b) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    return d, d - Z * se, d + Z * se


def sign_test(deltas):
    up = sum(1 for x in deltas if x > 0)
    dn = sum(1 for x in deltas if x < 0)
    return up, dn, mcnemar_exact(up, dn)


# ---------- behaviour ----------

def turn_fraction(r, cfg):
    """Share of commanded actions that were turns.

    Exact when the run logged `actions`. Otherwise inferred as
    steps - path_length/0.25, which is only valid when turns are pivots that
    cover no ground. Under Ackermann each turn advances ~0.17 m, so the
    inference under-reports turns badly -- refuse it rather than mislead.
    """
    acts = r.get("actions")
    if isinstance(acts, dict) and sum(acts.values()):
        tot = sum(acts.values())
        return (acts.get("TURN_LEFT", 0) + acts.get("TURN_RIGHT", 0)) / tot, "exact"
    if cfg.get("ackermann_radius"):
        return None, "unavailable"
    st = clean(r.get("steps"))
    if st <= 0:
        return None, "unavailable"
    return max(0.0, (st - clean(r.get("path_length")) / FORWARD_STEP) / st), "inferred"


def failure_modes(rows):
    n = len(rows)
    if not n:
        return {}
    f = lambda pred: sum(1 for r in rows if pred(r))
    return {
        "success": f(lambda r: clean(r.get("success")) >= .5),
        "reached_no_stop": f(lambda r: clean(r.get("oracle_success")) >= .5
                             and clean(r.get("success")) < .5),
        "never_close": f(lambda r: clean(r.get("oracle_success")) < .5),
        "step_cap": f(lambda r: clean(r.get("steps")) >= 395),
        "quit_early": f(lambda r: clean(r.get("steps")) <= 10),
        "n": n,
    }


def describe(cfg):
    if not cfg:
        return "(no run_config.json)"
    pitch = "?"
    if cfg.get("sensor_orientation"):
        pitch = "%+.0f" % math.degrees(cfg["sensor_orientation"][0])
    ack = cfg.get("ackermann_radius")
    return ("h=%.2fm pitch=%sdeg steering=%s slide=%s"
            % (cfg.get("cam_height", float("nan")), pitch,
               ("arc R=%.2f" % ack) if ack else "pivot",
               cfg.get("allow_sliding")))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", default=None, help="comma-separated names")
    ap.add_argument("--min-scene", type=int, default=8,
                    help="minimum episodes for a scene to be listed separately")
    a = ap.parse_args()

    data, cfgs, labels = [], [], []
    for i, d in enumerate(a.runs):
        rows, cfg = load_run(d)
        if not rows:
            raise SystemExit(f"no stats in {d}")
        data.append(rows); cfgs.append(cfg)
        labels.append(os.path.basename(os.path.dirname(d.rstrip("/"))))
    if a.labels:
        labels = a.labels.split(",")

    shared = set(data[0])
    for r in data[1:]:
        shared &= set(r)
    shared = sorted(shared, key=lambda x: int(x) if x.isdigit() else x)
    N = len(shared)

    print("=" * 78)
    for lbl, rows, cfg in zip(labels, data, cfgs):
        print(f"  {lbl:<20} {len(rows):>4} eps   {describe(cfg)}")
    print(f"\n  comparing the {N} episodes present in all runs")
    srv = [json.dumps(c.get("server", {}).get("config", {}), sort_keys=True) for c in cfgs]
    if len(set(srv)) > 1:
        print("  !! WARNING: server configs differ between runs. Model-side settings "
              "(lam, cache size, history) must match or these are not comparable:")
        for lbl, s in zip(labels, srv):
            print(f"       {lbl}: {s}")
    else:
        print(f"  server config identical across runs: {srv[0]}")
    print("=" * 78)

    # ---- headline metrics with intervals ----
    print("\nMETRICS (95% Wilson interval for rates)")
    print(f"{'':<20}" + "".join(f"{l:>19}" for l in labels))
    for key, lbl, is_rate in (("success", "SR", True), ("oracle_success", "OS", True),
                              ("spl", "SPL", False), ("distance_to_goal", "DTG (m)", False),
                              ("path_length", "PL (m)", False), ("steps", "steps", False)):
        cells = []
        for rows in data:
            vals = [clean(rows[k].get(key)) for k in shared]
            if is_rate:
                p, lo, hi = wilson(sum(1 for v in vals if v >= .5), N)
                cells.append(f"{p*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]")
            else:
                m = sum(vals) / N
                cells.append(f"{m:17.2f}")
        print(f"{lbl:<20}" + "".join(f"{c:>19}" for c in cells))

    # ---- turn fraction ----
    print("\nTURN FRACTION (share of commanded actions that were turns)")
    for lbl, rows, cfg in zip(labels, data, cfgs):
        fr = [(turn_frac := turn_fraction(rows[k], cfg))[0] for k in shared]
        src = turn_fraction(rows[shared[0]], cfg)[1]
        ok = [x for x in fr if x is not None]
        if ok:
            print(f"  {lbl:<20} {sum(ok)/len(ok)*100:5.1f}%   ({src}, {len(ok)}/{N} episodes)")
        else:
            print(f"  {lbl:<20}   n/a   ({src} -- rerun with action logging to get this)")

    # ---- nested pairwise ----
    pairs = [(i, i + 1) for i in range(len(data) - 1)]
    if len(data) > 2:
        pairs.append((0, len(data) - 1))
    print(f"\nPAIRED COMPARISONS   (Bonferroni threshold for {len(pairs)} tests: "
          f"p < {0.05/len(pairs):.4f})")
    for i, j in pairs:
        A, B = data[i], data[j]
        b = c = both = neither = 0
        for k in shared:
            sa = clean(A[k].get("success")) >= .5
            sb = clean(B[k].get("success")) >= .5
            if sa and sb: both += 1
            elif sa: b += 1
            elif sb: c += 1
            else: neither += 1
        p = mcnemar_exact(b, c)
        d, lo, hi = paired_diff_ci(b, c, N)
        tag = "TOTAL" if (i, j) == (0, len(data) - 1) else "step"
        verdict = ("significant" if p < 0.05 / len(pairs) else
                   "nominal only" if p < 0.05 else "not significant")
        print(f"\n  [{tag}] {labels[i]} -> {labels[j]}")
        print(f"    both {both:>3}   only {labels[i]} {b:>3}   only {labels[j]} {c:>3}   neither {neither:>3}")
        print(f"    SR change {d*100:+5.1f} pp  95% CI [{lo*100:+.1f}, {hi*100:+.1f}]  "
              f"McNemar p={p:.4f}  ({verdict})")
        if b + c < 15:
            print(f"    note: only {b+c} discordant pairs -- underpowered")

    # ---- success pattern across all runs ----
    print(f"\nSUCCESS PATTERN across {len(data)} runs  (1 = succeeded)")
    pat = Counter()
    for k in shared:
        pat["".join("1" if clean(r[k].get("success")) >= .5 else "0" for r in data)] += 1
    print("  " + "  ".join(f"{l[:12]:>12}" for l in labels) + "    count")
    for key, cnt in sorted(pat.items(), key=lambda kv: -kv[1]):
        print("  " + "  ".join(f"{ch:>12}" for ch in key) + f"    {cnt:>5}")
    allp = pat.get("1" * len(data), 0)
    alln = pat.get("0" * len(data), 0)
    print(f"\n  robust (succeed everywhere): {allp}/{N} = {allp/N*100:.1f}%")
    print(f"  hopeless (fail everywhere):  {alln}/{N} = {alln/N*100:.1f}%")
    print(f"  fragile (outcome flips):     {N-allp-alln}/{N} = {(N-allp-alln)/N*100:.1f}%")
    print("  A large fragile share means single-factor effects are hard to resolve:\n"
          "  those episodes sit near a decision boundary and flip on any perturbation.")

    # ---- failure modes ----
    print("\nFAILURE MODES")
    print(f"{'':<20}{'success':>9}{'reached,':>10}{'never':>8}{'step':>7}{'quit':>7}")
    print(f"{'':<20}{'':>9}{'no STOP':>10}{'close':>8}{'cap':>7}{'early':>7}")
    for lbl, rows in zip(labels, data):
        f = failure_modes([rows[k] for k in shared])
        print(f"{lbl:<20}{f['success']:>9}{f['reached_no_stop']:>10}"
              f"{f['never_close']:>8}{f['step_cap']:>7}{f['quit_early']:>7}")

    # ---- stairs split + per scene ----
    meta = load_episodes()
    if meta:
        stairs = [k for k in shared if "stair" in meta.get(k, ("", ""))[1].lower()]
        rest = [k for k in shared if k not in set(stairs)]
        print(f"\nSTAIRS SPLIT  ({len(stairs)} of {N} instructions mention stairs)")
        print("  A 1/10-scale car cannot climb stairs, so those episodes are unwinnable")
        print("  on the real robot regardless of the policy. The 'no stairs' column is")
        print("  the number that matters for a JetRacer.")
        print(f"\n{'':<20}{'SR all':>9}{'SR stairs':>11}{'SR no stairs':>14}")
        for lbl, rows in zip(labels, data):
            sr = lambda ks: (sum(1 for k in ks if clean(rows[k].get("success")) >= .5) / len(ks) * 100
                             if ks else float("nan"))
            print(f"{lbl:<20}{sr(shared):>8.1f}%{sr(stairs):>10.1f}%{sr(rest):>13.1f}%")

        print(f"\nPER-SCENE SR  (scenes with >= {a.min_scene} episodes)")
        byscene = defaultdict(list)
        for k in shared:
            byscene[meta.get(k, ("?", ""))[0]].append(k)
        print(f"{'scene':<16}{'n':>4}" + "".join(f"{l[:12]:>13}" for l in labels))
        for scene, ks in sorted(byscene.items(), key=lambda kv: -len(kv[1])):
            if len(ks) < a.min_scene:
                continue
            cells = []
            for rows in data:
                cells.append("%12.0f%%" % (sum(1 for k in ks if clean(rows[k].get("success")) >= .5)
                                           / len(ks) * 100))
            print(f"{scene:<16}{len(ks):>4}" + "".join(f"{c:>13}" for c in cells))

    # ---- attribution ----
    if len(data) >= 3:
        sr = [sum(1 for k in shared if clean(r[k].get("success")) >= .5) / N for r in data]
        total = sr[0] - sr[-1]
        print("\nATTRIBUTION of the total SR drop")
        if abs(total) < 1e-9:
            print("  no net drop to attribute")
        else:
            # Sign convention matches the paired section: negative = worse.
            for i in range(len(data) - 1):
                step = sr[i + 1] - sr[i]
                share = abs(step) / abs(total) * 100
                print(f"  {labels[i]} -> {labels[i+1]}: {step*100:+5.1f} pp "
                      f"({share:5.0f}% of total)")
            print(f"  {'TOTAL':<38} {-total*100:+5.1f} pp")
            print("  Shares only mean something if the steps are individually significant --\n"
                  "  check the paired tests above before quoting them.")


if __name__ == "__main__":
    main()
