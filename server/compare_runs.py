"""Paired comparison of two evaluation runs.

    python deploy/compare_runs.py outputs/eval/r2r/baseline/val_unseen \
                                  outputs/eval/r2r/cam050/val_unseen

Both runs must cover the same episodes -- eval/run.py seeds episode selection
with np.random.seed(42), so equal --max-episodes gives an identical set.

Why paired rather than comparing means: at 50 episodes the sampling noise on SR
is roughly +/-7 points, which swamps most single-factor effects. But the SAME
episodes ran in both configs, so the per-episode success flips carry far more
information than the two averages do. An exact McNemar test on the discordant
pairs is the right statistic, and it needs no extra episodes.

The OS-vs-SR contrast is the main diagnostic:
  oracle_success = 1, success = 0  -> came within 3 m but failed to STOP there;
                                      a distance-judgement failure, not navigation
  oracle_success = 0               -> never got close; navigation or comprehension
"""

import argparse
import glob
import json
import math
import os


def load(run_dir):
    """episode_id -> stats dict, from <run_dir>/log/stats_*.json"""
    rows = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "log", "stats_*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        rows[str(r.get("id", os.path.basename(f)[6:-5]))] = r
    return rows


def clean(v):
    v = float(v or 0)
    return 0.0 if (math.isnan(v) or math.isinf(v)) else v


def mean(rows, key):
    return sum(clean(r.get(key)) for r in rows) / max(len(rows), 1)


def mcnemar_exact(b, c):
    """Two-sided exact binomial p for discordant pairs (b wins vs c wins).

    Under H0 each discordant pair is a coin flip, so the count follows
    Binomial(b+c, 0.5). Exact rather than chi-square because the discordant
    counts here are small.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="baseline run directory")
    ap.add_argument("run_b", help="ablation run directory")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--show-flips", type=int, default=10,
                    help="how many flipped episodes to list")
    a = ap.parse_args()

    la = a.label_a or os.path.basename(os.path.dirname(a.run_a.rstrip("/")))
    lb = a.label_b or os.path.basename(os.path.dirname(a.run_b.rstrip("/")))

    A, B = load(a.run_a), load(a.run_b)
    if not A or not B:
        raise SystemExit(f"no stats found ({len(A)} in A, {len(B)} in B)")

    shared = sorted(set(A) & set(B), key=lambda x: int(x) if x.isdigit() else x)
    if not shared:
        raise SystemExit("no overlapping episode ids -- were both runs given the "
                         "same --max-episodes? selection is seeded, so they should match")
    print(f"{la}: {len(A)} episodes   {lb}: {len(B)} episodes   overlap: {len(shared)}")
    if len(shared) < max(len(A), len(B)):
        print(f"  (comparing only the {len(shared)} episodes present in both)")

    ra = [A[k] for k in shared]
    rb = [B[k] for k in shared]

    print(f"\n{'metric':<18}{la:>12}{lb:>12}{'delta':>10}")
    print("-" * 52)
    for key, label, pct in [
        ("success", "SR", True), ("oracle_success", "OS", True),
        ("spl", "SPL", True), ("ndtw", "nDTW", True),
        ("distance_to_goal", "DTG (m)", False),
        ("path_length", "PL (m)", False), ("steps", "steps", False),
    ]:
        if not any(key in r for r in ra + rb):
            continue
        ma, mb = mean(ra, key), mean(rb, key)
        if pct:
            print(f"{label:<18}{ma*100:>11.1f}%{mb*100:>11.1f}%{(mb-ma)*100:>+9.1f}")
        else:
            print(f"{label:<18}{ma:>12.2f}{mb:>12.2f}{mb-ma:>+10.2f}")

    # ---- turn fraction, on concordant episodes only ----
    # Far more powerful than SR: measured per step rather than per episode, so a
    # 50-episode run carries thousands of observations instead of fifty. Restrict
    # to episodes with the same outcome in both runs, otherwise a shift in which
    # episodes succeeded confounds the behaviour comparison.
    # path_length / FORWARD_STEP_SIZE recovers the number of forward actions;
    # everything else in `steps` was a turn.
    FORWARD_STEP = 0.25
    conc = [k for k in shared
            if (clean(A[k].get("success")) >= 0.5) == (clean(B[k].get("success")) >= 0.5)]

    def turn_frac(r):
        # Prefer exact tallies when the run recorded them. The path_length
        # fallback assumes turns are pivots that cover no ground, which is false
        # under Ackermann -- there each turn also advances ~0.17 m, so the
        # fallback under-reports turns badly. Runs from before action logging
        # still fall back, so never compare a logged run against a fallback one.
        acts = r.get("actions")
        if isinstance(acts, dict) and sum(acts.values()):
            turns = acts.get("TURN_LEFT", 0) + acts.get("TURN_RIGHT", 0)
            return turns / float(sum(acts.values()))
        st = clean(r.get("steps"))
        return (st - clean(r.get("path_length")) / FORWARD_STEP) / st if st else 0.0

    def frac_source(rows):
        exact = sum(1 for r in rows if isinstance(r.get("actions"), dict))
        return "exact" if exact == len(rows) else ("mixed" if exact else "estimated")

    if conc:
        deltas = [turn_frac(B[k]) - turn_frac(A[k]) for k in conc]
        up = sum(1 for d in deltas if d > 0)
        down = sum(1 for d in deltas if d < 0)
        sa = frac_source([A[k] for k in conc]); sb = frac_source([B[k] for k in conc])
        print(f"\nturn fraction ({len(conc)} concordant episodes)")
        if sa != sb:
            print(f"  WARNING: {la} counts are {sa}, {lb} counts are {sb}. "
                  f"An estimated count under-reports turns whenever turns translate "
                  f"(Ackermann), so this comparison is not apples to apples.")
        for lbl, R, src in ((la, A, sa), (lb, B, sb)):
            st = sum(clean(R[k].get("steps")) for k in conc)
            frac = sum(turn_frac(R[k]) * clean(R[k].get("steps")) for k in conc) / st if st else 0
            print(f"  {lbl:<16} {frac*100:5.1f}%  of {st:.0f} steps  ({src})")
        print(f"  per-episode: {up} up, {down} down, "
              f"mean {sum(deltas)/len(deltas)*100:+.1f} pp   "
              f"sign test p = {mcnemar_exact(up, down):.4f}")

    # ---- paired success flips ----
    both = kept = lost = neither = 0
    lost_ids, gained_ids = [], []
    for k in shared:
        sa, sb = clean(A[k].get("success")) >= 0.5, clean(B[k].get("success")) >= 0.5
        if sa and sb:       both += 1
        elif sa and not sb: lost += 1; lost_ids.append(k)
        elif sb and not sa: kept += 1; gained_ids.append(k)
        else:               neither += 1

    print(f"\npaired success ({len(shared)} episodes)")
    print(f"  both succeeded        {both:>4}")
    print(f"  only {la:<15} {lost:>4}   <- lost by the change")
    print(f"  only {lb:<15} {kept:>4}   <- gained")
    print(f"  both failed           {neither:>4}")
    p = mcnemar_exact(lost, kept)
    verdict = "significant at p<0.05" if p < 0.05 else "NOT significant"
    print(f"  McNemar exact p = {p:.4f}   ({verdict})")
    if lost + kept < 10:
        print("  note: few discordant pairs -- underpowered. Rerun with more episodes\n"
              "        before concluding the change is harmless.")

    # ---- failure-mode shift ----
    print("\nfailure modes")
    for lbl, rows in ((la, ra), (lb, rb)):
        n = len(rows)
        reached_no_stop = sum(1 for r in rows
                              if clean(r.get("oracle_success")) >= 0.5
                              and clean(r.get("success")) < 0.5)
        never_close = sum(1 for r in rows if clean(r.get("oracle_success")) < 0.5)
        capped = sum(1 for r in rows if clean(r.get("steps")) >= 395)
        early = sum(1 for r in rows if clean(r.get("steps")) <= 10)
        print(f"  {lbl:<16} reached-but-didn't-stop {reached_no_stop:>3}/{n}   "
              f"never-got-close {never_close:>3}/{n}   "
              f"hit-step-cap {capped:>3}/{n}   quit-early {early:>3}/{n}")
    print("\n  A rise in 'reached-but-didn't-stop' with OS roughly flat means the\n"
          "  model still navigates there but can no longer tell it has arrived --\n"
          "  the signature of a viewpoint/distance-judgement problem.")

    if lost_ids and a.show_flips:
        print(f"\nepisodes lost by {lb} (first {a.show_flips}):")
        for k in lost_ids[:a.show_flips]:
            x, y = A[k], B[k]
            print(f"  ep {k:>5}  DTG {clean(x.get('distance_to_goal')):5.2f} -> "
                  f"{clean(y.get('distance_to_goal')):5.2f} m   "
                  f"steps {int(clean(x.get('steps'))):3d} -> {int(clean(y.get('steps'))):3d}   "
                  f"OS {int(clean(y.get('oracle_success')))}")


if __name__ == "__main__":
    main()
