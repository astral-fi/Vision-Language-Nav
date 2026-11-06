"""Detailed 4-way comparison on the 400 stair-free val_unseen subset (ids_sf400.txt).

Arms: futurenav (reused base500 + capture_pool), lora (reused lora_v1_C_val300 + new lora_v1_C_sf400rest),
discrete_head and lora_waypoint_head (new, <arm>_sf400). Writes results.json and report.md and prints the report.
Only episodes finished by every arm enter the comparison (n is printed)."""
import gzip
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
R2R = ROOT / "outputs/eval/r2r"
ARMS = {"futurenav": ["base500", "capture_pool"],
        "lora": ["lora_v1_C_val300", "lora_v1_C_sf400rest"],
        "discrete_head": ["discrete_sf400"],
        "lora_waypoint_head": ["lora_waypoint_sf400"]}
MAX_STEPS = 400
ids = [l.strip() for l in open(HERE / "ids_sf400.txt") if l.strip()]
eps = {str(e["episode_id"]): e for e in json.load(
    gzip.open(ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"))["episodes"]}


def load(tags):
    rows = {}
    for t in tags:
        f = R2R / t / "val_unseen/summary_split_0.json"
        if f.exists():
            for r in json.load(open(f))["results"]:
                rows[str(r["id"])] = r
    return rows


data = {a: load(t) for a, t in ARMS.items()}
done = [i for i in ids if all(i in d for d in data.values())]
print(f"episodes in subset {len(ids)}, finished by all four arms {len(done)}")
if not done:
    raise SystemExit("nothing to compare yet")
rng = np.random.default_rng(0)
col = lambda a, k: np.array([float(data[a][i].get(k, 0) or 0) for i in done])
for a in data:
    for i in done:  # unreachable goals give inf distances; clip like the eval summary does
        r = data[a][i]
        if not math.isfinite(r.get("distance_to_goal", 0)):
            r["distance_to_goal"] = 0.0
meta = {"geo": np.array([eps[i]["info"]["geodesic_distance"] for i in done]),
        "words": np.array([len(eps[i]["instruction"]["instruction_text"].split()) for i in done]),
        "scene": np.array([eps[i]["scene_id"].split("/")[1] for i in done])}


def ci(x, n=10000):
    b = rng.integers(0, len(x), (n, len(x)))
    m = x[b].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):  # exact two-sided on discordant pairs
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    return n01, n10, p


out, md = {"n": len(done), "of": len(ids)}, []
P = md.append
P(f"# Four-model comparison, stair-free val_unseen subset (n={len(done)} of {len(ids)})\n")

# --- 1 headline
P("## 1. Headline metrics (mean, 95% bootstrap CI)\n")
P("| arm | SR | SPL | oracle SR | distance to goal (m) | path length (m) | steps |\n|---|---|---|---|---|---|---|")
out["headline"] = {}
for a in data:
    s, p, o = col(a, "success"), col(a, "spl"), col(a, "oracle_success")
    lo, hi = ci(s); plo, phi = ci(p)
    out["headline"][a] = dict(SR=s.mean(), SR_ci=[lo, hi], SPL=p.mean(), SPL_ci=[plo, phi], OS=o.mean(),
                              DTG=col(a, "distance_to_goal").mean(), PL=col(a, "path_length").mean(), steps=col(a, "steps").mean())
    P(f"| {a} | {s.mean():.3f} [{lo:.3f}, {hi:.3f}] | {p.mean():.3f} [{plo:.3f}, {phi:.3f}] | {o.mean():.3f} | "
      f"{col(a, 'distance_to_goal').mean():.2f} | {col(a, 'path_length').mean():.2f} | {col(a, 'steps').mean():.0f} |")

# --- 2 paired
P("\n## 2. Paired comparisons (same episodes)\n")
P("| A vs B | SR A | SR B | A only | B only | exact McNemar p | dSPL (A-B) [95% CI] |\n|---|---|---|---|---|---|---|")
out["paired"] = {}
for a, b in combinations(data, 2):
    sa, sb = col(a, "success"), col(b, "success")
    n01, n10, p = mcnemar(sa, sb)
    d = col(a, "spl") - col(b, "spl"); lo, hi = ci(d)
    out["paired"][f"{a}__{b}"] = dict(a_only=n10, b_only=n01, p=p, dSPL=d.mean(), dSPL_ci=[lo, hi])
    P(f"| {a} vs {b} | {sa.mean():.3f} | {sb.mean():.3f} | {n10} | {n01} | {p:.3g} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")

# --- 3 overlap
P("\n## 3. Who solves what\n")
S = np.stack([col(a, "success") for a in data]).astype(int)
names = list(data)
P(f"- solved by all four: {int((S.sum(0) == 4).sum())}; by none: {int((S.sum(0) == 0).sum())}; by exactly one: {int((S.sum(0) == 1).sum())}")
for k, a in enumerate(names):
    only = int(((S[k] == 1) & (S.sum(0) == 1)).sum())
    P(f"- only {a} succeeds: {only}")
out["overlap"] = {"all4": int((S.sum(0) == 4).sum()), "none": int((S.sum(0) == 0).sum())}

# --- 4 failure modes
P("\n## 4. How each model ends (counts per arm)\n")
P("| arm | success | stopped far (>3 m, STOP issued) | stopped near but failed | never stopped (hit step cap) | oracle reached but failed |\n|---|---|---|---|---|---|")
out["modes"] = {}
for a in data:
    s, d, st = col(a, "success") > 0, col(a, "distance_to_goal"), col(a, "steps")
    stop = np.array([(data[a][i].get("actions") or {}).get("STOP", 0) > 0 for i in done])
    cap = (~stop) & (st >= MAX_STEPS - 1)
    far, near = (~s) & stop & (d > 3), (~s) & stop & (d <= 3)
    ofail = (col(a, "oracle_success") > 0) & ~s
    out["modes"][a] = dict(success=int(s.sum()), stop_far=int(far.sum()), stop_near=int(near.sum()), cap=int(cap.sum()), oracle_fail=int(ofail.sum()))
    P(f"| {a} | {s.sum()} | {far.sum()} | {near.sum()} | {cap.sum()} | {ofail.sum()} |")

# --- 5 behaviour
P("\n## 5. Action mix and efficiency\n")
P("| arm | forward % | left % | right % | turns per forward | success-episode steps | path length / reference geodesic |\n|---|---|---|---|---|---|---|")
out["behaviour"] = {}
for a in data:
    tot = np.array([[(data[a][i].get("actions") or {}).get(k, 0) for k in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")] for i in done], float)
    f, l, r = tot.sum(0) / max(tot.sum(), 1) * 100
    s = col(a, "success") > 0
    ratio = (col(a, "path_length")[s] / meta["geo"][s]).mean() if s.any() else float("nan")
    out["behaviour"][a] = dict(fwd=f, left=l, right=r, turns_per_fwd=(l + r) / f if f else 0, ratio=ratio)
    P(f"| {a} | {f:.1f} | {l:.1f} | {r:.1f} | {(l + r) / f:.2f} | {col(a, 'steps')[s].mean() if s.any() else float('nan'):.0f} | {ratio:.2f} |")

# --- 6 slices
def slice_table(title, labels, order):
    P(f"\n## {title}\n")
    P("| slice | n | " + " | ".join(f"{a} SR" for a in data) + " |\n|---|---|" + "---|" * len(data))
    res = {}
    for lab in order:
        m = labels == lab
        if m.sum() == 0:
            continue
        sr = [col(a, "success")[m].mean() for a in data]
        res[str(lab)] = dict(n=int(m.sum()), SR=sr)
        P(f"| {lab} | {int(m.sum())} | " + " | ".join(f"{x:.3f}" for x in sr) + " |")
    return res

tert = lambda x: np.where(x <= np.quantile(x, 1 / 3), "short", np.where(x <= np.quantile(x, 2 / 3), "medium", "long"))
out["by_distance"] = slice_table("6a. By goal distance (geodesic terciles)", tert(meta["geo"]), ["short", "medium", "long"])
out["by_words"] = slice_table("6b. By instruction length (terciles)", tert(meta["words"]), ["short", "medium", "long"])
out["by_scene"] = slice_table("6c. By building", meta["scene"], sorted(set(meta["scene"])))

# --- 7 head server stats
P("\n## 7. Head-server bookkeeping\n")
for a, f in (("discrete_head", "discrete_sf400"), ("lora_waypoint_head", "lora_waypoint_sf400")):
    h = ROOT / f"outputs/head_eval/health_{f}.json"
    if h.exists():
        st = json.load(open(h)).get("stats", {})
        q, s = st.get("queries", 0), st.get("steps", 1)
        P(f"- {a}: {s} steps, {q} head queries ({s / max(q, 1):.1f} steps per query), {st.get('dropped', 0)} queue drops (collisions), "
          f"{st.get('invalid', 0)} invalid outputs")
        out.setdefault("server", {})[a] = st

(HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
(HERE / "report.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
