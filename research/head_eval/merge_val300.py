"""val300 benchmark (outputs/motion_head/bench/val300_ids.txt): LoRA waypoint head with the discrete executor vs the
continuous executor (val300.sh), next to FutureNav (base500) and LoRA mode C (lora_v1_C_val300) on the same ids.
Writes val300_report.md and val300_results.json. Only episodes finished by every arm are compared.

Behaviour columns come from traces (when an arm has them): turn = heading change summed over the episode;
reversal = a heading change of >= 5 deg opposite to the previous one of >= 5 deg (zig-zag); blocked = a step
that should have moved >= 5 cm but moved < half of it; lateral = sideways distance travelled (continuous only)."""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
R2R = ROOT / "outputs/eval/r2r"
ARMS = {"FutureNav": ["base500"], "LoRA (text plan)": ["lora_v1_C_val300"],
        "waypoint head, discrete exec": ["lora_waypoint_sf400", "lora_waypoint_val300rest"],
        "waypoint head, continuous exec": ["lora_waypoint_cont_val300"]}
ids = [l.strip() for l in open(ROOT / "outputs/motion_head/bench/val300_ids.txt") if l.strip()]
rng = np.random.default_rng(0)


def load(tags):
    rows, traces = {}, {}
    for t in tags:
        base = R2R / t / "val_unseen"
        f = base / "summary_split_0.json"
        if f.exists():
            for r in json.load(open(f))["results"]:
                rows[str(r["id"])] = r
        for p in (base / "log").glob("stats_*.json"):   # finished episodes of a run still in progress
            r = json.load(open(p))
            rows.setdefault(str(r["id"]), r)
        for p in (base / "trace").glob("*.json"):
            traces[p.stem] = p
    return rows, traces


def behaviour(path):
    J = json.load(open(path))
    S = [s for s in J["steps"] if "executed" in s]
    turn = rev = blocked = 0.0
    lateral, last_sign = 0.0, 0
    for a, b in zip(S, S[1:]):
        dyaw = (b["yaw_deg"] - a["yaw_deg"] + 180) % 360 - 180
        turn += abs(dyaw)
        if abs(dyaw) >= 5:
            sign = 1 if dyaw > 0 else -1
            rev += last_sign == -sign
            last_sign = sign
        m = (a.get("head") or {}).get("motion")
        want = math.hypot(m[0], m[1]) if m else (0.25 if a["executed"] == "MOVE_FORWARD" else 0)
        if want >= 0.05 and a["moved"] < 0.5 * want:
            blocked += 1
        if m:
            lateral += abs(m[1])
    return {"turn_deg": turn, "reversals": rev, "blocked": blocked, "lateral_m": lateral,
            "step_cap": float(J.get("stop_by") == "harness_step_cap")}


data = {a: load(t) for a, t in ARMS.items()}
done = [i for i in ids if all(i in d[0] for d in data.values())]
print(f"val300: {len(ids)} ids, finished by all arms {len(done)}")
if not done:
    raise SystemExit("nothing to compare yet")


def col(a, k):
    v = np.array([float(data[a][0][i].get(k, 0) or 0) for i in done])
    return np.where(np.isfinite(v), v, 0.0)


def ci(x, n=10000):
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def mcnemar(a, b):
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    k, n = min(n01, n10), n01 + n10
    return n01, n10, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n))


out, md = {"n": len(done)}, []
P = md.append
P(f"# Waypoint head, discrete vs continuous execution: val300 benchmark (n={len(done)})\n")
P("| arm | SR [95% CI] | SPL [95% CI] | oracle SR | NE (m) | path (m) | steps |\n|---|---|---|---|---|---|---|")
for a in data:
    s, p = col(a, "success"), col(a, "spl")
    out[a] = {k: float(col(a, k).mean()) for k in ("success", "spl", "oracle_success", "distance_to_goal", "path_length", "steps")}
    (sl, sh), (pl, ph) = ci(s), ci(p)
    P(f"| {a} | {s.mean():.3f} [{sl:.3f}, {sh:.3f}] | {p.mean():.3f} [{pl:.3f}, {ph:.3f}] | {col(a, 'oracle_success').mean():.3f} "
      f"| {col(a, 'distance_to_goal').mean():.2f} | {col(a, 'path_length').mean():.2f} | {col(a, 'steps').mean():.0f} |")

P("\n## Paired (exact McNemar on success, bootstrap on SPL difference)\n")
P("| A vs B | only A wins | only B wins | p | ΔSPL (A−B) [95% CI] |\n|---|---|---|---|---|")
C = "waypoint head, continuous exec"
out["paired"] = {}
for a in [x for x in data if x != C]:
    n01, n10, p = mcnemar(col(C, "success"), col(a, "success"))
    d = col(C, "spl") - col(a, "spl")
    lo, hi = ci(d)
    out["paired"][a] = {"continuous_only": n10, "other_only": n01, "p": p, "dSPL": float(d.mean())}
    P(f"| continuous vs {a} | {n10} | {n01} | {p:.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")

P("\n## Behaviour (per episode, from traces; mean over episodes that have a trace)\n")
P("| arm | heading change (deg) | zig-zag reversals | blocked steps | sideways travel (m) | step-cap endings |\n|---|---|---|---|---|---|")
for a, (rows, traces) in data.items():
    B = [behaviour(traces[i]) for i in done if i in traces]
    if not B:
        P(f"| {a} | – | – | – | – | – |")
        continue
    m = {k: float(np.mean([b[k] for b in B])) for k in B[0]}
    out[a]["behaviour"] = m
    P(f"| {a} | {m['turn_deg']:.0f} | {m['reversals']:.1f} | {m['blocked']:.1f} | "
      f"{m['lateral_m']:.2f} | {m['step_cap']:.0%} |")
    out[a]["traced"] = len(B)

(HERE / "val300_results.json").write_text(json.dumps(out, indent=1))
(HERE / "val300_report.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
