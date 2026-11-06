"""Test 0: is the waypoint head's failure at the fork a decoding problem (probability split between turn modes,
min-risk picks a compromise) or a head problem (mass on the wrong direction)?

Input: outputs/eval/r2r/lora_waypoint_t0_full traces (head.probs = 256 anchor probabilities per query), the
52 LoRA-won lost episodes. Anchors are binned by the net heading of their first 4 intervals (what is executed):
  bigL >= 45 deg, smallL 15..45, straight < 15, smallR, bigR; 'stop' = stop flag within the first 4.
Decodings compared on the same probabilities: minrisk (what ran), argmax, twostage (heaviest bin, then minrisk
inside it). Judged against the geodesic oracle's next action and LoRA's own plan at that query.
Caveat: only states the min-risk policy visited; after the fork the other decodings would see different states."""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
TR = ROOT / "outputs/eval/r2r/lora_waypoint_t0_full/val_unseen/trace"
src = open(ROOT / "deploy/head_eval_server.py").read()
ns = {"math": math}
exec(src[src.index("PRIMITIVES ="):src.index("def step(")], ns)
to_actions = ns["poses_to_actions"]
C = torch.load(Path.home() / "lora_waypoint_20261104/anchors.pt", weights_only=True).float()
D = torch.cdist(C, C).numpy()
NAME = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
BINS = ["bigL", "smallL", "straight", "smallR", "bigR", "stop"]


def plan_of(k):
    r = C[k].numpy()
    xy, yaw, st = r[:16].reshape(8, 2), r[16:24] / .5, (r[24:32] / 2) > .5
    return to_actions(np.concatenate([xy, yaw[:, None]], 1).tolist(), st.tolist())


def bin_of_plan(p):
    p4 = p[:4]
    if 0 in p4:
        return "stop"
    t = 15 * sum({2: 1, 3: -1}.get(a, 0) for a in p4)
    return "bigL" if t >= 45 else "smallL" if t >= 15 else "bigR" if t <= -45 else "smallR" if t <= -15 else "straight"


PLANS = [plan_of(k) for k in range(len(C))]
ABIN = np.array([BINS.index(bin_of_plan(p)) for p in PLANS])
side = lambda b: "L" if b in ("bigL", "smallL") else "R" if b in ("bigR", "smallR") else b


def decode(p):
    p = np.asarray(p)
    mr = int((p @ D).argmin())
    am = int(p.argmax())
    mass = np.array([p[ABIN == b].sum() for b in range(len(BINS))])
    b = int(mass.argmax())
    idx = np.nonzero(ABIN == b)[0]
    ts = int(idx[(p @ D[:, idx]).argmin()])
    return {"minrisk": mr, "argmax": am, "twostage": ts}, mass


def raw_plan(raw):
    return [NAME[w] for w in re.findall(r"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT", raw or "")]


lost = {e["id"]: e for e in json.load(open(HERE / "lost_episodes.json")) if e["success"]["lora"]}
rows = []
for i, e in lost.items():
    f = TR / f"{i}.json"
    if not f.exists():
        continue
    S = json.load(open(f))["steps"]
    fork = e.get("fork", {}).get("step")
    dep = e.get("departure_step")
    fq = max((j for j in range(len(S)) if S[j]["head"].get("queried") and (fork is None or j <= fork)), default=None)
    for s in S:
        h = s["head"]
        if not h.get("queried") or "probs" not in h:
            continue
        dec, mass = decode(h["probs"])
        assert dec["minrisk"] == h["anchor"], (i, s["step"])
        rp = raw_plan(h.get("raw"))
        rows.append(dict(ep=i, step=s["step"], is_fork=s["step"] == fq, pre_dep=dep is None or s["step"] <= dep,
                         oracle=NAME.get(s.get("oracle")), mass=mass, top1=float(max(h["probs"])),
                         lora_bin=bin_of_plan(rp) if rp else None, lora_first=rp[0] if rp else None,
                         **{f"{k}_bin": bin_of_plan(PLANS[v]) for k, v in dec.items()},
                         **{f"{k}_first": PLANS[v][0] for k, v in dec.items()}))
print(f"episodes {len({r['ep'] for r in rows})}, queries {len(rows)}, fork queries {sum(r['is_fork'] for r in rows)}")


def report(title, R):
    print(f"\n== {title} (n={len(R)})")
    if not R:
        return
    M = np.stack([r["mass"] for r in R])
    print("  mean mass per bin:", {b: round(float(m), 3) for b, m in zip(BINS, M.mean(0))})
    L, Rr = M[:, 0] + M[:, 1], M[:, 3] + M[:, 4]
    print(f"  bimodal (L >= 0.25 and R >= 0.25): {np.mean((L >= .25) & (Rr >= .25)):.0%};  top-1 anchor prob median {np.median([r['top1'] for r in R]):.3f}")
    o = [r for r in R if r["oracle"] in (2, 3)]
    if o:
        good = [M[k, 0] + M[k, 1] if r["oracle"] == 2 else M[k, 3] + M[k, 4] for k, r in enumerate(R) if r["oracle"] in (2, 3)]
        bad = [M[k, 3] + M[k, 4] if r["oracle"] == 2 else M[k, 0] + M[k, 1] for k, r in enumerate(R) if r["oracle"] in (2, 3)]
        print(f"  when oracle turns (n={len(o)}): mass on oracle side {np.mean(good):.2f}, opposite side {np.mean(bad):.2f}, "
              f"straight {np.mean([M[k, 2] for k, r in enumerate(R) if r['oracle'] in (2, 3)]):.2f}; "
              f"oracle side is the heaviest side {np.mean(np.array(good) > np.array(bad)):.0%}")
    for k in ("minrisk", "argmax", "twostage"):
        acc = np.mean([r[f"{k}_first"] == r["oracle"] for r in R if r["oracle"] is not None])
        lo = np.mean([side(r[f"{k}_bin"]) == side(r["lora_bin"]) for r in R if r["lora_bin"]])
        print(f"  {k:8s}: first action = oracle {acc:.0%}; same turn side as LoRA plan {lo:.0%}; bins {dict(Counter(r[f'{k}_bin'] for r in R))}")
    acc = np.mean([r["lora_first"] == r["oracle"] for r in R if r["oracle"] is not None and r["lora_first"] is not None])
    print(f"  LoRA plan: first action = oracle {acc:.0%}; bins {dict(Counter(r['lora_bin'] for r in R))}")


report("fork queries (the query that produced the first action differing from LoRA)", [r for r in rows if r["is_fork"]])
report("fork queries where LoRA's plan is a big turn", [r for r in rows if r["is_fork"] and r["lora_bin"] in ("bigL", "bigR")])
report("all queries before departure from the route", [r for r in rows if r["pre_dep"]])
report("all queries", rows)
print("\n== per fork: oracle | LoRA bin | mass bigL smallL straight smallR bigR stop | minrisk / argmax / twostage")
for r in sorted((r for r in rows if r["is_fork"]), key=lambda r: int(r["ep"])):
    print(f"  ep {r['ep']:>5s} {('SFLR'[r['oracle']] if r['oracle'] is not None else '?')} {str(r['lora_bin']):8s} "
          + " ".join(f"{m:.2f}" for m in r["mass"]) + f" | {r['minrisk_bin']} / {r['argmax_bin']} / {r['twostage_bin']}")
