"""Follow-ups to analyze_lost.py on the LoRA waypoint head (all 400 subset episodes, head run only):
  1. plan-level: head's executable plan (first 4) vs LoRA's own plan (raw, first 4) at every query
  2. collisions: what the head and LoRA's plan say right after a forward step that did not move
  3. stops: head STOP vs LoRA plan STOP, and how far from the goal
  4. the 'lost' fork cases where the head plan diverged mid-queue"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
R2R = HERE.parents[1] / "outputs/eval/r2r"
NAME = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
SYM = "SFLR"
ids = [l.strip() for l in open(HERE / "ids_sf400.txt") if l.strip()]
lost = {e["id"] for e in json.load(open(HERE / "lost_episodes.json"))}
W = {i: json.load(open(R2R / "lora_waypoint_sf400/val_unseen/trace" / f"{i}.json")) for i in ids}
won = {i for i in ids if W[i]["final"]["success"]}
plan = lambda raw: [NAME[w] for w in re.findall(r"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT", raw or "")]
cut = lambda p: p[:p.index(0) + 1] if 0 in p else p
net = lambda p: 15 * sum({2: 1, 3: -1}.get(a, 0) for a in p)
s4 = lambda p: "".join(SYM[a] for a in p)

# 1. plan level
rows = []
for i in ids:
    for s in W[i]["steps"]:
        h = s["head"]
        if not h.get("queried"):
            continue
        hp, rp = cut(h["actions"])[:4], cut(plan(h.get("raw")))[:4]
        rows.append(dict(ep=i, step=s["step"], hp=hp, rp=rp, grp="lost" if i in lost else "won" if i in won else "other",
                         oracle=NAME.get(s.get("oracle")), dtg=s["dtg"]))
print("== 1. head plan vs LoRA plan at queries (first 4 executable actions)")
for g in ("won", "lost"):
    R = [r for r in rows if r["grp"] == g]
    same = np.mean([r["hp"] == r["rp"] for r in R])
    first = np.mean([r["hp"][:1] == r["rp"][:1] for r in R])
    turnh, turnr = np.array([net(r["hp"]) for r in R]), np.array([net(r["rp"]) for r in R])
    both_turn = (turnr != 0)
    print(f"  {g:5s} queries={len(R):5d} identical plan {same:.0%}, same first action {first:.0%}; |net turn| head {np.abs(turnh).mean():.1f} "
          f"vs LoRA {np.abs(turnr).mean():.1f} deg; when LoRA turns: head turns less {np.mean(np.abs(turnh[both_turn]) < np.abs(turnr[both_turn])):.0%}, "
          f"more {np.mean(np.abs(turnh[both_turn]) > np.abs(turnr[both_turn])):.0%}, opposite sign {np.mean(np.sign(turnh[both_turn]) * np.sign(turnr[both_turn]) < 0):.0%}")
    fw = lambda p: sum(a == 1 for a in p)
    print(f"        forward steps in plan: head {np.mean([fw(r['hp']) for r in R]):.2f} vs LoRA {np.mean([fw(r['rp']) for r in R]):.2f}; "
          f"plan length head {np.mean([len(r['hp']) for r in R]):.2f} vs LoRA {np.mean([len(r['rp']) for r in R]):.2f}")
    print("        most common (LoRA plan -> head plan) disagreements:",
          Counter((s4(r["rp"]), s4(r["hp"])) for r in R if r["hp"] != r["rp"]).most_common(8))

# 2. collisions
print("\n== 2. after a forward step that did not move (collision)")
for g, S in (("won", won), ("lost", lost)):
    nxt = Counter()
    stuck_runs = []
    for i in S:
        st = W[i]["steps"]
        run = 0
        for j, s in enumerate(st):
            hit = s["executed"] == "MOVE_FORWARD" and s.get("moved", 1) < 0.05
            run = run + 1 if hit else 0
            if hit and j + 1 < len(st):
                n = st[j + 1]
                h = n["head"]
                if h.get("queried"):
                    nxt[(SYM[h["actions"][0]], SYM[(plan(h.get("raw")) or [1])[0]], n.get("oracle", "?")[:6])] += 1
            if run:
                stuck_runs.append(run)
    hits = sum(1 for i in S for s in W[i]["steps"] if s["executed"] == "MOVE_FORWARD" and s.get("moved", 1) < .05)
    print(f"  {g:5s} blocked forwards {hits} ({hits / len(S):.1f}/episode), longest blocked run median "
          f"{np.median(stuck_runs) if stuck_runs else 0}, max {max(stuck_runs) if stuck_runs else 0}")
    print("        next query (head first, LoRA first, oracle):", nxt.most_common(6))

# 3. stops
print("\n== 3. STOP decisions")
for g, S in (("won", won), ("lost", lost)):
    hs = rs = both = 0
    far = []
    for i in S:
        for s in W[i]["steps"]:
            h = s["head"]
            if not h.get("queried"):
                continue
            a = 0 in cut(h["actions"])[:4]
            b = 0 in cut(plan(h.get("raw")))[:4]
            hs += a; rs += b; both += a and b
            if a:
                far.append(s["dtg"])
    print(f"  {g:5s} queries with STOP in plan: head {hs}, LoRA plan {rs}, both {both}; head-STOP dtg median {np.median(far) if far else float('nan'):.2f} m")

# 4. what the lost forks look like
print("\n== 4. lost episodes, LoRA won: the head plan at the query that produced the fork")
L = [e for e in json.load(open(HERE / "lost_episodes.json")) if e["success"]["lora"] and "fork" in e]
for e in L[:52]:
    i, k = e["id"], e["fork"]["step"]
    st = W[i]["steps"]
    q = max(j for j in range(k + 1) if st[j]["head"].get("queried"))
    h = st[q]["head"]
    print(f"  ep {i:>5s} fork@{k:2d} (query@{q:2d}) oracle={e['fork']['oracle'][:6]:6s} LoRA plan {s4(cut(plan(h.get('raw')))):9s} "
          f"head {s4(cut(h['actions'])):9s} anchor {h.get('anchor'):3d} -> {e['ending']}, dtg_end {e['dtg_final']}")
