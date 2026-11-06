"""Why does the LoRA waypoint head fail on episodes other models solve? (stair-free subset, ids_sf400.txt)

Per-step traces have the executed action, position, distance to goal, the geodesic oracle's next action and,
for the head, `raw` = the LoRA model's own 8-action plan from the same forward pass plus the head's 4-8 actions.
LoRA (chunk_server mode C) and the head share backbone, generation and execution rules (<=4 actions, re-query
after 30 deg), so both trajectories coincide until the first step whose executed action differs: the fork.

Writes lost_episodes.json (per-episode features) and prints the aggregate tables."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
R2R = HERE.parents[1] / "outputs/eval/r2r"
TAGS = {"wp": ["lora_waypoint_sf400"], "lora": ["lora_v1_C_val300", "lora_v1_C_sf400rest"],
        "fn": ["base500", "capture_pool"], "disc": ["discrete_sf400"]}
NAME = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
ids = [l.strip() for l in open(HERE / "ids_sf400.txt") if l.strip()]


def trace(arm, i):
    for t in TAGS[arm]:
        f = R2R / t / "val_unseen/trace" / f"{i}.json"
        if f.exists():
            return json.load(open(f))
    return None


def off_route(steps, ref):
    """Distance (x, z) from each step's position to the reference polyline."""
    ref = np.array(ref)[:, [0, 2]]
    out = []
    for s in steps:
        p = np.array(s["pos"])[[0, 2]]
        best = np.inf
        for a, b in zip(ref[:-1], ref[1:]):
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            best = min(best, np.linalg.norm(p - (a + t * ab)))
        out.append(best)
    return np.array(out)


def departure(off, far=2.0, back=1.0):
    """First step after which the agent is > far m off the route and never returns within back m."""
    for k in range(len(off)):
        if off[k] > far and (off[k:] > back).all():
            return k
    return None


def raw_plan(raw):
    import re
    return [NAME[w] for w in re.findall(r"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT", raw or "")]


def episode(i):
    T = {a: trace(a, i) for a in TAGS}
    w = T["wp"]
    S = w["steps"]
    res = {a: (T[a]["final"]["success"] if T[a] else None) for a in TAGS}
    f = w["final"]
    off = off_route(S, w["reference_path"])
    dep = departure(off)
    stop = S[-1]["action"] == "STOP"
    e = {"id": i, "scene": w["scene"], "success": res, "steps": len(S), "dtg_final": round(f["distance_to_goal"], 2),
         "oracle_success": f["oracle_success"], "stopped": stop, "stop_by": w["stop_by"],
         "min_dtg": round(min(s["dtg"] for s in S), 2), "departure_step": dep,
         "departure_frac": None if dep is None else round(dep / len(S), 2), "max_off_route": round(float(off.max()), 2)}
    e["ending"] = ("success" if res["wp"] else "stop_after_oracle" if f["oracle_success"] and stop else
                   "overshoot_no_stop" if f["oracle_success"] else "stop_far" if stop else "step_cap")
    # oracle agreement of executed actions
    orc = [(s["executed"], s.get("oracle")) for s in S if s.get("oracle")]
    e["oracle_agree"] = round(np.mean([a == o for a, o in orc]), 3) if orc else None
    # head vs LoRA plan at queried steps, judged against the oracle (first action only)
    q = [s for s in S if s["head"].get("queried")]
    cat = Counter()
    pre = Counter()
    for s in q:
        h, r = s["head"]["actions"][0] if s["head"].get("actions") else None, raw_plan(s["head"].get("raw"))
        r = r[0] if r else None
        o = NAME.get(s.get("oracle"))
        if o is None or h is None or r is None:
            continue
        k = ("agree_right" if h == r == o else "agree_wrong" if h == r else "head_overrides_correct_lora" if r == o
             else "head_fixes_lora" if h == o else "both_wrong_differ")
        cat[k] += 1
        if dep is not None and s["step"] <= dep:
            pre[k] += 1
    e["query_cats"], e["query_cats_pre_departure"] = dict(cat), dict(pre)
    e["queries"], e["drops"] = len(q), sum("dropped" in s["head"] for s in S)
    # premature stop: STOP issued while far; did LoRA's own plan stop at that query?
    if stop:
        last_q = [s for s in q if s["step"] <= S[-1]["step"]][-1]
        e["stop_query_raw_has_stop"] = 0 in raw_plan(last_q["head"].get("raw"))[:4]
        e["stop_query_oracle"] = S[-1].get("oracle")
    # fork against LoRA
    L = T["lora"]
    if L:
        ls = L["steps"]
        k = next((j for j in range(min(len(S), len(ls))) if S[j]["executed"] != ls[j]["executed"]), None)
        if k is not None:
            e["fork"] = {"step": k, "wp": S[k]["executed"], "lora": ls[k]["executed"], "oracle": S[k].get("oracle"),
                         "dtg": S[k]["dtg"], "queried": S[k]["head"].get("queried"),
                         "lora_raw": S[k]["head"].get("raw") if S[k]["head"].get("queried") else None,
                         "head_actions": S[k]["head"].get("actions"), "off_route": round(float(off[k]), 2)}
            # did the head's plan near the fork under-rotate? turns executed in the 8 steps after the fork
            turn = lambda st: sum({"TURN_LEFT": 1, "TURN_RIGHT": -1}.get(x["executed"], 0) for x in st)
            e["fork"]["net_turn_next8_deg"] = {"wp": 15 * turn(S[k:k + 8]), "lora": 15 * turn(ls[k:k + 8])}
    return e


E = [episode(i) for i in ids]
by = {e["id"]: e for e in E}
lost = [e for e in E if not e["success"]["wp"] and any(e["success"][a] for a in ("lora", "fn", "disc"))]
lora_won = [e for e in lost if e["success"]["lora"]]
won = [e for e in E if e["success"]["wp"]]
(HERE / "lost_episodes.json").write_text(json.dumps(lost, indent=1, default=str))


def table(title, groups, fn):
    print(f"\n== {title}")
    for name, g in groups:
        print(f"  {name:28s} n={len(g):3d}  {fn(g)}")


G = [("head lost, others won", lost), ("  of which LoRA won", lora_won), ("head won (control)", won)]
table("ending", G, lambda g: dict(Counter(e["ending"] for e in g)))
table("departure from route (>2 m, never back <1 m)", G, lambda g: {
    "departed": sum(e["departure_step"] is not None for e in g),
    "median_step": float(np.median([e["departure_step"] for e in g if e["departure_step"] is not None] or [np.nan])),
    "in_first_25%": sum((e["departure_frac"] or 1) <= .25 for e in g if e["departure_step"] is not None)})
table("executed action == oracle", G, lambda g: round(float(np.mean([e["oracle_agree"] for e in g if e["oracle_agree"] is not None])), 3))


def cats(g, key):
    c = Counter()
    for e in g:
        c.update(e[key])
    n = sum(c.values())
    return {k: f"{v} ({v / n:.0%})" for k, v in sorted(c.items())} if n else {}


table("head vs LoRA plan at queries (first action, judged by oracle), all steps", G, lambda g: cats(g, "query_cats"))
table("... before the departure step", G, lambda g: cats(g, "query_cats_pre_departure"))
table("collision queue drops per episode", G, lambda g: round(float(np.mean([e["drops"] for e in g])), 2))
st = [e for e in lost if e["stopped"] and not e["oracle_success"]]
print(f"\n== stop-far failures in 'head lost' (n={len(st)}): LoRA's own plan also had STOP in its first 4 at that query: "
      f"{sum(e['stop_query_raw_has_stop'] for e in st)}; oracle at the stop step: {dict(Counter(e['stop_query_oracle'] for e in st))}")
F = [e for e in lora_won if "fork" in e]
print(f"\n== fork vs LoRA, episodes LoRA won (n={len(F)} with a fork)")
print("  fork step median", float(np.median([e["fork"]["step"] for e in F])), "| off-route at fork median",
      float(np.median([e["fork"]["off_route"] for e in F])))
print("  who matched the oracle at the fork:", dict(Counter(
    "head" if e["fork"]["wp"] == e["fork"]["oracle"] else "lora" if e["fork"]["lora"] == e["fork"]["oracle"] else "neither" for e in F)))
print("  (head, lora) actions at fork:", dict(Counter((e["fork"]["wp"], e["fork"]["lora"]) for e in F)))
print("  head step was a fresh query:", sum(bool(e["fork"]["queried"]) for e in F))
