"""Per-episode diagnostics from run.py traces (--exp-save data,trace[,frames]).

    python deploy/analyze_traces.py outputs/eval/r2r/trace_base_never_close/val_unseen

Writes <run>/trace/analysis/<id>.json (metrics) and <id>.png (top-down map, distance
curves, and the frames the model saw around the moment it left the route).
Run in the habitat env: it needs habitat_sim for the navmesh and matplotlib.
"""

import argparse
import json
import math
import os
import textwrap

import numpy as np

ROOM = {"a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining room", "e": "entryway",
        "f": "family room", "g": "garage", "h": "hallway", "i": "library", "j": "laundry room",
        "k": "kitchen", "l": "living room", "m": "meeting room", "n": "lounge", "o": "office",
        "p": "porch/deck", "r": "rec room", "s": "stairs", "t": "toilet", "u": "utility room",
        "v": "tv room", "w": "gym", "x": "outdoor", "y": "balcony", "z": "other room",
        "B": "bar", "C": "classroom", "D": "dining booth", "S": "spa", "Z": "junk", "-": "unlabeled"}

OFF_ROUTE_M = 1.5


def load_regions(scene_dir, scene):
    """MP3D region boxes, converted from the house file's z-up frame to habitat's y-up frame."""
    regions = []
    with open(os.path.join(scene_dir, scene, scene + ".house")) as fh:
        for line in fh:
            if not line.startswith("R "):
                continue
            t = line.split()
            label = t[5]
            xlo, ylo, zlo, xhi, yhi, zhi = map(float, t[9:15])
            # habitat (x, y, z) = mp3d (x, z, -y)
            regions.append({"label": ROOM.get(label, label), "level": int(t[2]),
                            "lo": np.array([xlo, zlo, -yhi]), "hi": np.array([xhi, zhi, -ylo])})
    return regions


def room_of(p, regions):
    p = np.asarray(p)
    hits = [r for r in regions if np.all(p >= r["lo"] - [0, 0.5, 0]) and np.all(p <= r["hi"] + [0, 0.5, 0])]
    if not hits:
        return "outside labeled rooms"
    return min(hits, key=lambda r: np.prod(r["hi"] - r["lo"]))["label"]


def project(p, poly):
    """Distance from p to the reference polyline and the arc-length fraction of the closest point."""
    seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    total = seg_len.sum() or 1.0
    best = (1e9, 0.0)
    acc = 0.0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        u = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0, 1))
        q = a + u * ab
        # Vertical error counts double: being on the wrong floor is not "near the route".
        d = math.sqrt((p[0] - q[0]) ** 2 + (p[2] - q[2]) ** 2 + (2 * (p[1] - q[1])) ** 2)
        if d < best[0]:
            best = (d, (acc + u * seg_len[i]) / total)
        acc += seg_len[i]
    return best


def wrap(a):
    return (a + 180) % 360 - 180


def analyse(t, regions):
    steps = t["steps"]
    poly = np.array(t["reference_path"], dtype=float)
    goal = np.array(t["goal"], dtype=float)
    pos = np.array([s["pos"] for s in steps] + [t["final_pos"]], dtype=float)
    dtg = np.array([s["dtg"] for s in steps] + [t["final"]["distance_to_goal"]], dtype=float)
    proj = [project(p, poly) for p in pos]
    off = np.array([d for d, _ in proj])
    prog = np.array([f for _, f in proj])
    n = len(steps)
    acts = [s["action"] for s in steps]

    on = np.where(off < OFF_ROUTE_M)[0]
    # Divergence: the last moment the agent was on the route before leaving it for good.
    if off[-1] < OFF_ROUTE_M:
        div = None
    else:
        div = int(on[-1]) if len(on) else 0
    max_prog_on = float(prog[on].max()) if len(on) else 0.0

    moved = np.array([s["moved"] for s in steps])
    fwd = np.array([a == "MOVE_FORWARD" for a in acts])
    blocked = int(np.sum(fwd & (moved < 0.05)))
    turns = int(sum(a in ("TURN_LEFT", "TURN_RIGHT") for a in acts))
    flips = sum(1 for i in range(1, n) if {acts[i], acts[i - 1]} == {"TURN_LEFT", "TURN_RIGHT"})
    streak = cur = 0
    for i, a in enumerate(acts):
        cur = cur + 1 if a in ("TURN_LEFT", "TURN_RIGHT") and i and acts[i - 1] == a else (1 if a in ("TURN_LEFT", "TURN_RIGHT") else 0)
        streak = max(streak, cur)

    # Heading of the first metre actually travelled vs the first metre of the reference route.
    start = pos[0]
    travelled = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1)])
    k = int(np.searchsorted(travelled, 1.0))
    agent_dir = ref_dir = None
    if k < len(pos):
        v = pos[k] - start
        agent_dir = math.degrees(math.atan2(v[0], -v[2]))
    rl = np.cumsum(np.r_[0, np.linalg.norm(np.diff(poly[:, [0, 2]], axis=0), axis=1)])
    j = int(np.searchsorted(rl, 1.0))
    j = min(j, len(poly) - 1)
    v = poly[j] - poly[0]
    ref_dir = math.degrees(math.atan2(v[0], -v[2]))
    first_move_err = abs(wrap(agent_dir - ref_dir)) if agent_dir is not None else None
    steps_before_moving = int(np.argmax(moved > 0.05)) if np.any(moved > 0.05) else n

    rooms = []
    for p in pos[::2]:
        r = room_of(p, regions)
        if not rooms or rooms[-1] != r:
            rooms.append(r)

    y0, yg, yf = pos[0, 1], goal[1], pos[-1, 1]
    path_len = float(travelled[-1])
    ref_len = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))

    m = {
        "id": t["episode_id"], "scene": t["scene"], "instruction": t["instruction"].strip(),
        "steps": n, "stop_by": t["stop_by"],
        "start_dtg": float(dtg[0]), "final_dtg": float(dtg[-1]), "min_dtg": float(dtg.min()), "min_dtg_step": int(dtg.argmin()),
        "path_len": path_len, "ref_len": ref_len,
        "divergence_step": div, "divergence_progress": float(prog[div]) if div is not None else None,
        "max_progress_on_route": max_prog_on, "final_off_route": float(off[-1]), "final_progress": float(prog[-1]),
        "first_move_heading_err": first_move_err, "steps_before_moving": steps_before_moving,
        "turns": turns, "turn_frac": turns / max(n, 1), "lr_flips": flips, "max_turn_streak": streak, "blocked_forward": blocked,
        "start_room": room_of(pos[0], regions), "goal_room": room_of(goal, regions), "final_room": room_of(pos[-1], regions),
        "rooms_visited": rooms,
        "floor_change_needed": bool(abs(yg - y0) > 1.0), "floor_final_vs_goal": float(yf - yg), "max_climb": float(pos[:, 1].max() - y0),
        "max_descent": float(y0 - pos[:, 1].min()), "goal_dy": float(yg - y0),
    }
    m["category"] = categorise(m)
    return m, pos, dtg, off, prog


def categorise(m):
    wrong_floor = abs(m["floor_final_vs_goal"]) > 1.0
    goal_up = m["goal_dy"] > 1.0
    goal_down = m["goal_dy"] < -1.0
    toward = m["max_climb"] if goal_up else m["max_descent"]
    away = m["max_descent"] if goal_up else m["max_climb"]
    if m["floor_change_needed"] and wrong_floor:
        if away > 0.4 and toward < 0.4:
            return "wrong direction on stairs"
        if toward < 0.4:
            return "never changed floor"
        return "incomplete floor change"
    if wrong_floor and not m["floor_change_needed"]:
        return "changed floor wrongly"
    if m["stop_by"] == "harness_stuck":
        return "stuck / spinning"
    if m["divergence_step"] is None:
        return "stopped short on route"
    if m["max_progress_on_route"] < 0.2:
        return "wrong way from start"
    return "left route midway"


def sheet(t, m, pos, dtg, off, frame_dir, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import habitat_sim

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(os.path.join("data/scene_datasets/mp3d", t["scene"], t["scene"] + ".navmesh"))
    lo, _ = pf.get_bounds()
    mpp = 0.05
    poly = np.array(t["reference_path"])
    goal = np.array(t["goal"])
    pts = np.vstack([pos, poly, goal[None]])
    floors = sorted({round(float(y), 1) for y in (pos[0, 1], goal[1])})
    if len(floors) == 2 and abs(floors[0] - floors[1]) < 1.0:
        floors = [floors[0]]

    fig = plt.figure(figsize=(16, 11), dpi=80)
    gs = fig.add_gridspec(3, 8, height_ratios=[1.3, 1.3, 1.0], hspace=0.3, wspace=0.12, top=0.86)
    div = m["divergence_step"]
    for fi, fy in enumerate(floors[:2]):
        span = 6 if len(floors) == 1 else 3
        ax = fig.add_subplot(gs[0:2, fi * span:(fi + 1) * span])
        tv = pf.get_topdown_view(mpp, fy + 0.1)
        x0, x1 = pts[:, 0].min() - 2, pts[:, 0].max() + 2
        z0, z1 = pts[:, 2].min() - 2, pts[:, 2].max() + 2
        c0, c1 = int((x0 - lo[0]) / mpp), int((x1 - lo[0]) / mpp)
        r0, r1 = int((z0 - lo[2]) / mpp), int((z1 - lo[2]) / mpp)
        crop = tv[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
        ax.imshow(crop, cmap="Greys", vmin=-0.4, vmax=1.6, origin="upper",
                  extent=[lo[0] + max(c0, 0) * mpp, lo[0] + max(c1, 0) * mpp, lo[2] + max(r1, 0) * mpp, lo[2] + max(r0, 0) * mpp])
        same = np.abs(pos[:, 1] - fy) < 1.0
        refm = np.abs(poly[:, 1] - fy) < 1.0
        ax.plot(poly[:, 0], poly[:, 2], "--", color="#1a9850", lw=2.5, alpha=.9, label="reference route")
        ax.scatter(poly[refm, 0], poly[refm, 2], s=18, color="#1a9850", zorder=3)
        ax.scatter(pos[same, 0], pos[same, 2], c=np.arange(len(pos))[same], cmap="plasma", s=9, zorder=4, label="agent (dark→bright = time)")
        ax.plot(pos[:, 0], pos[:, 2], color="#555", lw=.6, alpha=.6, zorder=3)
        ax.add_patch(plt.Circle((goal[0], goal[2]), 3.0, fill=False, ls=":", color="#1a9850", lw=1.5))
        ax.scatter([goal[0]], [goal[2]], marker="*", s=260, color="#1a9850", edgecolor="k", zorder=6, label="goal (3 m ring)")
        ax.scatter([pos[0, 0]], [pos[0, 2]], marker="o", s=110, color="white", edgecolor="k", zorder=6, label="start")
        ax.scatter([pos[-1, 0]], [pos[-1, 2]], marker="X", s=150, color="#d73027", edgecolor="k", zorder=6, label="end")
        # start heading arrow
        q = t["start_rotation"]
        yaw = 2 * math.atan2(q[1], q[3])
        ax.arrow(pos[0, 0], pos[0, 2], -math.sin(yaw) * 1.0, -math.cos(yaw) * 1.0, width=.06, color="k", zorder=7)
        if div is not None:
            ax.scatter([pos[div, 0]], [pos[div, 2]], marker="D", s=90, color="#fdae61", edgecolor="k", zorder=6, label=f"left route (step {div})")
        ax.set_xlim(x0, x1); ax.set_ylim(z1, z0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"floor at y≈{fy:.1f} m", fontsize=10)
        if fi == 0:
            ax.legend(loc="lower left", fontsize=7.5, framealpha=.85)

    ax = fig.add_subplot(gs[0, 6:8])
    ax.plot(dtg, color="#d73027", lw=1.8, label="distance to goal (m)")
    ax.plot(off, color="#4575b4", lw=1.2, label="distance off route (m)")
    ax.axhline(3, color="#1a9850", ls=":", lw=1)
    if div is not None:
        ax.axvline(div, color="#fdae61", lw=1.5)
    ax.set_xlabel("step"); ax.legend(fontsize=7.5); ax.grid(alpha=.25)
    ax = fig.add_subplot(gs[1, 6:8])
    ax.plot(pos[:, 1] - pos[0, 1], color="#762a83", lw=1.5, label="height vs start (m)")
    ax.axhline(t["goal"][1] - pos[0, 1], color="#1a9850", ls="--", lw=1, label="goal height")
    ax.set_xlabel("step"); ax.legend(fontsize=7.5); ax.grid(alpha=.25)
    n = len(t["steps"])
    picks = [0]
    anchor = div if div is not None else max(n - 12, 0)
    picks += [anchor - 6, anchor - 3, anchor, anchor + 3, anchor + 6]
    picks += [(anchor + n) // 2, n - 1]
    picks = sorted({min(max(p, 0), n - 1) for p in picks})
    while len(picks) < 8:
        extra = [p for p in np.linspace(0, n - 1, 12).astype(int) if p not in picks]
        if not extra:
            break
        picks = sorted(picks + [extra[len(extra) // 2]])
    for i, p in enumerate(picks[:8]):
        ax = fig.add_subplot(gs[2, i])
        fp = os.path.join(frame_dir, f"{p:03d}.jpg")
        if os.path.exists(fp):
            ax.imshow(Image.open(fp))
        s = t["steps"][p]
        ax.set_title(f"step {p}: {s['action'].replace('MOVE_FORWARD', 'FWD').replace('TURN_', '')}\n{s['dtg']:.1f} m to goal",
                     fontsize=8.5, color="#b35806" if p == div else "k")
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.01, 0.955, "\n".join([
        f"stop: {m['stop_by']}   steps {m['steps']}   path {m['path_len']:.1f} m / route {m['ref_len']:.1f} m",
        f"closest {m['min_dtg']:.1f} m @ step {m['min_dtg_step']}   max progress on route {m['max_progress_on_route']*100:.0f}%",
        f"first-metre heading error {m['first_move_heading_err'] if m['first_move_heading_err'] is None else round(m['first_move_heading_err'])}°   turns {m['turn_frac']*100:.0f}%   L/R flips {m['lr_flips']}   blocked fwd {m['blocked_forward']}",
        f"rooms: {' → '.join(m['rooms_visited'])[:110]}",
        f"goal room: {m['goal_room']}   final room: {m['final_room']}   category: {m['category']}",
    ]), va="top", fontsize=9, family="monospace")

    fig.suptitle(f"ep {t['episode_id']} · {t['scene']} · " + "\n".join(textwrap.wrap(t["instruction"].strip(), 150)),
                 fontsize=11, x=0.01, ha="left", y=0.995, fontweight="bold")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    tdir = os.path.join(a.run, "trace")
    out = os.path.join(tdir, "analysis")
    os.makedirs(out, exist_ok=True)
    region_cache = {}
    rows = []
    for f in sorted(os.listdir(tdir), key=lambda s: (len(s), s)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(tdir, f)) as fh:
            t = json.load(fh)
        if t["scene"] not in region_cache:
            region_cache[t["scene"]] = load_regions("data/scene_datasets/mp3d", t["scene"])
        m, pos, dtg, off, prog = analyse(t, region_cache[t["scene"]])
        with open(os.path.join(out, f"{m['id']}.json"), "w") as fh:
            json.dump(m, fh, indent=1)
        if not a.no_sheets:
            sheet(t, m, pos, dtg, off, os.path.join(tdir, "frames", m["id"]), os.path.join(out, f"{m['id']}.png"))
        rows.append(m)
        print(f"{m['id']:>5} {m['category']:<26} div {str(m['divergence_step']):>4} maxprog {m['max_progress_on_route']*100:3.0f}% "
              f"err1m {str(None if m['first_move_heading_err'] is None else round(m['first_move_heading_err'])):>4} stop {m['stop_by']:<14} "
              f"{m['start_room']} -> {m['final_room']} (goal {m['goal_room']})")
    with open(os.path.join(out, "all.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
