"""Offline checks of eval/s1_local.py's LocalPlanner in a kinematic 2D world (no Habitat, no GPU).

The scan is computed by ray/segment intersection against wall segments given in configuration space (the robot
centre must stay `clearance` away from them), matching what the navmesh LiDAR returns.

    ~/miniforge3/envs/futurenav/bin/python eval/check_s1_local.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from s1_local import LocalPlanner  # noqa: E402


def scan(pl, pose, walls):
    f, l = pl.axes(pose[3])
    pts = []
    for th in np.linspace(-math.pi, math.pi, pl.rays, endpoint=False):
        d, best = f * math.cos(th) + l * math.sin(th), None
        for x0, z0, x1, z1 in walls:
            A = np.array([[d[0], x0 - x1], [d[1], z0 - z1]])
            if abs(np.linalg.det(A)) < 1e-9:
                continue
            r, u = np.linalg.solve(A, np.array([x0 - pose[0], z0 - pose[2]]))
            if r > 0 and 0 <= u <= 1 and r < pl.max_range:
                best = r if best is None else min(best, r)
        if best:
            pts.append((best * math.cos(th), best * math.sin(th)))
    return np.array(pts).reshape(-1, 2)


def run(pl, pose, walls, steps=60):
    traj = [pose]
    for t in range(steps):
        if pl.need_query(pose):
            return traj, f"query ({pl.reason}) at step {t}"
        kind, m = pl.control(pose, scan(pl, pose, walls))
        if kind == "stop":
            return traj, f"stop at step {t}"
        dx, dy, dyaw = m
        f, l = pl.axes(pose[3])
        p = np.array([pose[0], pose[2]]) + dx * f + dy * l
        pose = [p[0], 0, p[1], pose[3] + math.radians(dyaw)]
        traj.append(pose)
    return traj, "ran out"


def seg_dist(p, w):
    a, b = np.array(w[:2]), np.array(w[2:])
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return float(np.linalg.norm(a + t * (b - a) - p))


def min_wall(traj, walls):
    return min(seg_dist(np.array([p[0], p[2]]), w) for p in traj for w in walls)


fwd = lambda n, stop_from=None: {"poses": [[0.25 * (k + 1), 0, 0] for k in range(n)],
                                 "stopped": [stop_from is not None and k >= stop_from for k in range(n)]}
checks = []

# 1. a 0.6 m wide block 1 m ahead on a straight 2 m plan: go around it (or ask System 2), never touch it
walls = [(-0.3, -1.0, 0.3, -1.0)]
pl = LocalPlanner(); pose = [0, 0, 0, 0.0]; pl.set_plan(fwd(8), pose)
tr, end = run(pl, pose, walls)
checks.append(("block ahead: keeps clearance", min_wall(tr, walls) >= pl.clearance - 1e-6, f"{end}, min wall {min_wall(tr, walls):.3f} m, "
               f"reached z {min(p[2] for p in tr):.2f}"))
checks.append(("block ahead: gets past it", min(p[2] for p in tr) < -1.0, ""))
# 2. turn-in-place plan to 90 deg left: turns, does not translate
pl = LocalPlanner(); pose = [0, 0, 0, 0.0]
pl.set_plan({"poses": [[0, 0, math.radians(min(15 * (k + 1), 90))] for k in range(8)], "stopped": [False] * 8}, pose)
tr, end = run(pl, pose, [])
yaw = math.degrees(tr[-1][3]); mv = math.hypot(tr[-1][0], tr[-1][2])
checks.append(("turn plan reaches 90 deg in place", abs(yaw - 90) < 10 and mv < 0.05, f"{end}, yaw {yaw:.1f}, moved {mv:.3f} m"))
# 3. stop flag after 4 forward poses: drives ~1 m then STOP
pl = LocalPlanner(); pose = [0, 0, 0, 0.0]; pl.set_plan(fwd(8, stop_from=4), pose)
tr, end = run(pl, pose, [])
d = math.hypot(tr[-1][0], tr[-1][2])
checks.append(("stop plan stops near 1 m", end.startswith("stop") and 0.8 <= d <= 1.15, f"{end}, travelled {d:.2f} m"))
# 4. stop flag on the first pose: immediate STOP
pl = LocalPlanner(); pose = [0, 0, 0, 0.0]; pl.set_plan(fwd(8, stop_from=0), pose)
tr, end = run(pl, pose, [])
checks.append(("stop-first plan stops at once", end == "stop at step 0", end))
# 5. plan that cuts into a corridor wall: stays inside with clearance
walls = [(-0.25, 1, -0.25, -6), (0.25, 1, 0.25, -6)]
pl = LocalPlanner(); pose = [0, 0, 0, 0.0]
pl.set_plan({"poses": [[0.25 * (k + 1), 0.1 * (k + 1), 0] for k in range(8)], "stopped": [False] * 8}, pose)
tr, end = run(pl, pose, walls)
checks.append(("corridor: keeps clearance", min_wall(tr, walls) >= pl.clearance - 1e-6, f"{end}, min wall {min_wall(tr, walls):.3f} m"))
# 6. dead end (wall across the corridor at 0.6 m): asks System 2 (blocked / off plan), never touches it
walls = [(-0.25, 1, -0.25, -6), (0.25, 1, 0.25, -6), (-0.25, -0.6, 0.25, -0.6)]
pl = LocalPlanner(); pose = [0, 0, 0, 0.0]; pl.set_plan(fwd(8), pose)
tr, end = run(pl, pose, walls)
checks.append(("dead end: asks System 2, no contact", end.startswith("query") and min_wall(tr, walls) >= pl.clearance - 1e-6,
               f"{end}, min wall {min_wall(tr, walls):.3f} m"))
# 7. plan used up: straight 2 m plan in free space ends with a 'plan used up' query
pl = LocalPlanner(max_steps=40); pose = [0, 0, 0, 0.0]; pl.set_plan(fwd(8), pose)
tr, end = run(pl, pose, [])
checks.append(("plan used up -> query", "plan used up" in end, f"{end}, travelled {math.hypot(tr[-1][0], tr[-1][2]):.2f} m"))

fails = 0
for name, ok, info in checks:
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")
print(f"{len(checks) - fails}/{len(checks)} passed")
sys.exit(1 if fails else 0)
