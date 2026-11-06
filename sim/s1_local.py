"""System 1 for the dual-system eval: a fast LiDAR local planner that tracks System 2's pose plan and decides
when System 2 must be asked again.

System 2 (deploy/head_eval_server.py --exec plan) returns the waypoint head's 8 robot-relative poses only when
asked. This module, run inside eval/run.py (--s1), plays the onboard role:

  plan     the 8 poses are anchored in the world (odometry) frame at the pose they were asked from
  scan     a simulated 2D LiDAR: rays cast along the navmesh with try_step_no_sliding, i.e. the distance the
           robot centre can travel in each direction (Habitat's navmesh is already eroded by the agent radius,
           so this is a real LiDAR range minus the robot radius, in configuration space). Idealised: it sees
           every obstacle the navmesh knows, with no overhang / below-plane misses, noise optional.
  control  receding-horizon arc sampling (DWA-like): forward step x turn per step, rolled out over a short
           horizon, rejected if it comes within `clearance` of a scan point, scored by distance to a lookahead
           point on the plan, heading error and clearance; the first step is executed through
           eval/waypoint_exec.py (MOVE_FORWARD carrying (dx, 0, dyaw)).
  triggers System 2 is queried when: no plan yet; the plan is used up (lookahead reached its end); the robot
           is off the plan by > `deviation`; the planner has been blocked for `blocked_steps` steps; or
           `max_steps` steps passed since the last query. A plan whose stop flag is reached ends the episode
           with STOP (a stop flag on the first pose stops immediately).
"""
import math

import numpy as np


class LocalPlanner:
    def __init__(self, lookahead=0.6, far=1.5, det_look=0.3, grid=2.5, clearance=0.05, horizon=4, max_steps=16, deviation=0.5,
                 blocked_steps=3, rays=180, max_range=5.0, scan_noise=0.0, seed=0):
        self.lookahead, self.far, self.det_look, self.grid = lookahead, far, det_look, grid
        self.clearance, self.horizon = clearance, horizon
        self.max_steps, self.deviation, self.blocked_steps = max_steps, deviation, blocked_steps
        self.rays, self.max_range, self.scan_noise = rays, max_range, scan_noise
        self.rng = np.random.default_rng(seed)
        self.V = np.array([0.0, 0.08, 0.16, 0.25])                  # metres per step
        self.W = np.radians([0, 7.5, -7.5, 15, -15, 22.5, -22.5, 30, -30, 45, -45])   # turn per step
        self.reset()

    def reset(self):
        self.path, self.yaws, self.stop_at_end = None, None, False
        self.s = 0.0
        self.since_query, self.blocked, self.queries = 0, 0, 0
        self.last_cmd, self.last_pose, self.reason = None, None, "start"

    # ------------------------------------------------------------------ geometry helpers (Habitat frames)
    @staticmethod
    def axes(yaw):
        """forward and left unit vectors in world (x, z); Habitat yaw is about +y, forward = -Z at yaw 0."""
        return np.array([-math.sin(yaw), -math.cos(yaw)]), np.array([-math.cos(yaw), math.sin(yaw)])

    def to_robot(self, pose, pts):
        f, l = self.axes(pose[3])
        d = pts - np.array([pose[0], pose[2]])
        return np.stack([d @ f, d @ l], -1)

    # ------------------------------------------------------------------ LiDAR
    def scan(self, sim, pose):
        """Robot-frame (forward, left) hit points of `rays` navmesh rays; rays that hit nothing are dropped."""
        # try_step_no_sliding over long segments is unreliable (end points off the mesh or on another floor can
        # return 0 m), so each ray is marched in `chunk` steps from a start snapped onto the navmesh, and the ray
        # ends where a chunk is cut short.
        pf = sim.pathfinder
        p0 = np.array(pf.snap_point(np.array(sim.get_agent_state().position, dtype=np.float32)), dtype=np.float32)
        f, l = self.axes(pose[3])
        pts, chunk = [], 0.1
        for th in np.linspace(-math.pi, math.pi, self.rays, endpoint=False):
            dxz = f * math.cos(th) + l * math.sin(th)
            d = np.array([dxz[0], 0, dxz[1]], np.float32)
            p, r = p0, 0.0
            while r < self.max_range - 1e-6:
                step = min(chunk, self.max_range - r)
                q = np.array(pf.try_step_no_sliding(p, p + d * step), dtype=np.float32)
                got = float(np.hypot(q[0] - p[0], q[2] - p[2]))
                r += got
                if got < step - 1e-3:
                    break
                p = q
            if r < self.max_range - 1e-3:
                if self.scan_noise:
                    r = max(0.0, r + self.rng.normal(0, self.scan_noise))
                pts.append((r * math.cos(th), r * math.sin(th)))
        return np.array(pts).reshape(-1, 2)

    # ------------------------------------------------------------------ plan handling
    def set_plan(self, plan, pose):
        """plan = {"poses": [[x fwd, y left, yaw], ...] robot-relative, "stopped": [bool]*8}, asked from `pose`."""
        f, l = self.axes(pose[3])
        o = np.array([pose[0], pose[2]])
        pts, yaws, stop = [o], [pose[3]], False
        for (x, y, yw), st in zip(plan["poses"], plan["stopped"]):
            if st:
                stop = True
                break
            pts.append(o + x * f + y * l)
            yaws.append(pose[3] + yw)
        self.path, self.yaws, self.stop_at_end = np.array(pts), np.array(yaws), stop
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.s, self.since_query, self.blocked = 0.0, 0, 0
        self.queries += 1

    def _project(self, xy):
        """Arc length of the closest point on the path at or after the current progress."""
        best, best_d = self.s, None
        for k in range(len(self.path) - 1):
            a, b = self.path[k], self.path[k + 1]
            L = self.cum[k + 1] - self.cum[k]
            t = 0.0 if L < 1e-6 else float(np.clip((xy - a) @ (b - a) / L ** 2, 0, 1))
            s = self.cum[k] + t * L
            d = float(np.linalg.norm(a + t * (b - a) - xy))
            if s >= self.s - 1e-6 and (best_d is None or d < best_d):
                best, best_d = s, d
        return best, (best_d if best_d is not None else float(np.linalg.norm(self.path[-1] - xy)))

    def _point_at(self, s):
        k = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.path) - 2)) if len(self.path) > 1 else 0
        if len(self.path) == 1:
            return self.path[0], self.yaws[0]
        L = self.cum[k + 1] - self.cum[k]
        t = 0.0 if L < 1e-6 else float(np.clip((s - self.cum[k]) / L, 0, 1))
        return self.path[k] + t * (self.path[k + 1] - self.path[k]), self.yaws[k] + t * (self.yaws[k + 1] - self.yaws[k])

    # ------------------------------------------------------------------ trigger + control
    def need_query(self, pose):
        """Called before every step. True means: ask System 2 for a new plan this step."""
        if self.last_cmd is not None and self.last_pose is not None:
            moved = math.hypot(pose[0] - self.last_pose[0], pose[2] - self.last_pose[2])
            if self.last_cmd[0] > 0.05 and moved < 0.5 * self.last_cmd[0]:
                self.blocked += 1
            elif self.last_cmd[0] > 0.05:
                self.blocked = 0
        if self.path is None:
            self.reason = "no plan"
            return True
        xy = np.array([pose[0], pose[2]])
        self.s, dev = self._project(xy)
        end_xy, end_yaw = self.path[-1], self.yaws[-1]
        at_end = (self.cum[-1] - self.s < 0.1 and np.linalg.norm(end_xy - xy) < 0.2
                  and abs((pose[3] - end_yaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(10))
        if at_end and not self.stop_at_end:
            self.reason = "plan used up"
            return True
        if dev > self.deviation:
            self.reason = "off plan"
            return True
        if self.blocked >= self.blocked_steps:
            self.reason = "blocked"
            return True
        if self.since_query >= self.max_steps:
            self.reason = "max steps"
            return True
        return False

    def _detour_target(self, scan_pts, goal):
        """Local A* on an occupancy grid built from the scan (robot frame, 5 cm cells, +-GRID m), from the robot to
        `goal` (robot frame; snapped to the nearest free cell). Returns the point ~`det_look` along the grid path,
        or None when no path exists. Unseen space counts as free."""
        import heapq
        cell, half = 0.05, self.grid
        n = int(2 * half / cell)
        occ = np.zeros((n, n), bool)
        if len(scan_pts):
            infl = self.clearance + 0.05
            r = int(math.ceil(infl / cell))
            ij = np.floor((scan_pts + half) / cell).astype(int)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if (di * cell) ** 2 + (dj * cell) ** 2 <= infl ** 2:
                        q = ij + (di, dj)
                        ok = (q >= 0).all(1) & (q < n).all(1)
                        occ[q[ok, 0], q[ok, 1]] = True
        to_ij = lambda p: tuple(np.clip(np.floor((np.asarray(p) + half) / cell).astype(int), 0, n - 1))
        s, g = to_ij((0.0, 0.0)), to_ij(np.clip(goal, -half + 1e-3, half - 1e-3))
        occ[s] = False
        if occ[g]:
            free = np.argwhere(~occ)
            g = tuple(free[np.argmin(np.hypot(*(free - g).T))])
        dist = np.full((n, n), np.inf)
        dist[s] = 0.0
        prev, pq = {}, [(0.0, s)]
        steps = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while pq:
            _, u = heapq.heappop(pq)
            if u == g:
                break
            for di, dj, c in steps:
                v = (u[0] + di, u[1] + dj)
                if 0 <= v[0] < n and 0 <= v[1] < n and not occ[v] and dist[u] + c < dist[v]:
                    dist[v] = dist[u] + c
                    prev[v] = u
                    heapq.heappush(pq, (dist[v] + math.hypot(v[0] - g[0], v[1] - g[1]), v))
        if g != s and g not in prev:
            return None
        path, u = [g], g
        while u != s:
            u = prev[u]
            path.append(u)
        pts = (np.array(path[::-1]) + 0.5) * cell - half
        seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        return pts[min(np.searchsorted(seg, self.det_look), len(pts) - 1)]

    def _front_clear(self, scan_pts):
        """Clearance of the robot's current position (used to score turning in place like the arcs)."""
        return float(np.min(np.hypot(*scan_pts.T))) if len(scan_pts) else self.max_range

    def control(self, pose, scan_pts):
        """Returns ("stop", None) or ("move", (dx, dy, dyaw_deg)) for this step."""
        self.since_query += 1
        xy = np.array([pose[0], pose[2]])
        if self.path is None:
            return "stop", None
        if self.stop_at_end and len(self.path) == 1:
            return "stop", None
        end_xy = self.path[-1]
        if self.stop_at_end and self.cum[-1] - self.s < 0.15 and np.linalg.norm(end_xy - xy) < 0.2:
            return "stop", None
        tgt_xy, tgt_yaw = self._point_at(min(self.s + self.lookahead, self.cum[-1]))
        tgt = self.to_robot(pose, tgt_xy[None])[0]
        dist = float(np.hypot(*tgt))
        if dist > 0.15:
            # steer toward the A* path to a farther plan point, so obstacles are bypassed instead of stalled at
            far_xy, _ = self._point_at(min(self.s + self.far, self.cum[-1]))
            det = self._detour_target(scan_pts, self.to_robot(pose, far_xy[None])[0])
            if det is None:
                self.blocked += 1
            elif np.hypot(*det) > 0.1:
                tgt = det
                dist = float(np.hypot(*tgt))
        # near the target position: align with the plan's heading there (pure turn plans, final heading)
        aim = math.atan2(tgt[1], tgt[0]) if dist > 0.15 else (tgt_yaw - pose[3] + math.pi) % (2 * math.pi) - math.pi
        best, best_cost = None, None
        for v in self.V:
            for w in self.W:
                # Roll the command out for up to `horizon` steps. Only the first step is executed (the planner runs
                # again next step, and turning in place is always safe), so only the first step must keep
                # `clearance`; the rollout ends at its first unsafe step, is scored at its best safe step (arcs that
                # would overshoot the target still count), and short safe prefixes are penalised.
                x = y = th = 0.0
                ok, clear, cost, kfree = True, self.max_range, None, self.horizon
                for k in range(self.horizon):
                    th += w / 2
                    x0, y0 = x, y
                    x, y = x + v * math.cos(th), y + v * math.sin(th)
                    th += w / 2
                    if len(scan_pts) and v > 0:
                        # swept check: points every <= 4 cm along the step, so a thin wall cannot be stepped over
                        n = max(2, int(math.ceil(v / 0.04)) + 1)
                        sx, sy = np.linspace(x0, x, n)[1:], np.linspace(y0, y, n)[1:]
                        c = float(np.min(np.hypot(scan_pts[:, 0][None] - sx[:, None], scan_pts[:, 1][None] - sy[:, None])))
                        if c < self.clearance:
                            ok, kfree = k > 0, k
                            break
                        clear = min(clear, c)
                    if dist > 0.15:
                        d = math.hypot(tgt[0] - x, tgt[1] - y)
                        h = abs((math.atan2(tgt[1] - y, tgt[0] - x) - th + math.pi) % (2 * math.pi) - math.pi) if d > 0.15 else 0.0
                        ck = d + 0.3 * h
                    else:
                        ck = abs((aim - th + math.pi) % (2 * math.pi) - math.pi) + 2.0 * math.hypot(x, y)
                    ck += 0.02 * k                    # prefer getting there sooner
                    cost = ck if cost is None else min(cost, ck)
                if not ok:
                    continue
                cost += 0.3 * (self.horizon - kfree) / self.horizon
                cost -= 0.3 * min(clear, 0.5) if v > 0 else 0.3 * min(self._front_clear(scan_pts), 0.5)
                if best_cost is None or cost < best_cost:
                    best, best_cost = (v, w), cost
        if best is None or best == (0.0, 0.0):
            # nothing useful is safe: count it, and turn in place toward the target rather than idle
            if dist > 0.15:
                self.blocked += 1
            best = (0.0, math.copysign(math.radians(30), aim if abs(aim) > 1e-3 else 1.0))
        v, w = best
        self.last_cmd, self.last_pose = (v, w), pose
        return "move", (v * math.cos(w / 2), v * math.sin(w / 2), math.degrees(w))
