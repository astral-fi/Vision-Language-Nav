"""Path-following expert: the action that gets back onto, and along, the described route.

The route is the episode's dense `locations` from {split}_gt.json.gz (one point
per 0.25 m forward step of the native R2R-CE expert). At each state:

  1. RouteTracker finds the progress index i: the nearest route point within a
     window [i-BACK, i+AHEAD] of the last match, so loops and parallel corridors
     cannot make it jump. If that point is more than REACQUIRE_M away it searches
     every point from i-BACK on (still never behind), and the step is flagged
     `fallback` when even that is beyond OFFROUTE_M.
  2. The target is the route point LOOKAHEAD points (0.25 m) past i. habitat-sim's
     greedy geodesic follower gives the next action toward it on the navmesh,
     with walls, doors and sliding simulated exactly as the agent moves. If the
     agent already sits on the target, the target moves further along.
  3. STOP when within STOP_M of the last route point and i is near the end.
     That is tighter than the 3 m success radius on purpose: labels teach
     stopping at the goal, not anywhere success would still count.

Constants were tuned on 20 distinct train paths, teacher-forced on the native
expert's states: a lookahead of 1 point and STOP_M 0.2 reproduce its actions on
99.8% of steps (nDTW 1.000). A 1 m lookahead cut corners (84%); STOP_M 0.5
stopped a step early.

chunk() rolls the same expert H steps ahead by moving the agent body with
agent.act() (no rendering) and restores the state, so action and chunk labels
always come from one expert. Poses are in the current agent frame.
"""
import numpy as np

from simenv import FORWARD, LEFT, RIGHT, STOP, agent_pose, to_local

BACK, AHEAD = 2, 12            # tracker window, route points (0.25 m each)
REACQUIRE_M = 1.5
TIE_M = 0.1
OFFROUTE_M = 3.0
LOOKAHEAD = 1                  # route points; more cuts corners
STOP_M = 0.2
FOLLOWER_RADIUS = 0.15         # < STOP_M, so the follower never stops first


class RouteTracker(object):
    def __init__(self, locations):
        self.L = np.asarray(locations, dtype=np.float64)
        self.i = 0

    def update(self, pos):
        d = np.linalg.norm(self.L - np.asarray(pos, dtype=np.float64), axis=1)
        lo = max(self.i - BACK, 0)
        hi = min(self.i + AHEAD, len(self.L) - 1)
        j = self._pick(d, lo, hi)
        if d[j] > REACQUIRE_M:
            k = self._pick(d, lo, len(self.L) - 1)
            if d[k] < d[j]:
                j = k
        self.i = j
        return j, float(d[j])

    def _pick(self, d, lo, hi):
        # Routes that double back (36 of 10,819 train paths) put two indices on the
        # same spot. Among near-ties take the first one at or after the current
        # progress; argmin's earliest index would send progress backwards forever.
        w = d[lo:hi + 1]
        tie = lo + np.flatnonzero(w <= w.min() + TIE_M)
        ahead = tie[tie >= self.i]
        return int(ahead[0] if len(ahead) else tie[-1])


class Expert(object):
    def __init__(self, sim, locations):
        self.sim = sim
        self.tracker = RouteTracker(locations)
        self.L = self.tracker.L
        self.follower = sim.make_greedy_follower(
            0, FOLLOWER_RADIUS, stop_key=STOP, forward_key=FORWARD, left_key=LEFT, right_key=RIGHT)

    def label(self):
        """Expert action at the current agent state, with diagnostics."""
        pos, _ = agent_pose(self.sim)
        i, off = self.tracker.update(pos)
        last = len(self.L) - 1
        info = {"progress": i, "progress_frac": i / max(last, 1), "off_route_m": off,
                "fallback": off > OFFROUTE_M, "error": False}
        if np.linalg.norm(pos - self.L[last]) < STOP_M and i >= last - LOOKAHEAD:
            return STOP, info
        k = i + LOOKAHEAD
        while True:
            t = min(k, last)
            try:
                a = self.follower.next_action_along(self.L[t].astype(np.float32))
            except Exception:  # habitat_sim.errors.GreedyFollowerError: no path to the target
                a = None
                info["error"] = True
            if a is not None and a != STOP:
                return int(a), info
            if t == last:
                # On the final point already (within FOLLOWER_RADIUS < STOP_M), or unreachable.
                return STOP, info
            k += LOOKAHEAD

    def chunk(self, H):
        """Roll the expert H steps ahead without rendering, then restore the state.

        Returns actions (<= H, ends at the first STOP), local poses [(x, y, dyaw)]
        after each action, and per-step collision flags.
        """
        agent = self.sim.get_agent(0)
        saved, saved_i = agent.get_state(), self.tracker.i
        p0, y0 = agent_pose(self.sim)
        acts, poses, hits = [], [], []
        try:
            for _ in range(H):
                a, _ = self.label()
                acts.append(a)
                if a == STOP:
                    poses.append(poses[-1] if poses else (0.0, 0.0, 0.0))
                    hits.append(False)
                    break
                hits.append(bool(agent.act(a)))
                p, y = agent_pose(self.sim)
                poses.append(to_local(p0, y0, p, y))
        finally:
            agent.set_state(saved)
            self.tracker.i = saved_i
        return acts, poses, hits
