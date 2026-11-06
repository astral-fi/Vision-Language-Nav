"""Per-episode perturbations for data collection (STAGE_A.md A3).

The executed action differs from the expert's only inside a perturbation; the
label is always the expert's, so the data shows mistakes and how to undo them.
Teleports are left out: they put a jump into the frame history that a real
agent never sees. A detour reaches the same off-route states physically.
Detours were lengthened after a 45-episode sample: at 1-2.5 m only half got >1 m off-route.

  heading   start facing a uniform random direction (the wrong-start case)
  jitter    each step, with probability eps, 1-3 random moves
  detour    at chosen route progress: turn 60-150 deg to a random side, then
            walk 6-16 steps forward (1.5-4 m); collisions end it early

Mix per episode (CLEAN_P clean, the rest independent draws) so the data keeps
plenty of clean driving next to the recovery cases. No new perturbation starts
within END_GUARD route points of the goal, so stop examples stay clean.
"""
import numpy as np

from simenv import FORWARD, LEFT, RIGHT

CLEAN_P = 0.2
HEADING_P = 0.5
JITTER_EPS = (0.0, 0.03, 0.08)
DETOUR_MEAN = 1.0              # Poisson mean of detours per episode
END_GUARD = 6                  # route points (1.5 m)
TURN_DEG = 15


def sample(rng, n_route):
    """Draw an episode's perturbation config; route has n_route points."""
    if rng.random() < CLEAN_P:
        return {"clean": True, "heading": None, "jitter_eps": 0.0, "detours": []}
    last = max(n_route - 1 - END_GUARD, 1)
    k = int(rng.poisson(DETOUR_MEAN))
    at = sorted(int(x) for x in rng.integers(1, last + 1, size=k)) if k else []
    return {"clean": False,
            "heading": float(rng.uniform(-np.pi, np.pi)) if rng.random() < HEADING_P else None,
            "jitter_eps": float(rng.choice(JITTER_EPS)),
            "detours": [{"at": a, "turns": int(rng.integers(4, 11)), "side": int(rng.choice((LEFT, RIGHT))),
                         "forward": int(rng.integers(6, 17))} for a in at]}


class Perturber(object):
    """Yields the action to execute in place of the expert's, or None."""

    def __init__(self, cfg, rng, n_route):
        self.cfg, self.rng, self.n_route = cfg, rng, n_route
        self.queue, self.kind = [], None
        self.pending = list(cfg["detours"])

    def next(self, progress, last_collided):
        if self.queue and self.kind == "detour" and last_collided and self.queue[0] == FORWARD:
            self.queue = []                      # walked into a wall: the detour ends there
        if not self.queue:
            self.kind = None
            if progress < self.n_route - 1 - END_GUARD:
                if self.pending and progress >= self.pending[0]["at"]:
                    d = self.pending.pop(0)
                    self.queue, self.kind = [d["side"]] * d["turns"] + [FORWARD] * d["forward"], "detour"
                elif self.cfg["jitter_eps"] and self.rng.random() < self.cfg["jitter_eps"]:
                    n = int(self.rng.integers(1, 4))
                    self.queue = [int(a) for a in self.rng.choice((FORWARD, LEFT, RIGHT), size=n)]
                    self.kind = "jitter"
        if self.queue:
            return self.queue.pop(0), self.kind
        return None, None
