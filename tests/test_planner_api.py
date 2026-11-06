"""Small API-level tests for the LocalPlanner (sim/s1_local.py)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from s1_local import LocalPlanner  # noqa: E402


def straight_plan(n=8):
    return {"poses": [[0.25 * (k + 1), 0, 0] for k in range(n)], "stopped": [False] * n}


def test_needs_query_without_plan():
    pl = LocalPlanner()
    assert pl.need_query([0, 0, 0, 0.0])


def test_no_query_right_after_plan():
    pl = LocalPlanner()
    pose = [0, 0, 0, 0.0]
    pl.set_plan(straight_plan(), pose)
    assert not pl.need_query(pose)


def test_free_space_moves_forward():
    pl = LocalPlanner()
    pose = [0, 0, 0, 0.0]
    pl.set_plan(straight_plan(), pose)
    kind, motion = pl.control(pose, np.zeros((0, 2)))
    assert kind != "stop"
    dx, dy, dyaw = motion
    assert dx > 0 and abs(dy) < 1e-6 and abs(dyaw) < 1.0


def test_axes_are_orthonormal():
    pl = LocalPlanner()
    for yaw in np.linspace(-math.pi, math.pi, 7):
        f, l = pl.axes(yaw)
        assert abs(np.linalg.norm(f) - 1) < 1e-9 and abs(np.linalg.norm(l) - 1) < 1e-9
        assert abs(float(np.dot(f, l))) < 1e-9
