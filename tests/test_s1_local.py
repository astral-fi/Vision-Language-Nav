"""Run the System 1 local-planner checks (sim/check_s1_local.py) as a test.

The checks drive the LocalPlanner in a kinematic 2D world with ray-cast LiDAR against wall segments: no Habitat,
no GPU. They cover obstacle bypass with clearance, turning in place, STOP handling, corridor following,
dead-end escalation to System 2 and plan exhaustion.
"""
import subprocess
import sys
from pathlib import Path

import pytest

CHECKS = Path(__file__).resolve().parents[1] / "sim" / "check_s1_local.py"


@pytest.fixture(scope="module")
def result():
    return subprocess.run([sys.executable, str(CHECKS)], capture_output=True, text=True, timeout=300)


def test_all_checks_pass(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_check_reported(result):
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith(("PASS", "FAIL"))]
    assert len(lines) >= 8
    assert not [ln for ln in lines if ln.startswith("FAIL")]
