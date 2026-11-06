"""Habitat env and pose helpers shared by the data engine (habitat env, habitat 0.1.7).

make_env builds the exact eval simulator (eval/config/futurenav_r2r.yaml) on any
R2R-CE split, dropping the task sensors we do not need. Episodes are served in
the order given via env_episode(), not habitat's iterator.

Poses: habitat agents face -Z; yaw is the rotation about +Y, the same convention
as eval/run.py's yaw_deg (2 * atan2(q.y, q.w)), so TURN_LEFT increases yaw.
"""
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, "VLN_CE")):
    if p not in sys.path:
        sys.path.insert(0, p)

R2R = os.path.join(ROOT, "data/datasets/R2R_VLNCE_v1-3_preprocessed")
CONFIG = os.path.join(ROOT, "eval/config/futurenav_r2r.yaml")
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")


def load_gt(split):
    with gzip.open(os.path.join(R2R, split, "%s_gt.json.gz" % split)) as f:
        return json.load(f)


def make_env(split, ids=None, gpu=0):
    from habitat import Env
    from habitat.datasets import make_dataset
    from VLN_CE.vlnce_baselines.config.default import get_config
    os.chdir(ROOT)
    config = get_config(CONFIG)
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu
    config.TASK_CONFIG.TASK.SENSORS = ["INSTRUCTION_SENSOR"]
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 100000
    config.freeze()
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    if ids is not None:
        wanted = set(map(str, ids))
        dataset.episodes = [e for e in dataset.episodes if str(e.episode_id) in wanted]
    env = Env(config.TASK_CONFIG, dataset)
    env.by_id = {str(e.episode_id): e for e in dataset.episodes}
    return env


def env_episode(env, eid):
    """Reset env onto episode eid (habitat 0.1.7 reads the next episode from its iterator)."""
    env._episode_iterator = iter([env.by_id[str(eid)]])
    return env.reset()


def yaw_of(q):
    return 2.0 * math.atan2(q.y, q.w)


def quat_of(yaw):
    import quaternion  # numpy-quaternion, habitat's rotation type
    return quaternion.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def agent_pose(sim):
    st = sim.get_agent(0).get_state()
    return np.array(st.position, dtype=np.float64), yaw_of(st.rotation)


def to_local(p0, yaw0, p, yaw):
    """Pose (p, yaw) in the frame of (p0, yaw0): x forward, y left, dyaw (rad)."""
    d = np.asarray(p, dtype=np.float64) - p0
    fwd = np.array([-math.sin(yaw0), 0.0, -math.cos(yaw0)])
    left = np.array([-math.cos(yaw0), 0.0, math.sin(yaw0)])
    return float(d @ fwd), float(d @ left), wrap(yaw - yaw0)


def set_pose(sim, p, yaw):
    agent = sim.get_agent(0)
    st = agent.get_state()
    st.position = np.asarray(p, dtype=np.float32)
    st.rotation = quat_of(yaw)
    agent.set_state(st)


def ndtw(path, ref, d_th=3.0):
    """Normalised DTW (Ilharco et al. 2019) between agent path and reference, both N x 3."""
    P, R = np.asarray(path, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    D = np.linalg.norm(P[:, None] - R[None], axis=2)
    C = np.full((len(P) + 1, len(R) + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, len(P) + 1):
        for j in range(1, len(R) + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return float(np.exp(-C[-1, -1] / (len(R) * d_th)))
