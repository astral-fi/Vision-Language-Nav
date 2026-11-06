"""Collect RGB + ground-truth depth along expert trajectories, for checking VGGT depth (eval_depth.py).

One val_unseen episode per scene (the first by id), driven by the geodesic oracle (SHORTEST_PATH_SENSOR) until
STOP or --max-steps. The depth sensor has the RGB camera's exact geometry (640x480, HFOV 79, 1.25 m, level), is
added only here, and stores metres (NORMALIZE_DEPTH False, range 0-10 m; 0 = no hit).

    PATH=~/miniforge3/envs/habitat/bin:$PATH python experiments/vggt_depth/collect.py
Output: outputs/vggt_depth/frames/<episode>.npz  rgb [T,480,640,3] uint8, depth [T,480,640] float16 m,
        pos [T,3], yaw [T] rad, action [T] (oracle action taken after the frame).
"""
import argparse
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "VLN_CE")]
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np  # noqa: E402
from habitat import Env  # noqa: E402
from habitat.datasets import make_dataset  # noqa: E402
from VLN_CE.vlnce_baselines.config.default import get_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--max-steps", type=int, default=120)
ap.add_argument("--per-scene", type=int, default=1)
ap.add_argument("--out", default=str(ROOT / "outputs/vggt_depth/frames"))
ap.add_argument("--cam-height", type=float, default=None, help="override RGB + depth sensor height (m)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

config = get_config(str(ROOT / "eval/config/futurenav_r2r.yaml"))
config.defrost()
config.TASK_CONFIG.DATASET.SPLIT = "val_unseen"
sim = config.TASK_CONFIG.SIMULATOR
sim.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
if a.cam_height is not None:
    sim.RGB_SENSOR.POSITION = [0, a.cam_height, 0]
for k in ("WIDTH", "HEIGHT", "HFOV", "POSITION"):
    setattr(sim.DEPTH_SENSOR, k, getattr(sim.RGB_SENSOR, k))
sim.DEPTH_SENSOR.NORMALIZE_DEPTH = False
sim.DEPTH_SENSOR.MIN_DEPTH = 0.0
sim.DEPTH_SENSOR.MAX_DEPTH = 10.0
config.freeze()

dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
dataset.episodes.sort(key=lambda e: int(e.episode_id))
by_scene = OrderedDict()
for e in dataset.episodes:
    by_scene.setdefault(e.scene_id, [])
    if len(by_scene[e.scene_id]) < a.per_scene:
        by_scene[e.scene_id].append(e)
dataset.episodes = [e for v in by_scene.values() for e in v]
print(f"{len(dataset.episodes)} episodes from {len(by_scene)} scenes; "
      f"depth {sim.DEPTH_SENSOR.WIDTH}x{sim.DEPTH_SENSOR.HEIGHT} HFOV {sim.DEPTH_SENSOR.HFOV} at {list(sim.DEPTH_SENSOR.POSITION)}")

env = Env(config.TASK_CONFIG, dataset)
for _ in range(len(dataset.episodes)):
    obs = env.reset()
    ep = env.current_episode
    rows = {k: [] for k in ("rgb", "depth", "pos", "yaw", "action")}
    for t in range(a.max_steps):
        st = env.sim.get_agent_state()
        act = int(obs["shortest_path_sensor"][0])
        rows["rgb"].append(obs["rgb"])
        rows["depth"].append(obs["depth"][..., 0].astype(np.float16))
        rows["pos"].append(np.array(st.position, np.float32))
        rows["yaw"].append(2 * math.atan2(st.rotation.y, st.rotation.w))
        rows["action"].append(act)
        if act == 0 or env.episode_over:
            break
        obs = env.step({"action": act})
    out = {k: np.stack(v) if k in ("rgb", "depth", "pos") else np.array(v) for k, v in rows.items()}
    np.savez_compressed(Path(a.out) / f"{ep.episode_id}.npz", scene=ep.scene_id.split("/")[-2], **out)
    print(f"episode {ep.episode_id} ({ep.scene_id.split('/')[-2]}): {len(out['rgb'])} frames", flush=True)
env.close()
