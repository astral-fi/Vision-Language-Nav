"""Expert + perturbation data collection, simulator only (STAGE_A.md A4). Habitat env.

Every episode on the given side of split.json runs once with a perturbation
config from perturb.sample (seeded by episode id). At every step the expert
labels the state; the executed action is the expert's unless a perturbation
overrides it. The episode ends when the expert says STOP outside a
perturbation, or at the step cap.

One file per episode, outputs/data_engine/expert_v1/<scene>/<eid>.npz:
  jpg_blob, jpg_off   JPEG frames (q=JPEG_Q) concatenated; frame t is
                      jpg_blob[jpg_off[t]:jpg_off[t+1]]
  pos (T,3), yaw (T,)               agent pose when frame t was taken
  label (T,)                        expert action at t (0 STOP 1 FWD 2 L 3 R)
  executed (T,)                     action actually taken at t (-1 after the last)
  perturb (T,)                      0 none, 1 jitter, 2 detour, 3 start heading
  chunk_act (T,H), chunk_pose (T,H,3)  expert H steps ahead: actions (-1 pad) and
                      local poses (x fwd, y left, dyaw), padded with the stop pose
  progress (T,), off_route (T,), fallback (T,), error (T,), collided (T,)
  meta_json           episode id, scene, instruction, trajectory id, perturbation
                      config, outcome (distance to goal, success, steps)
Files are written to a temp name and renamed, so a killed run resumes cleanly.

    python collect.py --side train --shard 0 --shards 6
"""
from __future__ import print_function

import argparse
import json
import os
import time

import cv2
import numpy as np

import expert as X
import perturb as Pt
from simenv import ROOT, STOP, agent_pose, env_episode, load_gt, make_env, set_pose
from validate import episodes

OUT = os.path.join(ROOT, "outputs/data_engine/expert_v1")
H = 16
JPEG_Q = 90
MIN_FREE_GB = 20


def cap(n_actions):
    return min(3 * n_actions + 60, 500)


def run_episode(env, ep, g, rng):
    obs = env_episode(env, ep["episode_id"])
    ex = X.Expert(env.sim, g["locations"])
    cfg = Pt.sample(rng, len(ex.L))
    pert = Pt.Perturber(cfg, rng, len(ex.L))
    first = None
    if cfg["heading"] is not None:
        p, _ = agent_pose(env.sim)
        set_pose(env.sim, p, cfg["heading"])
        obs = env.sim.get_observations_at()
        first = 3
    rec = {k: [] for k in ("pos", "yaw", "label", "executed", "perturb", "chunk_act", "chunk_pose",
                           "progress", "off_route", "fallback", "error", "collided")}
    frames, collided = [], False
    for t in range(cap(len(g["actions"]))):
        p, y = agent_pose(env.sim)
        a, info = ex.label()
        acts, poses, _ = ex.chunk(H)
        ok, buf = cv2.imencode(".jpg", obs["rgb"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        assert ok
        frames.append(buf.tobytes())
        override, kind = pert.next(info["progress"], collided)
        do = a if override is None else override
        rec["pos"].append(p)
        rec["yaw"].append(y)
        rec["label"].append(a)
        rec["perturb"].append(first if (first and t == 0) else {"jitter": 1, "detour": 2}.get(kind, 0))
        rec["chunk_act"].append(acts + [-1] * (H - len(acts)))
        rec["chunk_pose"].append(poses + [poses[-1]] * (H - len(poses)))
        for k in ("progress", "off_route", "fallback", "error"):
            rec[k].append(info[k if k != "off_route" else "off_route_m"])
        rec["collided"].append(collided)
        if do == STOP:
            rec["executed"].append(STOP)
            break
        rec["executed"].append(do)
        obs = env.step(do)
        collided = bool(env.sim.previous_step_collided)
    m = env.get_metrics()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["pos"], out["yaw"] = out["pos"].astype(np.float32), out["yaw"].astype(np.float32)
    out["chunk_pose"] = out["chunk_pose"].astype(np.float32)
    out["off_route"] = out["off_route"].astype(np.float32)
    for k in ("label", "executed", "perturb", "chunk_act", "progress"):
        out[k] = out[k].astype(np.int16)
    off = np.cumsum([0] + [len(f) for f in frames]).astype(np.int64)
    out["jpg_blob"], out["jpg_off"] = np.frombuffer(b"".join(frames), dtype=np.uint8), off
    meta = {"episode_id": str(ep["episode_id"]), "scene": ep["scene_id"], "trajectory_id": ep["trajectory_id"],
            "instruction": ep["instruction"]["instruction_text"], "perturbation": cfg, "H": H,
            "steps": len(frames), "expert_stopped": bool(rec["executed"][-1] == STOP),
            "distance_to_goal": float(m["distance_to_goal"]), "success": bool(m["distance_to_goal"] < 3.0)}
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="train", choices=("train", "holdout"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with open(os.path.join(ROOT, "outputs/data_engine/split.json")) as f:
        ids = json.load(f)[a.side]
    eps, gt = episodes("train"), load_gt("train")
    ids = sorted(ids, key=lambda i: (eps[i]["scene_id"], int(i)))
    scenes = sorted(set(eps[i]["scene_id"] for i in ids))
    mine = set(scenes[a.shard::a.shards])            # shard by building: one scene load each
    ids = [i for i in ids if eps[i]["scene_id"] in mine]
    if a.limit:
        ids = ids[:a.limit]
    todo = [i for i in ids if not os.path.exists(path_of(eps[i]))]
    print("shard %d/%d: %d buildings, %d episodes, %d to do" % (a.shard, a.shards, len(mine), len(ids), len(todo)),
          flush=True)
    if not todo:
        return
    env = make_env("train", todo)
    t0, n_steps = time.time(), 0
    for n, eid in enumerate(todo):
        st = os.statvfs(OUT if os.path.isdir(OUT) else ROOT)
        if st.f_bavail * st.f_frsize < MIN_FREE_GB * 2 ** 30:
            print("STOP: less than %d GB free" % MIN_FREE_GB, flush=True)
            return
        ep = eps[eid]
        out, meta = run_episode(env, ep, gt[eid], np.random.default_rng(int(eid)))
        p = path_of(ep)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p + ".tmp.npz", **out)
        os.replace(p + ".tmp.npz", p)
        n_steps += meta["steps"]
        if n % 50 == 0 or n == len(todo) - 1:
            el = time.time() - t0
            print("shard %d  %d/%d  %.1f steps/s  eta %.0f min" % (
                a.shard, n + 1, len(todo), n_steps / el, el / (n + 1) * (len(todo) - n - 1) / 60), flush=True)
    print("done shard %d" % a.shard, flush=True)


def path_of(ep):
    return os.path.join(OUT, ep["scene_id"].split("/")[1], "%s.npz" % ep["episode_id"])


if __name__ == "__main__":
    main()
