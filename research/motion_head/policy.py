"""Closed-loop motion-head policy: frozen FutureNav (System 2) + the head (System 1).

One step() per executed action (MOTION_HEAD_DESIGN.md):
  1. Every step the frame joins FutureNav's history and VGGT memory
     (advance_vggt); every R steps the language model also runs, giving the
     L18/22/26/final states and FutureNav's own action. Hooks (cache_s2.Hooks)
     pick these up exactly as when the training data was cached.
  2. The head samples K pose chunks from the latest FutureNav context (age =
     steps since that call) and the VGGT tokens of this and the last PAST frames.
  3. Consistency: keep the sample closest to the previous plan over their
     overlap. If even the closest is more than SWITCH_M away, every sample has
     changed its mind, so take the medoid (the majority mode) and start over.
  4. Blending: the plans of the last BLEND steps that agree with the kept one
     (within AGREE_M at the next step) are averaged at the next time step,
     newer plans weighted more (DECAY per step of age).
  5. Tracker: the move (forward / left / right) whose exact next pose is
     closest to the blended next pose.
  6. STOP: p(stop) > STOP_P, or remaining < STOP_TAU_M while FutureNav's last
     call said STOP. Never before MIN_STEPS. Also stops when circling near the
     goal: < CIRCLE_R_M of motion over CIRCLE_N steps with remaining < 3 m.
Plans are kept in the world frame using the pose the client sends each step.
"""
import math
import os
import sys
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "experiments/data_engine"), os.path.join(ROOT, "deploy")):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M  # noqa: E402
from cache_s2 import LAYERS, R, Hooks  # noqa: E402
from data import PAST  # noqa: E402
from train import next_action  # noqa: E402

K = 8
SWITCH_M = 1.0
AGREE_M = 0.3
BLEND = 4
DECAY = 0.5
STOP_P = 0.5
STOP_TAU_M = 1.0
MIN_STEPS = 5
CIRCLE_N, CIRCLE_R_M = 20, 0.75
# HEAD_SELECT=medoid: diagnostic, no consistency and no blending (medoid of this step's samples only)
SELECT = os.environ.get("HEAD_SELECT", "consistent")
NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")


def local_to_world(poses, p, yaw):
    """poses (...,3) local x fwd, y left, dyaw -> (...,3) world x, z, yaw."""
    fwd = np.array([-math.sin(yaw), -math.cos(yaw)])
    left = np.array([-math.cos(yaw), math.sin(yaw)])
    xz = p[[0, 2]] + poses[..., :1] * fwd + poses[..., 1:2] * left
    return np.concatenate([xz, (yaw + poses[..., 2:3])], -1)


def world_to_local(w, p, yaw):
    d = w[..., :2] - p[[0, 2]]
    fwd = np.array([-math.sin(yaw), -math.cos(yaw)])
    left = np.array([-math.cos(yaw), math.sin(yaw)])
    dy = (w[..., 2] - yaw + math.pi) % (2 * math.pi) - math.pi
    return np.stack([d @ fwd, d @ left, dy], -1)


class HeadPolicy(object):
    def __init__(self, ckpt, fn_policy, seed=0, k=K):
        self.fn = fn_policy
        self.fn.enable_hidden_capture()
        self.hooks = Hooks(self.fn.model)
        self.head, info = M.load(ckpt)
        self.ckpt_info = {"path": ckpt, "mode": self.head.config.mode, "eval": info.get("eval")}
        self.k = k
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.reset()

    def reset(self):
        self.fn.reset()
        self.hooks.v = []
        self.t = 0
        self.ctx = None
        self.vis = deque(maxlen=PAST + 1)
        self.plans = deque(maxlen=BLEND)        # (t made, world chunk (H,3))
        self.positions = deque(maxlen=CIRCLE_N)

    @torch.no_grad()
    def step(self, rgb, instruction, pose):
        """rgb HxWx3 uint8; pose [x, y, z, yaw_rad] of the agent at this frame."""
        p, yaw = np.asarray(pose[:3], dtype=np.float64), float(pose[3])
        if self.t % R == 0:
            self.hooks.on, self.hooks.h = True, {}
            try:
                res = self.fn.act(rgb, instruction, want_hidden=True)
            finally:
                self.hooks.on = False
            h = np.stack([self.hooks.h[k] for k in LAYERS] + [self.fn._hidden.astype(np.float32)])
            self.ctx = (torch.from_numpy(h).half().cuda()[None], int(res["action"]), self.t)
            cache_ok, cache_frames = res["vggt_cache_ok"], res["vggt_cache_frames"]
        else:
            self.fn.advance_vggt(rgb)
            cache_frames = self.fn.vggt_cache_frames()
            cache_ok = cache_frames == min(self.fn.step, self.fn.model.kv_cache_vggt.cache_size)
        if len(self.hooks.v) != 1:
            raise RuntimeError("expected one VGGT frame this step, got %d" % len(self.hooks.v))
        self.vis.appendleft(torch.from_numpy(self.hooks.v.pop()).half())
        h, fn_act, t_call = self.ctx
        n = len(self.vis)
        vis = torch.zeros(1, PAST + 1, *self.vis[0].shape, dtype=torch.float16)
        vis[0, :n] = torch.stack(list(self.vis))
        mask = torch.arange(PAST + 1)[None] < n
        poses, aux = self.head.sample(h, torch.tensor([fn_act], device="cuda"),
                                      torch.tensor([self.t - t_call], device="cuda"),
                                      vis.cuda(), mask.cuda(), k=self.k, generator=self.gen)
        cand = local_to_world(poses[0].float().cpu().numpy(), p, yaw)          # K x H x 3
        if SELECT == "medoid":
            self.plans.clear()
        chosen, switched = self._choose(cand)
        plan = cand[chosen]
        self.plans = deque([(tm, w) for tm, w in self.plans
                            if np.linalg.norm(w[min(self.t - tm, len(w) - 1), :2] - plan[0, :2]) < AGREE_M],
                           maxlen=BLEND) if not switched else deque(maxlen=BLEND)
        self.plans.append((self.t, plan))
        nxt, wsum = np.zeros(3), 0.0
        for tm, w in self.plans:
            i = self.t - tm                       # this plan's entry for time t+1
            if i < len(w):
                wt = DECAY ** (self.t - tm)
                e = w[i].copy()
                e[2] = plan[0, 2] + ((e[2] - plan[0, 2] + math.pi) % (2 * math.pi) - math.pi)
                nxt += wt * e
                wsum += wt
        nxt /= wsum
        local = world_to_local(nxt[None], p, yaw)[0]
        action = int(next_action(torch.from_numpy(local).float()[None])[0])
        p_stop = float(torch.sigmoid(aux["stop_logit"][0]))
        remaining = float(aux["remaining_m"][0])
        self.positions.append(p[[0, 2]])
        circling = (len(self.positions) == CIRCLE_N and remaining < 3.0 and
                    float(np.linalg.norm(np.asarray(self.positions) - np.mean(self.positions, 0), axis=1).max())
                    < CIRCLE_R_M)
        stop_by = None
        if self.t >= MIN_STEPS:
            if p_stop > STOP_P:
                stop_by = "head"
            elif remaining < STOP_TAU_M and fn_act == 0:
                stop_by = "head+futurenav"
            elif circling:
                stop_by = "circling"
        if stop_by:
            action = 0
        ends = cand[:, -1, :2]
        spread = float(np.linalg.norm(ends[:, None] - ends[None], axis=-1).mean())
        self.t += 1
        return {"action": action, "action_name": NAMES[action], "fallback": False,
                "vggt_cache_ok": bool(cache_ok), "vggt_cache_frames": int(cache_frames), "step": self.t,
                "head": {"p_stop": round(p_stop, 3), "remaining_m": round(remaining, 2),
                         "off_route_m": round(float(aux["off_route_m"][0]), 2),
                         "progress": round(float(aux["progress"][0]), 3), "fn_action": NAMES[fn_act],
                         "age": self.t - 1 - t_call, "spread_m": round(spread, 2), "switched": switched,
                         "blended": len(self.plans), "stop_by": stop_by,
                         "next_local": [round(float(x), 3) for x in local]}}

    def _choose(self, cand):
        """Index of the sample to follow, and whether all samples changed mode."""
        prev = self.plans[-1] if self.plans else None
        if prev is not None:
            tm, w = prev
            i = self.t - tm                        # prev plan's index for time t+1
            n = min(len(w) - i, cand.shape[1])
            if n > 0:
                d = np.linalg.norm(cand[:, :n, :2] - w[None, i:i + n, :2], axis=-1).mean(1)
                j = int(d.argmin())
                if d[j] <= SWITCH_M:
                    return j, False
        D = np.linalg.norm(cand[:, None, :, :2] - cand[None, :, :, :2], axis=-1).mean(-1)
        return int(D.sum(1).argmin()), prev is not None
