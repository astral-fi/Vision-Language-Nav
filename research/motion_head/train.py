"""Train the motion head (flow or regression) on cached windows.

    python train.py --mode flow --steps 20000 --tag flow_v1
    python train.py --mode reg  --steps 20000 --tag reg_v1

Writes outputs/motion_head/<tag>/{best.pt,last.pt,history.json}. Validation
(trajectory-held-out windows, deterministic subset) reports, on samples drawn
the way the agent will draw them:
  first_action_acc   tracker(first predicted pose) == expert label, non-STOP steps
  turn_side_acc      sign of the chunk's net heading change vs the expert's,
                     where the expert turns > 30 deg within the chunk
  ade_m / fde_m      mean / final position error over the chunk (m)
  stop_f1            p(stop) > 0.5 vs the expert's STOP label
  remaining_mae_m, off_route_mae_m
"""
from __future__ import print_function

import argparse
import json
import math
import os
import time

import torch

import model as M
from data import Windows

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TURN = math.radians(15.0)


def next_action(pose1):
    """Nearest-motion tracker. pose1 (..., 3) predicted next pose (x fwd, y left, dyaw).
    Returns 1 FORWARD, 2 LEFT, 3 RIGHT: the move whose exact next pose is closest."""
    cand = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.0, TURN], [0.0, 0.0, -TURN]], device=pose1.device)
    d = (pose1[..., None, :2] - cand[:, :2]).norm(dim=-1) / 0.25 + (pose1[..., None, 2] - cand[:, 2]).abs() / TURN
    return d.argmin(-1) + 1


def step_weights(H, device):
    w = torch.ones(H, device=device)
    w[:8] = 2.0
    return w


@torch.no_grad()
def evaluate(net, val, idx, bs=512):
    net.eval()
    g = torch.Generator(device=val.device).manual_seed(0)
    agg = {k: [] for k in ("fa", "fa_n", "ts", "ts_n", "ade", "fde", "tp", "fp", "fn", "rem", "off")}
    for i in range(0, len(idx), bs):
        b = val.batch(idx[i:i + bs], generator=g)
        poses, aux = net.sample(b["h"], b["fn_action"], b["age"], b["vis"], b["vis_mask"], k=1, generator=g)
        p, y = poses[:, 0], b["pose"]
        lab = val_label(y)
        moving = ~b["stop"]
        agg["fa"].append(((next_action(p[:, 0]) == lab) & moving).sum())
        agg["fa_n"].append(moving.sum())
        big = (y[:, -1, 2].abs() > math.radians(30)) & moving
        agg["ts"].append(((p[:, -1, 2].sign() == y[:, -1, 2].sign()) & big).sum())
        agg["ts_n"].append(big.sum())
        err = (p[..., :2] - y[..., :2]).norm(dim=-1)
        agg["ade"].append(err.mean(1).sum())
        agg["fde"].append(err[:, -1].sum())
        ps = aux["stop_logit"] > 0
        agg["tp"].append((ps & b["stop"]).sum())
        agg["fp"].append((ps & ~b["stop"]).sum())
        agg["fn"].append((~ps & b["stop"]).sum())
        agg["rem"].append((aux["remaining_m"] - b["remaining_m"]).abs().sum())
        agg["off"].append((aux["off_route_m"] - b["off_route_m"]).abs().sum())
    s = {k: float(torch.stack(v).sum()) for k, v in agg.items()}
    n = len(idx)
    net.train()
    return {"first_action_acc": s["fa"] / max(s["fa_n"], 1), "turn_side_acc": s["ts"] / max(s["ts_n"], 1),
            "turn_side_n": int(s["ts_n"]), "ade_m": s["ade"] / n, "fde_m": s["fde"] / n,
            "stop_f1": 2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1),
            "remaining_mae_m": s["rem"] / n, "off_route_mae_m": s["off"] / n}


def val_label(y):
    """Expert's first action from its chunk: the move matching the first pose exactly."""
    return next_action(y[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("flow", "reg"), default="flow")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0, help="episodes, for smoke tests")
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()
    out = os.path.join(ROOT, "outputs/motion_head", a.tag)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    tr, va = Windows("train", limit=a.limit), Windows("val", limit=max(a.limit // 10, 0))
    print("windows: train %d (%d episodes)  val %d (%d episodes)  load %.0f s"
          % (len(tr), len(tr.episodes), len(va), len(va.episodes), time.time() - t0), flush=True)
    net = M.MotionHead(M.Config(mode=a.mode)).cuda()
    with torch.no_grad():                                    # pose normalisation from training targets
        p = tr.pose[torch.randint(len(tr.pose), (min(len(tr.pose), 50000),), device=tr.pose.device)]
        p4 = torch.stack([p[..., 0], p[..., 1], p[..., 2].sin(), p[..., 2].cos()], -1).flatten(0, 1)
        net.pose_mean.copy_(p4.mean(0))
        net.pose_std.copy_(p4.std(0).clamp(min=1e-3))
    print("parameters %.1fM" % (sum(x.numel() for x in net.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    sw = step_weights(net.config.horizon, "cuda")
    val_idx = torch.randperm(len(va), generator=torch.Generator().manual_seed(1))[:20000].cuda()
    hist, best = [], -1.0
    # weighted sampling: decision steps DECISION_W times as likely
    probs = tr.weight / tr.weight.sum()
    for step in range(1, a.steps + 1):
        idx = torch.multinomial(probs, a.bs, replacement=True)
        b = tr.batch(idx)
        b["weight"] = torch.ones_like(b["weight"])           # weighting is in the sampling already
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(b, sw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print("step %d  loss %.4f  %s  %.0f s" % (step, float(loss), " ".join("%s %.3f" % kv for kv in parts.items()),
                                                    time.time() - t0), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(net, va, val_idx)
            ev["step"] = step
            hist.append(ev)
            print("EVAL " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}), flush=True)
            M.save(net, os.path.join(out, "last.pt"), {"args": vars(a), "eval": ev})
            score = ev["first_action_acc"] + ev["turn_side_acc"]
            if score > best:
                best = score
                M.save(net, os.path.join(out, "best.pt"), {"args": vars(a), "eval": ev})
            with open(os.path.join(out, "history.json"), "w") as f:
                json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main()
