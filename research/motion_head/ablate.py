"""Offline diagnostics on the validation windows (trajectory-held-out).

For each head and input ablation, first-action accuracy against the expert,
split by step category, plus FutureNav's own action on the same windows and a
head-vs-FutureNav disagreement table.

    python ablate.py outputs/motion_head/flow_v1/best.pt outputs/motion_head/reg_v1/best.pt
"""
import json
import sys

import torch

import model as M
from data import Windows
from train import next_action, val_label

CATS = ("straight", "turn", "off_route", "near_goal", "perturbed")


def categories(val, idx, b):
    lab = val_label(b["pose"])
    return {"straight": (lab == 1) & (b["off_route_m"] < 0.5) & (b["remaining_m"] >= 2),
            "turn": (lab != 1) & (b["off_route_m"] < 0.5) & (b["remaining_m"] >= 2),
            "off_route": b["off_route_m"] >= 0.5,
            "near_goal": b["remaining_m"] < 2,
            "all": torch.ones_like(lab, dtype=torch.bool)}


@torch.no_grad()
def run(net, val, idx, ablation, bs=512):
    g = torch.Generator(device=val.device).manual_seed(0)
    rows = []
    for i in range(0, len(idx), bs):
        b = val.batch(idx[i:i + bs], generator=g)
        h, fa, age, vis = b["h"], b["fn_action"], b["age"], b["vis"]
        perm = torch.randperm(len(h), device=h.device, generator=g)
        if ablation == "no_s2_states":
            h = torch.zeros_like(h)
        elif ablation == "shuffled_s2":          # states + action from another window
            h, fa = h[perm], fa[perm]
        elif ablation == "shuffled_fn_action":
            fa = fa[perm]
        elif ablation == "no_vision":
            vis = torch.zeros_like(vis)
        elif ablation == "stale_age7":
            age = torch.full_like(age, 7)
        poses, aux = net.sample(h, fa, age, vis, b["vis_mask"], k=1, generator=g)
        lab = val_label(b["pose"])
        rows.append({"head": next_action(poses[:, 0, 0]), "lab": lab, "fn": b["fn_action"],
                     "stop": b["stop"], **{k: v for k, v in categories(val, idx, b).items()}})
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]}


def acc(r, who, cat):
    m = r[cat] & ~r["stop"]
    return float(((r[who] == r["lab"]) & m).sum() / m.sum()), int(m.sum())


def main():
    val = Windows("val")
    idx = torch.arange(len(val), device=val.device)
    out = {}
    for ck in sys.argv[1:]:
        net, _ = M.load(ck)
        name = ck.split("/")[-2]
        for abl in ("none", "no_s2_states", "shuffled_s2", "shuffled_fn_action", "no_vision", "stale_age7"):
            r = run(net, val, idx, abl)
            out[(name, abl)] = r
            print("%-8s %-19s " % (name, abl) + "  ".join("%s %.3f" % (c, acc(r, "head", c)[0])
                                                        for c in ("all", "straight", "turn", "off_route", "near_goal")))
        r = out[(name, "none")]
    print("%-28s " % "FutureNav action (same ctx)" + "  ".join("%s %.3f" % (c, acc(r, "fn", c)[0])
                                                                for c in ("all", "straight", "turn", "off_route", "near_goal")))
    print("counts: " + "  ".join("%s %d" % (c, acc(r, "fn", c)[1]) for c in ("all", "straight", "turn", "off_route", "near_goal")))
    for ck in sys.argv[1:]:
        name = ck.split("/")[-2]
        r = out[(name, "none")]
        m = ~r["stop"]
        agree = (r["head"] == r["fn"]) & m
        dis = (r["head"] != r["fn"]) & m
        print("%s: agree with FN %.3f (expert-correct when agree %.3f) | disagree: head right %.3f, FN right %.3f, neither %.3f"
              % (name, float(agree.sum() / m.sum()), float(((r["head"] == r["lab"]) & agree).sum() / agree.sum()),
                 float(((r["head"] == r["lab"]) & dis).sum() / dis.sum()), float(((r["fn"] == r["lab"]) & dis).sum() / dis.sum()),
                 float(((r["head"] != r["lab"]) & (r["fn"] != r["lab"]) & dis).sum() / dis.sum())))
        for c in ("straight", "turn", "off_route"):
            d = dis & r[c]
            print("   %-10s disagree %.2f of steps: head right %.2f, FN right %.2f"
                  % (c, float(d.sum() / (m & r[c]).sum()), float(((r["head"] == r["lab"]) & d).sum() / d.sum()),
                     float(((r["fn"] == r["lab"]) & d).sum() / d.sum())))


if __name__ == "__main__":
    main()
