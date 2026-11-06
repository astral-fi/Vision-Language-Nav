"""Steps 1-2 after test 0: which LoRA layer carries the turn side, and does a waypoint head reading that layer
turn the right way more often?

Data: the waypoint head's own cache (outputs/pose_head/feats, h_lora at layers 18, 22, 26, 36=final norm) and its
exact split (train / dev = head-held-out buildings / test = val trajectories held out of LoRA tuning).
Side label: net heading of the expert's first 4 actions; big-turn states are |net| >= 45 deg, label L vs R.
Step 1: logistic probe for side on big-turn states, per layer (standardised features, L2 picked on dev).
Step 2: retrain the waypoint head (same recipe as ~/lora_waypoint_20261104/train.py) on L22, and on L22+L26+final;
compare side accuracy (min-risk and argmax decodings) and pose error with the original head on the same states."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WP = Path.home() / "lora_waypoint_20261104"
sys.path.insert(0, str(WP))
import train as T  # noqa: E402  (split logic, reference integrate / pose_metrics)
from heads import ChunkHead  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "layer_probe_results.json"
LAYERS = (18, 22, 26, 36)
integrate, vector, pose_metrics = T.reference_functions()
C = torch.load(WP / "anchors.pt", weights_only=True).float()
D = torch.cdist(C, C).cuda()

# anchor -> net heading of its first 4 executed actions, same conversion as the eval server / test0_analyze.py
src = open(T.PROJECT / "deploy/head_eval_server.py").read()
ns = {"math": math}
exec(src[src.index("PRIMITIVES ="):src.index("def step(")], ns)


def anchor_turn4(k):
    r = C[k].numpy()
    plan = ns["poses_to_actions"](np.concatenate([r[:16].reshape(8, 2), (r[16:24] / .5)[:, None]], 1).tolist(),
                                  ((r[24:32] / 2) > .5).tolist())[:4]
    return None if 0 in plan else 15 * sum({2: 1, 3: -1}.get(a, 0) for a in plan)


ATURN = np.array([t if t is not None else 0 for t in map(anchor_turn4, range(len(C)))])


def load():
    held = T.old_validation_scenes()
    rows = {"train": [], "dev": [], "test": []}
    for split in ("train", "val"):
        for r in json.loads((T.S2 / f"samples_{split}.json").read_text()):
            p = T.CACHE / f'{r["eid"]}.npz'
            if not p.exists():
                continue
            z = np.load(p)
            e = np.load(T.EXPERT / r["scene"] / f'{r["eid"]}.npz')
            t = z["t"].astype(int)
            prev = np.where(t == 0, 4, e["executed"][np.maximum(t - 1, 0)]).astype(np.int64)
            key = ("dev" if r["scene"] in held else "train") if split == "train" else "test"
            rows[key].append((z["h_lora"].astype(np.float32), prev, z["act"].astype(np.int64)))
    return {k: dict(zip(("h", "prev", "act"), map(np.concatenate, zip(*v)))) for k, v in rows.items()}


def turn4(act):
    a = act[:, :4]
    stop = (a <= 0).cumsum(1) > 0
    return 15 * (((a == 2) & ~stop).sum(1) - ((a == 3) & ~stop).sum(1))


def probe(Xtr, ytr, Xdv, ydv, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    f = lambda X: torch.from_numpy((X - mu) / sd).cuda()
    Xtr_, Xdv_, Xte_ = f(Xtr), f(Xdv), f(Xte)
    ytr_ = torch.from_numpy(ytr).float().cuda()
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        w = torch.zeros(Xtr.shape[1], device="cuda", requires_grad=True)
        b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xtr_ @ w + b, ytr_) + wd * w.square().sum() / 2
            loss.backward()
            return loss
        opt.step(closure)
        acc = lambda X, y: float(((X @ w + b > 0).cpu().numpy() == y).mean())
        cand = (acc(Xdv_, ydv), wd, acc(Xtr_, ytr), acc(Xte_, yte))
        best = max(best, cand) if best else cand
    return {"dev": best[0], "test": best[3], "train": best[2], "l2": best[1]}


def train_head(srcs, prev, act, seed=0):
    poses, stopped = integrate(act)
    q = torch.softmax(-torch.cdist(torch.from_numpy(vector(poses, stopped)).float(), C).square() / (2 * T.SIGMA2), 1).cuda()
    hs = [torch.from_numpy(s).cuda() for s in srcs]
    pv = torch.from_numpy(prev).cuda()
    torch.manual_seed(seed)
    m = ChunkHead(n_sources=len(srcs), queries=1, out=256).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=T.LR, weight_decay=.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, T.LR, total_steps=T.EPOCHS * math.ceil(len(pv) / T.BATCH))
    for _ in range(T.EPOCHS):
        m.train()
        for idx in torch.randperm(len(pv), device="cuda").split(T.BATCH):
            loss = -(q[idx] * F.log_softmax(m([h[idx] for h in hs], pv[idx])[:, 0], -1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
    return m.eval()


@torch.no_grad()
def evaluate(m, srcs, d):
    p = torch.cat([m([torch.from_numpy(s[b]).cuda() for s in srcs], torch.from_numpy(d["prev"][b]).cuda())[:, 0].softmax(-1)
                   for b in np.array_split(np.arange(len(d["prev"])), 8)])
    ids = {"minrisk": (p @ D).argmin(1).cpu().numpy(), "argmax": p.argmax(1).cpu().numpy()}
    tgt, tst = integrate(d["act"])
    t4 = turn4(d["act"])
    big = np.abs(t4) >= 45
    P = C.numpy()
    apose = np.concatenate([P[:, :16].reshape(-1, 8, 2), (P[:, 16:24] / .5)[..., None]], -1)
    astop = (P[:, 24:32] / 2) > .5
    res = {}
    for k, v in ids.items():
        side_ok = np.sign(ATURN[v]) == np.sign(t4)
        mass_side = np.where(t4[:, None] > 0, (p.cpu().numpy() * (ATURN > 0)).sum(1, keepdims=True),
                             (p.cpu().numpy() * (ATURN < 0)).sum(1, keepdims=True))[:, 0]
        pm = pose_metrics(apose[v], astop[v], tgt, tst)
        res[k] = {"big_turn_side_acc": float(side_ok[big].mean()), "big_turn_n": int(big.sum()),
                  "mass_on_true_side_big": float(mass_side[big].mean()),
                  "pose": {kk: (round(float(vv), 4) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in pm.items()}}
    return res


def main():
    d = load()
    print({k: len(v["prev"]) for k, v in d.items()}, flush=True)
    out = {"n": {k: len(v["prev"]) for k, v in d.items()}, "probe": {}, "heads": {}}
    for k, v in d.items():
        t4 = turn4(v["act"])
        out["n"][k + "_bigturn"] = int((np.abs(t4) >= 45).sum())
        out["n"][k + "_bigturn_left_frac"] = float((t4[np.abs(t4) >= 45] > 0).mean())
    print(out["n"], flush=True)

    # step 1: per-layer side probe on big-turn states (with and without the previous action as an extra input)
    sel = {k: np.abs(turn4(v["act"])) >= 45 for k, v in d.items()}
    y = {k: (turn4(v["act"])[sel[k]] > 0).astype(np.int64) for k, v in d.items()}
    for li, L in enumerate(LAYERS):
        X = {k: v["h"][sel[k], li] for k, v in d.items()}
        out["probe"][f"L{L}"] = probe(X["train"], y["train"], X["dev"], y["dev"], X["test"], y["test"])
        print("probe", L, out["probe"][f"L{L}"], flush=True)
    out["probe"]["majority_test"] = float(max(y["test"].mean(), 1 - y["test"].mean()))

    # step 2: retrain the head on different layers; also re-evaluate the shipped head
    shipped = ChunkHead(queries=1, out=256).cuda()
    shipped.load_state_dict(torch.load(WP / "waypoint_lora_v2.pt", weights_only=True)["model"])
    shipped.eval()
    configs = {"final (shipped ckpt)": None, "final (retrained)": [3], "L22": [1], "L22+L26+final": [1, 2, 3],
               "L18+L22+L26+final": [0, 1, 2, 3]}
    for name, li in configs.items():
        m = shipped if li is None else train_head([d["train"]["h"][:, i] for i in li], d["train"]["prev"], d["train"]["act"])
        li = li or [3]
        out["heads"][name] = {k: evaluate(m, [d[k]["h"][:, i] for i in li], d[k]) for k in ("dev", "test")}
        r = out["heads"][name]
        print(name, {k: (r[k]["minrisk"]["big_turn_side_acc"], r[k]["argmax"]["big_turn_side_acc"],
                         r[k]["minrisk"]["mass_on_true_side_big"]) for k in r}, flush=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
