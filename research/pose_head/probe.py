"""Which hidden state predicts the future pose better: frozen FutureNav or lora_v1?

Features from extract.py (last prompt token, layers 18/22/26/final). Target: the
expert's next 8 local poses (x fwd, y left, dyaw), flattened to 24 numbers.
Predictors, fit on train steps, scored on the 1,000 val steps:
  ridge  closed form, alpha picked on a 10% train holdout
  mlp    2 x 512, MSE, early stop on the same holdout
Feature sets: base_L<d>, lora_L<d>, base_cat / lora_cat (all four layers),
vggt (pooled current-frame VGGT only: what vision alone gives).
References: straight (8 x forward), lora_text (lora_v1's own greedy 8-action
answer, from s2_lora/offline_eval), base_text (FutureNav's one action, then nothing).
Metrics: ADE / FDE (m), final heading error (deg); all, decision, bend (|net turn| >= 3).

    python probe.py
Writes outputs/pose_head/probe.json
"""
import glob
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FEATS = os.path.join(ROOT, "outputs/pose_head/feats")
S2 = os.path.join(ROOT, "outputs/s2_lora")
LAYERS = (18, 22, 26, 36)


def load(split):
    with open(os.path.join(S2, "samples_%s.json" % split)) as f:
        rows = json.load(f)
    cols = {k: [] for k in ("h_base", "h_lora", "vg", "pose", "act", "t")}
    eids, dec = [], []
    for r in rows:
        p = os.path.join(FEATS, r["eid"] + ".npz")
        if not os.path.exists(p):   # train: only the episodes extract.py was run on
            continue
        z = np.load(p)
        for k in cols:
            cols[k].append(z[k])
        eids += [r["eid"]] * len(z["t"])
        dec += r["decision"]
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["eid"], out["decision"] = np.array(eids), np.array(dec, bool)
    return out


def net(acts):
    n = 0
    for a in acts:
        if a == 0:
            break
        n += int(a == 2) - int(a == 3)
    return n


def integrate(acts):
    """Local poses after each of up to 8 actions (0.25 m, 15 deg); holds after STOP."""
    x = y = h = 0.0
    out = []
    for i in range(8):
        a = acts[i] if i < len(acts) else 0
        if a == 1:
            x, y = x + 0.25 * math.cos(h), y + 0.25 * math.sin(h)
        elif a in (2, 3):
            h += math.radians(15) * (1 if a == 2 else -1)
        if a == 0:
            acts = []
        out.append((x, y, h))
    return np.array(out)


def metrics(pred, gt):
    d = np.linalg.norm(pred[:, :, :2] - gt[:, :, :2], axis=-1)
    dh = np.abs((pred[:, -1, 2] - gt[:, -1, 2] + np.pi) % (2 * np.pi) - np.pi)
    return {"ade": d.mean(1), "fde": d[:, -1], "head": np.degrees(dh)}


def summarise(m, masks):
    return {name: {k: round(float(v[mk].mean()), 4) for k, v in m.items()} | {"n": int(mk.sum())}
            for name, mk in masks.items()}


def ridge(Xtr, Ytr, Xho, Yho, Xva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float64, device="cuda")
    A, B, H = f(Xtr), torch.tensor(Ytr, dtype=torch.float64, device="cuda"), f(Xho)
    ym = B.mean(0)
    G, R = A.T @ A, A.T @ (B - ym)
    best = None
    for alpha in (1e1, 1e2, 1e3, 1e4, 1e5):
        W = torch.linalg.solve(G + alpha * torch.eye(G.shape[0], dtype=G.dtype, device="cuda"), R)
        err = float(((H @ W + ym - torch.tensor(Yho, device="cuda")) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, alpha, W)
    return (f(Xva) @ best[2] + ym).cpu().numpy(), best[1]


def mlp(Xtr, Ytr, Xho, Yho, Xva, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
    t = lambda X: torch.tensor(X, dtype=torch.float32, device="cuda")
    A, B = t((Xtr - mu) / sd), t((Ytr - ymu) / ysd)
    H, HY = t((Xho - mu) / sd), t((Yho - ymu) / ysd)
    net_ = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(A.shape[1], 512), torch.nn.GELU(),
                               torch.nn.Dropout(0.1), torch.nn.Linear(512, 512), torch.nn.GELU(),
                               torch.nn.Linear(512, B.shape[1])).cuda()
    opt = torch.optim.AdamW(net_.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, bad = 1e9, None, 0
    for ep in range(200):
        net_.train()
        for idx in torch.randperm(len(A), device="cuda").split(256):
            loss = ((net_(A[idx]) - B[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net_.eval()
        with torch.no_grad():
            err = float(((net_(H) - HY) ** 2).mean())
        if err < best - 1e-4:
            best, state, bad = err, {k: v.clone() for k, v in net_.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    net_.load_state_dict(state)
    with torch.no_grad():
        return net_(t((Xva - mu) / sd)).cpu().numpy() * ysd + ymu


def main():
    tr, va = load("train"), load("val")
    print("train %d steps, val %d steps" % (len(tr["t"]), len(va["t"])), flush=True)
    # holdout by episode
    ueid = np.unique(tr["eid"])
    ho_e = set(np.random.default_rng(0).choice(ueid, size=len(ueid) // 10, replace=False))
    ho = np.array([e in ho_e for e in tr["eid"]])
    gt = va["pose"].astype(np.float64)
    Y = tr["pose"].reshape(len(tr["pose"]), -1).astype(np.float64)
    bend = np.array([abs(net(a)) >= 3 for a in va["act"]])
    masks = {"all": np.ones(len(gt), bool), "decision": va["decision"], "bend": bend, "cruise": ~va["decision"]}
    res = {}

    # references
    res["straight"] = summarise(metrics(np.stack([integrate([1] * 8)] * len(gt)), gt), masks)
    rows = json.load(open(os.path.join(S2, "lora_v1", "offline_eval_rows.json")))
    txt = {(r["eid"], r["t"]): r for r in rows}
    keys = list(zip(va["eid"], va["t"].tolist()))
    if all(k in txt for k in keys):
        res["lora_text"] = summarise(metrics(np.stack([integrate(txt[k]["lora"]["pred"]) for k in keys]), gt), masks)
        res["base_text_1step"] = summarise(metrics(np.stack([integrate(txt[k]["base"]["pred"][:1]) for k in keys]), gt), masks)

    sets = {}
    for arm in ("base", "lora"):
        for i, d in enumerate(LAYERS):
            sets["%s_L%d" % (arm, d)] = (tr["h_" + arm][:, i], va["h_" + arm][:, i])
        sets["%s_cat" % arm] = (tr["h_" + arm].reshape(len(Y), -1), va["h_" + arm].reshape(len(gt), -1))
    sets["vggt"] = (tr["vg"].mean(1), va["vg"].mean(1))
    sets["vggt_grid"] = (tr["vg"].reshape(len(Y), 24, 2560)[:, :, ::8].reshape(len(Y), -1),
                         va["vg"].reshape(len(gt), 24, 2560)[:, :, ::8].reshape(len(gt), -1))
    per_step = {}
    for name, (Xtr, Xva) in sets.items():
        Xtr, Xva = Xtr.astype(np.float32), Xva.astype(np.float32)
        p, alpha = ridge(Xtr[~ho], Y[~ho], Xtr[ho], Y[ho], Xva)
        m = metrics(p.reshape(gt.shape), gt)
        res["ridge_" + name] = summarise(m, masks) | {"alpha": alpha}
        per_step["ridge_" + name] = m["ade"]
        p = mlp(Xtr[~ho], Y[~ho], Xtr[ho], Y[ho], Xva)
        m = metrics(p.reshape(gt.shape), gt)
        res["mlp_" + name] = summarise(m, masks)
        per_step["mlp_" + name] = m["ade"]
        print("%-16s ridge ADE %.3f  mlp ADE %.3f  (bend mlp ADE %.3f)" % (
            name, res["ridge_" + name]["all"]["ade"], res["mlp_" + name]["all"]["ade"],
            res["mlp_" + name]["bend"]["ade"]), flush=True)
    # paired comparison lora vs base, per layer and predictor
    from scipy.stats import wilcoxon
    res["paired"] = {}
    for pr in ("ridge", "mlp"):
        for suf in ["L%d" % d for d in LAYERS] + ["cat"]:
            a, b = per_step["%s_lora_%s" % (pr, suf)], per_step["%s_base_%s" % (pr, suf)]
            res["paired"]["%s_%s" % (pr, suf)] = {"lora_minus_base_ade": round(float((a - b).mean()), 4),
                                                  "lora_better_frac": round(float((a < b).mean()), 3),
                                                  "p": float(wilcoxon(a, b).pvalue)}
    with open(os.path.join(ROOT, "outputs/pose_head/probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["paired"], indent=1))


if __name__ == "__main__":
    main()
