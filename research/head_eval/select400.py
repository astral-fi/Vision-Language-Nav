"""Stratified 400-episode subset of the STAIR-FREE R2R-CE val_unseen pool (1,407 of 1,839 episodes; stairs =
route height range > 0.5 m, same rule as experiments/baseline500).

Strata: scene x geodesic-distance tercile (terciles of the stair-free pool). Allocation is proportional (largest remainder). Within strata the draw
is random; we take the best of N seeded draws by the largest two-sample KS distance to the full
set over continuous covariates. Deterministic: same inputs -> same ids.
Writes ids_sf400.txt, subset_sf400.json (diagnostics)."""
import gzip, json, sys
from pathlib import Path
import numpy as np

N, TRIES, SEED = 400, 5000, 0
OUT = Path(__file__).parent
d = json.load(gzip.open(Path(__file__).parents[2] / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"))
eps_all = d["episodes"]
eps = [e for e in eps_all if np.ptp(np.array(e["reference_path"])[:, 1]) <= 0.5]
assert len(eps) == 1407, len(eps)

def feats(e):
    rp = np.array(e["reference_path"]); seg = np.linalg.norm(np.diff(rp, axis=0), axis=1)
    return dict(scene=e["scene_id"].split("/")[1], geo=e["info"]["geodesic_distance"], plen=float(seg.sum()),
                nwp=len(rp), words=len(e["instruction"]["instruction_text"].split()),
                hrange=float(rp[:, 1].max() - rp[:, 1].min()))
F = [feats(e) for e in eps]
ids = np.array([e["episode_id"] for e in eps])
geo = np.array([f["geo"] for f in F]); q = np.quantile(geo, [1/3, 2/3])
strata = np.array([f"{f['scene']}|{int(np.searchsorted(q, f['geo']))}" for f in F])
cov = {k: np.array([f[k] for f in F], float) for k in ("geo", "plen", "nwp", "words", "hrange")}

keys, inv, cnt = np.unique(strata, return_inverse=True, return_counts=True)
exact = cnt * N / len(eps); alloc = np.floor(exact).astype(int)
for i in np.argsort(-(exact - alloc))[:N - alloc.sum()]: alloc[i] += 1
alloc = np.minimum(alloc, cnt)
members = [np.nonzero(inv == k)[0] for k in range(len(keys))]

def ks(a, b):
    g = np.sort(np.concatenate([a, b])); return np.abs(np.searchsorted(np.sort(a), g, "right") / len(a) - np.searchsorted(np.sort(b), g, "right") / len(b)).max()

rng = np.random.default_rng(SEED); best = None
for t in range(TRIES):
    pick = np.concatenate([rng.choice(m, a, replace=False) for m, a in zip(members, alloc) if a])
    s = max(ks(cov[k][pick], cov[k]) for k in cov)
    if best is None or s < best[0]: best = (s, pick)
score, pick = best; pick = np.sort(pick)
assert len(pick) == N and len(set(ids[pick])) == N

diag = {"n": N, "pool": len(eps), "pool_def": "stair-free val_unseen", "tries": TRIES, "seed": SEED, "max_ks": round(float(score), 4), "covariates": {}}
for k in cov:
    diag["covariates"][k] = dict(full_mean=round(float(cov[k].mean()), 3), sub_mean=round(float(cov[k][pick].mean()), 3),
                                 full_median=round(float(np.median(cov[k])), 3), sub_median=round(float(np.median(cov[k][pick])), 3),
                                 ks=round(float(ks(cov[k][pick], cov[k])), 4))
scenes = sorted(set(f["scene"] for f in F))
diag["scenes"] = {s: dict(full=int(sum(f["scene"] == s for f in F)), sub=int(sum(F[i]["scene"] == s for i in pick))) for s in scenes}
(OUT / "ids_sf400.txt").write_text("\n".join(str(i) for i in ids[pick]) + "\n")
(OUT / "subset_sf400.json").write_text(json.dumps(diag, indent=1))
print(json.dumps(diag, indent=1))
