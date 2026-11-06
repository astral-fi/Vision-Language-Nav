"""Step 7 of PLAN.md: offline eval on the val steps, LoRA vs frozen FutureNav.

Greedy generation on every val step (same inputs and injected VGGT tensor as
training), once with the adapter and once with it disabled (= frozen FutureNav).
Metrics vs the expert's next K actions:
  first_acc     first action word matches
  chunk_exact   whole chunk (cut at STOP) matches
  end_pos_m     position error after executing the chunk (0.25 m / 15 deg steps)
  end_head_deg  heading error after the chunk
  turn_err      |net predicted turns - net expert turns| when the expert turns >= 3 net
  side_acc      same turn side when the expert turns >= 3 net
Frozen FutureNav answers one word, so its chunk is that word alone: its end-pose
numbers are for reference only; first_acc is the fair comparison.
Reported for all steps and for decision steps.

    python offline_eval.py --adapter outputs/s2_lora/lora_v1/adapter_final
Writes <adapter>/offline_eval.json and offline_eval_rows.json
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch

from common import K, NAMES, OUT, Episode, episode_path, load_policy

WORD = re.compile(r"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT")


def words(raw):
    return [NAMES.index(w) for w in WORD.findall(raw or "")]


def end_pose(acts):
    x = y = h = 0.0
    for a in acts[:K]:
        if a == 0:
            break
        if a == 1:
            x, y = x + 0.25 * math.cos(h), y + 0.25 * math.sin(h)
        else:
            h += math.radians(15) * (1 if a == 2 else -1)
    return x, y, h


def net(acts):
    acts = acts[:acts.index(0) + 1] if 0 in acts else acts
    return sum(a == 2 for a in acts) - sum(a == 3 for a in acts)


def score(pred, gt):
    px, py, ph = end_pose(pred)
    gx, gy, gh = end_pose(gt)
    dh = abs((ph - gh + math.pi) % (2 * math.pi) - math.pi)
    r = {"first": bool(pred) and pred[0] == gt[0], "exact": pred[:len(gt)] == gt,
         "pos": math.hypot(px - gx, py - gy), "head": math.degrees(dh), "unparsed": not pred}
    if abs(net(gt)) >= 3:
        r["turn_err"] = abs(net(pred) - net(gt))
        r["side"] = np.sign(net(pred)) == np.sign(net(gt))
    return r


def summarise(rows):
    def agg(rs):
        out = {"n": len(rs)}
        for k, name in (("first", "first_acc"), ("exact", "chunk_exact"), ("pos", "end_pos_m"),
                        ("head", "end_head_deg"), ("unparsed", "unparsed"), ("turn_err", "turn_err"),
                        ("side", "side_acc")):
            v = [float(r[k]) for r in rs if k in r]
            out[name] = round(float(np.mean(v)), 4) if v else None
        return out
    return {"all": agg(rows), "decision": agg([r for r in rows if r["decision"]]),
            "cruise": agg([r for r in rows if not r["decision"]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(OUT, "lora_v1/adapter_final"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from peft import PeftModel
    with open(os.path.join(OUT, "samples_val.json")) as f:
        val = [(r["scene"], r["eid"], t, d) for r in json.load(f) for t, d in zip(r["steps"], r["decision"])]
    val = val[:a.limit] if a.limit else val
    policy = load_policy()
    base = policy.model
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    tok = policy.tokenizer
    rows, t0, ep_cache = [], time.time(), {}
    for i, (scene, eid, t, dec) in enumerate(val):
        if eid not in ep_cache:
            ep_cache = {eid: (Episode(episode_path(scene, eid)),
                              dict(zip(*(lambda v: (v["t"].tolist(), v["x"]))(np.load(os.path.join(OUT, "vggt", eid + ".npz"))))))}
        ep, vg = ep_cache[eid]
        x = torch.from_numpy(vg[t]).to(base.device, torch.bfloat16)
        inputs, _, _ = policy.build_inputs(ep.shown(t), ep.meta["instruction"])
        inputs["images_vggt"] = None
        gt = words(ep.answer(t))
        row = {"eid": eid, "t": t, "decision": dec, "gt": gt}
        for arm in ("lora", "base"):
            ctx = model.disable_adapter() if arm == "base" else torch.autocast("cuda", enabled=False)
            with torch.no_grad(), ctx, base.vggt_feature_reuse(inject=x):
                out = model.generate(**inputs, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                     do_sample=False, temperature=0, top_p=None, num_beams=1, max_new_tokens=48)
            raw = policy.processor.batch_decode([out[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]
            row[arm] = {"raw": raw, "pred": words(raw), **score(words(raw), gt)}
        rows.append(row)
        if i % 50 == 0:
            print("%d/%d  %.2f s/step" % (i + 1, len(val), (time.time() - t0) / (i + 1)), flush=True)
    res = {arm: summarise([dict(r[arm], decision=r["decision"]) for r in rows]) for arm in ("lora", "base")}
    res["adapter"] = a.adapter
    print(json.dumps(res, indent=1))
    d = os.path.dirname(a.adapter.rstrip("/"))
    with open(os.path.join(d, "offline_eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(d, "offline_eval_rows.json"), "w") as f:
        json.dump(rows, f, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
