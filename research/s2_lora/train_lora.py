"""Step 5-6 of PLAN.md: LoRA fine-tune FutureNav-4B to answer with the next K actions.

Inputs are built with FutureNavPolicy.build_inputs (the server's own code) and the
precomputed streaming VGGT tensor is injected through vggt_feature_reuse, so VGGT
never runs. LoRA on the language model's attention and MLP projections only.
Loss: cross-entropy on the answer tokens (+ end-of-turn token).

Writes outputs/s2_lora/<tag>/: adapter_step<k>/ (peft), adapter_final/, log.jsonl

    python train_lora.py --tag lora_v1                      # full run
    python train_lora.py --tag smoke --max-samples 64 --accum 4 --eval-every 8 --val-samples 16
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from common import OUT, Episode, episode_path, load_policy

TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def samples(split, limit=0, seed=0):
    with open(os.path.join(OUT, "samples_%s.json" % split)) as f:
        rows = json.load(f)
    vdir = os.path.join(OUT, "vggt")
    out = [(r["scene"], r["eid"], t) for r in rows if os.path.exists(os.path.join(vdir, r["eid"] + ".npz"))
           for t in r["steps"]]
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


class Source(object):
    """Episode frames + VGGT tensors, with a small cache of open episodes."""

    def __init__(self, n=64):
        self.ep, self.vg, self.n = {}, {}, n

    def get(self, scene, eid, t):
        if eid not in self.ep:
            if len(self.ep) >= self.n:
                k = next(iter(self.ep))
                self.ep.pop(k), self.vg.pop(k)
            self.ep[eid] = Episode(episode_path(scene, eid))
            v = np.load(os.path.join(OUT, "vggt", eid + ".npz"))
            self.vg[eid] = dict(zip(v["t"].tolist(), v["x"]))
        return self.ep[eid], self.vg[eid][t]


def make_batch(policy, src, scene, eid, t):
    ep, x = src.get(scene, eid, t)
    inputs, _, _ = policy.build_inputs(ep.shown(t), ep.meta["instruction"])
    answer = ep.answer(t)
    ans = policy.tokenizer(answer + policy.tokenizer.eos_token, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(inputs.input_ids.device)
    # tokens of the first action word (TURN_LEFT and TURN_RIGHT share their first token)
    n1 = len(policy.tokenizer(answer.split()[0], add_special_tokens=False).input_ids)
    ids = torch.cat([inputs.input_ids, ans], 1)
    labels = torch.full_like(ids, -100)
    labels[:, -ans.shape[1]:] = ans
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
             "pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"],
             "images_vggt": None}
    return batch, torch.from_numpy(x).to(inputs.input_ids.device, torch.bfloat16), ans.shape[1], n1


def forward(model, base, batch, x):
    with base.vggt_feature_reuse(inject=x), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**batch, use_cache=False)


@torch.no_grad()
def evaluate(model, base, policy, src, val):
    model.eval()
    tot = first = full = 0.0
    for scene, eid, t in val:
        batch, x, n, n1 = make_batch(policy, src, scene, eid, t)
        out = forward(model, base, batch, x)
        tot += float(out.loss)
        # teacher-forced argmax over the answer positions vs the expert's tokens
        L = batch["input_ids"].shape[1]
        ok = out.logits[0, L - n - 1:L - 1].argmax(-1) == batch["input_ids"][0, L - n:]
        first += float(ok[:n1].all())
        full += float(ok.all())
    model.train()
    return {"val_loss": tot / len(val), "val_first_action_acc": first / len(val), "val_chunk_exact": full / len(val)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--train-split", default="train", help="val only for the smoke test")
    ap.add_argument("--val-samples", type=int, default=200)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=125, help="optimizer steps")
    ap.add_argument("--save-every", type=int, default=250, help="optimizer steps")
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model

    out_dir = os.path.join(OUT, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    train, val = samples(a.train_split, a.max_samples), samples("val", a.val_samples, seed=1)
    print("train %d samples, val %d" % (len(train), len(val)), flush=True)

    policy = load_policy()
    base = policy.model
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                            target_modules=TARGETS, bias="none"))
    for n_, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = max(1, len(train) // a.accum)
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    src = Source()
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    t0, step, run_loss = time.time(), 0, 0.0
    for i, (scene, eid, t) in enumerate(train):
        batch, x, _, _ = make_batch(policy, src, scene, eid, t)
        loss = forward(model, base, batch, x).loss / a.accum
        loss.backward()
        run_loss += float(loss)
        if (i + 1) % a.accum:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": run_loss, "lr": sched.get_last_lr()[0],
               "s_per_sample": (time.time() - t0) / (i + 1),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        run_loss = 0.0
        if step % a.eval_every == 0 or step == total:
            try:                       # a metric bug must not kill an overnight run
                rec.update(evaluate(model, base, policy, src, val))
            except Exception as exc:
                model.train()
                rec["eval_error"] = repr(exc)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % a.save_every == 0:
            model.save_pretrained(os.path.join(out_dir, "adapter_step%d" % step))
    model.save_pretrained(os.path.join(out_dir, "adapter_final"))
    print("done: %d optimizer steps in %.1f h" % (step, (time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    main()
