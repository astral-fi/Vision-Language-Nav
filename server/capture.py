"""Read-only capture of FutureNav internals, one record per executed step.

Used by experiments/capture500 through deploy/server.py. Everything here is a
forward hook or a computation on tensors the model already produced; nothing
writes back into the model, so the executed policy is unchanged and every run
must reproduce base500 action for action (checked by capture500/check_capture.py).

Per step, on the prefill of the EXECUTED call only (probe arms are never captured):

  last        [38, 2560]  last prompt token's residual stream:
                          0 = input to layer 1, k = output of layer k (1..36),
                          37 = final RMSNorm output (the vector lm_head reads)
  pooled      [3, S, 2560] mean hidden state over each prompt segment at the
                          outputs of layers POOL_LAYERS (12, 24, 36)
  seg_count   [S]         tokens in each segment (0 = segment absent this step)
  attn        [36, 32, S] attention mass the last token puts on each segment,
                          per layer and head (recomputed from q_last and K; the
                          run itself stays on sdpa)
  vggt_grid   [21, 2048]  VGGT aggregator features of the current frame:
                          row 0 = mean over patches, rows 1..20 = a 4x5 grid
  vggt_merged [2560]      mean of the projected geometry tokens (merger output)
  first_logits[3]         raw logits of MOVE, TURN, STOP at the first decode position
  first_lse               logsumexp over the full vocabulary at that position
  top20_ids / top20_logits  first position
  gen2_logits [3], gen2_lse   second position as generated (after the greedy
                          first token): _FORWARD, _LEFT, _RIGHT
  turn_logits [3], turn_lse   second position with TURN forced, every step, so
                          p(LEFT|TURN) exists even when the model walked
  gen_tokens  [4]         first generated token ids (-1 padded)
  attn_check  [36]        relative error of the attention reconstruction on
                          validation steps (NaN otherwise); see _attn_hook
  turn_check              |forced-TURN logits - generated logits| when the
                          model itself chose TURN (NaN otherwise)
  n_images, seq_len, cache_frames, hist_idx[9]

Segments (S = 14 + K):
  0 sys   1..9 image slots, right-aligned (slot 9 = current frame)
  10 pre ("These images are ... Your task is to")   11 tail (" You should take ...")
  12 gen (assistant header; holds the last token)    13 other (vision start/end tokens)
  14.. instruction chunks, split at , . ; ! ? and before " and " / " then "
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

POOL_LAYERS = (12, 24, 36)
LITE_DEPTHS = (22, 37)       # profile "lite" (experiments/failure_detection/x4_derail): last-token state at the
                             # X2 peak and the final norm, plus every logit; nothing else
FIXED = ["sys"] + ["img%d" % i for i in range(1, 10)] + ["pre", "tail", "gen", "other"]
N_FIXED = len(FIXED)                     # 14
TOK = {"MOVE": 29116, "TURN": 16313, "STOP": 50669,
       "_FORWARD": 64005, "_LEFT": 19013, "_RIGHT": 21199}
FIRST = ("MOVE", "TURN", "STOP")
SECOND = ("_FORWARD", "_LEFT", "_RIGHT")
TASK_PREFIX = "Your task is to "
_CHUNK_END = re.compile(r"[,.;!?]+")
_CHUNK_START = re.compile(r"\s(?=(?:and|then)\b)", re.I)


def split_instruction(text: str):
    """Fine clause spans (start, end) covering the whole instruction. Any coarser
    grouping on these boundaries can be rebuilt exactly from the chunk means
    and token counts, so fine is the safe choice for storage."""
    cuts = {0, len(text)}
    for m in _CHUNK_END.finditer(text):
        cuts.add(m.end())
    for m in _CHUNK_START.finditer(text):
        cuts.add(m.start() + 1)
    cuts = sorted(cuts)
    spans = [(a, b) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]
    if not spans:
        return [(0, len(text))]
    # glue whitespace-only gaps onto the previous span so every char is covered
    out = []
    for a, b in spans:
        if out and a > out[-1][1]:
            out[-1] = (out[-1][0], a)
        out.append((a, b))
    out[0] = (0, out[0][1])
    out[-1] = (out[-1][0], len(text))
    # a bare connective ("and", "then", "and then") joins the clause it introduces
    merged = []
    for a, b in out:
        if merged and text[merged[-1][0]:merged[-1][1]].strip().lower() in ("and", "then", "and then"):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


class Capture:
    def __init__(self, policy):
        self.policy = policy
        m = policy.model
        self.model = m
        self.lm = m.model.language_model
        self.layers = list(self.lm.layers)
        self.n_layers = len(self.layers)
        cfg = m.config
        self.image_id = cfg.image_token_id
        self.vstart = cfg.vision_start_token_id
        self.vend = cfg.vision_end_token_id
        self.active = False
        self.validate = False
        self.lite = False
        self.step = None           # the record being filled
        self.buf = []              # records of the current episode
        self.episode_meta = None
        self._seg = None
        self._hooks = []
        self._install()

    # ------------------------------------------------------------ hooks ----
    def _install(self):
        self._hooks.append(self.layers[0].register_forward_pre_hook(self._pre0, with_kwargs=True))
        for k, layer in enumerate(self.layers):
            self._hooks.append(layer.register_forward_hook(self._make_layer_hook(k + 1)))
            self._hooks.append(layer.self_attn.register_forward_hook(
                self._make_attn_hook(k), with_kwargs=True))
        self._hooks.append(self.lm.register_forward_hook(self._final))
        self._hooks.append(self.model.merger_vggt.register_forward_hook(self._vggt))

    def _prefill(self, h):
        return self.active and self.step is not None and h is not None and h.dim() == 3 and h.shape[1] > 1

    def _pre0(self, module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if self._prefill(h):
            self.step["last"][0] = h[0, -1].float().cpu()

    def _make_layer_hook(self, k):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if not self._prefill(h):
                return
            self.step["last"][k] = h[0, -1].float().cpu()
            if k in POOL_LAYERS and not self.lite:
                seg, cnt = self._seg["ids"], self._seg["count"]
                S = cnt.shape[0]
                acc = torch.zeros(S, h.shape[-1], device=h.device, dtype=torch.float32)
                acc.index_add_(0, seg, h[0].float())
                mean = acc / cnt.clamp(min=1).unsqueeze(1).float()
                mean[cnt == 0] = float("nan")
                self.step["pooled"][POOL_LAYERS.index(k)] = mean.cpu()
        return hook

    def _make_attn_hook(self, idx):
        def hook(module, args, kwargs, output):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if self.lite or not self._prefill(h):
                return
            cos, sin = kwargs["position_embeddings"]
            L, hd = h.shape[1], module.head_dim
            q = module.q_norm(module.q_proj(h[:, -1:]).view(1, 1, -1, hd)).transpose(1, 2)
            k = module.k_norm(module.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
            cq, sq = cos[:, -1:].unsqueeze(1), sin[:, -1:].unsqueeze(1)
            ck, sk = cos.unsqueeze(1), sin.unsqueeze(1)
            q = q * cq + _rotate_half(q) * sq
            k = k * ck + _rotate_half(k) * sk
            rep = module.num_key_value_groups
            kf = k.float().repeat_interleave(rep, dim=1)                      # [1,32,L,hd]
            p = torch.softmax((q.float() @ kf.transpose(-1, -2)) * module.scaling, dim=-1)[0, :, 0]   # [32,L]
            S = self._seg["count"].shape[0]
            mass = torch.zeros(p.shape[0], S, device=p.device, dtype=torch.float32)
            mass.index_add_(1, self._seg["ids"], p)
            self.step["attn"][idx] = mass.cpu()
            if self.validate:
                # Rebuild the layer's own attention output for the last token from
                # these weights. A match proves the weights are the ones sdpa used.
                v = module.v_proj(h).view(1, L, -1, hd).transpose(1, 2).float().repeat_interleave(rep, dim=1)
                o = (p.unsqueeze(1) @ v[0]).reshape(1, 1, -1)                 # [1,1,32*hd]
                o = module.o_proj(o.to(h.dtype)).float()[0, 0]
                ref = (output[0] if isinstance(output, (tuple, list)) else output)[0, -1].float()
                self.step["attn_check"][idx] = float((o - ref).norm() / ref.norm().clamp(min=1e-6))
        return hook

    def _final(self, module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else getattr(output, "last_hidden_state", None)
        if self._prefill(h):
            self.step["last"][self.n_layers + 1] = h[0, -1].float().cpu()

    def _vggt(self, module, args, output):
        if self.lite or not (self.active and self.step is not None):
            return
        f = args[0]                                   # [n, h2, w2, 2, 2, C]
        if f.dim() != 6:
            return
        g = f.float().mean(dim=(3, 4))[0]             # [h2, w2, C]
        grid = F.adaptive_avg_pool2d(g.permute(2, 0, 1).unsqueeze(0), (4, 5))[0]   # [C,4,5]
        rows = torch.cat([g.reshape(-1, g.shape[-1]).mean(0, keepdim=True),
                          grid.reshape(grid.shape[0], -1).T], 0)                   # [21,C]
        self.step["vggt_grid"] = rows.cpu()
        self.step["vggt_merged"] = output.float().mean(0).cpu()

    # ---------------------------------------------------------- per step ----
    def begin(self, input_ids, text: str, instruction: str, tokenizer, hist_idx,
              validate: bool, profile: str = "full"):
        """Build this step's segment map and arm the hooks for the prefill."""
        if isinstance(text, (list, tuple)):          # apply_chat_template returns a batch
            text = text[0]
        ids = input_ids[0].tolist()
        L = len(ids)
        seg = [13] * L                                            # 'other' by default
        is_img = [t == self.image_id for t in ids]
        runs, i = [], 0
        while i < L:
            if is_img[i]:
                j = i
                while j + 1 < L and is_img[j + 1]:
                    j += 1
                runs.append((i, j + 1))
                i = j + 1
            else:
                i += 1
        n_img = len(runs)
        for r, (a, b) in enumerate(runs):
            for p in range(a, b):
                seg[p] = 1 + (9 - n_img) + r                      # right-aligned
        first_vs = ids.index(self.vstart) if self.vstart in ids else 0
        for p in range(first_vs):
            seg[p] = 0
        last_ve = L - 1 - ids[::-1].index(self.vend) if self.vend in ids else first_vs
        spans = split_instruction(instruction)
        K = len(spans)
        ok, err = True, None
        tail_txt = text[text.rfind("<|vision_end|>") + len("<|vision_end|>"):]
        enc = tokenizer(tail_txt, add_special_tokens=False, return_offsets_mapping=True)
        if enc["input_ids"] != ids[last_ve + 1:]:
            ok, err = False, "suffix tokenisation mismatch"
            for p in range(last_ve + 1, L):
                seg[p] = 10
        else:
            i0 = tail_txt.find(TASK_PREFIX)
            if i0 < 0 or tail_txt.find(instruction, i0) != i0 + len(TASK_PREFIX):
                ok, err = False, "instruction not found in prompt"
            i0 += len(TASK_PREFIX)
            i1 = i0 + len(instruction)
            gen0 = tail_txt.rfind("<|im_start|>assistant")
            for n, (a, b) in enumerate(enc["offset_mapping"]):
                p = last_ve + 1 + n
                c = max(a, b - 1)                                  # a token's last char decides
                if gen0 >= 0 and a >= gen0:
                    seg[p] = 12
                elif c < i0:
                    seg[p] = 10
                elif c >= i1:
                    seg[p] = 11
                else:
                    rel = c - i0
                    seg[p] = N_FIXED + next(k for k, (s, e) in enumerate(spans) if s <= rel < e or k == K - 1)
        dev = input_ids.device
        ids_t = torch.tensor(seg, device=dev, dtype=torch.long)
        count = torch.bincount(ids_t, minlength=N_FIXED + K)
        self._seg = {"ids": ids_t, "count": count}
        S = N_FIXED + K
        nan = float("nan")
        self.lite = profile == "lite"
        hid = self.model.config.text_config.hidden_size
        self.step = {
            "last": torch.full((self.n_layers + 2, hid), nan),
            "pooled": None if self.lite else torch.full((len(POOL_LAYERS), S, hid), nan),
            "attn": None if self.lite else torch.full(
                (self.n_layers, self.model.config.text_config.num_attention_heads, S), nan),
            "attn_check": np.full(self.n_layers, np.nan, np.float32),
            "seg_count": count.cpu().numpy().astype(np.int32),
            "vggt_grid": None, "vggt_merged": None,
            "n_images": n_img, "seq_len": L,
            "hist_idx": list(hist_idx) + [-1] * (9 - len(hist_idx)),
            "layout_ok": ok, "layout_err": err,
        }
        if self.episode_meta is None:
            self.episode_meta = {"instruction": instruction,
                                 "chunks": [instruction[a:b] for a, b in spans],
                                 "chunk_spans": spans}
        self.validate = bool(validate)
        self.active = True

    def end(self, gen_out, forced_turn_logits, cache_frames):
        """Disarm, then add the logits and store the record."""
        self.active = False
        rec, self.step = self.step, None
        lg = gen_out.logits
        z0 = lg[0][0].float()
        rec["first_logits"] = np.array([float(z0[TOK[t]]) for t in FIRST], np.float32)
        rec["first_lse"] = float(torch.logsumexp(z0, 0))
        top = torch.topk(z0, 20)
        rec["top20_ids"] = top.indices.cpu().numpy().astype(np.int32)
        rec["top20_logits"] = top.values.cpu().numpy().astype(np.float32)
        seq = gen_out.sequences[0, -len(lg):].tolist()
        rec["gen_tokens"] = np.array((seq + [-1] * 4)[:4], np.int32)
        if len(lg) > 1:
            z1 = lg[1][0].float()
            rec["gen2_logits"] = np.array([float(z1[TOK[t]]) for t in SECOND], np.float32)
            rec["gen2_lse"] = float(torch.logsumexp(z1, 0))
        else:
            rec["gen2_logits"] = np.full(3, np.nan, np.float32)
            rec["gen2_lse"] = float("nan")
        zt = forced_turn_logits.float()
        rec["turn_logits"] = np.array([float(zt[TOK[t]]) for t in SECOND], np.float32)
        rec["turn_lse"] = float(torch.logsumexp(zt, 0))
        rec["turn_check"] = (float(np.abs(rec["turn_logits"] - rec["gen2_logits"]).max())
                             if seq and seq[0] == TOK["TURN"] and len(lg) > 1 else float("nan"))
        rec["cache_frames"] = int(cache_frames)
        self.buf.append(rec)
        return {"layout_ok": rec["layout_ok"],
                "attn_check_max": (float(np.nanmax(rec["attn_check"]))
                                   if np.isfinite(rec["attn_check"]).any() else None),
                "turn_check": None if math.isnan(rec["turn_check"]) else rec["turn_check"]}

    def _flush_lite(self, path, episode_id, t0):
        T = len(self.buf)
        arrays = {"last": np.stack([r["last"][list(LITE_DEPTHS)].numpy().astype(np.float16) for r in self.buf])}
        arrays.update(_logit_arrays(self.buf))
        meta = {"episode_id": str(episode_id), "steps": T, "profile": "lite",
                "last_depths": list(LITE_DEPTHS), "tokens": TOK,
                "instruction": (self.episode_meta or {}).get("instruction")}
        arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        os.makedirs(path, exist_ok=True)
        final = os.path.join(path, "%s.npz" % episode_id)
        tmp = final + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, final)
        size = os.path.getsize(final)
        self.buf = []
        self.episode_meta = None
        return {"path": final, "steps": T, "bytes": size, "bytes_per_step": size // T,
                "write_s": round(time.time() - t0, 2)}

    def abort_step(self):
        self.active = False
        self.step = None

    def reset(self):
        self.abort_step()
        self.buf = []
        self.episode_meta = None

    # ------------------------------------------------------------- flush ----
    def flush(self, path: str, episode_id: str) -> dict:
        """Write the episode to <path>/<episode_id>.npz atomically; clear the buffer."""
        if not self.buf:
            raise RuntimeError("no captured steps to flush")
        t0 = time.time()
        T = len(self.buf)
        if self.buf[0]["pooled"] is None:
            return self._flush_lite(path, episode_id, t0)
        S = max(r["pooled"].shape[1] for r in self.buf)
        f16 = lambda x: x.numpy().astype(np.float16) if torch.is_tensor(x) else np.asarray(x, np.float16)
        pooled = np.full((T, len(POOL_LAYERS), S, self.buf[0]["pooled"].shape[2]), np.nan, np.float16)
        attn = np.full((T, self.n_layers, self.buf[0]["attn"].shape[1], S), np.nan, np.float16)
        segc = np.zeros((T, S), np.int32)
        for t, r in enumerate(self.buf):
            s = r["pooled"].shape[1]
            pooled[t, :, :s] = f16(r["pooled"])
            attn[t, :, :, :s] = f16(r["attn"])
            segc[t, :s] = r["seg_count"]
        vg = [r["vggt_grid"] for r in self.buf]
        vm = [r["vggt_merged"] for r in self.buf]
        arrays = {
            "last": np.stack([f16(r["last"]) for r in self.buf]),
            "pooled": pooled, "attn": attn, "seg_count": segc,
            "vggt_grid": np.stack([f16(x) if x is not None else np.full((21, 2048), np.nan, np.float16) for x in vg]),
            "vggt_merged": np.stack([f16(x) if x is not None else np.full(2560, np.nan, np.float16) for x in vm]),
            "first_logits": np.stack([r["first_logits"] for r in self.buf]),
            "first_lse": np.array([r["first_lse"] for r in self.buf], np.float32),
            "top20_ids": np.stack([r["top20_ids"] for r in self.buf]),
            "top20_logits": np.stack([r["top20_logits"] for r in self.buf]),
            "gen_tokens": np.stack([r["gen_tokens"] for r in self.buf]),
            "gen2_logits": np.stack([r["gen2_logits"] for r in self.buf]),
            "gen2_lse": np.array([r["gen2_lse"] for r in self.buf], np.float32),
            "turn_logits": np.stack([r["turn_logits"] for r in self.buf]),
            "turn_lse": np.array([r["turn_lse"] for r in self.buf], np.float32),
            "attn_check": np.stack([r["attn_check"] for r in self.buf]),
            "turn_check": np.array([r["turn_check"] for r in self.buf], np.float32),
            "n_images": np.array([r["n_images"] for r in self.buf], np.int16),
            "seq_len": np.array([r["seq_len"] for r in self.buf], np.int32),
            "cache_frames": np.array([r["cache_frames"] for r in self.buf], np.int16),
            "hist_idx": np.array([r["hist_idx"] for r in self.buf], np.int32),
            "layout_ok": np.array([r["layout_ok"] for r in self.buf], bool),
        }
        meta = dict(self.episode_meta or {})
        meta.update({"episode_id": str(episode_id), "steps": T,
                     "segments": FIXED + ["chunk%d" % k for k in range(S - N_FIXED)],
                     "pool_layers": list(POOL_LAYERS),
                     "last_index": "0=input to layer 1, k=output of layer k, %d=final norm" % (self.n_layers + 1),
                     "tokens": TOK, "layout_errors": sorted({r["layout_err"] for r in self.buf if r["layout_err"]})})
        arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        os.makedirs(path, exist_ok=True)
        final = os.path.join(path, "%s.npz" % episode_id)
        tmp = final + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, final)
        size = os.path.getsize(final)
        self.buf = []
        self.episode_meta = None
        return {"path": final, "steps": T, "bytes": size, "bytes_per_step": size // T,
                "write_s": round(time.time() - t0, 2)}


def _logit_arrays(buf):
    return {
        "first_logits": np.stack([r["first_logits"] for r in buf]),
        "first_lse": np.array([r["first_lse"] for r in buf], np.float32),
        "top20_ids": np.stack([r["top20_ids"] for r in buf]),
        "top20_logits": np.stack([r["top20_logits"] for r in buf]),
        "gen_tokens": np.stack([r["gen_tokens"] for r in buf]),
        "gen2_logits": np.stack([r["gen2_logits"] for r in buf]),
        "gen2_lse": np.array([r["gen2_lse"] for r in buf], np.float32),
        "turn_logits": np.stack([r["turn_logits"] for r in buf]),
        "turn_lse": np.array([r["turn_lse"] for r in buf], np.float32),
        "turn_check": np.array([r["turn_check"] for r in buf], np.float32),
        "n_images": np.array([r["n_images"] for r in buf], np.int16),
        "seq_len": np.array([r["seq_len"] for r in buf], np.int32),
        "cache_frames": np.array([r["cache_frames"] for r in buf], np.int16),
        "hist_idx": np.array([r["hist_idx"] for r in buf], np.int32),
        "layout_ok": np.array([r["layout_ok"] for r in buf], bool),
    }


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
