"""Motion head (System 1): pose chunks from frozen-FutureNav states + VGGT tokens.

Design: experiments/data_engine/MOTION_HEAD_DESIGN.md.

Conditioning tokens, all width D:
  4  FutureNav last-token residuals (L18, L22, L26, final), one projection each
  1  FutureNav's action at its last call (one-hot)
  1  context age: steps since that call (0..MAX_AGE)
  48 x (1 + PAST)  pooled 2048-d VGGT tokens of the current and PAST previous
     frames (grid position + frame embeddings; missing frames are masked)
  1  readout token, whose output feeds the auxiliary heads
A small encoder fuses them. The denoiser (H pose tokens: x, y, sin dyaw, cos dyaw
in the current agent frame, normalised) cross-attends to the fused tokens.

mode "flow": rectified flow. x_t = (1-t) noise + t target; predict the velocity
target - noise; sample with Euler steps. mode "reg": the same denoiser with
zero input and t = 1 predicts the poses directly (the regression baseline).

Auxiliary outputs from the readout token: stop logit, remaining route distance
(m), progress fraction, off-route distance (m).
"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    mode: str = "flow"
    width: int = 256
    heads: int = 8
    enc_layers: int = 2
    dec_layers: int = 4
    horizon: int = 16
    s2_layers: int = 4
    s2_dim: int = 2560
    vis_tokens: int = 48
    vis_dim: int = 2048
    past: int = 3
    max_age: int = 7
    sample_steps: int = 8


def _layer(d, heads, decoder):
    cls = nn.TransformerDecoderLayer if decoder else nn.TransformerEncoderLayer
    return cls(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)


class MotionHead(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = self.config = config or Config()
        d = c.width
        self.s2_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(c.s2_dim), nn.Linear(c.s2_dim, d))
                                      for _ in range(c.s2_layers)])
        self.s2_layer_emb = nn.Parameter(torch.zeros(c.s2_layers, d))
        self.fn_action = nn.Linear(4, d)
        self.age = nn.Embedding(c.max_age + 1, d)
        self.vis_proj = nn.Sequential(nn.LayerNorm(c.vis_dim), nn.Linear(c.vis_dim, d))
        self.vis_pos = nn.Parameter(torch.zeros(c.vis_tokens, d))
        self.vis_frame = nn.Parameter(torch.zeros(1 + c.past, d))
        self.readout = nn.Parameter(torch.zeros(1, d))
        self.encoder = nn.ModuleList([_layer(d, c.heads, False) for _ in range(c.enc_layers)])
        self.enc_norm = nn.LayerNorm(d)
        self.pose_in = nn.Linear(4, d)
        self.pose_pos = nn.Parameter(torch.zeros(c.horizon, d))
        self.time = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.decoder = nn.ModuleList([_layer(d, c.heads, True) for _ in range(c.dec_layers)])
        self.pose_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))
        self.aux = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        for p in (self.s2_layer_emb, self.vis_pos, self.vis_frame, self.readout, self.pose_pos):
            nn.init.normal_(p, std=0.02)
        freq = torch.exp(-math.log(10000.0) * torch.arange(d // 2).float() / (d // 2))
        self.register_buffer("freq", freq, persistent=False)
        self.register_buffer("pose_mean", torch.zeros(4))
        self.register_buffer("pose_std", torch.ones(4))

    # ---- conditioning ----
    def encode(self, h, fn_action, age, vis, vis_mask):
        """h (B,4,2560)  fn_action (B,) long  age (B,) long  vis (B,1+past,48,256)
        vis_mask (B,1+past) bool, True = frame present. Returns tokens, key padding mask, readout."""
        B = h.shape[0]
        s2 = torch.stack([p(h[:, i].float()) for i, p in enumerate(self.s2_proj)], 1) + self.s2_layer_emb
        fa = self.fn_action(F.one_hot(fn_action.long(), 4).float())[:, None]
        ag = self.age(age.clamp(max=self.config.max_age).long())[:, None]
        v = self.vis_proj(vis.float()) + self.vis_pos + self.vis_frame[None, :, None]
        v = v.flatten(1, 2)
        ro = self.readout.expand(B, 1, -1)
        tok = torch.cat([ro, s2, fa, ag, v], 1)
        pad = torch.cat([torch.zeros(B, 1 + s2.shape[1] + 2, dtype=torch.bool, device=h.device),
                         (~vis_mask)[:, :, None].expand(-1, -1, self.config.vis_tokens).flatten(1)], 1)
        for layer in self.encoder:
            tok = layer(tok, src_key_padding_mask=pad)
        tok = self.enc_norm(tok)
        return tok, pad, tok[:, 0]

    def aux_heads(self, readout):
        a = self.aux(readout)
        return {"stop_logit": a[:, 0], "remaining_m": F.softplus(a[:, 1]) * 5.0,
                "progress": torch.sigmoid(a[:, 2]), "off_route_m": F.softplus(a[:, 3])}

    # ---- denoiser ----
    def _time(self, t):
        ang = t[:, None] * 1000.0 * self.freq[None]
        return self.time(torch.cat([ang.sin(), ang.cos()], 1))

    def denoise(self, x, t, tok, pad):
        q = self.pose_in(x) + self.pose_pos + self._time(t)[:, None]
        for layer in self.decoder:
            q = layer(q, tok, memory_key_padding_mask=pad)
        return self.pose_out(q)

    # ---- poses <-> normalised targets ----
    def to_target(self, pose):
        """pose (B,H,3) x, y, dyaw  ->  normalised (B,H,4) x, y, sin, cos."""
        p = torch.stack([pose[..., 0], pose[..., 1], pose[..., 2].sin(), pose[..., 2].cos()], -1)
        return (p - self.pose_mean) / self.pose_std

    def from_target(self, y):
        p = y * self.pose_std + self.pose_mean
        return torch.stack([p[..., 0], p[..., 1], torch.atan2(p[..., 2], p[..., 3])], -1)

    # ---- training loss ----
    def loss(self, batch, step_w):
        tok, pad, ro = self.encode(batch["h"], batch["fn_action"], batch["age"], batch["vis"], batch["vis_mask"])
        y = self.to_target(batch["pose"])
        B = y.shape[0]
        if self.config.mode == "flow":
            t = torch.rand(B, device=y.device)
            x0 = torch.randn_like(y)
            xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * y
            err = (self.denoise(xt, t, tok, pad) - (y - x0)) ** 2
        else:
            pred = self.denoise(torch.zeros_like(y), torch.ones(B, device=y.device), tok, pad)
            err = (pred - y).abs()
        pose_loss = (err.mean(-1) * step_w).sum(1) / step_w.sum()
        aux = self.aux_heads(ro)
        w = batch["weight"]
        losses = {
            "pose": pose_loss,
            "stop": F.binary_cross_entropy_with_logits(aux["stop_logit"], batch["stop"].float(), reduction="none"),
            "remaining": (aux["remaining_m"] - batch["remaining_m"]).abs() / 5.0,
            "progress": (aux["progress"] - batch["progress"]).abs(),
            "off_route": (aux["off_route_m"] - batch["off_route_m"]).abs(),
        }
        total = sum((v * w).sum() / w.sum() for v in losses.values())
        return total, {k: float((v * w).sum() / w.sum()) for k, v in losses.items()}

    # ---- inference ----
    @torch.no_grad()
    def sample(self, h, fn_action, age, vis, vis_mask, k=1, generator=None):
        """Returns poses (B,k,H,3) and aux. k samples per input share one encoding."""
        tok, pad, ro = self.encode(h, fn_action, age, vis, vis_mask)
        aux = self.aux_heads(ro)
        B, H = h.shape[0], self.config.horizon
        tok_k, pad_k = tok.repeat_interleave(k, 0), pad.repeat_interleave(k, 0)
        if self.config.mode == "flow":
            x = torch.randn(B * k, H, 4, device=h.device, generator=generator)
            n = self.config.sample_steps
            for i in range(n):
                t = torch.full((B * k,), i / n, device=h.device)
                x = x + self.denoise(x, t, tok_k, pad_k) / n
        else:
            x = self.denoise(torch.zeros(B * k, H, 4, device=h.device), torch.ones(B * k, device=h.device),
                             tok_k, pad_k)
        return self.from_target(x).view(B, k, H, 3), aux


def save(model, path, extra=None):
    torch.save({"config": asdict(model.config), "model": model.state_dict(), **(extra or {})}, path)


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MotionHead(Config(**ck["config"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
