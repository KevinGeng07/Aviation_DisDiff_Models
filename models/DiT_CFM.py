"""
Diffusion Transformer for conditional flow matching (CFM) — the CITRIS `TrajectoryFlowV5`
architecture (CITRIS-modern-transformer-encoding, models/flow_v5.py), ported to this repo's
interface so it can be a teacher (configs/teacher_dit_cfm_*.json), a student or a baseline.

    [observed tokens (43) | noisy future tokens (43)]  → (86, d_model)
    + learned position embedding + t_rel MLP (sinusoidal → 2·d_model → d_model)
    → n_layers × AdaLN block (GELU feed-forward 4·d_model, residual dropout),
      conditioned on the flow time t (sinusoidal(1000·t) → 4·d_model → d_model)
    → final LayerNorm → Linear(d_model → 6) on the future tokens = velocity

Observed tokens cannot attend to future tokens (block-causal mask).

Sizes used by CITRIS (same code, different width/depth):
    flow_v3   d_model 128, 5 layers, 4 heads   ~1.5 M params
    flow_v4   d_model 256, 6 layers, 8 heads    7.13 M params   (configs/teacher_dit_cfm_7M.json)
    flow_v5   d_model 384, 8 layers, 8 heads   20.73 M params   (configs/teacher_dit_cfm_20M.json)

Differences from CITRIS: forward(x_obs, x_t, t, t_rel) argument order, as every model in this
repo (theirs is forward(x_t, t, x_obs, t_rel)); `in_dim` accepted as the name of n_features.
Parameter names are identical to TrajectoryFlowV5, so a CITRIS checkpoint's "ema_model" /
"model" state dict loads directly (models/registry.py: model_weights()).
"""
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t, d_model):
    """t: (B,) float → (B, d_model)."""
    half  = d_model // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args  = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def sinusoidal_embedding_seq(x, d_model):
    """x: (B, T) float → (B, T, d_model)."""
    half  = d_model // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args  = x.unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLNBlock(nn.Module):
    """Pre-LN transformer block; the flow-time vector sets each LayerNorm's scale/shift."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_model))
        self.drop  = nn.Dropout(dropout)
        self.adaLN = nn.Linear(d_model, 4 * d_model, bias=True)
        nn.init.zeros_(self.adaLN.weight)          # conditioning starts as the identity
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, cond, attn_mask=None):
        s1, b1, s2, b2 = (c.unsqueeze(1) for c in self.adaLN(cond).chunk(4, dim=-1))
        h = self.norm1(x) * (1.0 + s1) + b1
        # need_weights=False: don't build the (B, T, T) attention-weight tensor we never use
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(attn_out)
        h = self.norm2(x) * (1.0 + s2) + b2
        return x + self.drop(self.ff(h))


class TrajectoryDiT(nn.Module):
    def __init__(self, obs_len=43, fut_len=43, in_dim=None, d_model=384, n_heads=8, n_layers=8,
                 dropout=0.1, use_pos_emb=True, use_attn_mask=True, n_features=None):
        super().__init__()
        n_features = in_dim or n_features or 6
        self.obs_len, self.fut_len = obs_len, fut_len
        self.use_pos_emb, self.use_attn_mask = use_pos_emb, use_attn_mask
        seq_len = obs_len + fut_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb    = nn.Embedding(seq_len, d_model) if use_pos_emb else None
        self.trel_mlp   = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.SiLU(),
                                        nn.Linear(2 * d_model, d_model))
        self.tflow_mlp  = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(),
                                        nn.Linear(4 * d_model, d_model))
        self.blocks     = nn.ModuleList([AdaLNBlock(d_model, n_heads, 4 * d_model, dropout)
                                         for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out        = nn.Linear(d_model, n_features)

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[:obs_len, obs_len:] = True                    # observed tokens can't see the future
        self.register_buffer("_attn_mask", mask)
        self.register_buffer("_positions", torch.arange(seq_len), persistent=False)

    def forward(self, x_obs, x_t, t, t_rel):
        """x_obs, x_t: (B, 43, F); t: (B,) flow time in [0, 1]; t_rel: (B, 86) normalized
        timestamps. Returns the predicted velocity (B, 43, F)."""
        tokens = torch.cat([self.input_proj(x_obs), self.input_proj(x_t)], dim=1)
        if self.use_pos_emb:
            tokens = tokens + self.pos_emb(self._positions)
        tokens = tokens + self.trel_mlp(sinusoidal_embedding_seq(t_rel, tokens.shape[-1]))
        cond   = self.tflow_mlp(sinusoidal_embedding(t * 1000.0, tokens.shape[-1]))
        mask   = self._attn_mask if self.use_attn_mask else None
        for block in self.blocks:
            tokens = block(tokens, cond, attn_mask=mask)
        return self.out(self.final_norm(tokens[:, self.obs_len:]))
