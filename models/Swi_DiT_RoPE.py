"""
SWI_DiT with rotary position embeddings (RoPE) — pipeline stage 4.

Identical to models/Swi_DiT.py (SwiGLU feed-forward, AdaLN blocks, causal-to-the-past mask,
same widths), except for how a token's position enters the model. RoPE rotates each
attention head's queries and keys by an angle proportional to the token's position, so
attention scores depend on the *difference* between two positions rather than on learned
absolute position vectors.

    rope="index"      RoPE over token indices 0..85 replaces the learned position embedding;
                      the timestamps still enter additively through the t_rel MLP, as in SWI_DiT
    rope="timestamp"  RoPE over the real (normalized) timestamps t_rel; no learned position
                      embedding and no t_rel MLP: time enters only through the rotations,
                      so irregular sampling gaps are seen directly as angle differences

t_rel is normalized to roughly ±1.7, so in timestamp mode it is multiplied by rope_ts_scale
(default 25) to span about the same range as the token indices 0..85; otherwise even the
fastest RoPE frequency would barely rotate across the window.

Config: {"arch": "swi_dit_rope", "model": {..., "rope": "index" | "timestamp"}}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Swi_DiT import SwiGLU, sinusoidal_embedding, sinusoidal_embedding_seq

ROPE_MODES = ("index", "timestamp")


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, base=10000.0):
        super().__init__()
        half = d_head // 2
        self.register_buffer("freqs", 1.0 / (base ** (torch.arange(half).float() / half)),
                             persistent=False)

    def forward(self, x, positions):
        """x: (B, H, T, d_head); positions: (T,) or (B, T). Returns x rotated."""
        angles = positions.unsqueeze(-1).float() * self.freqs          # (T, half) or (B, T, half)
        if angles.dim() == 2:
            angles = angles[None, None]                                 # (1, 1, T, half)
        else:
            angles = angles[:, None]                                    # (B, 1, T, half)
        cos, sin = angles.cos(), angles.sin()
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE on Q and K (same parameter count as nn.MultiheadAttention)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0 and (d_model // n_heads) % 2 == 0
        self.n_heads, self.d_head, self.dropout = n_heads, d_model // n_heads, dropout
        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope     = RotaryEmbedding(self.d_head)

    def forward(self, x, positions, mask=None):
        """x: (B, T, d_model); positions: (T,) or (B, T); mask: (T, T) bool, True = ignore."""
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, positions), self.rope(k, positions)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else ~mask,      # SDPA: True = attend
            dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class AdaLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, ff_expansion=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = RoPEAttention(d_model, n_heads, dropout)
        self.ff    = SwiGLU(d_model, expansion=ff_expansion, bias=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 4 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)          # conditioning starts as the identity
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, positions, mask=None):
        s1, b1, s2, b2 = (c.unsqueeze(1) for c in self.adaLN(cond).chunk(4, dim=-1))
        x = x + self.attn(self.norm1(x) * (1 + s1) + b1, positions, mask=mask)
        return x + self.ff(self.norm2(x) * (1 + s2) + b2)


class TrajectoryDiT(nn.Module):
    def __init__(self, obs_len=43, fut_len=43, in_dim=6, d_model=256, n_heads=8, n_layers=4,
                 dropout=0.1, ff_expansion=3, rope="index", rope_ts_scale=25.0):
        super().__init__()
        if rope not in ROPE_MODES:
            raise ValueError(f"rope must be one of {ROPE_MODES}, got {rope!r}")
        self.obs_len, self.fut_len = obs_len, fut_len
        self.rope, self.rope_ts_scale = rope, rope_ts_scale
        seq_len = obs_len + fut_len

        self.input_proj = nn.Linear(in_dim, d_model)
        if rope == "index":                             # timestamps still enter additively
            self.trel_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                          nn.Linear(d_model, d_model))
        self.tflow_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                       nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([AdaLNBlock(d_model, n_heads, dropout, ff_expansion)
                                     for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, in_dim)

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[:obs_len, obs_len:] = True                 # observed tokens can't see the future
        self.register_buffer("attn_mask", mask)
        self.register_buffer("index_pos", torch.arange(seq_len).float(), persistent=False)

    def forward(self, x_obs, x_t, t, t_rel):
        """x_obs, x_t: (B, 43, 6); t: (B,) flow time; t_rel: (B, 86). Returns (B, 43, 6) velocity."""
        tokens = torch.cat([self.input_proj(x_obs), self.input_proj(x_t)], dim=1)   # (B, 86, D)
        if self.rope == "index":
            tokens    = tokens + self.trel_mlp(sinusoidal_embedding_seq(t_rel, tokens.shape[-1]))
            positions = self.index_pos                                            # (86,)
        else:
            positions = t_rel * self.rope_ts_scale                                # (B, 86)
        cond = self.tflow_mlp(sinusoidal_embedding(t.float() * 1000.0, tokens.shape[-1]))
        for block in self.blocks:
            tokens = block(tokens, cond, positions, mask=self.attn_mask)
        return self.output_head(self.output_norm(tokens[:, self.obs_len:]))
