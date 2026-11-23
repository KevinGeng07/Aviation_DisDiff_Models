"""
Which model class a config builds. A config's optional "arch" field picks the file;
without it, it's the original SWI_DiT, so all existing configs and checkpoints still load.

    {"arch": "gru_swi_dit", "model": {...TrajectoryDiT kwargs...}}

model_weights(ckpt) picks the weights to evaluate / distil from out of a checkpoint dict, in
this repo's format (ema_state / model_state) or CITRIS's (ema_model / model).

A config with "amp": true (e.g. configs/teacher_dit_cfm_20M.json) runs that model in fp16
autocast on CUDA when it is the frozen teacher in distillation: maybe_amp() wraps it. Evaluation
(eval/) ignores "amp" and samples in fp32, unless the config also sets "eval_amp": true or
--amp is given. On CPU / MPS it runs in full precision (half precision is slower there).
"""
import importlib

import torch
import torch.nn as nn

ARCHS = {
    "swi_dit":     "models.Swi_DiT",
    "gru_swi_dit": "models.GRU_Swi_DiT",
    "gru_swi_dit_2": "models.GRU_Swi_DiT_2",
    "gru_swi_dit_3": "models.GRU_Swi_DiT_3",
    "swi_dit_rope":  "models.Swi_DiT_RoPE",     # model kwarg rope = "index" | "timestamp"
    "dit_cfm":       "models.DiT_CFM",          # CITRIS TrajectoryFlowV5 (GELU DiT), CFM
}

WEIGHT_KEYS = ("ema_state", "ema_model", "model_state", "model")   # EMA first, like validation


def build_model(cfg):
    """TrajectoryDiT from the file named by cfg["arch"] (default swi_dit), with cfg["model"] kwargs."""
    arch = cfg.get("arch", "swi_dit")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {', '.join(ARCHS)}")
    return importlib.import_module(ARCHS[arch]).TrajectoryDiT(**cfg["model"])


class AutocastModel(nn.Module):
    """Runs a frozen model's forward in fp16 autocast on CUDA, returning float32 outputs.
    Elsewhere (CPU / MPS) it is a plain pass-through."""

    def __init__(self, model, dtype=torch.float16):
        super().__init__()
        self.model, self.dtype = model, dtype

    def forward(self, *args, **kwargs):
        if args[0].is_cuda:
            with torch.autocast("cuda", dtype=self.dtype):
                return self.model(*args, **kwargs).float()
        return self.model(*args, **kwargs)


def maybe_amp(model, cfg):
    """Wrap model in AutocastModel if its config asks for amp (inference / frozen-teacher use only)."""
    return AutocastModel(model) if cfg.get("amp") else model


def model_weights(ckpt):
    """State dict to load from a checkpoint: EMA weights if saved, else the raw model."""
    for key in WEIGHT_KEYS:
        if ckpt.get(key):
            return ckpt[key]
    raise KeyError(f"checkpoint has none of {WEIGHT_KEYS}; keys: {list(ckpt)[:10]}")
