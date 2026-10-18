"""
Which model class a config builds. A config's optional "arch" field picks the file;
without it, it's the original SWI_DiT, so all existing configs and checkpoints still load.

    {"arch": "gru_swi_dit", "model": {...TrajectoryDiT kwargs...}}
"""
import importlib

ARCHS = {
    "swi_dit":     "models.SWI_DiT",
    "gru_swi_dit": "models.GRU_SWi_DiT",
}


def build_model(cfg):
    """TrajectoryDiT from the file named by cfg["arch"] (default swi_dit), with cfg["model"] kwargs."""
    arch = cfg.get("arch", "swi_dit")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {', '.join(ARCHS)}")
    return importlib.import_module(ARCHS[arch]).TrajectoryDiT(**cfg["model"])
