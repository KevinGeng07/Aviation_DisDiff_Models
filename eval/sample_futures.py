"""
Sample futures from every finished model for a few random test windows → one CSV.

Usage (from the repo root):
    python eval/sample_futures.py                                  # every model in finished_ckpt/
    python eval/sample_futures.py finished_ckpt/6_4_256_SWiDiT     # or folders given on the command line
    python eval/sample_futures.py --n_traj 10 --n_samples 20 --out eval/futures.csv

Picks --n_traj random test windows (the same ones for every model), samples
--n_samples futures from each model, and writes one row per point:

    model, window, future, step, x_m, y_m

window = index of the window in the .nc file, future = 1..K (0 = the true future),
step = 1..43, x_m / y_m = position in metres. Every model starts from the same
sampling noise. Progressive-distillation students (checkpoints that store n_steps)
are sampled with their own step count and labelled e.g. 6_4_256_Progressive_4step.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DataLoaders.ADSBdataset import get_dataloaders
from models.cfm import euler_sample
from evaluate import load_model, find_run_folders, model_name


def sample_futures(model, test_ds, picks, n_samples, n_steps, device):
    """n_samples futures for the picked test windows, (x, y) in metres: (K, n_picks, 43, 2)."""
    items = [test_ds[i] for i in picks]
    obs   = torch.stack([it["obs"] for it in items]).to(device)
    t_rel = torch.stack([it["t_rel"] for it in items]).to(device)
    preds = euler_sample(model, obs, t_rel, n_samples=n_samples,
                         n_steps=n_steps, device=str(device))           # (K, n, 43, 6) normalized
    mean, std = test_ds.stats["feature_mean"][:2], test_ds.stats["feature_std"][:2]
    return preds[..., :2].cpu().numpy() * std + mean


def main():
    parser = argparse.ArgumentParser(description="Sampled futures for random test windows → CSV")
    parser.add_argument("folders", nargs="*", help="run folders (default: every model in finished_ckpt/)")
    parser.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    parser.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--n_traj", type=int, default=10, help="random test windows")
    parser.add_argument("--n_samples", type=int, default=20, help="futures per window per model")
    parser.add_argument("--n_steps", type=int, default=20, help="Euler steps (progressive students use their own)")
    parser.add_argument("--traj_seed", type=int, default=0, help="which windows get picked")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "eval" / "futures.csv"))
    args = parser.parse_args()

    folders = [Path(f) if Path(f).is_absolute() or Path(f).exists() else ROOT / f
               for f in args.folders] or find_run_folders(args.ckpt)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | {len(folders)} models")

    # seed 42 split, as in training, so these aircraft were never trained on
    _, _, test_loader = get_dataloaders(args.nc_path, seed=42, num_workers=args.num_workers)
    test_ds = test_loader.dataset
    picks   = random.Random(args.traj_seed).sample(range(len(test_ds)), args.n_traj)
    windows = [test_ds.indices[i] for i in picks]           # window index in the .nc file
    print(f"Windows: {windows}")

    mean, std = test_ds.stats["feature_mean"][:2], test_ds.stats["feature_std"][:2]
    rows = []
    for w, i in zip(windows, picks):                        # true futures, labelled future 0
        true_xy = test_ds[i]["fut"][:, :2].numpy() * std + mean
        rows += [("ground_truth", w, 0, s + 1, x, y) for s, (x, y) in enumerate(true_xy)]

    for folder in folders:
        print(f"\n=== {folder} ===")
        model, seed, ckpt_steps = load_model(folder, args.ckpt, device)
        if seed != 42:
            print(f"  warning: trained with seed {seed}, so its test split differs from seed 42's")
        name = model_name(folder, ckpt_steps)

        torch.manual_seed(0)                                # same sampling noise for every model
        futures = sample_futures(model, test_ds, picks, args.n_samples,
                                 ckpt_steps or args.n_steps, device)
        for k in range(futures.shape[0]):
            for j, w in enumerate(windows):
                rows += [(name, w, k + 1, s + 1, x, y) for s, (x, y) in enumerate(futures[k, j])]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "window", "future", "step", "x_m", "y_m"])
        writer.writerows((m, w, k, s, f"{x:.1f}", f"{y:.1f}") for m, w, k, s, x, y in rows)
    print(f"\nSaved {len(rows):,} rows to {out}")


if __name__ == "__main__":
    main()
