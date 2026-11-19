"""
Per-axis error over the 43-step forecast horizon, with outliers trimmed to p1–p99.

Usage (from the repo root):
    python eval/error_by_step.py                                   # finished_ckpt/2_architecture/distill_4_256
    python eval/error_by_step.py finished_ckpt/3_gru/gru1_2L_256 --max_batches 20
    python eval/error_by_step.py --plots_only                       # re-plot from saved stats.csv
On Savio:
    SCRIPT=error_by_step sbatch savio/job_eval.sh
    SCRIPT=error_by_step EVAL_ARGS="--pick random" sbatch savio/job_eval.sh <run folder>

For every test window the model samples K futures (--n_samples), and one of them is scored:
    --pick best    (default) the sample with the lowest ADE, i.e. the one minADE scores
    --pick random  one sample chosen at random: what a typical single forecast looks like
Its error at every future step k = 1..43 and axis (X east, Y north, Z altitude) in metres:
    signed error   pred - true                     (bias: does it drift one way?)
    |error|        abs(pred - true)

For each (step, axis) the windows below that step's p1 or above its p99 are dropped, and the
mean and standard deviation of the rest are reported, so a few wild windows don't dominate.
The full spread (p1, p10, p25, p50, p75, p90, p99, untrimmed) is saved too.

Output, in --out_dir/<model>/ (default results/error_by_step/<model>/):
    stats.csv                one row per step × axis × kind (abs / signed): n kept, trimmed
                             mean & std, and the percentiles
    error_by_step.png        |error|: trimmed mean ± 1 std band, one panel per axis
    error_percentiles.png    |error|: median with p25–p75, p10–p90, p1–p99 bands
    signed_error_by_step.png signed error: trimmed mean ± 1 std band and the zero line
Panels have their own y-axis: altitude errors are on a different scale from X / Y.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DataLoaders.ADSBdataset import get_dataloaders
from evaluate import load_model, model_name, sample

AXES       = ("X", "Y", "Z")
AXIS_NAMES = {"X": "X (east)", "Y": "Y (north)", "Z": "Z (altitude)"}
PCTS       = (1, 10, 25, 50, 75, 90, 99)
FIELDS     = ["kind", "axis", "step", "n_kept", "trimmed_mean_m", "trimmed_std_m"] + [f"p{p}_m" for p in PCTS]

# Categorical slots 1–3 of the dataviz reference palette (validated: CVD-safe on light surface)
COLORS  = {"X": "#2a78d6", "Y": "#eb6834", "Z": "#1baf7a"}
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#2b2b29", "#6b6a64", "#e6e5e0"


# ── Computing the errors ──────────────────────────────────────────────────

def collect_errors(model, test_loader, std, device, n_samples, n_steps, pick, max_batches, chunk, amp=False):
    """Signed error (pred - true) in metres of one sample per window: (N windows, 43, 3)."""
    std_xyz = torch.tensor(std[:3], device=device, dtype=torch.float32)
    chunks  = []
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            if max_batches is not None and b >= max_batches:
                break
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)
            preds, chunk = sample(model, obs, t_rel, n_samples, n_steps, device, chunk, amp)  # (K, B, 43, 6)

            err = (preds[..., :3] - fut[None, ..., :3]) * std_xyz        # (K, B, 43, 3) metres
            if pick == "best":                                            # lowest ADE on (x, y)
                k = err[..., :2].norm(dim=-1).mean(dim=-1).argmin(dim=0)  # (B,)
            else:
                k = torch.randint(0, err.shape[0], (err.shape[1],), device=device)
            chunks.append(err[k, torch.arange(err.shape[1], device=device)].cpu())
            print(f"  batch {b + 1}/{len(test_loader)}", end="\r", flush=True)
    print()
    return torch.cat(chunks).double().numpy()


def step_stats(err):
    """Rows of stats.csv from signed errors (N, 43, 3)."""
    rows = []
    for kind, values in (("abs", np.abs(err)), ("signed", err)):
        for a, axis in enumerate(AXES):
            for s in range(values.shape[1]):
                v = values[:, s, a]
                pct = np.percentile(v, PCTS)
                kept = v[(v >= pct[0]) & (v <= pct[-1])]              # p1 ≤ v ≤ p99
                rows.append({"kind": kind, "axis": axis, "step": s + 1, "n_kept": len(kept),
                             "trimmed_mean_m": kept.mean(), "trimmed_std_m": kept.std(),
                             **{f"p{p}_m": x for p, x in zip(PCTS, pct)}})
    return rows


def write_stats(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def read_stats(path):
    with open(path) as f:
        return [{k: (v if k in ("kind", "axis") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


# ── Plots ─────────────────────────────────────────────────────────────────

def series(rows, kind, axis, field):
    sel = sorted((r for r in rows if r["kind"] == kind and r["axis"] == axis), key=lambda r: r["step"])
    return np.array([r["step"] for r in sel]), np.array([r[field] for r in sel])


def figure(title, subtitle):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.text(0.01, 0.905, subtitle, ha="left", fontsize=9.5, color=MUTED)
    for ax, axis in zip(axs, AXES):
        ax.set_facecolor(SURFACE)
        ax.set_title(AXIS_NAMES[axis], loc="left", fontsize=11, color=TEXT)   # names the series
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("future step")
        ax.set_xlim(1, 43)
        ax.set_xticks([1, 11, 22, 33, 43])
    axs[0].set_ylabel("metres")
    return fig, axs


def save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  saved {path}")


def make_plots(rows, out_dir, label):
    """The three figures described in the module docstring."""
    fig, axs = figure(f"{label}: absolute error by forecast step",
                      "line = mean |error| of windows between p1 and p99 at that step; band = ±1 std (floored at 0)")
    for ax, axis in zip(axs, AXES):
        s, m = series(rows, "abs", axis, "trimmed_mean_m")
        _, sd = series(rows, "abs", axis, "trimmed_std_m")
        ax.fill_between(s, np.maximum(m - sd, 0), m + sd, color=COLORS[axis], alpha=0.18, linewidth=0)
        ax.plot(s, m, color=COLORS[axis], linewidth=2)
        ax.set_ylim(bottom=0)
    save(fig, out_dir / "error_by_step.png")

    fig, axs = figure(f"{label}: spread of absolute error by forecast step",
                      "line = median |error|; bands = p25–p75 (darkest), p10–p90, p1–p99 (lightest), all windows")
    for ax, axis in zip(axs, AXES):
        s, med = series(rows, "abs", axis, "p50_m")
        for lo, hi, alpha in (("p1_m", "p99_m", 0.12), ("p10_m", "p90_m", 0.22), ("p25_m", "p75_m", 0.38)):
            ax.fill_between(s, series(rows, "abs", axis, lo)[1], series(rows, "abs", axis, hi)[1],
                            color=COLORS[axis], alpha=alpha, linewidth=0)
        ax.plot(s, med, color=COLORS[axis], linewidth=2)
        ax.set_ylim(bottom=0)
    save(fig, out_dir / "error_percentiles.png")

    fig, axs = figure(f"{label}: signed error (bias) by forecast step",
                      "line = mean of (predicted − true) for windows between p1 and p99; band = ±1 std; 0 = no bias")
    for ax, axis in zip(axs, AXES):
        s, m = series(rows, "signed", axis, "trimmed_mean_m")
        _, sd = series(rows, "signed", axis, "trimmed_std_m")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.fill_between(s, m - sd, m + sd, color=COLORS[axis], alpha=0.18, linewidth=0)
        ax.plot(s, m, color=COLORS[axis], linewidth=2)
    save(fig, out_dir / "signed_error_by_step.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-axis p1–p99 trimmed error over the forecast horizon")
    parser.add_argument("folders", nargs="*", default=["finished_ckpt/2_architecture/distill_4_256"],
                        help="run folders (default: finished_ckpt/2_architecture/distill_4_256)")
    parser.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    parser.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--n_samples", type=int, default=20, help="K futures sampled per window")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Euler steps (progressive students use their own)")
    parser.add_argument("--pick", choices=("best", "random"), default="best",
                        help="which of the K samples is scored per window")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=256, help="trajectories per model call (see evaluate.py)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="half-precision sampling; default from the model config (see evaluate.py)")
    parser.add_argument("--max_batches", type=int, default=None, help="limit for a quick check")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "error_by_step"))
    parser.add_argument("--plots_only", action="store_true",
                        help="skip the model; re-draw the plots from each <out_dir>/<model>/stats.csv")
    args = parser.parse_args()

    folders = [Path(f) if Path(f).is_absolute() or Path(f).exists() else ROOT / f for f in args.folders]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}

    for folder in folders:
        print(f"\n=== {folder} ===")
        if args.plots_only:
            label   = model_name(folder) + ("" if args.pick == "best" else f"_{args.pick}")
            out_dir = Path(args.out_dir) / label
            rows    = read_stats(out_dir / "stats.csv")
        else:
            model, seed, ckpt_steps = load_model(folder, args.ckpt, device)
            label   = model_name(folder, ckpt_steps) + ("" if args.pick == "best" else f"_{args.pick}")
            out_dir = Path(args.out_dir) / label
            out_dir.mkdir(parents=True, exist_ok=True)
            if seed not in loaders:   # training seed → same aircraft split, so these were never trained on
                _, _, loaders[seed] = get_dataloaders(args.nc_path, batch_size=args.batch_size,
                                                      seed=seed, num_workers=args.num_workers)
            std = loaders[seed].dataset.stats["feature_std"]
            torch.manual_seed(0)      # same sampling noise as evaluate.py
            err = collect_errors(model, loaders[seed], std, device, args.n_samples,
                                 ckpt_steps or args.n_steps, args.pick, args.max_batches, args.chunk,
                                 model.default_amp if args.amp is None else args.amp)
            rows = step_stats(err)
            write_stats(rows, out_dir / "stats.csv")
            print(f"  {err.shape[0]:,} windows | saved {out_dir / 'stats.csv'}")
            for axis in AXES:
                s, m = series(rows, "abs", axis, "trimmed_mean_m")
                print(f"  {AXIS_NAMES[axis]:13s} mean |error| step 1: {m[0]:8.1f} m   step 43: {m[-1]:8.1f} m")

        try:
            make_plots(rows, out_dir, label)
        except ImportError:
            print("  matplotlib not installed here: stats.csv is saved; copy it over and run "
                  "--plots_only where matplotlib is available")


if __name__ == "__main__":
    main()
