"""
Benchmark the distilled SWI_DiT students side by side: accuracy, size and speed.

Usage (on Savio, from the repo root):
    SCRIPT=benchmark sbatch savio_batchfiles/EvalSavio.sh                      # the 7 *_SWiDiT students
    SCRIPT=benchmark sbatch savio_batchfiles/EvalSavio.sh finished_ckpt/6_4_256_SWiDiT
    SCRIPT=benchmark EVAL_ARGS="--max_batches 10 --out results/benchmark.json" sbatch savio_batchfiles/EvalSavio.sh
    SCRIPT=benchmark EVAL_ARGS="--n_steps 20 16 8 4 2 1 --out results/benchmark_all.json" \
        sbatch savio_batchfiles/EvalSavio.sh finished_ckpt/*_SWiDiT finished_ckpt/6_4_256_Progressive
        # every student at every step count + each progressive round, in one file

For every model:
    minADE, minFDE, avgADE, avgFDE   test-set errors in metres, exactly as eval/evaluate.py
    params                           number of model parameters
    steps                            Euler steps used; with several --n_steps, each model is
                                     run once per value and named <model>_<N>step. Progressive
                                     students always use their own step count, once.
    latency                          time to sample ONE future for ONE test window, run
                                     --n_runs times (default 1000) on that same window;
                                     reported as mean / median / p95 in milliseconds
"""
import argparse
import json
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from DataLoaders.ADSBdataset import get_dataloaders
from models.cfm import euler_sample
from evaluate import load_model, evaluate, model_name


def sample_latency(model, test_loader, device, n_steps, n_runs, n_warmup=20):
    """Milliseconds to sample one future for the first test window, timed n_runs times."""
    batch = next(iter(test_loader))
    obs   = batch["obs"][:1].to(device)     # (1, 43, 6) the same window every run
    t_rel = batch["t_rel"][:1].to(device)   # (1, 86)

    def run():
        euler_sample(model, obs, t_rel, n_samples=1, n_steps=n_steps, device=str(device))
        if device.type == "cuda":
            torch.cuda.synchronize()        # GPU calls are async; wait so the timing is real

    for _ in range(n_warmup):               # first calls include CUDA setup, so don't time them
        run()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {"lat_mean_ms": times.mean(), "lat_median_ms": np.median(times),
            "lat_p95_ms": np.percentile(times, 95)}


def main():
    parser = argparse.ArgumentParser(description="ADE/FDE, parameter count and latency per model")
    parser.add_argument("folders", nargs="*",
                        help="run folders (default: every finished_ckpt/*_SWiDiT student)")
    parser.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    parser.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--n_samples", type=int, default=20, help="K futures per window for ADE/FDE")
    parser.add_argument("--n_steps", type=int, nargs="+", default=[20],
                        help="Euler steps; several values run every model at each (progressive students use their own)")
    parser.add_argument("--n_runs", type=int, default=1000, help="timed single-sample runs for latency")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=None, help="limit ADE/FDE batches for a quick check")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out", default=None, help="also save the results to this JSON file")
    args = parser.parse_args()

    folders = [Path(f) if Path(f).is_absolute() or Path(f).exists() else ROOT / f for f in args.folders] \
              or sorted((ROOT / "finished_ckpt").glob("*_SWiDiT"))
    missing = [str(f) for f in folders if not f.is_dir()]
    if missing:
        sys.exit("Folder(s) not found: " + ", ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {gpu}")
    ds = nc.Dataset(args.nc_path, "r")
    feat_mean = np.array(ds.feature_mean)
    feat_std  = np.array(ds.feature_std)
    ds.close()

    test_loaders = {}   # one per training seed; the seed decides the aircraft-level split
    all_results = {}
    for folder in folders:
        print(f"\n=== {folder} ===")
        model, seed, ckpt_steps = load_model(folder, args.ckpt, device)
        if seed not in test_loaders:
            _, _, test_loaders[seed] = get_dataloaders(args.nc_path, batch_size=args.batch_size,
                                                       seed=seed, num_workers=args.num_workers)

        # progressive students only work at their own step count
        for n_steps in [ckpt_steps] if ckpt_steps else args.n_steps:
            name = model_name(folder, ckpt_steps or (n_steps if len(args.n_steps) > 1 else None))
            print(f"--- {name}: {n_steps} Euler steps ---")

            torch.manual_seed(0)   # same sampling noise for every model → comparable numbers
            metrics, n, _ = evaluate(model, test_loaders[seed], feat_mean, feat_std, device,
                                     args.n_samples, n_steps, args.max_batches)
            results = {"steps": n_steps}
            results.update({k: metrics[k] for k in ("minADE", "minFDE", "avgADE", "avgFDE")})
            results["params"] = sum(p.numel() for p in model.parameters())
            results.update(sample_latency(model, test_loaders[seed], device, n_steps, args.n_runs))
            all_results[name] = results
            print("  " + " | ".join(f"{k} {v:,.2f}" for k, v in results.items()))

    # side-by-side summary
    print(f"\n{gpu} | test set: {n:,} windows | K={args.n_samples} | Euler steps {args.n_steps} "
          f"(progressive: own) | latency: 1 sample, 1 window, {args.n_runs} runs")
    cols = ["steps", "minADE", "minFDE", "avgADE", "avgFDE", "params", "lat_mean_ms", "lat_median_ms", "lat_p95_ms"]
    width = max(len(k) for k in all_results) + 2
    print("model".ljust(width) + "".join(f"{c:>15}" for c in cols))
    for name, results in all_results.items():
        print(name.ljust(width) + "".join(f"{results[c]:15,.0f}" if c in ("params", "steps") else f"{results[c]:15.2f}"
                                          for c in cols))

    if args.out:
        settings = {k: getattr(args, k) for k in ("ckpt", "n_samples", "n_steps", "n_runs", "max_batches", "nc_path")}
        payload = {"settings": settings, "device": gpu, "n_windows": n,
                   "results": {m: {k: float(v) for k, v in r.items()} for m, r in all_results.items()}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
