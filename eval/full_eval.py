"""
Accuracy (evaluate.py) and sampling latency (latency.py) for finished checkpoints, in one run.

Usage (on Savio, from the repo root):
    SCRIPT=full_eval sbatch savio/job_eval.sh finished_ckpt/3_gru
        # → results/eval/full_eval_b64.json
    SCRIPT=full_eval sbatch savio/job_eval.sh finished_ckpt/2_architecture/distill_4_256 finished_ckpt/3_gru
    SCRIPT=full_eval EVAL_ARGS="--lat_batch_size 1 --lat_n_calls 10000" sbatch savio/job_eval.sh finished_ckpt/3_gru
    python eval/full_eval.py finished_ckpt/2_architecture/distill_2_128 --max_batches 2 --lat_n_calls 20 --device cpu

Each model is loaded once, then:
  1. accuracy   minADE / minFDE / avgADE / avgFDE (+ turning, p1/p99) on the test set, exactly
                as evaluate.py: --n_samples futures (default 20), --n_steps, --batch_size, ...
  2. latency    --lat_n_calls timed calls (default 1,000), each sampling --lat_n_samples futures
                (default 1) for a batch of --lat_batch_size windows (default 64), exactly as
                latency.py; --lat_synthetic for random windows instead of test windows.
Progressive students use their own step count for both.

Results are rewritten to --out after each part of each model (default
results/eval/full_eval_b<lat_batch_size>[_synthetic].json). A run with the same --out skips
whatever is already in it, so a job cut off by its time limit resumes, even halfway through a
model (accuracy done, latency not). Settings, including the GPU name, must match.
"""
import argparse
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from DataLoaders.ADSBdataset import get_dataloaders
from evaluate import load_model, model_name, find_run_folders, pick_device, evaluate
from latency import real_windows, synthetic_windows, time_model, summarize, SYNTHETIC_POOL_BATCHES


def main():
    p = argparse.ArgumentParser(description="Test-set accuracy + sampling latency for finished checkpoints")
    p.add_argument("folders", nargs="*", help="run folders (default: every model in finished_ckpt/)")
    p.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    p.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    p.add_argument("--n_steps", type=int, default=20, help="Euler steps (progressive students use their own)")
    p.add_argument("--device", default="auto", help="auto (CUDA, else MPS, else CPU), cuda, mps or cpu")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="fp16 autocast; default: on for models whose config sets eval_amp (on CUDA), else fp32")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", default=None, help="JSON file (resumable); default results/eval/full_eval_b<B>.json")
    p.add_argument("--redo", nargs="+", default=[], metavar="NAME",
                   help="models already in --out to evaluate again (e.g. dit_cfm_20M); their old entries "
                        "are replaced. Use it only for the first submission: a resubmission with --redo "
                        "would throw away the new result too")
    # accuracy (evaluate.py)
    p.add_argument("--n_samples", type=int, default=20, help="accuracy: K futures sampled per window")
    p.add_argument("--batch_size", type=int, default=256, help="accuracy: test windows per batch")
    p.add_argument("--chunk", type=int, default=256, help="accuracy: trajectories per model call")
    p.add_argument("--max_batches", type=int, default=None, help="accuracy: limit for a quick check")
    p.add_argument("--turn_deg", type=float, default=45.0, help="accuracy: threshold for *_turn metrics")
    # latency (latency.py)
    p.add_argument("--lat_n_calls", type=int, default=1000, help="latency: timed calls per model")
    p.add_argument("--lat_batch_size", type=int, default=64, help="latency: windows per call")
    p.add_argument("--lat_n_samples", type=int, default=1, help="latency: futures per window")
    p.add_argument("--lat_n_warmup", type=int, default=50, help="latency: untimed calls first")
    p.add_argument("--lat_synthetic", action="store_true", help="latency: random windows instead of test windows")
    args = p.parse_args()

    if args.out is None:
        tag = f"_b{args.lat_batch_size}" + ("_synthetic" if args.lat_synthetic else "")
        args.out = str(ROOT / "results" / "eval" / f"full_eval{tag}.json")

    folders = [f if Path(f).is_absolute() or Path(f).exists() else ROOT / f
               for f in args.folders] or find_run_folders(args.ckpt)
    missing = [str(f) for f in folders if not Path(f).is_dir()]
    if missing:
        sys.exit("Folder(s) not found: " + ", ".join(missing))
    folders = [run for f in folders for run in (find_run_folders(args.ckpt, f) or [f])]
    if not folders:
        sys.exit("No checkpoints found")

    device = pick_device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False      # same kernels every call, steadier timings
    print(f"Device: {gpu} | accuracy K={args.n_samples} | latency {args.lat_n_calls:,} calls × "
          f"{args.lat_batch_size} {'synthetic' if args.lat_synthetic else 'test'} window(s) × "
          f"{args.lat_n_samples} future(s)")

    ds = nc.Dataset(args.nc_path, "r")
    feat_mean, feat_std = np.array(ds.feature_mean), np.array(ds.feature_std)
    ds.close()

    settings = {"ckpt": args.ckpt, "n_steps": args.n_steps, "amp": args.amp, "nc_path": args.nc_path,
                "device": gpu,
                "accuracy": {k: getattr(args, k) for k in ("n_samples", "max_batches", "turn_deg")},
                "latency": {"n_calls": args.lat_n_calls, "batch_size": args.lat_batch_size,
                            "n_samples": args.lat_n_samples, "n_warmup": args.lat_n_warmup,
                            "synthetic": args.lat_synthetic}}
    out = Path(args.out)
    state = {"settings": settings, "n_windows": None, "n_turning": None, "results": {}, "folders": {}}
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("settings") != settings:
            sys.exit(f"{out} was made with other settings ({prev.get('settings')}); "
                     f"use another --out, or delete it to start over")
        state = prev
        print(f"Resuming {out}: {len(state['results'])} model(s) started")
    results, done = state["results"], state["folders"]
    for name in args.redo:                 # forget these so they are evaluated again
        if results.pop(name, None) is None:
            sys.exit(f"--redo {name}: not in {out} (models in it: {', '.join(results) or 'none'})")
        done.pop(name, None)
        print(f"--redo: dropped {name}; it will be evaluated again")

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out)                       # never leaves a half-written file

    test_loaders, pools = {}, {}
    for folder in folders:
        res = Path(folder).resolve()
        key = str(res.relative_to(ROOT)) if res.is_relative_to(ROOT) else str(res)
        name = next((n for n, k in done.items() if k == key), None)
        if name and {"accuracy", "latency"} <= results.get(name, {}).keys():
            print(f"\n=== {folder} === already in {out}, skipping")
            continue
        print(f"\n=== {folder} ===")
        model, seed, ckpt_steps = load_model(folder, args.ckpt, device)
        name = model_name(folder, ckpt_steps)
        n_steps = ckpt_steps or args.n_steps
        amp = model.default_amp if args.amp is None else args.amp
        entry = results.setdefault(name, {})
        entry.update(params=sum(p.numel() for p in model.parameters()), steps=n_steps, amp=bool(amp))
        done[name] = key

        if "accuracy" not in entry:
            if seed not in test_loaders:   # same split seed as training → unseen aircraft
                _, _, test_loaders[seed] = get_dataloaders(args.nc_path, batch_size=args.batch_size,
                                                           seed=seed, num_workers=args.num_workers)
            print("Accuracy:")
            torch.manual_seed(0)           # same sampling noise for every model
            acc, state["n_windows"], state["n_turning"] = evaluate(
                model, test_loaders[seed], feat_mean, feat_std, device, args.n_samples, n_steps,
                args.max_batches, args.turn_deg, args.chunk, amp)
            entry["accuracy"] = acc
            save()
            print("  " + " | ".join(f"{m} {acc[m]:.1f} ± {acc[m + '_std']:.1f} m"
                                    for m in ("minADE", "minFDE", "avgADE", "avgFDE", "minFDE_turn")))

        if "latency" not in entry:
            pool_key = "synthetic" if args.lat_synthetic else seed
            if pool_key not in pools:
                pools[pool_key] = synthetic_windows(SYNTHETIC_POOL_BATCHES * args.lat_batch_size, device) \
                    if args.lat_synthetic else \
                    real_windows(args.nc_path, seed, args.lat_n_calls * args.lat_batch_size, device)
            print("Latency:")
            torch.manual_seed(0)
            times = time_model(model, *pools[pool_key], device, args.lat_batch_size, args.lat_n_calls,
                               args.lat_n_samples, n_steps, args.lat_n_warmup, amp)
            entry["latency"] = {k: float(v) for k, v in summarize(times, args.lat_batch_size).items()}
            save()
            r = entry["latency"]
            print(f"  mean {r['mean_ms']:.2f} ms/call | p50 {r['p50_ms']:.2f} | p99 {r['p99_ms']:.2f} | "
                  f"{r['ms_per_window']:.3f} ms/window | {r['windows_per_s']:,.0f} windows/s")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # side-by-side summary
    print(f"\n{gpu} | accuracy: {state['n_windows'] or 0:,} test windows, K={args.n_samples}, metres | "
          f"latency: batch {args.lat_batch_size} × {args.lat_n_calls:,} calls, ms")
    acc_cols = ["minADE", "minFDE", "avgADE", "avgFDE", "minFDE_turn"]
    lat_cols = ["mean_ms", "p50_ms", "p99_ms", "ms_per_window", "windows_per_s"]
    width = max(len(k) for k in results) + 2
    print("model".ljust(width) + f"{'params':>12}{'steps':>7}" + "".join(f"{c:>14}" for c in acc_cols + lat_cols))
    for name, r in results.items():
        a, l = r.get("accuracy", {}), r.get("latency", {})
        row = name.ljust(width) + f"{r.get('params', 0):12,}{r.get('steps', 0):7}"
        row += "".join(f"{a[c]:14.1f}" if c in a else f"{'-':>14}" for c in acc_cols)
        row += "".join((f"{l[c]:14,.0f}" if c == "windows_per_s" else f"{l[c]:14.3f}") if c in l
                       else f"{'-':>14}" for c in lat_cols)
        print(row)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
