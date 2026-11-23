"""
Inference latency per model: time for one sampling call, repeated many times.

Usage (on Savio, from the repo root):
    SCRIPT=latency sbatch savio/job_eval.sh finished_ckpt/2_architecture   # → results/eval/eval_latency.json
    SCRIPT=latency sbatch savio/job_eval.sh finished_ckpt/1_teacher/dit_cfm_20M finished_ckpt/2_architecture
    SCRIPT=latency EVAL_ARGS="--batch_size 64 --synthetic" sbatch savio/job_eval.sh finished_ckpt/2_architecture
        # 10,000 synthetic batches of 64 windows → results/eval/eval_latency_b64_synthetic.json
    SCRIPT=latency EVAL_ARGS="--n_samples 20" sbatch savio/job_eval.sh ...   # K=20 futures per window
    python eval/latency.py finished_ckpt/2_architecture/distill_4_256 --n_calls 500 --device mps

Each of --n_calls (default 10,000) timed calls samples --n_samples futures (default 1) for a
batch of --batch_size windows (default 1) with --n_steps Euler steps (default 20; progressive
students use their own); the whole batch goes through the model in one call (no chunking).
  real windows (default)  consecutive test windows, wrapping around the test set if needed
  --synthetic             random windows (N(0,1) features, evenly spaced timestamps) from a
                          fixed pool; the model's cost does not depend on the values, and no
                          dataset is needed beyond its normalization file being readable
Inputs are moved to the device beforehand and each batch is assembled before its clock
starts, so only the sampling is timed; on CUDA every call is synchronized so the time is the
real wall time. --n_warmup untimed calls come first (CUDA setup, kernel selection).

Reported per model: per-call mean, std, min, p50, p90, p95, p99, max (milliseconds), per-window
mean (ms_per_window = mean / batch_size), throughput (windows/s), params, steps and whether fp16
autocast was used (default fp32, or the config's eval_amp, like evaluate.py; --amp / --no-amp to force).

batch 1 measures responsiveness for one aircraft: it is dominated by per-layer launch overhead,
so it grows with depth, not width. Bigger batches give the GPU real work, so they measure
compute cost: there wider / larger models are slower per call but far faster per window.

Results are rewritten to --out after every model (default results/eval/eval_latency.json for
batch 1 real windows, else eval_latency_b<B>[_synthetic].json); a run with the same --out skips
the models already in it, so a job cut off by its time limit resumes. The settings, including
the device name, must match, so numbers from different GPUs or batch sizes never mix.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from DataLoaders.ADSBdataset import get_dataloaders
from models.cfm import euler_sample
from evaluate import load_model, model_name, find_run_folders, pick_device, AMP_DTYPES

SYNTHETIC_POOL_BATCHES = 16   # distinct synthetic batches, cycled


def real_windows(nc_path, split_seed, n_needed, device):
    """Up to n_needed consecutive test windows (obs, t_rel) on device, in a fixed order."""
    _, _, loader = get_dataloaders(nc_path, batch_size=256, seed=split_seed, num_workers=0)
    obs, t_rel, n = [], [], 0
    for batch in loader:
        obs.append(batch["obs"]); t_rel.append(batch["t_rel"]); n += batch["obs"].shape[0]
        if n >= n_needed:
            break
    obs, t_rel = torch.cat(obs)[:n_needed], torch.cat(t_rel)[:n_needed]
    if obs.shape[0] < n_needed:
        print(f"  {obs.shape[0]:,} test windows available for {n_needed:,} needed; batches wrap around")
    return obs.to(device), t_rel.to(device)


def synthetic_windows(n, device):
    """n random windows: N(0,1) normalized features, evenly spaced normalized timestamps."""
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(n, 43, 6, generator=g)
    t_rel = torch.linspace(-1.7, 1.7, 86).expand(n, 86).contiguous()
    return obs.to(device), t_rel.to(device)


def time_model(model, obs, t_rel, device, batch_size, n_calls, n_samples, n_steps, n_warmup, amp):
    """Per-call latencies in ms; call i samples windows [i*B, (i+1)*B) of the pool, wrapping."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    ctx  = (lambda: torch.autocast(device_type=device.type, dtype=AMP_DTYPES[device.type])) if amp \
           else torch.no_grad      # euler_sample already runs under no_grad; this is just a no-op context
    pool = obs.shape[0]
    offsets = torch.arange(batch_size, device=device)

    def batch(i):
        idx = (i * batch_size + offsets) % pool
        return obs.index_select(0, idx), t_rel.index_select(0, idx)

    def run(o, tr):
        with ctx():
            euler_sample(model, o, tr, n_samples=n_samples, n_steps=n_steps, device=str(device),
                         chunk=n_samples * batch_size)   # whole batch in one model call
        sync()                      # GPU calls are asynchronous: wait so the timing is real

    for i in range(n_warmup):       # first calls include CUDA setup / kernel selection
        run(*batch(i))

    times = np.empty(n_calls)
    for i in range(n_calls):
        o, tr = batch(i)
        sync()                      # batch assembly finished before the clock starts
        start = time.perf_counter()
        run(o, tr)
        times[i] = (time.perf_counter() - start) * 1000
        if (i + 1) % 1000 == 0 or i + 1 == n_calls:
            print(f"  {i + 1}/{n_calls} calls | running mean {times[:i + 1].mean():.2f} ms",
                  end="\r", flush=True)
    print()
    return times


def summarize(times, batch_size):
    p50, p90, p95, p99 = np.percentile(times, [50, 90, 95, 99])
    return {"mean_ms": times.mean(), "std_ms": times.std(), "min_ms": times.min(),
            "p50_ms": p50, "p90_ms": p90, "p95_ms": p95, "p99_ms": p99, "max_ms": times.max(),
            "ms_per_window": times.mean() / batch_size,
            "windows_per_s": 1000.0 * batch_size / times.mean()}


def main():
    parser = argparse.ArgumentParser(description="Sampling latency per call for finished checkpoints")
    parser.add_argument("folders", nargs="*", help="run folders (default: every model in finished_ckpt/)")
    parser.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    parser.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--n_calls", "--n_windows", "--n_batches", dest="n_calls", type=int, default=10000,
                        help="timed calls per model")
    parser.add_argument("--batch_size", type=int, default=1, help="windows per call")
    parser.add_argument("--synthetic", action="store_true", help="random windows instead of test windows")
    parser.add_argument("--n_samples", type=int, default=1, help="futures sampled per window (1 = one trajectory)")
    parser.add_argument("--n_steps", type=int, default=20, help="Euler steps (progressive students use their own)")
    parser.add_argument("--n_warmup", type=int, default=50, help="untimed calls before timing")
    parser.add_argument("--device", default="auto", help="auto (CUDA, else MPS, else CPU), cuda, mps or cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="fp16 autocast; default: on for models whose config sets eval_amp (on CUDA), else fp32")
    parser.add_argument("--num_workers", type=int, default=0, help="unused; accepted for savio/job_eval.sh")
    parser.add_argument("--out", default=None,
                        help="JSON file, rewritten after every model; existing entries are skipped (resume). "
                             "Default results/eval/eval_latency.json, or eval_latency_b<B>[_synthetic].json")
    args = parser.parse_args()

    if args.out is None:
        tag = "" if args.batch_size == 1 and not args.synthetic else \
              f"_b{args.batch_size}" + ("_synthetic" if args.synthetic else "")
        args.out = str(ROOT / "results" / "eval" / f"eval_latency{tag}.json")

    folders = [f if Path(f).is_absolute() or Path(f).exists() else ROOT / f
               for f in args.folders] or find_run_folders(args.ckpt)
    missing = [str(f) for f in folders if not Path(f).is_dir()]
    if missing:
        sys.exit("Folder(s) not found: " + ", ".join(missing))
    folders = [run for f in folders for run in (find_run_folders(args.ckpt, f) or [f])]

    device = pick_device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    print(f"Device: {gpu} | {args.n_calls:,} calls × {args.batch_size} "
          f"{'synthetic' if args.synthetic else 'test'} window(s) × {args.n_samples} future(s)")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False      # same kernels every call, steadier timings

    # "n_windows" = timed calls (its name from when every call was one window; kept so older files resume)
    settings = {"ckpt": args.ckpt, "n_windows": args.n_calls, "n_samples": args.n_samples,
                "n_steps": args.n_steps, "n_warmup": args.n_warmup, "amp": args.amp,
                "device": gpu, "nc_path": args.nc_path,
                "batch_size": args.batch_size, "synthetic": args.synthetic}
    out = Path(args.out)
    results, done = {}, {}
    if out.exists():
        prev = json.loads(out.read_text())
        prev_settings = {"batch_size": 1, "synthetic": False, **prev.get("settings", {})}
        if prev_settings != settings:
            sys.exit(f"{out} was made with other settings ({prev_settings}); "
                     f"use another --out, or delete it to start over")
        results, done = prev["results"], prev.get("folders", {})
        print(f"Resuming {out}: {len(results)} model(s) already done")

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps({"settings": settings, "results": results, "folders": done}, indent=2))
        tmp.replace(out)

    pools = {}   # split seed (or "synthetic") → window pool on the device
    for folder in folders:
        res = Path(folder).resolve()
        key = str(res.relative_to(ROOT)) if res.is_relative_to(ROOT) else str(res)
        if key in done.values():
            print(f"\n=== {folder} === already in {out}, skipping")
            continue
        print(f"\n=== {folder} ===")
        model, seed, ckpt_steps = load_model(folder, args.ckpt, device)
        pool_key = "synthetic" if args.synthetic else seed
        if pool_key not in pools:
            pools[pool_key] = synthetic_windows(SYNTHETIC_POOL_BATCHES * args.batch_size, device) \
                if args.synthetic else real_windows(args.nc_path, seed, args.n_calls * args.batch_size, device)
        obs, t_rel = pools[pool_key]
        n_steps = ckpt_steps or args.n_steps
        amp = model.default_amp if args.amp is None else args.amp

        torch.manual_seed(0)
        times = time_model(model, obs, t_rel, device, args.batch_size, args.n_calls,
                           args.n_samples, n_steps, args.n_warmup, amp)
        name = model_name(folder, ckpt_steps)
        results[name] = {"steps": n_steps, "amp": bool(amp), "calls": int(times.shape[0]),
                         "batch_size": args.batch_size,
                         "params": sum(p.numel() for p in model.parameters()),
                         **{k: float(v) for k, v in summarize(times, args.batch_size).items()}}
        done[name] = key
        save()
        r = results[name]
        print(f"  mean {r['mean_ms']:.2f} ms/call | p50 {r['p50_ms']:.2f} | p99 {r['p99_ms']:.2f} | "
              f"{r['ms_per_window']:.3f} ms/window | {r['windows_per_s']:,.0f} windows/s | saved to {out}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        sys.exit("No models timed")
    print(f"\n{gpu} | {args.n_calls:,} calls per model × batch {args.batch_size} "
          f"({'synthetic' if args.synthetic else 'test windows'}) | K={args.n_samples} | "
          f"{args.n_steps} Euler steps (progressive: own) | milliseconds per call unless noted")
    cols = ["params", "steps", "mean_ms", "std_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms",
            "ms_per_window", "windows_per_s"]
    width = max(len(k) for k in results) + 2
    print("model".ljust(width) + "".join(f"{c:>14}" for c in cols))
    for name, r in results.items():
        print(name.ljust(width) + "".join(
            f"{r.get(c, 0):14,.0f}" if c in ("params", "steps", "windows_per_s") else f"{r.get(c, 0):14.3f}"
            for c in cols))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
