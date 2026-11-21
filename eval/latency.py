"""
Inference latency per model: time to sample futures for ONE test window, over many windows.

Usage (on Savio, from the repo root):
    SCRIPT=latency sbatch savio/job_eval.sh finished_ckpt/2_architecture   # → results/eval/eval_latency.json
    SCRIPT=latency sbatch savio/job_eval.sh finished_ckpt/1_teacher/dit_cfm_20M finished_ckpt/2_architecture
    SCRIPT=latency EVAL_ARGS="--n_samples 20" sbatch savio/job_eval.sh ...   # K=20 futures per call
    python eval/latency.py finished_ckpt/2_architecture/distill_4_256 --n_windows 500 --device mps

For every model, each of the first --n_windows (default 10,000) test windows is one timed call:
batch size 1, --n_samples futures (default 1), --n_steps Euler steps (default 20;
progressive students use their own). The inputs are moved to the device beforehand, so only
the sampling is timed; on CUDA every call is synchronized so the time is the real wall time.
--n_warmup untimed calls come first (CUDA setup, kernel selection).

Reported per model (milliseconds): mean, std, min, p50, p90, p95, p99, max, plus throughput
(windows/s), params, steps and whether fp16 autocast was used (default from the model config,
like evaluate.py; --amp / --no-amp to force). Same windows for every model.

Results are rewritten to --out after every model (default results/eval/eval_latency.json); a run
with the same --out skips the models already in it, so a job cut off by its time limit resumes.
The settings, including the device name, must match, so numbers from different GPUs never mix.
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


def test_windows(nc_path, split_seed, n_windows, device):
    """The first n_windows test windows (obs, t_rel) as tensors on device, in a fixed order."""
    _, _, loader = get_dataloaders(nc_path, batch_size=256, seed=split_seed, num_workers=0)
    obs, t_rel, n = [], [], 0
    for batch in loader:
        obs.append(batch["obs"]); t_rel.append(batch["t_rel"]); n += batch["obs"].shape[0]
        if n >= n_windows:
            break
    obs, t_rel = torch.cat(obs)[:n_windows], torch.cat(t_rel)[:n_windows]
    if obs.shape[0] < n_windows:
        print(f"  only {obs.shape[0]} test windows available; timing all of them")
    return obs.to(device), t_rel.to(device)


def time_model(model, obs, t_rel, device, n_samples, n_steps, n_warmup, amp):
    """Per-window latencies in ms: one euler_sample call per window, batch size 1."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    ctx  = (lambda: torch.autocast(device_type=device.type, dtype=AMP_DTYPES[device.type])) if amp \
           else torch.no_grad      # euler_sample already runs under no_grad; this is just a no-op context

    def run(i):
        with ctx():
            euler_sample(model, obs[i:i + 1], t_rel[i:i + 1], n_samples=n_samples,
                         n_steps=n_steps, device=str(device))
        sync()                      # GPU calls are asynchronous: wait so the timing is real

    for i in range(n_warmup):       # first calls include CUDA setup / kernel selection
        run(i % obs.shape[0])

    n, times = obs.shape[0], np.empty(obs.shape[0])
    for i in range(n):
        start = time.perf_counter()
        run(i)
        times[i] = (time.perf_counter() - start) * 1000
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} windows | running mean {times[:i + 1].mean():.2f} ms",
                  end="\r", flush=True)
    print()
    return times


def summarize(times):
    p50, p90, p95, p99 = np.percentile(times, [50, 90, 95, 99])
    return {"mean_ms": times.mean(), "std_ms": times.std(), "min_ms": times.min(),
            "p50_ms": p50, "p90_ms": p90, "p95_ms": p95, "p99_ms": p99, "max_ms": times.max(),
            "windows_per_s": 1000.0 / times.mean()}


def main():
    parser = argparse.ArgumentParser(description="Per-window sampling latency for finished checkpoints")
    parser.add_argument("folders", nargs="*", help="run folders (default: every model in finished_ckpt/)")
    parser.add_argument("--nc_path", default=str(ROOT / "Data" / "trajectories_adsblol_seq86_stage2.nc"))
    parser.add_argument("--ckpt", default="best.pt", help="best.pt or last.pt")
    parser.add_argument("--n_windows", type=int, default=10000, help="timed calls per model, one test window each")
    parser.add_argument("--n_samples", type=int, default=1, help="futures sampled per call (1 = one trajectory)")
    parser.add_argument("--n_steps", type=int, default=20, help="Euler steps (progressive students use their own)")
    parser.add_argument("--n_warmup", type=int, default=50, help="untimed calls before timing")
    parser.add_argument("--device", default="auto", help="auto (CUDA, else MPS, else CPU), cuda, mps or cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="fp16 autocast; default: on for models whose config says amp (on CUDA)")
    parser.add_argument("--num_workers", type=int, default=0, help="unused; accepted for savio/job_eval.sh")
    parser.add_argument("--out", default=str(ROOT / "results" / "eval" / "eval_latency.json"),
                        help="JSON file, rewritten after every model; existing entries are skipped (resume)")
    args = parser.parse_args()

    folders = [f if Path(f).is_absolute() or Path(f).exists() else ROOT / f
               for f in args.folders] or find_run_folders(args.ckpt)
    missing = [str(f) for f in folders if not Path(f).is_dir()]
    if missing:
        sys.exit("Folder(s) not found: " + ", ".join(missing))
    folders = [run for f in folders for run in (find_run_folders(args.ckpt, f) or [f])]

    device = pick_device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    print(f"Device: {gpu}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False      # same kernels every call, steadier timings

    settings = {"ckpt": args.ckpt, "n_windows": args.n_windows, "n_samples": args.n_samples,
                "n_steps": args.n_steps, "n_warmup": args.n_warmup, "amp": args.amp,
                "device": gpu, "nc_path": args.nc_path}
    out = Path(args.out)
    results, done = {}, {}
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("settings") != settings:
            sys.exit(f"{out} was made with other settings ({prev.get('settings')}); "
                     f"use another --out, or delete it to start over")
        results, done = prev["results"], prev.get("folders", {})
        print(f"Resuming {out}: {len(results)} model(s) already done")

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps({"settings": settings, "results": results, "folders": done}, indent=2))
        tmp.replace(out)

    windows = {}   # split seed → test windows on the device
    for folder in folders:
        res = Path(folder).resolve()
        key = str(res.relative_to(ROOT)) if res.is_relative_to(ROOT) else str(res)
        if key in done.values():
            print(f"\n=== {folder} === already in {out}, skipping")
            continue
        print(f"\n=== {folder} ===")
        model, seed, ckpt_steps = load_model(folder, args.ckpt, device)
        if seed not in windows:
            windows[seed] = test_windows(args.nc_path, seed, args.n_windows, device)
        obs, t_rel = windows[seed]
        n_steps = ckpt_steps or args.n_steps
        amp = model.default_amp if args.amp is None else args.amp

        torch.manual_seed(0)
        times = time_model(model, obs, t_rel, device, args.n_samples, n_steps, args.n_warmup, amp)
        name = model_name(folder, ckpt_steps)
        results[name] = {"steps": n_steps, "amp": bool(amp), "windows": int(times.shape[0]),
                         "params": sum(p.numel() for p in model.parameters()),
                         **{k: float(v) for k, v in summarize(times).items()}}
        done[name] = key
        save()
        r = results[name]
        print(f"  mean {r['mean_ms']:.2f} ms | p50 {r['p50_ms']:.2f} | p99 {r['p99_ms']:.2f} | "
              f"{r['windows_per_s']:.1f} windows/s | saved to {out}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        sys.exit("No models timed")
    print(f"\n{gpu} | {args.n_windows:,} windows per model, batch 1 | K={args.n_samples} | "
          f"{args.n_steps} Euler steps (progressive: own) | milliseconds")
    cols = ["params", "steps", "mean_ms", "std_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms", "windows_per_s"]
    width = max(len(k) for k in results) + 2
    print("model".ljust(width) + "".join(f"{c:>14}" for c in cols))
    for name, r in results.items():
        print(name.ljust(width) + "".join(f"{r[c]:14,.0f}" if c in ("params", "steps") else f"{r[c]:14.2f}"
                                          for c in cols))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
