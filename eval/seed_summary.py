"""
Mean ± std over training seeds, from eval result files.

Runs whose names differ only by a _s<seed> suffix are seeds of one model
(distill_4_256 = seed 42, distill_4_256_s43 = seed 43, ...).

Usage (from the repo root):
    python eval/seed_summary.py                                   # results/eval/full_eval_b64.json
    python eval/seed_summary.py results/eval/full_eval_b64.json results/eval/eval.json
    python eval/seed_summary.py --only distill_4_256 scratch_4_256 gru1_2L_256
    python eval/seed_summary.py --metrics minFDE minFDE_turn avgFDE ms_per_window
    python eval/seed_summary.py --csv results/eval/seed_summary.csv

Reads full_eval.py files (accuracy + latency per model), evaluate.py files (accuracy only) and
latency.py files (latency only). If a run appears in more than one file, the first file given
wins. std is the sample standard deviation (n − 1); with one seed it is shown as "-".
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ["minADE", "minFDE", "minFDE_turn", "avgADE", "avgFDE", "avgFDE_turn", "ms_per_window"]
SEED_RE = re.compile(r"^(.*)_s(\d+)$")


def load_runs(paths):
    """{run name: {metric: value}} merged over the files (first file wins)."""
    runs = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        if "results_m" in d:                       # evaluate.py
            entries = {n: {"accuracy": r} for n, r in d["results_m"].items()}
        else:                                      # full_eval.py / latency.py
            entries = {}
            for n, r in d["results"].items():
                if "accuracy" in r or "latency" in r:
                    entries[n] = r
                else:                              # latency.py: metrics at the top level
                    entries[n] = {"latency": r}
        for name, r in entries.items():
            flat = {**r.get("accuracy", {}), **r.get("latency", {})}
            if "params" in r:
                flat["params"] = r["params"]
            if name in runs:
                missing = {k: v for k, v in flat.items() if k not in runs[name]}
                runs[name].update(missing)         # fill in metrics the first file lacked
            else:
                runs[name] = flat
    return runs


def group_by_model(runs):
    """{model: {seed: metrics}}; a name without _s<seed> is seed 42."""
    models = {}
    for name, metrics in runs.items():
        m = SEED_RE.match(name)
        base, seed = (m.group(1), int(m.group(2))) if m else (name, 42)
        models.setdefault(base, {})[seed] = metrics
    return models


def main():
    p = argparse.ArgumentParser(description="Mean ± std over training seeds")
    p.add_argument("files", nargs="*", default=[str(ROOT / "results" / "eval" / "full_eval_b64.json")])
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--only", nargs="+", default=None, help="model names (without _s<seed>) to show")
    p.add_argument("--min_seeds", type=int, default=1, help="hide models with fewer seeds than this")
    p.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = p.parse_args()

    missing = [f for f in args.files if not Path(f).exists()]
    if missing:
        sys.exit("Not found: " + ", ".join(missing))
    models = group_by_model(load_runs(args.files))
    if args.only:
        unknown = [m for m in args.only if m not in models]
        if unknown:
            sys.exit(f"Not in the files: {', '.join(unknown)} (models: {', '.join(sorted(models))})")
        models = {m: models[m] for m in args.only}
    models = {m: s for m, s in models.items() if len(s) >= args.min_seeds}
    if not models:
        sys.exit("No models to show")

    rows = []
    for model, seeds in models.items():
        row = {"model": model, "n": len(seeds), "seeds": ",".join(map(str, sorted(seeds)))}
        for metric in args.metrics:
            x = np.array([s[metric] for s in seeds.values() if metric in s], dtype=float)
            row[metric] = (x.mean(), x.std(ddof=1) if len(x) > 1 else None, len(x)) if len(x) else None
        rows.append(row)

    width = max(len(r["model"]) for r in rows) + 2
    print(f"Mean ± std over seeds | files: {', '.join(Path(f).name for f in args.files)}")
    print("model".ljust(width) + f"{'n':>3}  {'seeds':<12}" + "".join(f"{m:>22}" for m in args.metrics))
    for r in rows:
        cells = []
        for m in args.metrics:
            v = r[m]
            if v is None:
                cells.append(f"{'n/a':>22}")
                continue
            mean, std, n = v
            prec = 3 if m.endswith("_ms") or m == "ms_per_window" else 1
            txt = f"{mean:.{prec}f} ± {std:.{prec}f}" if std is not None else f"{mean:.{prec}f} ± -"
            if n < r["n"]:
                txt += f" (n={n})"                 # metric missing for some seeds
            cells.append(f"{txt:>22}")
        print(r["model"].ljust(width) + f"{r['n']:>3}  {r['seeds']:<12}" + "".join(cells))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "n_seeds", "seeds"] + [f"{m}_{s}" for m in args.metrics for s in ("mean", "std", "n")])
            for r in rows:
                vals = []
                for m in args.metrics:
                    v = r[m]
                    vals += [v[0], "" if v[1] is None else v[1], v[2]] if v else ["", "", 0]
                w.writerow([r["model"], r["n"], r["seeds"]] + vals)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
