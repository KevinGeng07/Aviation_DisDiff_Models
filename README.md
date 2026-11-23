# Aviation DisDiff Models

Probabilistic **aircraft trajectory prediction** (San Francisco Bay Area ADS-B) with
**conditional flow matching (CFM)**. Given 43 observed steps (~2 min) of an aircraft's track,
a model samples many plausible 43-step futures. Every model is a **SWI_DiT**: a Diffusion
Transformer with SwiGLU feed-forward blocks and AdaLN conditioning on the flow time.

The work is one pipeline: train (or reuse) a large teacher, distil it into smaller students,
test architectural alternatives, cut the number of sampling steps, and finally re-weight the
loss towards the hard cases (turns, altitude).

| Stage | What | Submit (on Savio) | Output (`finished_ckpt/…`) |
|---|---|---|---|
| 1 | Teacher: CITRIS DiT, 20.7 M (default) or 7.1 M — their `best.pt` | `bash savio/stage1_teacher.sh [--retrain] [config]` | `1_teacher/dit_cfm_{20M,7M}/` |
| 2 | Architecture distillation: 7 students (2/4/6 layers × 128/256/384) | `bash savio/stage2_architectures.sh` | `2_architecture/distill_<L>_<d>/` |
| 3 | GRU alternatives at d_model 256 (3 designs) | `bash savio/stage3_gru.sh` | `3_gru/gru<k>_…/` |
| 4 | RoPE position embeddings (token index / real timestamps) | `bash savio/stage4_rope.sh` | `4_rope/rope_{index,ts}_4L_256/` |
| 5 | Stepwise distillation: progressive vs. consistency | `bash savio/stage5_stepwise.sh [base]` | `5_stepwise/{progressive,consistency}_<base>/` |
| 6 | Loss weighting: continuous / discrete turn weighting, Z-weighting | `bash savio/stage6_weighting.sh` | `6_weighting/<student>_<variant>/` |

Every stage script takes `SEED=<n>` for extra seeds (folders get a `_s<n>` suffix). The data
split is fixed separately (`--split_seed 42`), so all runs and all evaluations share one test set.

---

## Repository layout

```
configs/
  teacher_dit_cfm_{7M,20M}.json  CITRIS DiT teachers (finished_ckpt/1_teacher/dit_cfm_*/best.pt)
  students/                    one JSON per student: swi_dit_<L>L_<d> (stage 2), gru<k>_… (3),
                               rope_{index,ts}_4L_256 (4); "arch" picks the model file
models/
  Swi_DiT.py                   the SWI_DiT backbone (teacher and stage-2 students)
  GRU_Swi_DiT{,_2,_3}.py       stage-3 GRU variants
  Swi_DiT_RoPE.py              stage-4 RoPE variant (rope = "index" | "timestamp")
  DiT_CFM.py                   CITRIS Diffusion Transformer (TrajectoryFlowV5) — the 7.1 M / 20.7 M teachers
  registry.py                  config "arch" → model class
  cfm.py                       flow-time sampling, interpolation, Euler sampler
training/
  train_teacher.py             stage 1: CFM from scratch
  train_distill.py             stages 2/3/4/6: teacher → student (1 or more GPUs)
  train_stepwise.py            stage 5: progressive / consistency distillation
  loss_weighting.py            turn weighting (continuous / step) and Z-weighting
DataLoaders/ADSBdataset.py     load the .nc, normalize, split by aircraft, DataLoaders
Data/ADSBnetcdfbuilder.py      raw ADS-B CSVs → windowed .nc dataset
eval/
  evaluate.py                  test-set minADE/minFDE/avg…, turning-window and p1/p99 metrics
  benchmark.py                 accuracy + parameter count + latency, optionally per step count
  error_by_step.py             per-axis (X/Y/Z) error over the 43-step horizon + plots
  sample_futures.py            sampled futures for a few windows → CSV (eval/plots/ draws them)
savio/
  stage{1..6}_*.sh             submit one pipeline stage
  submit_distill_run.sh        submit one distillation run (used by stages 2/3/4/6)
  job_{teacher,distill,stepwise,eval}.sh   the Slurm job scripts
finished_ckpt/
  <stage>/<run>/model/         best.pt, last.pt, metrics.csv, student_config.json
  <stage>/<run>/logs/          Slurm .out / .err
  compress_ckpts.py            split .pt files over 99 MB into compressed parts (for GitHub);
  decompress_ckpts.py          eval reads the parts directly, decompress only to train on
checkpoints/swi_dit/           the earlier SWI_DiT teacher (Git LFS; no longer used by default)
results/                       eval / benchmark JSON (results/legacy/: earlier experiments)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch numpy netCDF4 matplotlib
# teachers: CITRIS best.pt files at finished_ckpt/1_teacher/dit_cfm_{20M,7M}/best.pt
```

On Savio the jobs use the `adsb` conda env and read the dataset from
`/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc`
(`NC_PATH` in `savio/job_*.sh`). Submit from the repo root; the jobs run in the folder they
were submitted from. Default account `ac_mixedav`; use `SBATCH_ACCOUNT=fc_mixedav` to charge
another one.

## Data

`Data/ADSBnetcdfbuilder.py` turns the raw ADS-B CSVs (`Data/adsb/`, gitignored) into
`trajectories_adsblol_seq86_stage2.nc`: windows of 86 steps (43 observed + 43 future) with
features `x, y, z` (metres, local Cartesian) and `vx, vy, vz` (m/s), timestamps, the aircraft
ICAO24 and normalization statistics. `DataLoaders/ADSBdataset.py` normalizes the features,
turns timestamps into normalized `t_rel`, and splits **by aircraft** (85 / 10 / 5 % train /
val / test, `split_seed` 42), so no test aircraft is ever trained on.

## Model and training objective

Tokens: the 43 observed steps and the 43 noisy future steps, each projected to `d_model`, plus
position information (learned position embedding + a `t_rel` MLP; stage 4 swaps these for
RoPE). Observed tokens cannot attend to future tokens. The flow time `t` conditions every
block through AdaLN; the head predicts a velocity for the 43 future tokens.

CFM: `x_t = (1 − t)·noise + t·future`, target velocity `future − noise`, `t` from a
logit-normal; sampling integrates the predicted velocity with Euler steps from noise (`t = 0`)
to data (`t = 1`), 20 steps unless stated. The loss weights the last quarter of the horizon ×2.

Distillation (stages 2–4, 6): the student matches `0.7 ×` the ground-truth CFM loss and
`0.3 ×` the frozen teacher's velocity, each normalized by its running size. Stage 5 instead
trains the selected student to sample in fewer steps: progressive distillation halves the
step count round by round (32 → 1); consistency distillation learns
`f(x_t, t) = x_t + (1 − t)·v(x_t, t)` to jump straight to the end of the teacher's path.
See the docstrings of `training/train_*.py` for every option.

## Evaluation

```bash
python eval/evaluate.py                                  # every model in finished_ckpt/
python eval/evaluate.py finished_ckpt/3_gru --device mps # one stage, on a Mac GPU
SCRIPT=benchmark sbatch savio/job_eval.sh finished_ckpt/2_architecture
SCRIPT=error_by_step sbatch savio/job_eval.sh            # per-axis error plots for 4_256
```

For each test window the model samples K = 20 futures. minADE / minFDE score the best of the
K, avgADE / avgFDE the average; `_turn` metrics use only windows turning ≥ 45°, and `_p1` /
`_p99` give the spread of the per-window minADE / minFDE. Consistency models store no step
count: pass `--n_steps` (or `benchmark.py --n_steps 1 2 4 8 16`).

## Notes

- 100 epochs of a 4-layer student take ~22 h on a GTX 2080 Ti; `submit_distill_run.sh`
  requests 24 h. A run cut off by the time limit continues with `--resume`.
- `submit_distill_run.sh --gpus 2` trains one run on two GPUs (`--batch_size` stays the total).
- The 20.7 M teacher runs in fp16 autocast on CUDA during distillation (`"amp": true` in its
  config). Evaluation samples every model in fp32 unless its config sets `"eval_amp": true` or
  `--amp` is given. On CPU / MPS it stays in full precision, where half precision is slower.
- The old SWI_DiT teacher checkpoint is a Git LFS file: without `git lfs pull` it is a 130-byte pointer and
  loading fails with `invalid load key, 'v'`.
- `Swi_DiT.py` and the GRU files define a top-level `adaLN` that `forward()` never uses; it is
  kept so existing checkpoints load, and counts towards their parameter numbers.
