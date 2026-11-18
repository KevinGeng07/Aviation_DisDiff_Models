#!/bin/bash
# ---------------------------------------------------------------------------
# Stepwise distillation of one model: progressive vs. consistency, compared side by side.
# Submits three Slurm jobs:
#   1. progressive_<base>   progressive distillation 32 → 16 → 8 → 4 → 2 → 1 steps   (1 GPU)
#   2. consistency_<base>   consistency distillation                                 (1 GPU)
#   3. compare_<base>       starts when 1 and 2 have both finished OK: eval/benchmark.py on
#                           the base model, every progressive round and the consistency model
#                           at 1/2/4/8/16 steps → results/stepwise_<base>.json + table in its log
#
# Usage (on Savio, from anywhere):
#   bash savio_batchfiles/submit_stepwise_distill.sh            # base 6_4_256
#   bash savio_batchfiles/submit_stepwise_distill.sh 6_4_128
#   CD_N=64 CD_LOSS=huber bash savio_batchfiles/submit_stepwise_distill.sh   # CD settings
# Any setting of StepwiseDistillSavio.sh (CD_EPOCHS, CD_N, LR, ...) can be set the same way.
# Re-running resumes: finished rounds / stages are skipped, unfinished ones continue.
# ---------------------------------------------------------------------------
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p logs results

BASE=${1:-6_4_256}
BASE_DIR=finished_ckpt/SWi_DiT/distill_$BASE
[ -d "$BASE_DIR" ] || echo "Note: $BASE_DIR not found; the job will try the older layouts for '$BASE'."
TIME_LIMIT=${TIME_LIMIT:-20:00:00}

submit() {   # $1 = method → prints the job id
    local name=${1}_$BASE dir=finished_ckpt/SWi_DiT/${1}_$BASE
    if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
        echo "ERROR: a job named $name is already queued/running" >&2; exit 1
    fi
    mkdir -p "$dir/logs"
    sbatch --parsable --job-name="$name" --time="$TIME_LIMIT" \
           --output="$dir/logs/%x_%j.out" --error="$dir/logs/%x_%j.err" \
           --export=ALL,METHOD="$1",BASE="$BASE",RUN_NAME="$name" \
           savio_batchfiles/StepwiseDistillSavio.sh
}

PD=$(submit progressive)
CD=$(submit consistency)
echo "progressive_$BASE  → job $PD   (finished_ckpt/SWi_DiT/progressive_$BASE/)"
echo "consistency_$BASE  → job $CD   (finished_ckpt/SWi_DiT/consistency_$BASE/)"

# Comparison: progressive rounds run at their own step count; the base and consistency
# models run at every --n_steps value (named <model>_<N>step).
CMP=$(SCRIPT=benchmark EVAL_ARGS="--n_steps 1 2 4 8 16 --out results/stepwise_$BASE.json" \
      sbatch --parsable --job-name="compare_$BASE" --dependency=afterok:$PD:$CD --time=12:00:00 \
             --output="logs/compare_${BASE}_%j.out" --error="logs/compare_${BASE}_%j.err" \
             savio_batchfiles/EvalSavio.sh "$BASE_DIR" \
             "finished_ckpt/SWi_DiT/progressive_$BASE" "finished_ckpt/SWi_DiT/consistency_$BASE")
echo "compare_$BASE      → job $CMP  (waits for both; results/stepwise_$BASE.json, logs/compare_${BASE}_$CMP.out)"

squeue -u "$USER" -o "%.10i %.24j %.3t %.10M %.12a %R"
