#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 5 — stepwise distillation of the selected model (the best stage 4 run): progressive vs. consistency.
# Submits three Slurm jobs:
#   1. progressive_<base>   progressive distillation 32 → 16 → 8 → 4 → 2 → 1 steps   (1 GPU)
#   2. consistency_<base>   consistency distillation                                 (1 GPU)
#   3. compare_<base>       starts when 1 and 2 have both finished OK: eval/benchmark.py on
#                           the base model, every progressive round and the consistency model
#                           at 1/2/4/8/16 steps → results/stage5_<base>.json + table in its log
# Output: finished_ckpt/5_stepwise_distillation/{progressive,consistency}_<base>[_s<seed>]/
#
# Usage (on Savio, from anywhere):
#   bash savio/stage5_stepwise_distillation.sh                                  # base 4_256
#   bash savio/stage5_stepwise_distillation.sh finished_ckpt/4_path_weighting/<best run>   # any run folder as base
#   for s in 42 43 44; do SEED=$s bash savio/stage5_stepwise_distillation.sh; done  # 3 seeds
#   CD_N=64 CD_LOSS=huber bash savio/stage5_stepwise_distillation.sh            # consistency settings
# Any setting of savio/job_stepwise.sh (CD_EPOCHS, CD_N, LR, TURN_WEIGHT, Z_WEIGHT, ...) can be
# set the same way. Re-running resumes: finished rounds / stages are skipped.
# ---------------------------------------------------------------------------
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p logs results

BASE=${1:-4_256}
if [ -d "$BASE" ]; then                     # a run folder: label it by its name
    BASE_DIR=$BASE; LABEL=$(basename "$BASE"); LABEL=${LABEL#distill_}
else                                        # a shorthand like 4_256
    BASE_DIR=finished_ckpt/2_architecture/distill_$BASE; LABEL=$BASE
fi
[ -d "$BASE_DIR" ] || { echo "ERROR: base model folder $BASE_DIR not found"; exit 1; }
SEED=${SEED:-42}
SUFFIX=$([ "$SEED" = 42 ] || echo "_s$SEED")
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
OUT=finished_ckpt/5_stepwise_distillation

submit() {   # $1 = method → prints the job id
    local name=${1}_$LABEL$SUFFIX
    if squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
        echo "ERROR: a job named $name is already queued/running" >&2; exit 1
    fi
    mkdir -p "$OUT/$name/logs"
    sbatch --parsable --job-name="$name" --time="$TIME_LIMIT" \
           --output="$OUT/$name/logs/%x_%j.out" --error="$OUT/$name/logs/%x_%j.err" \
           --export=ALL,METHOD="$1",BASE="$BASE_DIR",RUN_NAME="$name",SEED="$SEED",OUTPUT_DIR="$REPO/$OUT/$name/model" \
           savio/job_stepwise.sh
}

PD=$(submit progressive)
CD=$(submit consistency)
echo "progressive_$LABEL$SUFFIX  → job $PD   ($OUT/progressive_$LABEL$SUFFIX/)"
echo "consistency_$LABEL$SUFFIX  → job $CD   ($OUT/consistency_$LABEL$SUFFIX/)"

# Comparison: progressive rounds run at their own step count; the base and consistency
# models run at every --n_steps value (named <model>_<N>step).
CMP=$(SCRIPT=benchmark EVAL_ARGS="--n_steps 1 2 4 8 16 --out results/stage5_$LABEL$SUFFIX.json" \
      sbatch --parsable --job-name="compare_$LABEL$SUFFIX" --dependency=afterok:$PD:$CD --time=12:00:00 \
             --output="logs/compare_$LABEL${SUFFIX}_%j.out" --error="logs/compare_$LABEL${SUFFIX}_%j.err" \
             savio/job_eval.sh "$BASE_DIR" "$OUT/progressive_$LABEL$SUFFIX" "$OUT/consistency_$LABEL$SUFFIX")
echo "compare_$LABEL$SUFFIX      → job $CMP  (waits for both; results/stage5_$LABEL$SUFFIX.json)"

squeue -u "$USER" -o "%.10i %.24j %.3t %.10M %.12a %R"
