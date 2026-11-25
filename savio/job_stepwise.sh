#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: stepwise distillation of a CFM + SWI_DiT model (pipeline stage 5, stepwise_distillation), 1 GPU
#   METHOD=progressive  (default) halve the Euler step count round by round, 32 → 1
#   METHOD=consistency  one stage: learn to jump to the end of the teacher's path
#
# Easiest: savio/stage5_stepwise_distillation.sh submits both + the comparison.
# By hand (from the repo root; settings via --export, defaults below):
#   sbatch --job-name=<name> --export=ALL,METHOD=consistency,BASE=4_256 \
#          savio/job_stepwise.sh
#
# BASE is any model spec accepted by train_stepwise.py --base:
# a shorthand like 4_256 (→ finished_ckpt/2_architecture/distill_4_256), 'teacher', a run folder or a .pt.
# Output: finished_ckpt/5_stepwise_distillation/$RUN_NAME/model (progressive: one steps_NN folder per round).
# Resubmitting with the same RUN_NAME resumes (finished rounds / stages are skipped).
# ---------------------------------------------------------------------------
#SBATCH --job-name=stepwise_distill
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out      # %x = job name, %j = job id
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p logs

module load anaconda3
source activate adsb

PYTHON=$HOME/.conda/envs/adsb/bin/python
echo "Python: $PYTHON"

$PYTHON -c "import torch, numpy, netCDF4; print('torch', torch.__version__, '| netCDF4', netCDF4.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" \
    || { echo "Environment check failed (see .err log)"; exit 1; }

# ---------------------------------------------------------------------------
# Run settings (override with sbatch --export=ALL,VAR=value,...)
# ---------------------------------------------------------------------------
NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
# Repo = the folder sbatch was run from, so the job uses the clone you submitted from
REPO=${REPO:-$SLURM_SUBMIT_DIR}

METHOD=${METHOD:-progressive}
BASE=${BASE:-4_256}
RUN_NAME=${RUN_NAME:-${METHOD}_$BASE}
LR=${LR:-5e-5}
MIN_LR_FRAC=${MIN_LR_FRAC:-0.2}
START_STEPS=${START_STEPS:-32}       # progressive: round-1 teacher steps; consistency: teacher reference only
MIN_STEPS=${MIN_STEPS:-1}            # progressive only
EPOCHS_PER_ROUND=${EPOCHS_PER_ROUND:-10}   # progressive only
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}                     # training randomness
SPLIT_SEED=${SPLIT_SEED:-42}         # train/val/test aircraft split: must match the base model's
TURN_WEIGHT=${TURN_WEIGHT:-0}        # extra loss weight for turning windows (0 = off)
TURN_MODE=${TURN_MODE:-continuous}   # continuous | step
TURN_THRESHOLD=${TURN_THRESHOLD:-45}
Z_WEIGHT=${Z_WEIGHT:-1}              # altitude-feature loss weight (1 = off)
# consistency only (see `python training/train_stepwise.py --help`)
CD_EPOCHS=${CD_EPOCHS:-20}
CD_N=${CD_N:-32}
CD_SOLVER=${CD_SOLVER:-heun}
CD_TARGET_DECAY=${CD_TARGET_DECAY:-0.95}
CD_LOSS=${CD_LOSS:-l2}
CD_EVAL_STEPS=${CD_EVAL_STEPS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/finished_ckpt/5_stepwise_distillation/$RUN_NAME/model}

echo "Run: $RUN_NAME | method: $METHOD | base: $BASE | lr=$LR (floor ×$MIN_LR_FRAC) | batch=$BATCH_SIZE | seed=$SEED | split_seed=$SPLIT_SEED"
echo "Loss weighting: turn_weight=$TURN_WEIGHT ($TURN_MODE, threshold ${TURN_THRESHOLD}°) | z_weight=$Z_WEIGHT"
if [ "$METHOD" = consistency ]; then
    echo "Consistency: epochs=$CD_EPOCHS | N=$CD_N | solver=$CD_SOLVER | target_decay=$CD_TARGET_DECAY | loss=$CD_LOSS | val steps=$CD_EVAL_STEPS"
else
    echo "Progressive: steps $START_STEPS → $MIN_STEPS | epochs/round=$EPOCHS_PER_ROUND"
fi
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Output-folder guard: never let two jobs write the same folder
# ---------------------------------------------------------------------------
LOCK=$OUTPUT_DIR/.active_job
if [ -f "$LOCK" ]; then
    OTHER=$(cat "$LOCK")
    if [ "$OTHER" != "$SLURM_JOB_ID" ] && squeue -h -j "$OTHER" 2>/dev/null | grep -q .; then
        echo "ERROR: $OUTPUT_DIR is in use by job $OTHER (still queued/running)."
        echo "       Submit this run with a different RUN_NAME."
        exit 1
    fi
fi
echo "$SLURM_JOB_ID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$REPO"

# -u: unbuffered stdout so per-epoch lines appear in the log as they happen.
$PYTHON -u training/train_stepwise.py \
    --nc_path $NC_PATH \
    --method $METHOD \
    --base $BASE \
    --output_dir "$OUTPUT_DIR" \
    --lr $LR \
    --min_lr_frac $MIN_LR_FRAC \
    --start_steps $START_STEPS \
    --min_steps $MIN_STEPS \
    --epochs_per_round $EPOCHS_PER_ROUND \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --split_seed $SPLIT_SEED \
    --turn_weight $TURN_WEIGHT \
    --turn_mode $TURN_MODE \
    --turn_threshold $TURN_THRESHOLD \
    --z_weight $Z_WEIGHT \
    --cd_epochs $CD_EPOCHS \
    --cd_n $CD_N \
    --cd_solver $CD_SOLVER \
    --cd_target_decay $CD_TARGET_DECAY \
    --cd_loss $CD_LOSS \
    --cd_eval_steps $CD_EVAL_STEPS
STATUS=$?

echo "Job finished at $(date) with exit code $STATUS"
exit $STATUS    # non-zero → Slurm marks the job FAILED instead of COMPLETED
