#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: CFM + SWI_DiT knowledge distillation, frozen teacher → student
# (pipeline stages 2, 3, 4 and 6). Normally submitted by savio/submit_distill_run.sh,
# which the stage scripts call; settings come in as environment variables (defaults below).
# ---------------------------------------------------------------------------
#SBATCH --job-name=cfm_swi_dit_distill
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

# Call the env's interpreter by path so the job can't fall back to another Python
# (e.g. base anaconda, which has torch but not netCDF4) via the inherited environment.
PYTHON=$HOME/.conda/envs/adsb/bin/python
echo "Python: $PYTHON"

# Fail fast with a clear message if the env or GPU isn't usable.
$PYTHON -c "import torch, numpy, netCDF4; print('torch', torch.__version__, '| netCDF4', netCDF4.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" \
    || { echo "Environment check failed (see .err log)"; exit 1; }

# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
# Repo = the folder sbatch was run from (submit_distill_run.sh submits from the repo root),
# so the job always uses the clone you submitted from, whatever its folder name.
REPO=${REPO:-$SLURM_SUBMIT_DIR}
# The teacher checkpoint is stored with Git LFS: run `git lfs pull` (or copy it in) on a
# fresh clone, or pass TEACHER_CKPT=/path/to/best_swi_dit.pt.
TEACHER_CONFIG=${TEACHER_CONFIG:-configs/teacher_dit_cfm_20M.json}   # or configs/teacher_dit_cfm_7M.json
case "$TEACHER_CONFIG" in /*) ;; *) TEACHER_CONFIG="$REPO/$TEACHER_CONFIG" ;; esac
TEACHER_CKPT=${TEACHER_CKPT:-$REPO/$($PYTHON -c "import json; print(json.load(open('$TEACHER_CONFIG'))['checkpoint'])")}
[ -f "$TEACHER_CKPT" ] || { echo "Teacher checkpoint not found: $TEACHER_CKPT (set TEACHER_CKPT=...)"; exit 1; }

# Per-run settings (override with sbatch --export=ALL,VAR=value,...)
RUN_NAME=${RUN_NAME:?set RUN_NAME (submit with savio/submit_distill_run.sh)}
STUDENT_CONFIG=${STUDENT_CONFIG:?set STUDENT_CONFIG (a configs/students/*.json)}
ALPHA=${ALPHA:-0.7}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}                  # training randomness
SPLIT_SEED=${SPLIT_SEED:-42}      # train/val/test aircraft split: keep 42 everywhere
TURN_WEIGHT=${TURN_WEIGHT:-0}     # extra loss weight for turning windows (0 = off)
TURN_MODE=${TURN_MODE:-continuous}    # continuous | step
TURN_THRESHOLD=${TURN_THRESHOLD:-45}  # degrees, for TURN_MODE=step
VRATE_WEIGHT=${VRATE_WEIGHT:-0}   # extra loss weight for vertical-rate changes (0 = off)
Z_WEIGHT=${Z_WEIGHT:-1}           # altitude-feature loss weight (1 = off)
TIME_WEIGHTING=${TIME_WEIGHTING:-step}   # step (last quarter x2) | linear (continuous 1 -> 2)
NGPUS=${NGPUS:-1}                 # >1 → torchrun + DistributedDataParallel (submit_distill_run.sh --gpus N)
# one output folder per run: runs must never share last.pt / best.pt / metrics.csv
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR (submit_distill_run.sh passes <save_dir>/model)}

echo "Run: $RUN_NAME | student: $STUDENT_CONFIG | gpus=$NGPUS | alpha=$ALPHA | epochs=$EPOCHS | batch=$BATCH_SIZE | seed=$SEED | split_seed=$SPLIT_SEED"
echo "Loss weighting: turn_weight=$TURN_WEIGHT ($TURN_MODE, threshold ${TURN_THRESHOLD}°) | vrate_weight=$VRATE_WEIGHT | z_weight=$Z_WEIGHT | time_weighting=$TIME_WEIGHTING"
echo "Teacher: $TEACHER_CONFIG → $TEACHER_CKPT"
echo "Output: $OUTPUT_DIR"
[ -f "$STUDENT_CONFIG" ] || { echo "Student config not found: $STUDENT_CONFIG"; exit 1; }

mkdir -p $OUTPUT_DIR

# ---------------------------------------------------------------------------
# Output-folder guard: never let two jobs write the same last.pt / best.pt / metrics.csv
# ---------------------------------------------------------------------------
LOCK=$OUTPUT_DIR/.active_job
if [ -f "$LOCK" ]; then
    OTHER=$(cat "$LOCK")
    if [ "$OTHER" != "$SLURM_JOB_ID" ] && squeue -h -j "$OTHER" 2>/dev/null | grep -q .; then
        echo "ERROR: $OUTPUT_DIR is in use by job $OTHER (still queued/running)."
        echo "       Submit this run with a different RUN_NAME."
        exit 1
    fi
elif [ "${ALLOW_RECENT:-0}" != 1 ] && [ "${RESUME:-0}" != 1 ] && \
     [ -n "$(find "$OUTPUT_DIR" -maxdepth 1 \( -name last.pt -o -name metrics.csv \) -mmin -90 2>/dev/null)" ]; then
    # folder written by a job started before this lock existed (e.g. a run that is active right now)
    # (skipped for --resume: submit_distill_run.sh already checked no job of this name is queued/running)
    echo "ERROR: $OUTPUT_DIR was updated in the last 90 min but has no lock — it may belong to an active run."
    echo "       Submit with a different RUN_NAME, or add ALLOW_RECENT=1 if you are sure it's not running."
    exit 1
fi
echo "$SLURM_JOB_ID" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$REPO"

# -u: unbuffered stdout so per-epoch lines appear in logs/ as they happen.
# Per-epoch student metrics are also appended to $OUTPUT_DIR/metrics.csv.
# With NGPUS > 1, torchrun starts one training process per GPU on this node.
if [ "$NGPUS" -gt 1 ]; then
    LAUNCH=($PYTHON -u -m torch.distributed.run --standalone --nproc_per_node=$NGPUS)
else
    LAUNCH=($PYTHON -u)
fi
"${LAUNCH[@]}" training/train_distill.py \
    --nc_path $NC_PATH \
    --teacher_config "$TEACHER_CONFIG" \
    --student_config $STUDENT_CONFIG \
    --teacher_ckpt $TEACHER_CKPT \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --alpha $ALPHA \
    --seed $SEED \
    --split_seed $SPLIT_SEED \
    --turn_weight $TURN_WEIGHT \
    --turn_mode $TURN_MODE \
    --turn_threshold $TURN_THRESHOLD \
    --vrate_weight $VRATE_WEIGHT \
    --z_weight $Z_WEIGHT \
    --time_weighting $TIME_WEIGHTING
STATUS=$?

echo "Job finished at $(date) with exit code $STATUS"
exit $STATUS    # non-zero → Slurm marks the job FAILED instead of COMPLETED
