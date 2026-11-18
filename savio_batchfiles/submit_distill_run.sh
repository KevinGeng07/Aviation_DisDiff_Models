#!/bin/bash
# ---------------------------------------------------------------------------
# Submit ONE distillation run with custom parameters as its own Slurm job (1 GPU by default).
# Safe to use while other runs are training: it refuses to reuse an existing
# output folder or a job name that is already queued/running.
#
# Usage (on Savio, from anywhere):
#   bash savio_batchfiles/submit_distill_run.sh --name NAME --student CONFIG \
#        [--alpha 0.7] [--turn_weight 0] [--time_weighting step|linear] [--epochs 100] [--batch_size 256] [--seed 42] [--save_dir DIR] [--gpus 1] [--resume]
#
#   --name     unique run name → Slurm job name
#   --save_dir run folder (relative to the repo or absolute; default finished_ckpt/SWi_DiT/NAME)
#              checkpoints/metrics → DIR/model/, Slurm .out/.err → DIR/logs/NAME_<jobid>.out
#   --student  student config JSON (path relative to the repo, e.g. configs/students/swi_dit_2L_256.json)
#   --turn_weight  extra loss weight for turning windows (>= 90 deg turn counts 1 + this; 0 = off)
#   --time_weighting  loss weight over the future horizon: step (last quarter x2, default) | linear (1 -> 2 ramp)
#   --gpus     GPUs on one node (default 1). >1 trains with DistributedDataParallel; --batch_size
#              stays the total batch (split over the GPUs). Requests 2 CPUs per GPU; override the
#              GPU type / CPUs with GPU_TYPE=A40 CPUS_PER_GPU=8 (plus matching sbatch --partition/--qos)
#   --resume   continue an existing run of the same NAME from its last.pt
#
# Example:
#   bash savio_batchfiles/submit_distill_run.sh --name distill_2L_256_a05 \
#        --student configs/students/swi_dit_2L_256.json --alpha 0.5
# ---------------------------------------------------------------------------
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p logs        # Slurm writes logs/ relative to the submit directory

NAME=""; STUDENT=""; ALPHA=0.7; TURN_WEIGHT=0; TIME_WEIGHTING=step; EPOCHS=100; BATCH_SIZE=256; SEED=42; SAVE_DIR=""; RESUME=0; GPUS=1
usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
        --name)       NAME="$2"; shift 2 ;;
        --student)    STUDENT="$2"; shift 2 ;;
        --alpha)      ALPHA="$2"; shift 2 ;;
        --turn_weight) TURN_WEIGHT="$2"; shift 2 ;;
        --time_weighting) TIME_WEIGHTING="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --save_dir)   SAVE_DIR="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --resume)     RESUME=1; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[ -n "$NAME" ] && [ -n "$STUDENT" ] || { echo "ERROR: --name and --student are required."; usage; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --gpus must be a positive integer"; exit 1; }
(( BATCH_SIZE % GPUS == 0 )) || { echo "ERROR: --batch_size $BATCH_SIZE must divide evenly over $GPUS GPUs"; exit 1; }
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: --name may only contain letters, digits, _ . -"; exit 1; }
case "$STUDENT" in /*) ;; *) STUDENT="$REPO/$STUDENT" ;; esac
[ -f "$STUDENT" ] || { echo "ERROR: student config not found: $STUDENT"; exit 1; }

SAVE_DIR=${SAVE_DIR:-finished_ckpt/SWi_DiT/$NAME}
case "$SAVE_DIR" in /*) ;; *) SAVE_DIR="$REPO/$SAVE_DIR" ;; esac
OUT="$SAVE_DIR/model"
LOGDIR="$SAVE_DIR/logs"
if [ -e "$OUT" ] && [ "$RESUME" != 1 ]; then
    echo "ERROR: $OUT already exists (an earlier or active run)."
    echo "       Pick a new --name, or add --resume to continue that run."
    exit 1
fi
if squeue -h -u "$USER" -n "$NAME" 2>/dev/null | grep -q .; then
    echo "ERROR: a job named '$NAME' is already queued/running:"
    squeue -u "$USER" -n "$NAME"
    exit 1
fi

mkdir -p "$LOGDIR"   # Slurm will not create the --output directory itself

echo "Save dir: $SAVE_DIR  (model/ + logs/)"
echo "Submitting $NAME | gpus=$GPUS | student=$STUDENT | alpha=$ALPHA | turn_weight=$TURN_WEIGHT | time_weighting=$TIME_WEIGHTING | epochs=$EPOCHS | batch=$BATCH_SIZE | seed=$SEED | resume=$RESUME"
# Time limit given explicitly: Savio rejects jobs without one, and the job script's
# #SBATCH --time is not always picked up. Override with TIME_LIMIT=HH:MM:SS.
sbatch --job-name="$NAME" --time="${TIME_LIMIT:-20:00:00}" \
       --gres="gpu:${GPU_TYPE:-GTX2080TI}:$GPUS" --cpus-per-task=$(( ${CPUS_PER_GPU:-2} * GPUS )) \
       --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
       --export=ALL,NGPUS="$GPUS",RUN_NAME="$NAME",OUTPUT_DIR="$OUT",STUDENT_CONFIG="$STUDENT",ALPHA="$ALPHA",TURN_WEIGHT="$TURN_WEIGHT",TIME_WEIGHTING="$TIME_WEIGHTING",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",SEED="$SEED" \
       savio_batchfiles/CFMSwiDITDistillSavio.sh
