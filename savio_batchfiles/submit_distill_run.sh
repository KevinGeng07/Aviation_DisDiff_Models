#!/bin/bash
# ---------------------------------------------------------------------------
# Submit ONE distillation run with custom parameters as its own 1-GPU Slurm job.
# Safe to use while other runs are training: it refuses to reuse an existing
# output folder or a job name that is already queued/running.
#
# Usage (on Savio, from anywhere):
#   bash savio_batchfiles/submit_distill_run.sh --name NAME --student CONFIG \
#        [--alpha 0.7] [--turn_weight 0] [--time_weighting step|linear] [--epochs 100] [--batch_size 256] [--seed 42] [--save_dir DIR] [--resume]
#
#   --name     unique run name → Slurm job name
#   --save_dir run folder (relative to the repo or absolute; default finished_ckpt/SWi_DiT/NAME)
#              checkpoints/metrics → DIR/model/, Slurm .out/.err → DIR/logs/NAME_<jobid>.out
#   --student  student config JSON (path relative to the repo, e.g. configs/students/swi_dit_2L_256.json)
#   --turn_weight  extra loss weight for turning windows (>= 90 deg turn counts 1 + this; 0 = off)
#   --time_weighting  loss weight over the future horizon: step (last quarter x2, default) | linear (1 -> 2 ramp)
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

NAME=""; STUDENT=""; ALPHA=0.7; TURN_WEIGHT=0; TIME_WEIGHTING=step; EPOCHS=100; BATCH_SIZE=256; SEED=42; SAVE_DIR=""; RESUME=0
usage() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
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
        --resume)     RESUME=1; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[ -n "$NAME" ] && [ -n "$STUDENT" ] || { echo "ERROR: --name and --student are required."; usage; }
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
echo "Submitting $NAME | student=$STUDENT | alpha=$ALPHA | turn_weight=$TURN_WEIGHT | time_weighting=$TIME_WEIGHTING | epochs=$EPOCHS | batch=$BATCH_SIZE | seed=$SEED | resume=$RESUME"
sbatch --job-name="$NAME" \
       --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
       --export=ALL,RUN_NAME="$NAME",OUTPUT_DIR="$OUT",STUDENT_CONFIG="$STUDENT",ALPHA="$ALPHA",TURN_WEIGHT="$TURN_WEIGHT",TIME_WEIGHTING="$TIME_WEIGHTING",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",SEED="$SEED" \
       savio_batchfiles/CFMSwiDITDistillSavio.sh
