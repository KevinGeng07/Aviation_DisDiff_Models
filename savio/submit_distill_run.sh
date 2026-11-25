#!/bin/bash
# ---------------------------------------------------------------------------
# Submit ONE distillation run (frozen teacher → student) as its own Slurm job.
# The stage scripts (stage2_architectures.sh, stage3_gru.sh, extra_rope.sh,
# stage4_path_weighting.sh) call this once per run. It refuses to reuse an existing output
# folder or a job name that is already queued/running, unless --resume is given.
#
# Usage (on Savio, from anywhere):
#   bash savio/submit_distill_run.sh --name NAME --student CONFIG --save_dir DIR \
#        [--alpha 0.7] [--epochs 100] [--batch_size 256] [--seed 42] [--split_seed 42] \
#        [--turn_weight 0] [--turn_mode continuous|step] [--turn_threshold 45] [--vrate_weight 0] [--z_weight 1] \
#        [--time_weighting step|linear] [--gpus 1] [--resume]
#
#   --name      unique run name → Slurm job name and log prefix
#   --student   student config JSON (relative to the repo, e.g. configs/students/swi_dit_4L_256.json)
#   --save_dir  run folder (relative to the repo or absolute; default finished_ckpt/other/NAME):
#               checkpoints/metrics → DIR/model/, Slurm .out/.err → DIR/logs/
#   --seed      training randomness;  --split_seed  train/val/test aircraft split (keep 42)
#   --turn_weight S / --turn_mode / --turn_threshold   turning windows count up to 1 + S (0 = off)
#   --vrate_weight S_v   windows whose vertical rate changes by >= 10 m/s count up to 1 + S_v more (0 = off)
#   --z_weight W   altitude features (z, vz) count W× the horizontal ones (1 = off)
#   --gpus N    GPUs on one node (default 1; >1 = DistributedDataParallel, --batch_size stays the
#               total). 2 CPUs per GPU; other GPU types: GPU_TYPE=A40 CPUS_PER_GPU=8 + sbatch --qos
#   --resume    continue an existing run of the same NAME from its last.pt
# Environment: TEACHER_CONFIG (default configs/teacher_dit_cfm_20M.json; or configs/teacher_dit_cfm_7M.json),
#   TEACHER_CKPT (default: the config's "checkpoint"), TIME_LIMIT (default 24:00:00),
#   SBATCH_ACCOUNT, GPU_TYPE, CPUS_PER_GPU. They reach the job through sbatch --export=ALL.
# ---------------------------------------------------------------------------
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p logs        # Slurm writes logs/ relative to the submit directory

NAME=""; STUDENT=""; SAVE_DIR=""; ALPHA=0.7; EPOCHS=100; BATCH_SIZE=256; SEED=42; SPLIT_SEED=42
TURN_WEIGHT=0; TURN_MODE=continuous; TURN_THRESHOLD=45; VRATE_WEIGHT=0; Z_WEIGHT=1; TIME_WEIGHTING=step; GPUS=1; RESUME=0
usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
        --name)           NAME="$2"; shift 2 ;;
        --student)        STUDENT="$2"; shift 2 ;;
        --save_dir)       SAVE_DIR="$2"; shift 2 ;;
        --alpha)          ALPHA="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --split_seed)     SPLIT_SEED="$2"; shift 2 ;;
        --turn_weight)    TURN_WEIGHT="$2"; shift 2 ;;
        --turn_mode)      TURN_MODE="$2"; shift 2 ;;
        --turn_threshold) TURN_THRESHOLD="$2"; shift 2 ;;
        --vrate_weight)   VRATE_WEIGHT="$2"; shift 2 ;;
        --z_weight)       Z_WEIGHT="$2"; shift 2 ;;
        --time_weighting) TIME_WEIGHTING="$2"; shift 2 ;;
        --gpus)           GPUS="$2"; shift 2 ;;
        --resume)         RESUME=1; shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[ -n "$NAME" ] && [ -n "$STUDENT" ] || { echo "ERROR: --name and --student are required."; usage; }
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: --name may only contain letters, digits, _ . -"; exit 1; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --gpus must be a positive integer"; exit 1; }
(( BATCH_SIZE % GPUS == 0 )) || { echo "ERROR: --batch_size $BATCH_SIZE must divide evenly over $GPUS GPUs"; exit 1; }
case "$STUDENT" in /*) ;; *) STUDENT="$REPO/$STUDENT" ;; esac
[ -f "$STUDENT" ] || { echo "ERROR: student config not found: $STUDENT"; exit 1; }

SAVE_DIR=${SAVE_DIR:-finished_ckpt/other/$NAME}
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

echo "Submitting $NAME → $SAVE_DIR | student=$(basename "$STUDENT") | gpus=$GPUS | seed=$SEED | turn=$TURN_WEIGHT/$TURN_MODE | vrate=$VRATE_WEIGHT | z=$Z_WEIGHT | time=$TIME_WEIGHTING | resume=$RESUME"
# Time limit given explicitly: Savio rejects jobs without one. 100 epochs of a 4-layer
# student take ~22 h on a GTX 2080 Ti, so the default is 24 h. Override with TIME_LIMIT=HH:MM:SS.
sbatch --job-name="$NAME" --time="${TIME_LIMIT:-24:00:00}" \
       --gres="gpu:${GPU_TYPE:-GTX2080TI}:$GPUS" --cpus-per-task=$(( ${CPUS_PER_GPU:-2} * GPUS )) \
       --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
       --export=ALL,NGPUS="$GPUS",RUN_NAME="$NAME",OUTPUT_DIR="$OUT",STUDENT_CONFIG="$STUDENT",ALPHA="$ALPHA",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",SEED="$SEED",SPLIT_SEED="$SPLIT_SEED",TURN_WEIGHT="$TURN_WEIGHT",TURN_MODE="$TURN_MODE",TURN_THRESHOLD="$TURN_THRESHOLD",VRATE_WEIGHT="$VRATE_WEIGHT",Z_WEIGHT="$Z_WEIGHT",TIME_WEIGHTING="$TIME_WEIGHTING",RESUME="$RESUME" \
       savio/job_distill.sh
