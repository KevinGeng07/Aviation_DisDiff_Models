#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 1 — teacher, trained with CFM. A teacher config (configs/teacher*.json) names the
# architecture and the checkpoint the later stages distil from:
#   configs/teacher_dit_cfm_20M.json  CITRIS DiT, flow_v5 size    20.73 M  DEFAULT teacher
#   configs/teacher_dit_cfm_7M.json   CITRIS DiT, flow_v4 size     7.13 M
# Their checkpoints are CITRIS's best.pt files (finished_ckpt/1_teacher/dit_cfm_*/best.pt), which
# load as is; --retrain trains one from scratch instead. Later stages pick a teacher with
# TEACHER_CONFIG=<config> (default the 20M one).
#
# Usage (on Savio):  bash savio/stage1_teacher.sh [config]                # check its checkpoint
#                    bash savio/stage1_teacher.sh --retrain [config]      # train it (~1–2 days)
#   e.g. EPOCHS=200 BATCH_SIZE=256 bash savio/stage1_teacher.sh --retrain configs/teacher_dit_cfm_20M.json
# ---------------------------------------------------------------------------
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p logs

RETRAIN=0; [ "$1" = "--retrain" ] && { RETRAIN=1; shift; }
CONFIG=${1:-configs/teacher_dit_cfm_20M.json}
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
CKPT=$(python3 -c "import json; print(json.load(open('$CONFIG'))['checkpoint'])")
if [ "$RETRAIN" = 1 ]; then
    NAME=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('name', 'teacher'))")
    sbatch --job-name="teacher_$NAME" --time="${TIME_LIMIT:-48:00:00}" \
           --export=ALL,CONFIG="$CONFIG" savio/job_teacher.sh
    exit 0
fi

SIZE=$(wc -c < "$CKPT" 2>/dev/null || echo 0)
if [ "$SIZE" -lt 1000000 ]; then
    echo "Teacher checkpoint $CKPT is missing or a Git LFS pointer ($SIZE bytes)."
    echo "Preset teacher: git lfs install && git lfs pull.  Others: copy the checkpoint there,"
    echo "or train it: bash savio/stage1_teacher.sh --retrain $CONFIG"
    exit 1
fi
echo "Teacher OK: $CKPT ($((SIZE / 1000000)) MB). Continue with stage 2 (TEACHER_CONFIG=$CONFIG)."
