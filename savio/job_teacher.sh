#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: train a CFM teacher from scratch (pipeline stage 1), 1 GPU.
# Submitted by savio/stage1_teacher.sh. CONFIG picks the architecture (configs/teacher*.json);
# output: finished_ckpt/1_teacher/<config name>/model. Resubmitting continues from last.pt.
# ---------------------------------------------------------------------------
#SBATCH --job-name=teacher
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs
module load anaconda3
source activate adsb
PYTHON=$HOME/.conda/envs/adsb/bin/python
$PYTHON -c "import torch, netCDF4; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())" \
    || { echo "Environment check failed (see .err log)"; exit 1; }

NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
REPO=${REPO:-$SLURM_SUBMIT_DIR}
CONFIG=${CONFIG:-configs/teacher_dit_cfm_20M.json}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
echo "Teacher config: $CONFIG | epochs=$EPOCHS | batch=$BATCH_SIZE"

cd "$REPO"
$PYTHON -u training/train_teacher.py \
    --nc_path $NC_PATH \
    --config "$CONFIG" \
    ${OUTPUT_DIR:+--output_dir "$OUTPUT_DIR"} \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE
STATUS=$?
echo "Job finished at $(date) with exit code $STATUS"
exit $STATUS
