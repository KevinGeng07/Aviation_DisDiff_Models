#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: evaluate finished checkpoints, 1 GPU
#   SCRIPT=evaluate        (default) test-set ADE/FDE        → table in logs/eval_<jobid>.out
#   SCRIPT=sample_futures  sampled futures for 10 windows    → eval/futures.csv
#
# Usage (on Savio, from the repo root):
#   sbatch savio_batchfiles/EvalSavio.sh                                  # every model in finished_ckpt/
#   sbatch savio_batchfiles/EvalSavio.sh finished_ckpt/6_4_256_SWiDiT training/checkpoints/distill_4L_384_a07
#   EVAL_ARGS="--n_samples 5 --max_batches 50" sbatch savio_batchfiles/EvalSavio.sh   # extra script options
#   SCRIPT=sample_futures sbatch savio_batchfiles/EvalSavio.sh            # futures CSV for every model
# ---------------------------------------------------------------------------
#SBATCH --job-name=eval
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out      # %x = job name, %j = job id
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load anaconda3
source activate adsb
PYTHON=$HOME/.conda/envs/adsb/bin/python

NC_PATH=/global/scratch/users/kevingeng/aviation-bayen/data/trajectories_adsblol_seq86_stage2.nc
REPO=/global/scratch/users/kevingeng/aviation-bayen/AviationDiffusionModelsKishore
cd $REPO

SCRIPT=${SCRIPT:-evaluate}
echo "Script: eval/$SCRIPT.py | folders: ${*:-every model in finished_ckpt/} | extra args: ${EVAL_ARGS:-none}"
# -u: unbuffered so results show up in the log as each model finishes
$PYTHON -u eval/$SCRIPT.py "$@" --nc_path $NC_PATH --num_workers ${SLURM_CPUS_PER_TASK:-2} $EVAL_ARGS
