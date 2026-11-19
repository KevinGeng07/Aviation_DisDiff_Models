#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: evaluate finished checkpoints, 1 GPU
#   SCRIPT=evaluate        (default) test-set ADE/FDE (+ p1/p99) → table in logs/eval_<jobid>.out
#   SCRIPT=benchmark       ADE/FDE + params + latency            → table in logs/eval_<jobid>.out
#   SCRIPT=sample_futures  sampled futures for 10 windows        → eval/futures.csv
#   SCRIPT=error_by_step   per-axis error over the horizon + plots → results/error_by_step/
#
# Usage (on Savio, from the repo root — the job runs in the folder you submit from):
#   sbatch savio/job_eval.sh                                  # every model in finished_ckpt/
#   sbatch savio/job_eval.sh finished_ckpt/2_architecture     # one pipeline stage
#   sbatch savio/job_eval.sh finished_ckpt/2_architecture/distill_4_256 finished_ckpt/3_gru/gru1_2L_256
#   EVAL_ARGS="--n_samples 5 --max_batches 50" sbatch savio/job_eval.sh   # extra script options
#   EVAL_ARGS="--out results/eval.json" sbatch savio/job_eval.sh          # also save JSON
#   SCRIPT=benchmark sbatch savio/job_eval.sh                 # accuracy/size/speed table
#   SCRIPT=sample_futures sbatch savio/job_eval.sh            # futures CSV for every model
# Compressed checkpoints (best.pt.gz.part*) are read directly; no need to decompress first.
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
REPO=${REPO:-$SLURM_SUBMIT_DIR}   # override with REPO=/path/to/repo sbatch ...
cd "$REPO"

SCRIPT=${SCRIPT:-evaluate}
echo "Script: eval/$SCRIPT.py | folders: ${*:-every model in finished_ckpt/} | extra args: ${EVAL_ARGS:-none}"
# -u: unbuffered so results show up in the log as each model finishes
$PYTHON -u eval/$SCRIPT.py "$@" --nc_path $NC_PATH --num_workers ${SLURM_CPUS_PER_TASK:-2} $EVAL_ARGS
