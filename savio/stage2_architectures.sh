#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 2 — architecture distillation: the 6-layer teacher → 7 SWI_DiT students of
# different depth (2/4/6 layers) and width (128/256/384), 1 GPU each, in parallel.
# Output: finished_ckpt/2_architecture/distill_<layers>_<width>/
#
# Usage (on Savio):  bash savio/stage2_architectures.sh
#                    SEED=43 bash savio/stage2_architectures.sh     # another seed → *_s43
# Runs whose folder already exists are skipped (resume one with submit_distill_run.sh --resume).
# Then compare:      SCRIPT=benchmark sbatch savio/job_eval.sh finished_ckpt/2_architecture
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."
SEED=${SEED:-42}
SUFFIX=$([ "$SEED" = 42 ] || echo "_s$SEED")

RUNS=(   # <run name>        <student config>
  "distill_2_128     configs/students/swi_dit_2L_128.json"
  "distill_2_256     configs/students/swi_dit_2L_256.json"
  "distill_2_384     configs/students/swi_dit_2L_384.json"
  "distill_4_128     configs/students/swi_dit_4L_128.json"
  "distill_4_256     configs/students/swi_dit_4L_256.json"
  "distill_4_384     configs/students/swi_dit_4L_384.json"
  "distill_6_128     configs/students/swi_dit_6L_128.json"
)
for run in "${RUNS[@]}"; do
  read -r name cfg <<< "$run"
  bash savio/submit_distill_run.sh --name "$name$SUFFIX" --student "$cfg" --seed "$SEED" \
       --save_dir "finished_ckpt/2_architecture/$name$SUFFIX" || echo "  → skipped $name$SUFFIX"
done
squeue -u "$USER"
