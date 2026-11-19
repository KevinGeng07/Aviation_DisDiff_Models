#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 3 — GRU alternatives at the selected size (d_model 256), distilled from the same teacher:
#   gru1_2L_256   models/GRU_Swi_DiT.py     5.8 M params
#   gru2_2L_256   models/GRU_Swi_DiT_2.py   5.8 M params
#   gru3_1L_256   models/GRU_Swi_DiT_3.py   one GRU block + one 3-attention block per layer
# Compare against finished_ckpt/2_architecture/distill_4_256 (5.0 M params).
# Output: finished_ckpt/3_gru/<run>/
#
# Usage (on Savio):  bash savio/stage3_gru.sh
#                    SEED=43 bash savio/stage3_gru.sh               # another seed → *_s43
# Then compare:      SCRIPT=benchmark sbatch savio/job_eval.sh \
#                        finished_ckpt/2_architecture/distill_4_256 finished_ckpt/3_gru
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."
SEED=${SEED:-42}
SUFFIX=$([ "$SEED" = 42 ] || echo "_s$SEED")

for name in gru1_2L_256 gru2_2L_256 gru3_1L_256; do
  bash savio/submit_distill_run.sh --name "$name$SUFFIX" --student "configs/students/$name.json" \
       --seed "$SEED" --save_dir "finished_ckpt/3_gru/$name$SUFFIX" || echo "  → skipped $name$SUFFIX"
done
squeue -u "$USER"
