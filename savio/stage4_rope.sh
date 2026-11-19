#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 4 — RoPE position embeddings on the selected 4-layer, d_model 256 student
# (models/Swi_DiT_RoPE.py), distilled from the same teacher:
#   rope_index_4L_256   RoPE over token indices; timestamps still added through the t_rel MLP
#   rope_ts_4L_256      RoPE over the real timestamps; no learned positions, no t_rel MLP
# Baseline: finished_ckpt/2_architecture/distill_4_256 (learned absolute positions).
# Output: finished_ckpt/4_rope/<run>/
#
# Usage (on Savio):  bash savio/stage4_rope.sh
#                    SEED=43 bash savio/stage4_rope.sh              # another seed → *_s43
# Then compare:      SCRIPT=benchmark sbatch savio/job_eval.sh \
#                        finished_ckpt/2_architecture/distill_4_256 finished_ckpt/4_rope
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."
SEED=${SEED:-42}
SUFFIX=$([ "$SEED" = 42 ] || echo "_s$SEED")

for name in rope_index_4L_256 rope_ts_4L_256; do
  bash savio/submit_distill_run.sh --name "$name$SUFFIX" --student "configs/students/$name.json" \
       --seed "$SEED" --save_dir "finished_ckpt/4_rope/$name$SUFFIX" || echo "  → skipped $name$SUFFIX"
done
squeue -u "$USER"
