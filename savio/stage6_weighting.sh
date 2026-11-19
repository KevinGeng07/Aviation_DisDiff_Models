#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 6 — loss weighting on the selected student (training/loss_weighting.py):
#   turn_cont_S2    continuous turn weighting: weight 1 + 2·min(turn/90°, 1)
#   turn_step_S2    discrete turn weighting:   weight 3 if the turn is >= 45°, else 1
#   zw_2            altitude features (z, vz) weighted 2× the horizontal ones
#   turn_cont_S2_zw_2   both
# Baseline: the same student without weighting (stage 2 / 4 run).
# The weighting is applied while distilling the base model, where the loss includes the real
# trajectories; stepwise distillation (stage 5) only matches its teacher, so it can't fix turning.
# Output: finished_ckpt/6_weighting/<student>_<variant>/
#
# Usage (on Savio):  bash savio/stage6_weighting.sh                                   # 4L_256 student
#                    STUDENT=configs/students/rope_ts_4L_256.json bash savio/stage6_weighting.sh
#                    S=1 THRESHOLD=30 Z=3 SEED=43 bash savio/stage6_weighting.sh      # other settings
# Then compare (evaluate.py reports minFDE_turn / avgFDE_turn):
#   python eval/evaluate.py finished_ckpt/2_architecture/distill_4_256 finished_ckpt/6_weighting
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."
STUDENT=${STUDENT:-configs/students/swi_dit_4L_256.json}
S=${S:-2}; THRESHOLD=${THRESHOLD:-45}; Z=${Z:-2}; SEED=${SEED:-42}
BASE=$(basename "$STUDENT" .json)
SUFFIX=$([ "$SEED" = 42 ] || echo "_s$SEED")

RUNS=(   # <variant>                 <extra flags>
  "turn_cont_S$S                     --turn_weight $S --turn_mode continuous"
  "turn_step_S$S                     --turn_weight $S --turn_mode step --turn_threshold $THRESHOLD"
  "zw_$Z                             --z_weight $Z"
  "turn_cont_S${S}_zw_$Z             --turn_weight $S --turn_mode continuous --z_weight $Z"
)
for run in "${RUNS[@]}"; do
  read -r variant flags <<< "$run"
  name="${BASE}_$variant$SUFFIX"
  # shellcheck disable=SC2086
  bash savio/submit_distill_run.sh --name "$name" --student "$STUDENT" --seed "$SEED" \
       --save_dir "finished_ckpt/6_weighting/$name" $flags || echo "  → skipped $name"
done
squeue -u "$USER"
