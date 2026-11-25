#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 4 — path (loss) weighting on the selected student, as a 2×2 design
# (training/loss_weighting.py, training/train_distill.py):
#
#                      no turn weighting             continuous turn + vertical-rate weighting
#   step time weights  baseline (alpha-sweep run)    turn_cont_S<S>_V<SV>
#   linear time ramp   time_linear                   time_linear_turn_cont_S<S>_V<SV>
#
#   step time weights     last quarter of the horizon ×2 (the default everywhere)
#   linear time ramp      1 at the first future step → 2 at the last
#   turn + vertical rate  per window: w = 1 + S·min(turn/90°, 1) + SV·clamp((dvz − 1)/(10 − 1), 0, 1)
#                         turn = largest heading change from the last observed heading (degrees),
#                         dvz  = largest vertical-rate change from the last observed one (m/s);
#                         rescaled to batch mean 1
# Baseline: the same student at the same alpha, unweighted (the alpha-sweep run, e.g.
# finished_ckpt/alpha/distill_4_256_a0.3, or finished_ckpt/2_architecture/distill_4_256 for 0.7).
# Weighting comes before stage 5: stepwise distillation only copies its base model.
# Output: finished_ckpt/4_path_weighting/<student>_<variant>[_alpha<a>][_s<seed>]/
#
# Usage (on Savio):  ALPHA=0.3 bash savio/stage4_path_weighting.sh      # 3 runs at the chosen alpha
#                    bash savio/stage4_path_weighting.sh                # alpha 0.7 (default)
#                    S=4 SV=4 SEED=43 bash savio/stage4_path_weighting.sh   # other strengths / seed
#                    SV=0 bash savio/stage4_path_weighting.sh           # horizontal turns only
# Then compare:      SCRIPT=full_eval sbatch savio/job_eval.sh <baseline run folder> finished_ckpt/4_path_weighting
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."
STUDENT=${STUDENT:-configs/students/swi_dit_4L_256.json}
ALPHA=${ALPHA:-0.7}   # ground-truth weight in the distillation loss: use the alpha sweep's winner
S=${S:-2}             # turn strength: windows turning >= 90° count up to 1 + S
SV=${SV:-2}           # vertical-rate strength: vz changes >= 10 m/s add up to SV
SEED=${SEED:-42}
BASE=$(basename "$STUDENT" .json)
SUFFIX=$([ "$ALPHA" = 0.7 ] || echo "_alpha${ALPHA/./p}")$([ "$SEED" = 42 ] || echo "_s$SEED")
TURN="--turn_weight $S --turn_mode continuous --vrate_weight $SV"

RUNS=(   # <variant>                          <extra flags>
  "turn_cont_S${S}_V${SV}                     --time_weighting step   $TURN"
  "time_linear                                --time_weighting linear"
  "time_linear_turn_cont_S${S}_V${SV}         --time_weighting linear $TURN"
)
for run in "${RUNS[@]}"; do
  read -r variant flags <<< "$run"
  name="${BASE}_$variant$SUFFIX"
  # shellcheck disable=SC2086
  bash savio/submit_distill_run.sh --name "$name" --student "$STUDENT" --seed "$SEED" --alpha "$ALPHA" \
       --save_dir "finished_ckpt/4_path_weighting/$name" $flags || echo "  → skipped $name"
done
squeue -u "$USER"
