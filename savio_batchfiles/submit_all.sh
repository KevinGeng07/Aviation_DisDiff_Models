#!/bin/bash
# Submit the 7 SWi_DiT distillation runs + 3 GRU variants, all in parallel (1 GPU each).
# Usage, on Savio:  bash savio_batchfiles/submit_all.sh
# Existing run folders / already-queued names are skipped by submit_distill_run.sh.

cd "$(dirname "$0")/.."

OLD_SW=OLD_finished_ckpt/SWi_DiT
NEW_SW=finished_ckpt/SWi_DiT
RUNS=(   # <run_name>  <student_config>  [<save_dir>]   (save_dir omitted → $SW/<run_name>)
  "distill_6_2_128  $OLD_SW/6_2_128_SWiDiT/model/student_config.json  $NEW_SW/distill_6_2_128"
  "distill_6_2_256  $OLD_SW/6_2_256_SWiDiT/model/student_config.json  $NEW_SW/distill_6_2_256"
  "distill_6_2_384  $OLD_SW/6_2_384_SWiDiT/model/student_config.json  $NEW_SW/distill_6_2_384"
  "distill_6_4_128  $OLD_SW/6_4_128_SWiDiT/model/student_config.json  $NEW_SW/distill_6_4_128"
  "distill_6_4_256  $OLD_SW/6_4_256_SWiDiT/model/student_config.json  $NEW_SW/distill_6_4_256"
  "distill_6_4_384  $OLD_SW/6_4_384_SWiDiT/model/student_config.json  $NEW_SW/distill_6_4_384"
  "distill_6_6_128  $OLD_SW/6_6_128_SWiDiT/model/student_config.json  $NEW_SW/distill_6_6_128"
  "gru1_2L_256      configs/students/gru_swi_dit_2L_256.json      $NEW_SW/gru1_2L_256"
  "gru2_2L_256      configs/students/gru_swi_dit_2_2L_256.json    $NEW_SW/gru2_2L_256"
  "gru3_1L_256      configs/students/gru_swi_dit_3_1L_256.json    $NEW_SW/gru3_1L_256"
)

for run in "${RUNS[@]}"; do
  read -r name cfg out <<< "$run"
  bash savio_batchfiles/submit_distill_run.sh --name "$name" --student "$cfg" \
       ${out:+--save_dir "$out"} || echo "  → skipped $name"
done

squeue -u "$USER"
