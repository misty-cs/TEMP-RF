#!/bin/bash
# ---------------------------------------------------------------------------
# WiSig (full release, receiver node24-16, equalised)
#
#   Phase 1  session 1 (Mar 01)   initial embedding
#   Phase 2  session 2            fine-tuning
#   Phase 3  session 3            references + threshold (whole session;
#                                 never used for training)
#   Phase 4  session 4 (Mar 23)   open-set evaluation, enrolled + unknown
#
# Shared with the NEU and Pycom projects: model, four-phase pipeline, five
# seeds, 780 signals per device-session, 256-sample inputs, 150 fine-tuning
# samples per device, and the few-shot enrollment sweep in Phase 4.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-/usr/local/bin/python3.13}"
SEEDS="${SEEDS:-42 7 13 21 100}"
RES="${T2R_RES:-res}"

for seed in $SEEDS; do
  out="$RES/seed${seed}"
  grep -q 'traj_bn_adapted' "$out/results_phase4.txt" 2>/dev/null && { echo "seed $seed done"; continue; }
  mkdir -p "$out"
  echo "=== seed $seed start $(date)"
  T2R_FT_NTEST=60 $PY -u run_experiment.py \
      --dataset_layout wisig_full --data_root data \
      --wisig_full_receiver node24-16 \
      --output_root "$out" --seed "$seed" --device_shift 0 \
      --n_known 12 --n_unknown 4 \
      --init_day 1 --finetune_days 2 --traj_day 3 --test_day 4 \
      --num_slice "${NUM_SLICE:-780}" --slice_len 256 --ft_n_train "${FT_N_TRAIN:-150}" \
      --no_ft_always_reinit --skip_plots > "$out/run.log" 2>&1
  echo "=== seed $seed end $(date) rc=$?"
done
echo "WISIG COMPLETE $(date)"
