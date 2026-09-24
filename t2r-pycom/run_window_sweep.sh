#!/bin/bash
# Forced-window Pycom sweep for the multi-day prototype claim.
#
# Protocol:
#   Day 1: train
#   Days 2-3: fine-tune
#   Day 4: reference/calibration
#   Day 5: test
#
# Windows:
#   W=1 -> Day 4 only
#   W=2 -> Days 3-4
#   W=3 -> Days 2-4
#   W=4 -> Days 1-4
set -u
cd "$(dirname "$0")" || exit 1

PY="${PYTHON:-/usr/local/bin/python3.13}"
SEEDS="${SEEDS:-42 7 13 21 100}"
RES_PREFIX="${T2R_RES_PREFIX:-res_W}"
DR="${T2R_DATA_ROOT:-data/pycom_indoor}"

for w in 1 2 3 4; do
  for seed in $SEEDS; do
    out="${RES_PREFIX}${w}/seed${seed}"
    grep -q 'traj_bn_adapted' "$out/results_phase4.txt" 2>/dev/null && {
      echo "W=$w seed $seed done"
      continue
    }
    mkdir -p "$out"
    echo "=== Pycom W=$w seed $seed start $(date)"
    T2R_FORCE_DAYS_KEEP="$w" T2R_FT_NTEST=60 $PY -u run_experiment.py \
      --dataset_layout pycom_indoor --data_root "$DR" \
      --output_root "$out" --seed "$seed" --device_shift 0 \
      --n_known 16 --n_unknown 4 --ft_split_mode capture \
      --init_day 1 --finetune_days 2,3 --traj_day 4 --test_day 5 \
      --num_slice "${NUM_SLICE:-780}" --slice_len 256 \
      --ft_n_train "${FT_N_TRAIN:-150}" \
      --no_ft_always_reinit --skip_plots > "$out/run.log" 2>&1
    echo "=== Pycom W=$w seed $seed end $(date) rc=$?"
  done
done

echo "PYCOM WINDOW SWEEP COMPLETE $(date)"
