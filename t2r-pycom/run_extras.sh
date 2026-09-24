#!/bin/bash
# Forced reference window W on the res/ models: prototypes from the last W
# enrollment sessions (W=1 is the calibration session only). Phase 4 only
# (--resume_from 3); embedding, calibration procedure and seeds unchanged.
set -u
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-/usr/local/bin/python3.13}"
SEEDS="${SEEDS:-42 7 13 21 100}"
export T2R_FT_NTEST=60 T2R_MINIMAL=1
for seed in $SEEDS; do
  for w in 1 2 3 4; do
    out="res_w_sweep/seed${seed}_W${w}"
    grep -q 'traj_bn_adapted' "$out/results_phase4.txt" 2>/dev/null && { echo "W-sweep seed $seed W=$w done"; continue; }
    [ -d "res/seed${seed}/modelDir" ] || { echo "no models for seed $seed"; continue; }
    mkdir -p "$out/modelDir"
    for f in res/seed${seed}/modelDir/*; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$out/modelDir/$(basename "$f")"; done
    echo "=== W-sweep seed $seed W=$w start $(date)"
    T2R_FORCE_DAYS_KEEP="$w" $PY -u run_experiment.py --dataset_layout pycom_indoor --data_root data/pycom_indoor --device_shift 0 --n_known 16 --n_unknown 4 --ft_split_mode capture --init_day 1 --finetune_days 2,3 --traj_day 4 --test_day 5 \
        --output_root "$out" --seed "$seed" --num_slice 780 --slice_len 256 --ft_n_train 150 \
        --resume_from 3 --no_ft_always_reinit --skip_plots > "$out/run.log" 2>&1
    echo "=== W-sweep seed $seed W=$w end $(date) rc=$?"
  done
done
echo "W SWEEP COMPLETE $(date)"
