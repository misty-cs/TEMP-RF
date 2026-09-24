#!/bin/bash
# ---------------------------------------------------------------------------
# Pycom (raw time-domain indoor WiFi captures)
#
#   Phase 1  day 1            initial embedding
#   Phase 2  days 2, 3        sequential fine-tuning
#   Phase 3  day 4            references + threshold
#   Phase 4  day 5            open-set evaluation, enrolled + unknown
#
# Each device-day is five separate capture files, so splits are
# capture-disjoint: slices from a capture used for training never appear in
# validation or test. Captures are sampled across the whole recording rather
# than from its opening window, since a contiguous read would give every slice
# the same channel realisation.
#
# Shared with the WiSig and NEU projects: model, four-phase pipeline, five
# seeds, 780 slices per device-session, 256-sample inputs, 150 fine-tuning
# samples per device, and the few-shot enrollment sweep in Phase 4.
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-/usr/local/bin/python3.13}"
SEEDS="${SEEDS:-42 7 13 21 100}"
RES="${T2R_RES:-res}"
DR="${T2R_DATA_ROOT:-data/pycom_indoor}"

for seed in $SEEDS; do
  out="$RES/seed${seed}"
  grep -q 'traj_bn_adapted' "$out/results_phase4.txt" 2>/dev/null && { echo "seed $seed done"; continue; }
  mkdir -p "$out"
  echo "=== seed $seed start $(date)"
  T2R_FT_NTEST=60 $PY -u run_experiment.py \
      --dataset_layout pycom_indoor --data_root "$DR" \
      --output_root "$out" --seed "$seed" --device_shift 0 \
      --n_known 16 --n_unknown 4 --ft_split_mode capture \
      --init_day 1 --finetune_days 2,3 --traj_day 4 --test_day 5 \
      --num_slice "${NUM_SLICE:-780}" --slice_len 256 --ft_n_train "${FT_N_TRAIN:-150}" \
      --no_ft_always_reinit --skip_plots > "$out/run.log" 2>&1
  echo "=== seed $seed end $(date) rc=$?"
done
echo "PYCOM COMPLETE $(date)"
