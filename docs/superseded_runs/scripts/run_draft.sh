#!/bin/zsh
# DRAFT runs for the draft paper (protocol v3: Phase-1/2 defaults, no tuning;
# PROTOCOL.md section 11). Results: res_draft/, res_joint_draft/.
# Normal CPU priority (the full plan runs at lower priority).
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
EXPLORE=../explore/pycom_input_length/run_all.log
while [ -f "$EXPLORE" ] && ! grep -qE "=== ALL DONE|PREPARE FAILED" "$EXPLORE"; do sleep 60; done
echo "=== DRAFT START $(date)"
run_ds() {
  for s in 42 7 13 21 100; do
    $PY -u main.py --dataset $1 --seeds $s --allow-untuned --root res_draft --go
  done
  $PY -u main.py --dataset $1 --joint --allow-untuned --root res_draft --go
  echo "=== $1 DONE $(date)"
}
run_ds wisig > draft_wisig.log 2>&1 &
run_ds pycom > draft_pycom.log 2>&1 &
wait
$PY -u aggregate.py --tags wisig pycom --root res_draft > draft_aggregate.log 2>&1
echo "=== DRAFT DONE rc=$? $(date)"
