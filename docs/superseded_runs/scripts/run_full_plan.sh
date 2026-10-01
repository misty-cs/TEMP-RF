#!/bin/zsh
# FULL PLAN (protocol v2 + Pycom L=256): per-split Phase-1/2 tuning on
# validation data -> tuned pilot (split 0, seed 42) -> gate: both audits pass
# -> paper runs (5 seeds, all splits, joint) -> aggregate. Results: res_tune/,
# res_pilot_tuned/, res/, res_joint/. Lower CPU priority than the draft runs.
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
EXPLORE=../explore/pycom_input_length/run_all.log
while [ -f "$EXPLORE" ] && ! grep -qE "=== ALL DONE|PREPARE FAILED" "$EXPLORE"; do sleep 60; done
N=(nice -n 15)
echo "=== TUNING START $(date)"
for ds in wisig pycom; do
  "${N[@]}" $PY -u tune_phase1.py --dataset $ds --go > tune1_$ds.log 2>&1; echo "=== tune_phase1 $ds rc=$? $(date)"
  "${N[@]}" $PY -u tune_phase2.py --dataset $ds --go > tune2_$ds.log 2>&1; echo "=== tune_phase2 $ds rc=$? $(date)"
done
echo "=== TUNED PILOT START $(date)"
for ds in wisig pycom; do
  "${N[@]}" $PY -u main.py --dataset $ds --shifts 0 --seeds 42 --root res_pilot_tuned --go > pilot_tuned_$ds.log 2>&1
done
ok=1
for ds in wisig pycom; do
  a=res_pilot_tuned/$ds/shift0/seed42/audit.json
  if [ ! -f $a ] || ! grep -q '"all_pass": true' $a || [ -f res_pilot_tuned/$ds/shift0/seed42/AUDIT_FAILED ]; then ok=0; fi
done
if [ $ok = 0 ]; then echo "=== TUNED PILOT FAILED - paper runs NOT started $(date)"; exit 1; fi
echo "=== PAPER RUN START $(date)"
run_ds() {
  for s in 42 7 13 21 100; do "${N[@]}" $PY -u main.py --dataset $1 --seeds $s --go; done
  "${N[@]}" $PY -u main.py --dataset $1 --joint --go
  echo "=== $1 DONE $(date)"
}
run_ds wisig > paper_wisig.log 2>&1 &
run_ds pycom > paper_pycom.log 2>&1 &
wait
$PY -u aggregate.py --tags wisig pycom --root res > paper_aggregate.log 2>&1
echo "=== FULL PLAN DONE rc=$? $(date)"
