#!/bin/bash
# Run the ICC experiments for WiSig and Pycom.
#
# The script first performs a small smoke test for each dataset, then runs the
# five-seed protocol. Finished seeds are skipped by each dataset's run_all.sh.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/campaign_$(date +%Y%m%d_%H%M%S).log"
exec >> "$LOG" 2>&1

echo "CAMPAIGN START $(date)  log=$LOG"

ok() {
  grep -q 'traj_bn_adapted' "$1/results_phase4.txt" 2>/dev/null &&
    ! grep -q Traceback "$1/run.log" 2>/dev/null
}

echo "== smoke tests $(date)"
for d in wisig pycom; do
  SEEDS=42 T2R_RES=smoke NUM_SLICE=200 FT_N_TRAIN=60 bash "$ROOT/t2r-$d/run_all.sh"
  chk="$ROOT/t2r-$d/smoke/seed42"
  if ok "$chk"; then
    echo "smoke $d OK"
  else
    echo "smoke $d FAILED, see $chk/run.log"
    echo "CAMPAIGN ABORTED $(date)"
    exit 1
  fi
done

echo "== full runs $(date)"
for d in wisig pycom; do
  echo "== stage: $d $(date)"
  bash "$ROOT/t2r-$d/run_all.sh"
done

echo "CAMPAIGN COMPLETE $(date)"
