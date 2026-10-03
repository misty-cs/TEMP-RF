#!/bin/zsh
# LoRa tuning in parallel with the Pycom tuning of run_all.sh v1 (PROTOCOL.md 16.1).
# Same commands as run_all.sh tune(); then, when Pycom and LoRa tuning are both
# complete and nothing else runs, start run_all.sh v2 (pilots, finals, aggregation).
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
N=(nice -n 10)
LOG=run_all.log
say() { echo "=== [lora-parallel] $* $(date)" | tee -a $LOG; }
commit() { msg=$1; shift
  for t in 1 2 3 4 5; do
    for p in "$@"; do [ -e "$p" ] && git add -A -- "$p"; done
    git diff --cached --quiet && return 0
    git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && return 0
    sleep 30
  done; }
nshifts() { $PY -c "import datasets as D; print(len(D.default_shifts('$1')))"; }
selected_n() { [ -f res_tune/$1/selected_phase$2.json ] && $PY -c "import json; print(len(json.load(open('res_tune/$1/selected_phase$2.json'))))" || echo 0; }
waitlora() { until $PY data_prep/lora_dedup.py --check data/lora_indoor >/dev/null 2>&1; do echo "$(date) lora data missing, waiting" >> $LOG; sleep 120; done; }

say "START"
while pgrep -f "tune_phase.\.py --dataset lora" >/dev/null; do sleep 120; done   # an earlier start of this script still tunes
waitlora
$PY -u pool_check.py --dataset lora > pool_check_lora.log 2>&1 || { say "LoRa pool rule FAILED - see pool_check_lora.log; stopping"; exit 1; }
cp pool_check_lora.log docs/pool_check_lora.txt
commit "LoRa: de-duplicated tree complete, device-pool rule passes" docs/pool_check_lora.txt
for ph in 1 2; do
  for attempt in 1 2 3 4 5 6 7 8; do
    waitlora
    "${N[@]}" $PY -u tune_phase$ph.py --dataset lora --go >> tune${ph}_lora.log 2>&1
    [ "$(selected_n lora $ph)" = "$(nshifts lora)" ] && break
    say "tune_phase$ph lora incomplete after attempt $attempt ($(selected_n lora $ph)/$(nshifts lora) splits); retrying"
    sleep 60
  done
  [ "$(selected_n lora $ph)" = "$(nshifts lora)" ] || { say "tune_phase$ph lora FAILED after retries - stopping"; exit 1; }
  say "tune_phase$ph lora done"; commit "Tuning: lora Phase $ph selected for all splits" res_tune/lora
done
say "LoRa tuning done; waiting for Pycom tuning"
until [ "$(selected_n pycom 2)" = "$(nshifts pycom)" ] && ! pgrep -f "tune_phase|res_pilot_tuned" >/dev/null; do sleep 300; done   # WiSig finals (run_wisig_early.sh) may run
commit "Tuning: pycom Phase 1 and 2 (state at the start of run_all.sh v2)" res_tune/pycom
say "starting run_all.sh v2 (run_all.sh v1, PID 36842, only waits at its old LoRa check; it can be stopped)"
nohup ./run_all.sh >/dev/null 2>&1 &
