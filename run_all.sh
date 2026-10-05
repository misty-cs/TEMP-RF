#!/bin/zsh
# Complete study, protocol v4 (docs/PROTOCOL.md): WiSig, Pycom, LoRa.
#   1. Phase-1 and Phase-2 tuning per device split (validation data only),
#      WiSig -> Pycom -> LoRa (LoRa only after its download is complete and its
#      device-pool rule passes).
#   2. Tuned pilot (split 0, seed 42) per dataset; gate: every pilot audit passes.
#   3. Final runs only AFTER ALL tuning: all splits x 5 seeds + joint control,
#      the three datasets in parallel.
#   4. aggregate.py --tags wisig pycom lora.
# Robust to drive disconnects: every step waits for the data and is repeated
# until it is complete (failed attempts are kept as *.failedN, never deleted).
# After each stage the results are committed to git (code is frozen during the
# study; see docs/PROTOCOL.md section 16).
# v2 (2026-10-03, before any LoRa tuning; PROTOCOL.md 16.1): LoRa uses the
# de-duplicated tree (data_prep/lora_dedup.py); only the LoRa completeness
# check changed. Restarting repeats nothing that is complete.
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
N=(nice -n 10)
LOG=run_all.log
say() { echo "=== $* $(date)" | tee -a $LOG; }
datadir() { case $1 in wisig) echo data/wisig;; pycom) echo data/pycom_indoor/Day_1;; lora) echo data/lora_indoor/Day1;; esac; }
waitdata() { while [ ! -d "$(datadir $1)" ]; do echo "$(date) $1 data missing, waiting" >> $LOG; sleep 120; done; }
commit() {   # commit results of a finished stage (only files allowed by .gitignore)
  git add -A >/dev/null 2>&1
  git diff --cached --quiet || git commit -q -m "$1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}
nshifts() { $PY -c "import datasets as D; print(len(D.default_shifts('$1')))"; }
selected_n() { [ -f res_tune/$1/selected_phase$2.json ] && $PY -c "import json; print(len(json.load(open('res_tune/$1/selected_phase$2.json'))))" || echo 0; }

tune() {   # $1 dataset
  for ph in 1 2; do
    for attempt in 1 2 3 4 5 6 7 8; do
      waitdata $1
      "${N[@]}" $PY -u tune_phase$ph.py --dataset $1 --go >> tune${ph}_$1.log 2>&1
      [ "$(selected_n $1 $ph)" = "$(nshifts $1)" ] && break
      say "tune_phase$ph $1 incomplete after attempt $attempt ($(selected_n $1 $ph)/$(nshifts $1) splits); retrying"
      sleep 60
    done
    [ "$(selected_n $1 $ph)" = "$(nshifts $1)" ] || { say "tune_phase$ph $1 FAILED after retries - stopping"; exit 1; }
    say "tune_phase$ph $1 done"; commit "Tuning: $1 Phase $ph selected for all splits"
  done
}

ok_runs() {  # number of audited complete runs under $1/$2
  $PY - "$1" "$2" <<'EOF'
import sys, glob, json, os, manifest, datasets as D
root, ds = sys.argv[1], sys.argv[2]; n = 0
for a in glob.glob(f'{root}/{ds}/shift*/seed*/audit.json'):
    d = os.path.dirname(a); A = json.load(open(a))
    if A.get('all_pass') and not os.path.exists(f'{d}/AUDIT_FAILED') and \
       manifest.artifacts_match(d, A.get('artifacts'), D.DATASETS[ds]['traj_day'])[0]:
        n += 1
print(n)
EOF
}

runs() {   # $1 dataset, $2 root, $3 expected count, then main.py args
  ds=$1; root=$2; want=$3; shift 3
  for attempt in 1 2 3 4 5 6 7 8; do
    waitdata $ds
    "${N[@]}" $PY -u main.py --dataset $ds --root $root "$@" --go >> ${root}_${ds}.log 2>&1
    [ "$(ok_runs $root $ds)" -ge "$want" ] && return 0
    say "$root $ds: $(ok_runs $root $ds)/$want audited runs after attempt $attempt; retrying"
    sleep 60
  done
  return 1
}

say "START"
tune wisig
tune pycom
lora_complete() {   # de-duplicated LoRa tree = downloaded files minus the listed duplicates
  $PY data_prep/lora_dedup.py --check data/lora_indoor >/dev/null 2>&1 && echo yes || echo no
}
until [ "$(lora_complete)" = "yes" ]; do echo "$(date) waiting for the LoRa download" >> $LOG; sleep 600; done
waitdata lora
$PY -u pool_check.py --dataset lora > pool_check_lora.log 2>&1 || { say "LoRa pool rule FAILED - see pool_check_lora.log; stopping"; commit "LoRa pool check (failed)"; exit 1; }
cp pool_check_lora.log docs/pool_check_lora.txt
commit "LoRa: de-duplicated tree complete, device-pool rule passes"
tune lora

say "TUNED PILOTS"
for ds in wisig pycom lora; do runs $ds res_pilot_tuned 1 --shifts 0 --seeds 42 || { say "tuned pilot $ds failed"; exit 1; }; done
commit "Tuned pilots (split 0, seed 42): all audits pass"

say "FINAL RUNS"
# Stage A (2026-10-05, PROTOCOL 16.1): seeds 42, 7, 13 + joint control (seed 42,
# every split) for all datasets, then the 3-seed aggregation (the ICC results).
# Stage B: seeds 21 and 100, then the 5-seed aggregation.
final_seeds() {   # $1 dataset, then seeds: seed by seed, then repeat until complete
  ds=$1; shift; want=$(( $(nshifts $ds) * $# ))
  for s in "$@"; do waitdata $ds; "${N[@]}" $PY -u main.py --dataset $ds --root res --seeds $s --go >> res_$ds.log 2>&1; done
  runs $ds res $want --seeds "$@" || say "final $ds seeds $* incomplete"
}
stageA() {
  while pgrep -f run_wisig_joint.sh >/dev/null; do sleep 300; done   # WiSig joint (seed 42) started early
  final_seeds $1 42 7 13
  for attempt in 1 2 3; do waitdata $1; "${N[@]}" $PY -u main.py --dataset $1 --joint --seeds 42 --go >> res_joint_$1.log 2>&1; done
  say "final $1 stage A done (seeds 42 7 13 + joint seed 42)"
}
snapshot() {   # $1 name, then seeds: aggregate only these seeds (aggregate.py uses every run under
               # its root, so res_<name>/ holds relative links to the selected runs; nothing is copied)
  name=$1; shift; R=res_$name; J=res_joint_$name
  for r in $R $J; do [ -e $r ] && { k=1; while [ -e $r.old$k ]; do k=$((k+1)); done; mv $r $r.old$k; }; done
  for t in wisig pycom lora; do
    for d in res/$t/shift* res_joint/$t/shift*; do
      [ -d "$d" ] || continue
      dst=${d/#res_joint/$J}; dst=${dst/#res\//$R/}
      seeds=("$@"); [[ $d == res_joint/* ]] && seeds=(42)     # joint control: seed 42 only
      for sd in $seeds; do [ -d $d/seed$sd ] && { mkdir -p $dst; ln -s ../../../$d/seed$sd $dst/seed$sd; }; done
    done
  done
  $PY -u aggregate.py --tags wisig pycom lora --root $R --seeds "$@" > aggregate_$name.log 2>&1
  mkdir -p docs/results_$name; cp aggregate_$name.log docs/results_$name/aggregate.txt
  cp $R/PRIMARY_*.md docs/results_$name/ 2>/dev/null
  for t in wisig pycom lora; do cp $R/$t/SUMMARY.md docs/results_$name/SUMMARY_$t.md 2>/dev/null; done
}
stageA wisig & stageA pycom & stageA lora & wait
commit "Final runs stage A: WiSig, Pycom, LoRa (all splits x seeds 42, 7, 13 + joint seed 42)"
snapshot 3seeds 42 7 13
commit "Aggregated results, 3 seeds (docs/results_3seeds)"
say "STAGE A DONE (3-seed results)"
final_seeds wisig 42 7 13 21 100 & final_seeds pycom 42 7 13 21 100 & final_seeds lora 42 7 13 21 100 & wait
commit "Final runs stage B: seeds 21, 100"
$PY -u aggregate.py --tags wisig pycom lora --root res > aggregate_final.log 2>&1
mkdir -p docs/results_5seeds; cp aggregate_final.log docs/results_5seeds/aggregate.txt; cp res/PRIMARY_*.md docs/results_5seeds/ 2>/dev/null
for t in wisig pycom lora; do cp res/$t/SUMMARY.md docs/results_5seeds/SUMMARY_$t.md 2>/dev/null; done
commit "Aggregated final results, 5 seeds (res/*/SUMMARY.md, res/PRIMARY_*, docs/results_5seeds)"
say "ALL DONE"
