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
# v3 (2026-10-05, PROTOCOL.md 16.1): one pipeline per dataset (its tuning done ->
# its pilot -> stage A: seeds 42, 7, 13 + joint control), one worker per split/seed;
# 3-seed aggregation when all three are done; then stage B (seeds 21, 100) and
# the 5-seed aggregation. Tuning itself is run by run_all.sh v1 (Pycom) and
# run_lora_parallel.sh (LoRa); this script waits for it. tools/governor.py
# limits the number of training jobs.
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
N=(nice -n 10)
LOG=run_all.log
say() { echo "=== $* $(date)" | tee -a $LOG; }
datadir() { case $1 in wisig) echo data/wisig;; pycom) echo data/pycom_indoor/Day_1;; lora) echo data/lora_indoor/Day1;; esac; }
waitdata() { while [ ! -d "$(datadir $1)" ]; do echo "$(date) $1 data missing, waiting" >> $LOG; sleep 120; done; }
commit() {   # $1 message, then paths (default: all); only files allowed by .gitignore; parallel-safe
  msg=$1; shift; [ $# -eq 0 ] && set -- .
  until mkdir .commit_lock 2>/dev/null; do sleep 5; done
  for p in "$@"; do [ -e "$p" ] && git add -A -- "$p" >/dev/null 2>&1; done
  git diff --cached --quiet || git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  rmdir .commit_lock
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

mkdir .run_all_v3.lock 2>/dev/null || { echo "$(date) run_all.sh v3 already running; exit" >> $LOG; exit 0; }
say "START v3"

tuned() {   # $1 dataset: both tuning phases selected for every split and no tuning process left
  [ "$(selected_n $1 1)" = "$(nshifts $1)" ] && [ "$(selected_n $1 2)" = "$(nshifts $1)" ] && \
    ! pgrep -f "tune_phase.\.py --dataset $1" >/dev/null
}
shifts() { $PY -c "import datasets as D; print(*D.default_shifts('$1'))"; }
par_main() {   # $1 dataset, then main.py args: one worker per split, in parallel, then wait
  ds=$1; shift
  for sh in $(shifts $ds); do
    "${N[@]}" $PY -u main.py --dataset $ds --shifts $sh "$@" --go >> res_${ds}_s$sh.log 2>&1 &
  done
  wait
}
final_seeds() {   # $1 dataset, then seeds; repeated until every run is audited
  ds=$1; shift; want=$(( $(nshifts $ds) * $# ))
  for attempt in 1 2 3 4 5 6 7 8; do
    waitdata $ds; par_main $ds --root res --seeds "$@"
    [ "$(ok_runs res $ds)" -ge "$want" ] && return 0
    say "res $ds seeds $*: $(ok_runs res $ds)/$want audited runs after attempt $attempt; retrying"; sleep 60
  done
  say "final $ds seeds $* incomplete ($(ok_runs res $ds)/$want)"; return 1
}
joint_seeds() {   # $1 dataset, then seeds: joint control (split 0 only, main.py), one worker per seed
  ds=$1; shift
  for attempt in 1 2 3; do
    waitdata $ds
    for sd in "$@"; do "${N[@]}" $PY -u main.py --dataset $ds --joint --seeds $sd --go >> res_joint_${ds}_seed$sd.log 2>&1 & done
    wait
  done
}
pipeline() {   # $1 dataset: tuning -> pilot (gate) -> stage A
  ds=$1
  until tuned $ds; do sleep 300; done
  say "$ds tuning complete; tuned pilot"
  runs $ds res_pilot_tuned 1 --shifts 0 --seeds 42 || { say "tuned pilot $ds FAILED - no final runs for $ds"; return 1; }
  commit "Tuned pilot $ds (split 0, seed 42): audit passes" res_pilot_tuned/$ds
  final_seeds $ds 42 7 13
  [ $ds = wisig ] && joint_seeds $ds 42 7 13     # Pycom/LoRa joint control moved to stage B (2026-10-06)
  commit "Final runs stage A: $ds (all splits x seeds 42, 7, 13)" res/$ds res_joint/$ds
  say "final $ds stage A done ($(ok_runs res $ds) audited runs)"
}
snapshot() {   # $1 name, then seeds: aggregate only these seeds (aggregate.py uses every run under
               # its root, so res_<name>/ holds relative links to the selected runs; nothing is copied)
  name=$1; shift; R=res_$name; J=res_joint_$name
  for r in $R $J; do [ -e $r ] && { k=1; while [ -e $r.old$k ]; do k=$((k+1)); done; mv $r $r.old$k; }; done
  for t in wisig pycom lora; do
    for d in res/$t/shift* res_joint/$t/shift*; do
      [ -d "$d" ] || continue
      dst=${d/#res_joint/$J}; dst=${dst/#res\//$R/}
      for sd in "$@"; do [ -d $d/seed$sd ] && { mkdir -p $dst; ln -s ../../../$d/seed$sd $dst/seed$sd; }; done
    done
  done
  $PY -u aggregate.py --tags wisig pycom lora --root $R --seeds "$@" > aggregate_$name.log 2>&1
  mkdir -p docs/results_$name; cp aggregate_$name.log docs/results_$name/aggregate.txt
  cp $R/PRIMARY_*.md docs/results_$name/ 2>/dev/null
  for t in wisig pycom lora; do cp $R/$t/SUMMARY.md docs/results_$name/SUMMARY_$t.md 2>/dev/null; done
}

pipeline wisig & pipeline pycom & pipeline lora & wait
snapshot 3seeds 42 7 13
commit "Aggregated results, 3 seeds (docs/results_3seeds, res_3seeds)" docs/results_3seeds res_3seeds res_joint_3seeds
say "STAGE A DONE (3-seed results: docs/results_3seeds)"
(final_seeds wisig 42 7 13 21 100; joint_seeds wisig 21 100) &
for ds in pycom lora; do (final_seeds $ds 42 7 13 21 100; joint_seeds $ds 42 7 13 21 100) & done; wait
commit "Final runs stage B: seeds 21, 100 (+ joint control)" res res_joint
$PY -u aggregate.py --tags wisig pycom lora --root res > aggregate_final.log 2>&1
mkdir -p docs/results_5seeds; cp aggregate_final.log docs/results_5seeds/aggregate.txt; cp res/PRIMARY_*.md docs/results_5seeds/ 2>/dev/null
for t in wisig pycom lora; do cp res/$t/SUMMARY.md docs/results_5seeds/SUMMARY_$t.md 2>/dev/null; done
commit "Aggregated final results, 5 seeds (res/*/SUMMARY.md, res/PRIMARY_*, docs/results_5seeds)" res docs/results_5seeds
say "ALL DONE"
rmdir .run_all_v3.lock 2>/dev/null
