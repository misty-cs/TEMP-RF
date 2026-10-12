#!/bin/zsh
# Stage B and the remaining aggregation steps (PROTOCOL.md 16.1, 2026-10-11).
# Takes over from run_all.sh v3 (its parent shell is frozen, so it does not start its
# own stage B): LoRa stage B starts at once, in parallel with Pycom stage A; when
# Pycom stage A is complete, the 3-seed aggregation (same code as run_all.sh v3), then
# Pycom stage B, the joint controls and the 5-seed aggregation. Same commands as v3.
cd "$(dirname "$0")"
setopt null_glob   # a pattern without matches (e.g. no Pycom joint runs yet) must not stop the script
PY=${PYTHON:-/usr/local/bin/python3.13}
N=(nice -n 10)
LOG=run_all.log
say() { echo "=== [stageB] $* $(date)" | tee -a $LOG; }
datadir() { case $1 in wisig) echo data/wisig;; pycom) echo data/pycom_indoor/Day_1;; lora) echo data/lora_indoor/Day1;; esac; }
waitdata() { while [ ! -d "$(datadir $1)" ]; do echo "$(date) $1 data missing, waiting" >> $LOG; sleep 120; done; }
commit() { msg=$1; shift; [ $# -eq 0 ] && set -- .
  until mkdir .commit_lock 2>/dev/null; do sleep 5; done
  for p in "$@"; do [ -e "$p" ] && git add -A -- "$p" >/dev/null 2>&1; done
  git diff --cached --quiet || git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  rmdir .commit_lock; }
nshifts() { $PY -c "import datasets as D; print(len(D.default_shifts('$1')))"; }
shifts() { $PY -c "import datasets as D; print(*D.default_shifts('$1'))"; }
ok_runs() {
  $PY - "$1" "$2" <<'PYEOF'
import sys, glob, json, os, manifest, datasets as D
root, ds = sys.argv[1], sys.argv[2]; n = 0
for a in glob.glob(f'{root}/{ds}/shift*/seed*/audit.json'):
    d = os.path.dirname(a); A = json.load(open(a))
    if A.get('all_pass') and not os.path.exists(f'{d}/AUDIT_FAILED') and \
       manifest.artifacts_match(d, A.get('artifacts'), D.DATASETS[ds]['traj_day'])[0]:
        n += 1
print(n)
PYEOF
}
par_main() { ds=$1; shift
  for sh in $(shifts $ds); do "${N[@]}" $PY -u main.py --dataset $ds --shifts $sh "$@" --go >> res_${ds}_s$sh.log 2>&1 & done; wait; }
final_seeds() { ds=$1; shift; want=$(( $(nshifts $ds) * $# ))
  for attempt in 1 2 3 4 5 6 7 8; do
    waitdata $ds; par_main $ds --root res --seeds "$@"
    [ "$(ok_runs res $ds)" -ge "$want" ] && return 0
    say "res $ds seeds $*: $(ok_runs res $ds)/$want audited runs after attempt $attempt; retrying"; sleep 60
  done; say "final $ds seeds $* incomplete ($(ok_runs res $ds)/$want)"; return 1; }
joint_seeds() { ds=$1; shift
  for attempt in 1 2 3; do waitdata $ds
    for sd in "$@"; do "${N[@]}" $PY -u main.py --dataset $ds --joint --seeds $sd --go >> res_joint_${ds}_seed$sd.log 2>&1 & done; wait; done; }
snapshot() { name=$1; shift; R=res_$name; J=res_joint_$name
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
  for t in wisig pycom lora; do cp $R/$t/SUMMARY.md docs/results_$name/SUMMARY_$t.md 2>/dev/null; done; }

say "START (LoRa stage B now, parallel to Pycom stage A)${LORA_PID:+ [continuation]}"
if [ -n "$LORA_PID" ]; then   # continuation (2026-10-12): the LoRa block already runs as $LORA_PID
  say "continuation: LoRa stage B already running as PID $LORA_PID"
  ( while kill -0 $LORA_PID 2>/dev/null; do sleep 300; done
    for attempt in 1 2 3; do final_seeds lora 42 7 13 21 100 && break; done
    joint_seeds lora 42 7 13 21 100
    commit "Final runs stage B: lora (seeds 21, 100) + joint control" res/lora res_joint/lora
    say "lora stage B done" ) &
else
( final_seeds lora 42 7 13 21 100; joint_seeds lora 42 7 13 21 100
  commit "Final runs stage B: lora (seeds 21, 100) + joint control" res/lora res_joint/lora
  say "lora stage B done" ) &
fi
LORA=$!
# Pycom stage A is run by run_all.sh v3 (its pipeline subshell still runs); wait for it
until [ "$(ok_runs res pycom)" -ge "$(( $(nshifts pycom) * 3 ))" ] && ! pgrep -f "main.py --dataset pycom --shifts [0-9]* --root res --seeds 42 7 13" >/dev/null; do sleep 300; done
commit "Final runs stage A: pycom (all splits x seeds 42, 7, 13)" res/pycom
snapshot 3seeds 42 7 13
commit "Aggregated results, 3 seeds (docs/results_3seeds, res_3seeds)" docs/results_3seeds res_3seeds res_joint_3seeds
say "STAGE A DONE (3-seed results: docs/results_3seeds)"
( final_seeds pycom 42 7 13 21 100; joint_seeds pycom 42 7 13 21 100
  commit "Final runs stage B: pycom (seeds 21, 100) + joint control" res/pycom res_joint/pycom ) &
PYC=$!
joint_seeds wisig 21 100; commit "WiSig joint control seeds 21, 100" res_joint/wisig
wait $LORA $PYC
$PY -u aggregate.py --tags wisig pycom lora --root res > aggregate_final.log 2>&1
mkdir -p docs/results_5seeds; cp aggregate_final.log docs/results_5seeds/aggregate.txt; cp res/PRIMARY_*.md docs/results_5seeds/ 2>/dev/null
for t in wisig pycom lora; do cp res/$t/SUMMARY.md docs/results_5seeds/SUMMARY_$t.md 2>/dev/null; done
commit "Aggregated final results, 5 seeds (res/*/SUMMARY.md, res/PRIMARY_*, docs/results_5seeds)" res docs/results_5seeds
say "ALL DONE"
