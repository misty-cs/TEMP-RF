#!/bin/zsh
# WiSig final runs started before Pycom/LoRa tuning is complete (author's decision
# 2026-10-02, PROTOCOL.md 16.1): WiSig uses only its own tuning, which is complete.
# Gate: the WiSig tuned pilot audit passes. Same commands as run_all.sh final().
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
N=(nice -n 10)
LOG=run_all.log
say() { echo "=== [wisig-early] $* $(date)" | tee -a $LOG; }
commit() { msg=$1; shift
  for t in 1 2 3 4 5; do
    for p in "$@"; do [ -e "$p" ] && git add -A -- "$p"; done
    git diff --cached --quiet && return 0
    git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && return 0
    sleep 30
  done; }
waitdata() { while [ ! -d data/wisig ]; do echo "$(date) wisig data missing, waiting" >> $LOG; sleep 120; done; }
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
say "START (waiting for the WiSig tuned pilot)"
while pgrep -f "main.py --dataset wisig --root res_pilot_tuned" >/dev/null; do sleep 60; done
"${N[@]}" $PY -u main.py --dataset wisig --root res_pilot_tuned --shifts 0 --seeds 42 --go >> res_pilot_tuned_wisig.log 2>&1   # skips if complete
[ "$(ok_runs res_pilot_tuned wisig)" -ge 1 ] || { say "WiSig tuned pilot audit FAILED - no early finals"; exit 1; }
commit "Tuned pilot wisig (split 0, seed 42): audit passes" res_pilot_tuned/wisig
say "pilot passed; WiSig final runs"
want=$(( $($PY -c "import datasets as D; print(len(D.default_shifts('wisig')))") * 5 ))
for s in 42 7 13 21 100; do waitdata; "${N[@]}" $PY -u main.py --dataset wisig --root res --seeds $s --go >> res_wisig.log 2>&1; done
for attempt in 1 2 3 4 5 6 7 8; do
  [ "$(ok_runs res wisig)" -ge "$want" ] && break
  say "res wisig: $(ok_runs res wisig)/$want audited runs; repeating (attempt $attempt)"; sleep 60
  waitdata; "${N[@]}" $PY -u main.py --dataset wisig --root res --go >> res_wisig.log 2>&1
done
for attempt in 1 2 3; do waitdata; "${N[@]}" $PY -u main.py --dataset wisig --joint --go >> res_joint_wisig.log 2>&1; done
commit "Final runs: wisig (all splits x 5 seeds + joint control), started early" res/wisig res_joint/wisig
say "WiSig finals done ($(ok_runs res wisig)/$want audited runs)"
