#!/bin/zsh
# WiSig joint-training control, seed 42 for every split (PROTOCOL.md 16.1, 2026-10-05).
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/local/bin/python3.13}
LOG=run_all.log
for attempt in 1 2 3; do nice -n 10 $PY -u main.py --dataset wisig --joint --seeds 42 --go >> res_joint_wisig.log 2>&1; done
for p in res/wisig res_joint/wisig; do git add -A -- $p; done
git diff --cached --quiet || git commit -q -m "WiSig: final runs (4 splits x 5 seeds) and joint control (seed 42)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "=== [wisig-joint] done $(date)" >> $LOG
