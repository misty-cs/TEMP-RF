#!/bin/zsh
# After run_stageB.sh "ALL DONE" and the Pycom seed-21 re-runs: make sure the 5-seed
# aggregation contains all 25 Pycom runs; if not, run it again and commit (2026-10-13).
cd "$(dirname "$0")/.."
setopt null_glob
until grep -q "\[stageB\] ALL DONE" run_all.log && grep -q "\[rerun\] pycom seed 21" run_all.log; do sleep 300; done
r=(res/pycom/shift*/seed*/audit.json)
if ! grep -q "Audited runs used: 25" docs/results_5seeds/SUMMARY_pycom.md 2>/dev/null; then
  /usr/local/bin/python3.13 -u aggregate.py --tags wisig pycom lora --root res > aggregate_final.log 2>&1
  mkdir -p docs/results_5seeds; cp aggregate_final.log docs/results_5seeds/aggregate.txt; cp res/PRIMARY_*.md docs/results_5seeds/ 2>/dev/null
  for t in wisig pycom lora; do cp res/$t/SUMMARY.md docs/results_5seeds/SUMMARY_$t.md 2>/dev/null; done
  until mkdir .commit_lock 2>/dev/null; do sleep 5; done
  git add -A -- res docs/results_5seeds >/dev/null 2>&1
  git diff --cached --quiet || git commit -q -m "Aggregated final results, 5 seeds (complete: all Pycom runs)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  rmdir .commit_lock
fi
echo "=== [final-guard] 5-seed aggregation complete (${#r} Pycom runs) $(date)" >> run_all.log
