#!/bin/zsh
# Copy the run-control, tuning and pool-check logs into docs/logs/ so that they
# are part of the repository (top-level *.log files are excluded by .gitignore).
# Run before every push; per-run logs stay inside the run folders (audit.log is
# tracked, the long training logs are not). PROTOCOL.md 16.1.
cd "$(dirname "$0")/.."
setopt null_glob
mkdir -p docs/logs
for f in run_all.log tune1_*.log tune2_*.log pool_check_*.log res_*_s*.log res_*.log res_joint_*.log res_pilot_tuned_*.log aggregate_*.log; do
  [ -f "$f" ] && cp "$f" docs/logs/"${f%.log}.txt"
done
[ -f tools/governor.log ] && cp tools/governor.log docs/logs/governor.txt
ls docs/logs | wc -l | xargs echo "logs collected:"
