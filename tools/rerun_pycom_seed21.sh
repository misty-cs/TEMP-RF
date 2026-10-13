#!/bin/zsh
# Re-run the three Pycom seed-21 runs whose first attempt was ended before writing a
# manifest (2026-10-12); one worker per split, same command as run_stageB.sh.
cd "$(dirname "$0")/.."
for sh in 8 12 16; do nice -n 10 /usr/local/bin/python3.13 -u main.py --dataset pycom --shifts $sh --root res --seeds 21 --go >> res_pycom_s$sh.log 2>&1 & done; wait
echo "=== [rerun] pycom seed 21 splits 8, 12, 16 done $(date)" >> run_all.log
