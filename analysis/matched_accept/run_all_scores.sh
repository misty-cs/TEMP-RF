#!/bin/zsh
# Step 1 for every final run of the given datasets (default: wisig). Skips runs already reproduced.
cd "$(dirname $0)/../.."
for ds in ${@:-wisig}; do
  for d in res/$ds/shift*/seed*(N); do
    sd=$(basename $(dirname $d)); s=${sd#shift}; ed=$(basename $d); e=${ed#seed}; tag=${ds}_shift${s}_seed${e}
    grep -q '"reproduced": true' analysis/matched_accept/scores/$tag.json 2>/dev/null && continue
    /usr/local/bin/python3.13 -W ignore analysis/matched_accept/run_scores.py $ds $s $e 2>&1 | tail -1
  done
done
echo ALL_DONE
