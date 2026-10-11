#!/bin/zsh
# Pause the study when the battery is low and no charger is connected (PROTOCOL.md 16.1).
cd "$(dirname "$0")/.."
while true; do
  b=$(pmset -g batt); pct=$(echo "$b" | grep -oE '[0-9]+%' | head -1 | tr -d %)
  if echo "$b" | grep -q "Battery Power" && [ "${pct:-100}" -le 20 ]; then
    ./tools/study_pause.sh stop; echo "$(date) battery ${pct}% on battery power: study paused" >> tools/study_pause.log; exit 0
  fi
  sleep 120
done
