#!/bin/zsh
# Battery guard (PROTOCOL.md 16.1): pause the study at <=20% battery without a charger,
# and resume it automatically once the Mac is on AC power again with >=30% battery.
# Runs until stopped. Pauses and resumes are logged in tools/study_pause.log.
cd "$(dirname "$0")/.."
paused_by_me=0
while true; do
  b=$(pmset -g batt); pct=$(echo "$b" | grep -oE '[0-9]+%' | head -1 | tr -d %)
  if [ $paused_by_me = 0 ] && echo "$b" | grep -q "Battery Power" && [ "${pct:-100}" -le 20 ]; then
    ./tools/study_pause.sh stop; echo "$(date) battery ${pct}% on battery power: study paused" >> tools/study_pause.log; paused_by_me=1
  elif [ $paused_by_me = 1 ] && echo "$b" | grep -q "AC Power" && [ "${pct:-0}" -ge 30 ] && [ -d "/Volumes/TOSHIBA EXT/lora_oregon/Indoor_dedup/Day1" ]; then
    ./tools/study_pause.sh cont; echo "$(date) on AC power, battery ${pct}%, drive present: study resumed" >> tools/study_pause.log; paused_by_me=0
  fi
  sleep 120
done
