#!/bin/zsh
# Pause (stop) or resume (cont) the whole study without losing work (PROTOCOL.md 16.1).
#   tools/study_pause.sh stop   # freeze governor, schedulers and all training jobs (SIGSTOP)
#   tools/study_pause.sh cont   # continue them (SIGCONT); the governor re-applies its limit
#   tools/study_pause.sh status
# Frozen processes keep their state and use no CPU. Do not disconnect the data drive
# while a run is frozen in the middle of reading data; prefer to pause first, then
# disconnect, and reconnect before resuming.
cd "$(dirname "$0")/.."
LOG=tools/study_pause.log
pids() {   # governor, schedulers, every training job and all its descendants
  local roots=($(pgrep -f "tools/governor.py|run_all.sh|run_lora_parallel.sh|main.py --dataset|tune_phase[12].py"))
  local all=($roots) todo=($roots) p c
  while (( ${#todo} )); do p=$todo[1]; todo=(${todo[2,-1]}); for c in $(pgrep -P $p); do all+=($c); todo+=($c); done; done
  print -l ${(u)all}
}
case $1 in
  stop) P=($(pids)); kill -STOP $P 2>/dev/null; echo "$(date) PAUSED ${#P} processes" | tee -a $LOG ;;
  cont) P=($(pids)); kill -CONT $P 2>/dev/null; echo "$(date) RESUMED ${#P} processes" | tee -a $LOG ;;
  status) P=($(pids)); echo "${#P} study processes; stopped: $(ps -o stat= -p ${(j:,:)P} 2>/dev/null | grep -c T)" ;;
  *) echo "usage: $0 stop|cont|status"; exit 1 ;;
esac
