"""Load governor (run control only; PROTOCOL.md 16.1, 2026-10-05).

The Mac (10 cores, 16 GB) thrashes when more than a few training jobs run at
once. Every 60 s this keeps at most MAX_ACTIVE training jobs running and pauses
the rest with SIGSTOP (no work is lost; SIGCONT resumes exactly where the job
stopped). Results do not depend on it: jobs are seeded and deterministic.

A training job = a top-level tune_phase1.py / tune_phase2.py / main.py process
with all its descendants. Priority: LoRa is the critical path, so up to LORA_SLOTS
LoRa jobs first, then the others; within a group the worker with the fewest completed
final runs first, ties by the oldest worker (stable; changed 2026-10-10).

    python tools/governor.py [MAX_ACTIVE [LORA_SLOTS]]   # default 4 3; log: tools/governor.log
"""
import os, re, signal, subprocess, sys, time

MAX_ACTIVE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LORA_SLOTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'governor.log')
JOB = re.compile(r'python[\d.]*\s+-u\s+(tune_phase[12]\.py|main\.py)\s', re.I)


def log(msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def etime_s(t):                      # [[dd-]hh:]mm:ss -> seconds
    d, _, t = t.rpartition('-')
    p = [int(x) for x in t.split(':')]
    while len(p) < 3:
        p.insert(0, 0)
    return int(d or 0) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def snapshot():
    out = subprocess.run(['ps', '-axo', 'pid=,ppid=,etime=,stat=,command='], capture_output=True, text=True).stdout
    procs = {}
    for line in out.splitlines():
        f = line.split(None, 4)
        if len(f) == 5:
            procs[int(f[0])] = dict(ppid=int(f[1]), etime=etime_s(f[2]), stat=f[3], cmd=f[4])
    return procs


def descendants(procs, pid):
    kids, todo = [], [pid]
    while todo:
        p = todo.pop()
        c = [q for q, v in procs.items() if v['ppid'] == p]
        kids += c; todo += c
    return kids


def jobs(procs):
    tops = [p for p, v in procs.items() if JOB.search(v['cmd'])
            and not JOB.search(procs.get(v['ppid'], {}).get('cmd', ''))]
    J = []
    for p in tops:
        cmd = procs[p]['cmd']; desc = descendants(procs, p)
        youngest = min([procs[q]['etime'] for q in desc] or [procs[p]['etime']])
        lora = '--dataset lora' in cmd
        J.append(dict(pid=p, pids=[p] + desc, cmd=cmd, lora=lora, age=youngest))
    # LoRa is the critical path: up to LORA_SLOTS LoRa jobs first, then the other jobs, then
    # the remaining LoRa jobs. Within a group (2026-10-10): the worker with the fewest
    # completed final runs first, ties by the oldest worker (stable order). The earlier key
    # (youngest child process) demoted a run each time it started its next step, so the
    # single Pycom slot rotated between workers and no Pycom run finished.
    key = lambda j: (completed(j['cmd']), j['pid'])
    lo = sorted([j for j in J if j['lora']], key=key)
    ot = sorted([j for j in J if not j['lora']], key=key)
    return lo[:LORA_SLOTS] + ot + lo[LORA_SLOTS:]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def completed(cmd):
    """Number of audited runs of the split a main.py worker is responsible for."""
    ds = re.search(r'--dataset (\w+)', cmd); sh = re.search(r'--shifts (\d+)', cmd)
    root = re.search(r'--root (\S+)', cmd)
    if not (ds and sh) or 'main.py' not in cmd:
        return 0
    d = os.path.join(ROOT, root.group(1) if root else 'res', ds.group(1), 'shift' + sh.group(1))
    try:
        return sum(os.path.exists(os.path.join(d, s, 'audit.json')) for s in os.listdir(d) if s.startswith('seed') and '.' not in s)
    except FileNotFoundError:
        return 0


def send(job, sig):
    for p in job['pids']:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass


def short(cmd):
    m = re.search(r'(tune_phase[12]\.py|main\.py).*', cmd)
    return (m.group(0) if m else cmd)[:70]


paused = set()


def resume_all(*_):                                  # never leave jobs paused when the governor stops
    for j in jobs(snapshot()):
        send(j, signal.SIGCONT)
    log('stop: all jobs resumed'); sys.exit(0)


if __name__ != '__main__':
    raise SystemExit
signal.signal(signal.SIGTERM, resume_all); signal.signal(signal.SIGINT, resume_all); signal.signal(signal.SIGHUP, resume_all)
log(f'start, MAX_ACTIVE={MAX_ACTIVE}')
while True:
    try:
        procs = snapshot(); J = jobs(procs)
        for i, j in enumerate(J):
            if i < MAX_ACTIVE:
                send(j, signal.SIGCONT)            # also resumes children started while paused
                if j['pid'] in paused:
                    paused.discard(j['pid']); log(f'resume {j["pid"]} {short(j["cmd"])}')
            else:
                send(j, signal.SIGSTOP)
                if j['pid'] not in paused:
                    paused.add(j['pid']); log(f'pause  {j["pid"]} {short(j["cmd"])}')
        paused &= {j['pid'] for j in J}
    except Exception as e:                           # never stop governing because of one bad snapshot
        log(f'error {e!r}')
    time.sleep(60)
