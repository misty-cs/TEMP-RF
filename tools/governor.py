"""Load governor (run control only; PROTOCOL.md 16.1, 2026-10-05).

The Mac (10 cores, 16 GB) thrashes when more than a few training jobs run at
once. Every 60 s this keeps at most MAX_ACTIVE training jobs running and pauses
the rest with SIGSTOP (no work is lost; SIGCONT resumes exactly where the job
stopped). Results do not depend on it: jobs are seeded and deterministic.

A training job = a top-level tune_phase1.py / tune_phase2.py / main.py process
with all its descendants. Priority: Pycom/WiSig jobs first, then LoRa jobs
whose current candidate has run longest (nearly finished work is not paused).

    python tools/governor.py [MAX_ACTIVE]      # default 4; log: tools/governor.log
"""
import os, re, signal, subprocess, sys, time

MAX_ACTIVE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
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
        J.append(dict(pid=p, pids=[p] + desc, cmd=cmd, lora=lora,
                      key=(1 if lora else 0, -youngest)))   # non-LoRa first; then the longest-running candidate
    return sorted(J, key=lambda j: j['key'])


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
