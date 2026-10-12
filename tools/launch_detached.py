"""Start a command in its own session, detached from the calling shell, so that it
survives when the shell that started it is closed. Output goes to the given log.
    python tools/launch_detached.py LOG VAR=value ... -- command args"""
import os, subprocess, sys
log = sys.argv[1]; i = sys.argv.index('--'); env = dict(os.environ)
for kv in sys.argv[2:i]:
    k, v = kv.split('=', 1); env[k] = v
p = subprocess.Popen(sys.argv[i + 1:], stdout=open(log, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), start_new_session=True)
print(p.pid)
