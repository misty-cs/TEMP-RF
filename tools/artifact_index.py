"""Index of the large run artifacts that are not stored in git (PROTOCOL.md 16.1).

Trained models (*.keras, *.h5) and large arrays (prototype banks, calibration
scores) are too large for the repository. This writes docs/ARTIFACTS.tsv with
path, size and SHA-256 of every such file under the result folders, so that an
archived copy can be verified file by file.

    python tools/artifact_index.py
"""
import hashlib, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ['res', 'res_joint', 'res_pilot_tuned', 'res_tune', 'docs/superseded_runs']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def ignored(paths):
    out = subprocess.run(['git', 'check-ignore', '--stdin'], cwd=ROOT, input='\n'.join(paths),
                         capture_output=True, text=True).stdout
    return set(out.split('\n')) - {''}


files = []
for top in FOLDERS:
    for d, _, fs in os.walk(os.path.join(ROOT, top)):
        files += [os.path.relpath(os.path.join(d, f), ROOT) for f in fs]
skip = ignored(files)
rows = [(p, os.path.getsize(os.path.join(ROOT, p)), sha256(os.path.join(ROOT, p))) for p in sorted(skip)]
with open(os.path.join(ROOT, 'docs', 'ARTIFACTS.tsv'), 'w') as f:
    f.write('path\tbytes\tsha256\n')
    for r in rows:
        f.write('%s\t%d\t%s\n' % r)
print(f'{len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB indexed in docs/ARTIFACTS.tsv')
