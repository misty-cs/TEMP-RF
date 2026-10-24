"""Matched-acceptance diagnostic (requested 2026-10-23), step 1: per-signal test scores.

Runs an in-memory copy of the official common_eval.py for one finished run, with two
changes made to the copy only (common_eval.py itself is not modified, so code
identity of the audited runs is untouched):
  1. tuned_calibration() also stores the test-day predictions and scores of every rule;
  2. the script stops after the tuned-calibration section and writes nothing into the
     run folder (no common_eval.json, no manifest) - output goes to analysis/matched_accept/.
The copy must reproduce the official per-rule closed accuracy, AUROC and tuned metrics
exactly (checked here); otherwise the run is marked FAILED and not used.
Usage: python analysis/matched_accept/run_scores.py <ds> <shift> <seed>
"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ds, shift, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
run_dir = os.path.join(ROOT, 'res', ds, f'shift{shift}', f'seed{seed}')
OUT = os.path.join(ROOT, 'analysis', 'matched_accept', 'scores'); os.makedirs(OUT, exist_ok=True)
src = open(os.path.join(ROOT, 'common_eval.py')).read()
cut = src.index('# ── selection-weight sensitivity')
src = src[:cut]
a = "    pk, sk = _chunked(rule, T_k); _, su = _chunked(rule, T_u)\n"
assert src.count(a) == 1
src = src.replace(a, a + "    _DUMP[_DN] = dict(pk=pk, sk=sk, su=su, yT=yT, thr=float(cands[best]))\n")
b = "    res['rules'][name]['tuned'] = tuned_calibration("
assert src.count(b) == 1
src = src.replace(b, "    _DN = name\n" + b)
for nm in ("res['rules']['raw_centroid']['tuned'] = tuned_calibration(", "res['rules']['raw_centroid_cfo']['tuned'] = tuned_calibration("):
    assert src.count(nm) == 1
    src = src.replace(nm, f"_DN = '{nm.split(chr(39))[3]}'\n" + nm)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.argv = ['common_eval.py', ds, run_dir, str(shift), str(seed)]
sys.path.insert(0, ROOT); os.chdir(ROOT)
from datasets import DATASETS
for k, v in DATASETS[ds].get('env', {}).items(): os.environ[k] = str(v)
g = {'__name__': '__ce_copy__', '__file__': os.path.join(ROOT, 'common_eval.py'), '_DUMP': {}}
exec(compile(src, 'common_eval_copy', 'exec'), g)
res, dump = g['res'], g['_DUMP']
off = json.load(open(os.path.join(run_dir, 'common_eval.json')))['rules']
bad = []
for r in dump:
    o, n = off[r], res['rules'][r]
    for m in ('closed', 'auroc'):
        if abs(o[m] - n[m]) > 1e-9: bad.append((r, m, o[m], n[m]))
    for m in ('accept', 'unk_det', 'known_ok'):
        if abs(o['tuned'][m] - n['tuned'][m]) > 1e-9: bad.append((r, 'tuned_' + m, o['tuned'][m], n['tuned'][m]))
tag = f'{ds}_shift{shift}_seed{seed}'
np.savez_compressed(os.path.join(OUT, tag + '.npz'), **{f'{r}__{k}': np.asarray(v) for r, d in dump.items() for k, v in d.items()})
json.dump(dict(run=tag, reproduced=not bad, mismatches=bad, rules=sorted(dump)), open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
print(tag, 'REPRODUCED' if not bad else f'FAILED {bad[:3]}', flush=True)
