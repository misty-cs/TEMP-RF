import os, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('NB', '50000')
import step2_errvec as S
d2 = {}
for dev in sorted(glob.glob(f'{S.ROOT}/day2/equ/Device_*'), key=lambda p: int(p.split('_')[-1])):
    d2[int(dev.split('_')[-1])] = S.feats(sorted(glob.glob(f'{dev}/*.bin'))[0])
for kind in ('amplitude', 'errvec'):
    tr = {d: [{kind: f[kind][:len(f[kind]) // 2]}] for d, f in d2.items()}
    te = {d: [{kind: f[kind][len(f[kind]) // 2:]}] for d, f in d2.items()}
    c, k, n = S.evaluate(tr, te, kind)
    print(f'{kind:10s} within one file (1st half -> 2nd half): centroid {c:.3f} 5-NN {k:.3f} (n={n})')
