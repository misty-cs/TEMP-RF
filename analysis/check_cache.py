"""Compare cache-based scores with the official common_eval.json of the same run."""
import json, sys, numpy as np, os
from sklearn.metrics import roc_auc_score
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
ds = sys.argv[1]
for sh in [int(s) for s in sys.argv[2:]]:
    c = dict(np.load(f'analysis/cache/{ds}_shift{sh}.npz')); known = sorted(c['known'].tolist())
    e, eA, yA = l2n(c['e']), l2n(c['eA']), c['yA']; isu = np.isin(c['y'], c['unknown'])
    best = np.stack([(e @ eA[yA == i].T).max(1) for i in range(len(known))], 1)
    C = l2n(np.stack([eA[yA == i].mean(0) for i in range(len(known))])); sc = e @ C.T
    E = json.load(open(f'res/{ds}/shift{sh}/seed42/common_eval.json'))
    for name, S, off in [('MP-NN all', best, 'temp_rff_default'), ('static', sc, 'static')]:
        acc = np.mean(np.array(known)[S.argmax(1)][~isu] == c['y'][~isu])
        print(ds, sh, f'{name:10s} cache acc {acc:.3f} auroc {roc_auc_score(isu, 1 - S.max(1)):.3f} | official {off}: {E["rules"][off]["closed"]:.3f} {E["rules"][off]["auroc"]:.3f}')
    print('   n test', len(c['y']), 'known', known[:4], 'unk', c['unknown'], '| official unk', E['unknown'])
