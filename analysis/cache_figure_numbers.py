"""Numbers quoted in the text for the cache-based figures (E1a, E3a, E6a, E7).
Same computations as make_figures.py; writes analysis/cache_figure_numbers.json."""
import json, os, glob, numpy as np, warnings
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
out = {}
for ds in ['wisig', 'pycom', 'lora']:
    C = {int(f.split('shift')[1][:-4]): dict(np.load(f)) for f in glob.glob(f'analysis/cache/{ds}_shift*.npz')}
    r = out[ds] = {}
    # E3a
    r['E3a_auroc_vs_P'] = {}
    for P in [1, 5, 20, 50, 100, 'all']:
        a = []
        for sh, c in sorted(C.items()):
            e, eA, yA, K = l2n(c['e']), l2n(c['eA']), c['yA'], len(c['known']); best = np.full((len(e), K), -1.0)
            for k in range(K):
                X = eA[yA == k]
                B = X if P == 'all' or P >= len(X) else l2n(MiniBatchKMeans(P, random_state=0, n_init=3).fit(X).cluster_centers_)
                best[:, k] = (e @ B.T).max(1)
            a.append(roc_auc_score(np.isin(c['y'], c['unknown']), 1 - best.max(1)))
        r['E3a_auroc_vs_P'][str(P)] = round(float(np.mean(a)), 3)
    # E6a / E7 (MP-NN, P=all, W=1)
    devacc, devauc, fx = {}, {}, {}
    for sh, c in sorted(C.items()):
        known = sorted(c['known'].tolist()); e, eA, yA = l2n(c['e']), l2n(c['eA']), c['yA']
        S = np.stack([(e @ eA[yA == i].T).max(1) for i in range(len(known))], 1); s = 1 - S.max(1); pred = S.argmax(1)
        isu = np.isin(c['y'], c['unknown'])
        thr = np.quantile(1 - (l2n(c['eC']) @ eA.T).max(1), 0.9)
        fx[sh] = dict(known_accept=round(float((s[~isu] <= thr).mean()), 3), unknown_detect=round(float((s[isu] > thr).mean()), 3))
        for d in set(c['y'][~isu].tolist()):
            m = c['y'] == d; devacc[int(d)] = float((np.array(known)[pred[m]] == d).mean())
        for u in c['unknown']:
            m = (c['y'] == u) | ~isu; devauc[int(u)] = float(roc_auc_score(c['y'][m] == u, s[m]))
    r['E6a_fixed90'] = fx
    r['E7_acc_known'] = dict(min=round(min(devacc.values()), 3), max=round(max(devacc.values()), 3), n_below_0p5=sum(v < 0.5 for v in devacc.values()), n=len(devacc))
    r['E7_auroc_unknown'] = dict(min=round(min(devauc.values()), 3), max=round(max(devauc.values()), 3), n_below_0p5=sum(v < 0.5 for v in devauc.values()), n=len(devauc), per_device={k: round(v, 3) for k, v in sorted(devauc.items())})
json.dump(out, open('analysis/cache_figure_numbers.json', 'w'), indent=1)
print(json.dumps({d: {k: v for k, v in out[d].items() if k != 'E7_auroc_unknown'} | {'E7u': {k: v for k, v in out[d]['E7_auroc_unknown'].items() if k != 'per_device'}} for d in out}, indent=0))
print({d: out[d]['E7_auroc_unknown']['per_device'] for d in out})
