"""Holm-corrected tests restricted to the learned rules shown in ICC Table I
(tuned MP-NN minus MSP, OpenMax, static centroid, k-NN; 4 outcomes; 3 datasets
= 48 tests). Uses the raw split-level p-values of the pre-registered primary
analysis (res/PRIMARY_wisig_pycom_lora.json, paired t-test and Wilcoxon) and
the same Holm function as aggregate.py. Post-hoc subset; read-only. 2026-10-22."""
import json, os, sys, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = sys.argv[:1]
def holm(pv):
    ks = sorted(pv, key=lambda k_: 1.0 if np.isnan(pv[k_]) else pv[k_]); out, run_ = {}, 0.0
    for i, k_ in enumerate(ks):
        p_ = 1.0 if np.isnan(pv[k_]) else pv[k_]
        run_ = max(run_, min(1.0, (len(ks) - i) * p_)); out[k_] = run_
    return out
P = json.load(open('res/PRIMARY_wisig_pycom_lora.json'))
RULES = ('msp', 'openmax', 'static', 'knn'); res = {}
for kind in ('t_p', 'w_p'):
    pv = {f'{ds}|{k}': v[kind] for ds in ('wisig', 'pycom', 'lora') for k, v in P[ds].items() if k.split('|')[0] in RULES}
    h = holm(pv); best = min(h, key=h.get)
    res[kind] = dict(n_tests=len(pv), min_holm=round(h[best], 4), at=best, raw_min=round(min(x for x in pv.values() if not np.isnan(x)), 4))
json.dump(res, open('analysis/table1_tests.json', 'w'), indent=1); print(json.dumps(res, indent=1))
