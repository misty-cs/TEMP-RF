"""Mean and SD over all final runs for ICC Table I (closed, AUROC, tuned
unknown rejection, tuned known accepted-and-correct). Read-only. 2026-10-23."""
import json, glob, os, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = ['msp', 'openmax', 'static', 'knn', 'temp_rff', 'temp_rff_default', 'raw_centroid']; out = {}
for ds in ['wisig', 'pycom', 'lora']:
    E = [json.load(open(f))['rules'] for f in sorted(glob.glob(f'res/{ds}/shift*/seed*/common_eval.json'))]
    out[ds] = {'n_runs': len(E)}
    for r in R:
        v = {'closed': [e[r]['closed'] for e in E], 'auroc': [e[r]['auroc'] for e in E],
             'rej': [e[r]['tuned']['unk_det'] for e in E], 'ac': [e[r]['tuned']['known_ok'] for e in E]}
        out[ds][r] = {m: [round(float(np.mean(x)), 3), round(float(np.std(x, ddof=1)), 3)] for m, x in v.items()}
json.dump(out, open('analysis/table1_sd.json', 'w'), indent=1)
for ds in out:
    print(ds, out[ds]['n_runs'])
    for r in R: print('  ', r, out[ds][r])
