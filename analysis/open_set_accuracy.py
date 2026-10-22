"""Balanced open-set accuracy under the tuned thresholds, from the official
common_eval.json of all final runs: 0.5 x (known accepted and correct) + 0.5 x
(unknown-device detection). Read-only; added 2026-10-22 on request."""
import json, glob, os, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = {}
for ds in ['wisig', 'pycom', 'lora']:
    fs = sorted(glob.glob(f'res/{ds}/shift*/seed*/common_eval.json')); out[ds] = {'n_runs': len(fs)}
    for r in ['msp', 'probe', 'openmax', 'static', 'knn', 'temp_rff', 'temp_rff_default', 'raw_centroid']:
        T = [json.load(open(f))['rules'][r]['tuned'] for f in fs]
        b = [0.5 * (t['known_ok'] + t['unk_det']) for t in T]
        out[ds][r] = dict(known_ok=round(float(np.mean([t['known_ok'] for t in T])), 3),
                          unk_det=round(float(np.mean([t['unk_det'] for t in T])), 3),
                          balanced_osa=round(float(np.mean(b)), 3), sd=round(float(np.std(b, ddof=1)), 3))
json.dump(out, open('analysis/open_set_accuracy.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
