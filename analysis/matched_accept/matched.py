"""Matched-acceptance diagnostic, step 2 (2026-10-23).

For every reproduced run and rule: choose the threshold on the TEST day so that exactly a
share a of the known test signals is accepted (a = 0.80, 0.90, 0.95), then report the
unknown-signal rejection rate and the known accepted-and-correct rate at that threshold.
This uses test labels to place the operating point: it is a DIAGNOSTIC that compares rules
at equal known acceptance, not a deployable threshold (the deployable protocol is the
validation-selected threshold, ICC Table I). Rules: those in ICC Table I.
Output: analysis/matched_accept/matched.json and a printed table (mean +- sd over runs,
and tuned MP-NN minus each rule: mean difference and runs where MP-NN is higher)."""
import os, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ['msp', 'openmax', 'static', 'knn', 'temp_rff', 'temp_rff_default', 'raw_centroid']
ACC = [0.80, 0.90, 0.95]
out = {}
for ds in ['wisig', 'pycom', 'lora']:
    runs = []
    for jf in sorted(glob.glob(os.path.join(HERE, 'scores', f'{ds}_*.json'))):
        J = json.load(open(jf))
        if not J['reproduced']: continue
        Z = np.load(jf[:-5] + '.npz'); runs.append(Z)
    if not runs: continue
    R = {r: {a: {'rej': [], 'ac': []} for a in ACC} for r in RULES}
    for Z in runs:
        for r in RULES:
            sk, su, pk, yT = Z[f'{r}__sk'], Z[f'{r}__su'], Z[f'{r}__pk'], Z[f'{r}__yT']
            for a in ACC:
                t = np.quantile(sk, a)                     # accept known if score <= t
                R[r][a]['rej'].append(float(np.mean(su > t)))
                R[r][a]['ac'].append(float(np.mean((sk <= t) & (pk == yT))))
    o = out[ds] = {'n_runs': len(runs)}
    for r in RULES:
        o[r] = {str(a): {m: [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3)] for m, v in R[r][a].items()} for a in ACC}
    for r in RULES:
        if r == 'temp_rff': continue
        o[f'diff_temp_rff_minus_{r}'] = {str(a): dict(mean=round(float(np.mean(np.array(R['temp_rff'][a]['rej']) - np.array(R[r][a]['rej']))), 3),
                                                     higher=int(np.sum(np.array(R['temp_rff'][a]['rej']) > np.array(R[r][a]['rej']))), n=len(runs)) for a in ACC}
json.dump(out, open(os.path.join(HERE, 'matched.json'), 'w'), indent=1)
for ds, o in out.items():
    print(f"\n{ds} ({o['n_runs']} runs) - unknown rejection at matched known acceptance (mean +- sd)")
    print('rule'.ljust(18) + ''.join(f'acc={a:.2f}'.rjust(16) for a in ACC) + '   A&C@0.90')
    for r in RULES:
        print(r.ljust(18) + ''.join(f"{o[r][str(a)]['rej'][0]:.3f}+-{o[r][str(a)]['rej'][1]:.3f}".rjust(16) for a in ACC) + f"   {o[r]['0.9']['ac'][0]:.3f}")
    for r in RULES:
        if r == 'temp_rff': continue
        d = o[f'diff_temp_rff_minus_{r}']['0.9']; print(f"  tuned MP-NN - {r:16s} @0.90: {d['mean']:+.3f}  (MP-NN higher in {d['higher']}/{d['n']})")
