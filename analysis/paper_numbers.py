"""All numbers used in the paper text, from the audited final runs (writes analysis/paper_numbers.json)."""
import os, glob, json, collections, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = {}
for ds in ('wisig', 'pycom', 'lora'):
    E = [json.load(open(f)) for f in sorted(glob.glob(f'res/{ds}/shift*/seed*/common_eval.json')) if '.failed' not in f]
    rules = ['softmax', 'msp', 'probe', 'openmax', 'static', 'temp_rff_p1', 'knn', 'temp_rff', 'raw_centroid', 'temp_rff_default', 'knn_default', 'probe_default', 'openmax_default']
    m = lambda f: (float(np.mean(f)), float(np.std(f, ddof=1)))
    d = {'n': len(E), 'rules': {}, 'split': {}, 'stages': {}, 'W': {}, 'fewshot': {}, 'sel': {}}
    for r in rules:
        x = {'closed': m([e['rules'][r]['closed'] for e in E])}
        if E[0]['rules'][r].get('auroc') is not None:
            x['auroc'] = m([e['rules'][r]['auroc'] for e in E])
            for p in ('fixed90', 'objective', 'tuned'):
                x[p] = {k: float(np.mean([e['rules'][r][p][k] for e in E])) for k in ('accept', 'unk_det', 'known_ok')}
        d['rules'][r] = x
    by = collections.defaultdict(list)
    for e in E: by[e['shift']].append(e)
    for sh, es in by.items():
        d['split'][sh] = {'unknown': es[0]['unknown'], **{r: [float(np.mean([e['rules'][r]['closed'] for e in es])), float(np.mean([e['rules'][r]['auroc'] for e in es]))]
                          for r in ('msp', 'probe', 'openmax', 'static', 'knn', 'temp_rff', 'raw_centroid')}}
    for s in ('phase1', 'phase2', 'phase3'):
        d['stages'][s] = {k: m([e['stages'][s][k] for e in E]) for k in ('softmax', 'centroid', 'auroc')}
    for k in sorted({k for e in E for k in e['rules'] if k.startswith('temp_rff_W')}):
        d['W'][k] = {'closed': m([e['rules'][k]['closed'] for e in E]), 'auroc': m([e['rules'][k]['auroc'] for e in E])}
    for r in ('temp_rff', 'knn', 'static'):
        d['fewshot'][r] = {k: [float(np.mean([e['fewshot'][r][k]['closed'] for e in E])), float(np.mean([e['fewshot'][r][k]['auroc'] for e in E]))] for k in E[0]['fewshot'][r]}
    d['sel']['W'] = collections.Counter(e['sel_days_keep'] for e in E)
    d['sel']['decay'] = collections.Counter(e['sel_decay'] for e in E)
    # wins of MP-NN vs rules (run level)
    d['wins'] = {r: [int(sum(e['rules']['temp_rff']['auroc'] > e['rules'][r]['auroc'] for e in E)), len(E)] for r in ('msp', 'probe', 'openmax', 'static', 'knn', 'temp_rff_p1', 'raw_centroid')}
    J = [json.load(open(f)) for f in sorted(glob.glob(f'res_joint/{ds}/shift0/seed*/joint_eval.json')) if '.failed' not in f]
    d['joint_n'] = len(J)
    R[ds] = d
json.dump(R, open('analysis/paper_numbers.json', 'w'), indent=1, default=str)
for ds, d in R.items():
    print('==', ds, d['n'], 'runs; joint', d['joint_n'])
    for r, x in d['rules'].items():
        if 'auroc' in x: print(f"  {r:17s} {x['closed'][0]:.3f}±{x['closed'][1]:.3f}  {x['auroc'][0]:.3f}±{x['auroc'][1]:.3f}  tuned acc {x['tuned']['accept']:.3f} unk {x['tuned']['unk_det']:.3f} ok {x['tuned']['known_ok']:.3f} | f90 acc {x['fixed90']['accept']:.3f} unk {x['fixed90']['unk_det']:.3f} ok {x['fixed90']['known_ok']:.3f} | obj acc {x['objective']['accept']:.3f} unk {x['objective']['unk_det']:.3f} ok {x['objective']['known_ok']:.3f}")
    print('  stages', {s: [round(v[k][0], 3) for k in ('softmax', 'centroid', 'auroc')] for s, v in d['stages'].items()})
    print('  W', {k: [round(v['closed'][0], 3), round(v['auroc'][0], 3)] for k, v in d['W'].items()}, 'sel', dict(d['sel']['W']), dict(d['sel']['decay']))
    print('  fewshot MP-NN', {k: [round(a, 3) for a in v] for k, v in d['fewshot']['temp_rff'].items()})
    print('  fewshot kNN  ', {k: [round(a, 3) for a in v] for k, v in d['fewshot']['knn'].items()})
    print('  wins', d['wins'])
    print('  split', {sh: [v['unknown'], [round(a, 2) for a in v['temp_rff']], [round(a, 2) for a in v['knn']], [round(a, 2) for a in v['raw_centroid']]] for sh, v in sorted(d['split'].items())})
