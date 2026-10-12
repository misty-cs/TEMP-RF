"""Sanity checks of all finished final runs (read-only). Writes report_<ds>.txt."""
import os, sys, json, glob, collections, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
MAIN = ['msp', 'probe', 'openmax', 'static', 'knn', 'temp_rff', 'raw_centroid']
for ds in sys.argv[1:] or ['wisig', 'lora']:
    out = open(os.path.join(os.path.dirname(__file__), f'report_{ds}.txt'), 'w')
    P = lambda *a: (print(*a), print(*a, file=out))
    F = sorted(glob.glob(f'res/{ds}/shift*/seed*/common_eval.json')); F = [f for f in F if '.failed' not in f]
    R = []
    for f in F:
        d = os.path.dirname(f); A = json.load(open(d + '/audit.json')); E = json.load(open(f)); R.append((d, A, E))
    P(f'===== {ds}: {len(R)} runs')
    issues = []
    # 1 completeness and audits
    by = collections.defaultdict(list)
    for d, A, E in R: by[E['shift']].append(E['seed'])
    P('splits/seeds:', {k: sorted(v) for k, v in sorted(by.items())})
    bad = [d for d, A, E in R if not A.get('all_pass')]; P('audit failures:', bad or 'none'); issues += bad
    # 2 test set sizes
    ts = collections.Counter(json.dumps(E.get('test_set', {}).get('counts', E.get('test_set')), sort_keys=True)[:80] for d, A, E in R)
    P('test-set descriptors (distinct):', len(ts))
    for d, A, E in R:
        nk, nu = len(E['known']), len(E['unknown'])
        # 3 identities and ranges
        r = E['rules']
        if abs(r['msp']['closed'] - r['softmax']['closed']) > 1e-9: issues.append(f'{d}: msp closed != softmax closed')
        for name, x in r.items():
            for key in ('closed', 'auroc'):
                v = x.get(key)
                if v is not None and not (0 <= v <= 1): issues.append(f'{d}: {name}.{key}={v}')
            for p in ('fixed90', 'objective', 'tuned'):
                t = x.get(p)
                if isinstance(t, dict) and 'known_ok' in t and t['known_ok'] > t['accept'] + 1e-9: issues.append(f'{d}: {name}.{p} known_ok>accept')
            if x.get('closed') is not None and x['closed'] < 1.0 / nk: issues.append(f'{d}: {name} closed {x["closed"]:.3f} below chance {1/nk:.3f}')
        sel = E.get('window_rules', {})
        if sel.get('static_window', {}).get('W') == 1 and abs(r['static_window']['closed'] - r['static']['closed']) > 1e-9:
            issues.append(f'{d}: static_window (W=1) != static')
    # 4 summary per split (mean over seeds) and seed spread
    P('\nper split: closed / AUROC (mean over seeds) and max-min over seeds of AUROC')
    for sh in sorted(by):
        Es = [E for d, A, E in R if E['shift'] == sh]
        row = []
        for m in MAIN:
            a = [E['rules'][m].get('auroc') for E in Es if E['rules'][m].get('auroc') is not None]
            c = [E['rules'][m]['closed'] for E in Es]
            row.append(f'{m[:6]} {np.mean(c):.2f}/{np.mean(a):.2f}±{(max(a)-min(a)):.2f}')
        P(f' split {sh:2d} unk={Es[0]["unknown"]}: ' + ' | '.join(row))
    # 5 AUROC below 0.5 (worse than chance)
    low = collections.Counter(); lowsplits = collections.defaultdict(set)
    for d, A, E in R:
        for m in MAIN:
            a = E['rules'][m].get('auroc')
            if a is not None and a < 0.5: low[m] += 1; lowsplits[m].add(E['shift'])
    P('\nruns with AUROC < 0.5 per rule:', dict(low), {k: sorted(v) for k, v in lowsplits.items()})
    # 6 stages and few-shot monotonicity
    st = np.array([[E['stages'][s]['centroid'] for s in ('phase1', 'phase2', 'phase3')] for d, A, E in R])
    sa = np.array([[E['stages'][s]['auroc'] for s in ('phase1', 'phase2', 'phase3')] for d, A, E in R])
    P('\nstages centroid acc mean:', np.round(st.mean(0), 3), ' AUROC mean:', np.round(sa.mean(0), 3))
    P('runs where Phase 2 lowers centroid acc by >0.02:', int((st[:, 1] < st[:, 0] - 0.02).sum()), 'of', len(R))
    fs = [E.get('fewshot') for d, A, E in R if E.get('fewshot')]
    if fs:
        try:
            ks = sorted({int(k) for f in fs for k in f})
            P('few-shot keys:', ks[:10])
        except Exception as e:
            P('few-shot structure:', type(fs[0]).__name__, list(fs[0])[:5] if isinstance(fs[0], dict) else '')
    # 7 selections
    P('\nselected P/W/decay:', collections.Counter((E['selection'].get('P') if isinstance(E.get('selection'), dict) else None, E.get('sel_days_keep'), E.get('sel_decay')) for d, A, E in R).most_common(6))
    # 8 operating points of tuned thresholds
    for m in MAIN:
        acc = [E['rules'][m].get('tuned', {}).get('accept') for d, A, E in R]; acc = [a for a in acc if a is not None]
        if acc: P(f' tuned accept {m:12s} mean {np.mean(acc):.3f} min {min(acc):.3f} max {max(acc):.3f}')
    P('\nISSUES:', len(issues)); [P(' -', i) for i in issues[:40]]
