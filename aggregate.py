"""Aggregate the runs of one or more datasets into tables.

    python aggregate.py --tags wisig pycom      # pre-registered primary family (both datasets)
    python aggregate.py --tags wisig            # one dataset (not the pre-registered family)

Only runs whose audit passed (audit.json all_pass, no AUDIT_FAILED) are used;
excluded and missing (split, seed) runs are listed at the top of every summary.
Writes res/<tag>/SUMMARY.md and res/<tag>/summary.json per tag, and with several
tags also res/PRIMARY_<tags>.md / .json (the combined primary test):
  - every rule: closed accuracy, AUROC, and for each threshold protocol
    (fixed90, objective, tuned) accept rate, open accuracy, unknown detection,
    known-accepted-and-correct; mean +- std over all splits x seeds;
  - per-split means (closed, AUROC);
  - PRIMARY test (FAIRNESS.md section 0): TEMP-RFF minus every other rule at the
    split level (seeds averaged per split) for closed accuracy, AUROC, tuned
    known ok and tuned unknown detection; paired t-test and Wilcoxon, each
    Holm-corrected over the family PRIMARY_RULES x 4 outcomes x both tags;
    other rules: split-level estimates, uncorrected (exploratory);
  - secondary per-run test, forced windows, few-shot curve (with effective k),
    stage ablation, joint-training control vs Phase 2 (paired), selected settings.
"""
import argparse, glob, json, os, re
from collections import Counter
import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument('--tags', nargs='+', required=True, help='results folder names under res/, e.g. wisig pycom')
p.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 13, 21, 100], help='expected seeds')
p.add_argument('--root', default='res', help='results folder (res = final, res_draft = draft)')
a = p.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
import sys; sys.path.insert(0, here)
import datasets as D, manifest
PRIMARY_METRICS = ('closed', 'auroc', 'tuned_known_ok', 'tuned_unk_det')
key = lambda r: (r['shift'], r['seed'])
ms = lambda v: f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}' if len(v) > 1 else (f'{v[0]:.3f}' if v else '–')


def val(rd, m):
    return rd['tuned'][m[6:]] if m.startswith('tuned_') else rd[m]


def has(rd, m):
    return ('tuned' in rd and m[6:] in rd['tuned']) if m.startswith('tuned_') else m in rd


def holm(pv):
    """Holm step-down over a dict name -> p; NaN counts as p = 1 (conservative)."""
    ks = sorted(pv, key=lambda k_: 1.0 if np.isnan(pv[k_]) else pv[k_]); out, run_ = {}, 0.0
    for i, k_ in enumerate(ks):
        p_ = 1.0 if np.isnan(pv[k_]) else pv[k_]
        run_ = max(run_, min(1.0, (len(ks) - i) * p_)); out[k_] = run_
    return out


def dataset_of(tag):
    return max((n for n in D.DATASETS if tag == n or tag.startswith(n + '_')), key=len, default=None)


TUNED_VARS = ('T2R_SUPCON_LAMBDA', 'T2R_SUPCON_T', 'T2R_P1_EPOCHS', 'T2R_FT_REPLAY', 'T2R_FT_LR2')


def study_key(run_dir):
    """What must be identical for every run of one study: code (training and
    evaluation), T2R_* settings except the per-split tuned values, data
    inventory, software versions."""
    mt, me = (json.load(open(os.path.join(run_dir, f'manifest_{s_}.json'))) for s_ in ('train', 'eval'))
    env_ = {k: v for k, v in mt['env'].items() if k not in TUNED_VARS + manifest.REPRO_KEYS + ('T2R_JOINT_DAYS',)}
    ver = [{k: v for k, v in m_.get('versions', {}).items() if k != 'platform'} for m_ in (mt, me)]
    cfg_ = [manifest.norm_config(m_.get('config', {})) for m_ in (mt, me)]
    return json.dumps([mt['code_sha256_all'], me['code_sha256_all'], env_, mt.get('data'), ver, cfg_],
                      sort_keys=True, default=str)


def load(tag):
    """Audited runs of one tag whose result files are exactly the audited ones,
    plus the excluded and missing runs. All runs must share one study key."""
    files = sorted(glob.glob(f'{a.root}/{tag}/shift*/seed*/common_eval.json'))
    R, excl, keys = [], [], {}
    ds_ = dataset_of(tag)
    traj = D.DATASETS[ds_]['traj_day'] if ds_ else None
    for f in files:
        d_ = os.path.dirname(f); au = os.path.join(d_, 'audit.json')
        if not os.path.exists(au):
            excl.append((d_, 'no audit.json')); continue
        A = json.load(open(au))
        if not A.get('all_pass') or os.path.exists(os.path.join(d_, 'AUDIT_FAILED')):
            excl.append((d_, 'audit failed')); continue
        ok, why = manifest.artifacts_match(d_, A.get('artifacts'), traj)
        if not ok:
            excl.append((d_, why)); continue
        r = json.load(open(f)); r['_file'] = f; R.append(r)
        keys.setdefault(study_key(d_), []).append(d_)
    if len(keys) > 1:
        raise SystemExit(f'{a.root}/{tag}: runs come from {len(keys)} different studies (code/settings/data/versions): '
                         + ' | '.join(f'{len(v)} runs e.g. {v[0]}' for v in keys.values()))
    exp_shifts = D.default_shifts(ds_) if ds_ else sorted({r['shift'] for r in R})
    have = {key(r) for r in R}
    missing = [(s_, sd) for s_ in exp_shifts for sd in a.seeds if (s_, sd) not in have]
    return R, excl, missing, exp_shifts


def split_tests(R, rules):
    """Primary split-level test: raw p-values per (rule, metric)."""
    sp = sorted({r['shift'] for r in R}); T = {}
    sm = lambda n, m: np.array([np.mean([val(r['rules'][n], m) for r in R if r['shift'] == s_]) for s_ in sp])
    for n in rules:
        if n == 'temp_rff':
            continue
        for m in PRIMARY_METRICS:
            if not all(has(r['rules'][n], m) and has(r['rules']['temp_rff'], m) for r in R):
                continue
            x, y = sm('temp_rff', m), sm(n, m); d = x - y
            pt = float(stats.ttest_rel(x, y).pvalue) if len(sp) > 1 and np.any(d != 0) else float('nan')
            try:
                pw = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                pw = float('nan')
            T[f'{n}|{m}'] = dict(diff=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else float('nan'),
                                 wins=int((d > 0).sum()), n=len(d), t_p=pt, w_p=pw)
    return T, sp


DATA = {}
for tag in a.tags:
    R, excl, missing, exp_shifts = load(tag)
    if not R:
        raise SystemExit(f'no audited common_eval.json under res/{tag}/ (excluded: {excl})')
    rules = [n for n in R[0]['rules'] if not n.startswith('temp_rff_W')]
    T, sp = split_tests(R, rules)
    DATA[tag] = dict(R=R, excl=excl, missing=missing, exp_shifts=exp_shifts, rules=rules, T=T, sp=sp)
# one Holm family over every primary test of every tag given
# Primary family, fixed 2026-09-28 before any confirmatory run (PROTOCOL.md 7):
# TEMP-RFF vs the main competitors x 4 outcomes x both datasets = 56 tests.
# Every other rule (defaults, ablations, temp_rff_p1/nowt/W*, raw centroid) is
# secondary: split-level estimates and uncorrected p-values, exploratory only.
PRIMARY_RULES = ('msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window')
is_prim = lambda k_: k_.split('|')[0] in PRIMARY_RULES
fam_t = holm({f'{t}|{k_}': v['t_p'] for t in DATA for k_, v in DATA[t]['T'].items() if is_prim(k_)})
fam_w = holm({f'{t}|{k_}': v['w_p'] for t in DATA for k_, v in DATA[t]['T'].items() if is_prim(k_)})
FAMILY = (f'{len(fam_t)} tests: {len(PRIMARY_RULES)} rules {list(PRIMARY_RULES)} x {len(PRIMARY_METRICS)} outcomes '
          f'x tags {a.tags}' + ('' if set(a.tags) >= {'wisig', 'pycom', 'lora'} else
                                ' (NOT the pre-registered family: that needs --tags wisig pycom lora)'))
for t in DATA:
    for k_, v in DATA[t]['T'].items():
        v['primary'] = is_prim(k_)
        v['t_p_holm'] = fam_t.get(f'{t}|{k_}', float('nan')); v['w_p_holm'] = fam_w.get(f'{t}|{k_}', float('nan'))


def summarize(tag):
    X = DATA[tag]; R, rules = X['R'], X['rules']
    files = [r['_file'] for r in R]
    out, S = [], {'n_runs': len(R), 'runs': [list(key(r)) for r in R], 'excluded': X['excl'],
                  'missing': X['missing'], 'holm_family': FAMILY}
    status = 'COMPLETE' if not X['missing'] else f'INCOMPLETE ({len(X["missing"])} runs missing)'
    out.append(f'# Summary: {tag} — {status}\n\nAudited runs used: {len(R)} (splits {sorted({r["shift"] for r in R})}, '
               f'seeds {sorted({r["seed"] for r in R})}); expected splits {X["exp_shifts"]} x seeds {a.seeds}\n')
    if X['excl']:
        out.append('Excluded (not audited / audit failed): ' + ', '.join(f'{d_} ({why})' for d_, why in X['excl']) + '\n')
    if X['missing']:
        out.append('Missing (split, seed): ' + ', '.join(str(m_) for m_ in X['missing']) + '\n')
    protos = ['fixed90', 'objective', 'tuned']
    # ── main table ────────────────────────────────────────────────────────────
    out.append('## All rules (mean ± std over runs)\n')
    out.append('| Rule | Closed acc. | AUROC | ' + ' | '.join(f'{p_} accept | {p_} open | {p_} unk. det. | {p_} known ok' for p_ in protos) + ' |')
    out.append('|' + '---|' * (3 + 4 * len(protos)))
    S['rules'] = {}
    for n in rules:
        rr = [r['rules'][n] for r in R if n in r['rules']]
        row = {'closed': [x['closed'] for x in rr], 'auroc': [x['auroc'] for x in rr if 'auroc' in x]}
        cells = [ms(row['closed']), ms(row['auroc'])]
        for p_ in protos:
            for m in ('accept', 'open', 'unk_det', 'known_ok'):
                v = [x[p_][m] for x in rr if p_ in x and not np.isnan(x[p_][m])]
                row[f'{p_}_{m}'] = v; cells.append(ms(v))
        S['rules'][n] = {k: [float(x) for x in v] for k, v in row.items()}
        out.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    tm = Counter((n, r['rules'][n]['tuned']['method']) for r in R for n in rules if 'tuned' in r['rules'][n])
    out.append('\nTuned calibration: method chosen on Part B, count over runs: ' +
               ', '.join(f'{n}: ' + '/'.join(f'{m}×{c}' for (nn, m), c in sorted(tm.items()) if nn == n)
                         for n in rules if any(nn == n for nn, _ in tm)) + '\n')
    # ── per split ─────────────────────────────────────────────────────────────
    out.append('## Per device split (mean over seeds)\n')
    shifts = sorted({r['shift'] for r in R})
    main_rules = [n for n in ('softmax', 'msp', 'probe', 'openmax', 'static', 'static_window', 'knn', 'knn_window',
                              'temp_rff_p1', 'temp_rff', 'raw_centroid_cfo') if n in rules]
    out.append('| Rule | ' + ' | '.join(f'shift {s} closed / AUROC' for s in shifts) + ' |')
    out.append('|' + '---|' * (1 + len(shifts)))
    for n in main_rules:
        cells = []
        for s in shifts:
            rr = [r['rules'][n] for r in R if r['shift'] == s]
            c = np.mean([x['closed'] for x in rr]); au = [x['auroc'] for x in rr if 'auroc' in x]
            cells.append(f'{c:.3f} / {np.mean(au):.3f}' if au else f'{c:.3f} / –')
        out.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    # ── primary test: split level (runs of one split share its devices) ───────
    out.append('\n## PRIMARY test: TEMP-RFF minus rule at the split level\n')
    out.append('Seeds are averaged within each device split first (the seeds of a split share its '
               'devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, '
               'AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: '
               + FAMILY + '.\n')
    out.append('| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for k_ in sorted(k for k in X['T'] if is_prim(k)):
        v = X['T'][k_]; n_, met = k_.split('|')
        out.append(f"| {n_} | {met} | {v['diff']:+.3f} | {v['sd']:.3f} | {v['wins']}/{v['n']} | {v['t_p']:.3g} | "
                   f"{v['t_p_holm']:.3g} | {v['w_p']:.3g} | {v['w_p_holm']:.3g} |")
    out.append('\n## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected\n')
    out.append('| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |')
    out.append('|---|---|---|---|---|---|---|')
    for k_ in sorted(k for k in X['T'] if not is_prim(k)):
        v = X['T'][k_]; n_, met = k_.split('|')
        out.append(f"| {n_} | {met} | {v['diff']:+.3f} | {v['sd']:.3f} | {v['wins']}/{v['n']} | {v['t_p']:.3g} | {v['w_p']:.3g} |")
    S['paired_split'] = X['T']
    out.append(f"\nWith {len(X['sp'])} splits the smallest possible two-sided Wilcoxon p is {2 / 2 ** len(X['sp']):.3g}; "
               f'report effect sizes and wins, not only p-values.\n')
    # ── secondary: per run ────────────────────────────────────────────────────
    out.append('\n## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)\n')
    out.append('| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |')
    out.append('|---|---|---|---|---|---|')
    S['paired'] = {}
    base = {key(r): r['rules']['temp_rff'] for r in R}
    for n in rules:
        if n == 'temp_rff':
            continue
        for m in PRIMARY_METRICS:
            pairs = [(val(base[key(r)], m), val(r['rules'][n], m)) for r in R
                     if has(r['rules'][n], m) and has(base[key(r)], m)]
            if len(pairs) < 3:
                continue
            x, y = np.array(pairs).T; d = x - y
            pt = float(stats.ttest_rel(x, y).pvalue) if np.any(d != 0) else float('nan')
            try:
                pw = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                pw = float('nan')
            S['paired'][f'{n}|{m}'] = dict(diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d), t_p=pt, w_p=pw)
            out.append(f'| {n} | {m} | {d.mean():+.3f} | {(d > 0).sum()}/{len(d)} | {pt:.3g} | {pw:.3g} |')
    # ── windows ───────────────────────────────────────────────────────────────
    wn = sorted([n for n in R[0]['rules'] if n.startswith('temp_rff_W')], key=lambda s: int(s[10:]))
    out.append('## Forced prototype window (decay 1)\n')
    out.append('| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |')
    out.append('|---|---|---|---|---|')
    for n in wn:
        rr = [r['rules'][n] for r in R]
        out.append(f'| {n[10:]} | {ms([x["closed"] for x in rr])} | {ms([x["auroc"] for x in rr])} | '
                   f'{ms([x["tuned"]["accept"] for x in rr if "tuned" in x])} | {ms([x["tuned"]["unk_det"] for x in rr if "tuned" in x])} |')
    # ── few-shot ──────────────────────────────────────────────────────────────
    out.append('\n## Few-shot target-session enrollment (same support for every rule)\n')
    fsr = list(R[0]['fewshot'])
    out.append('| k requested | k effective | ' + ' | '.join(f'{n} closed / AUROC / tuned known ok' for n in fsr) + ' |')
    out.append('|' + '---|' * (2 + len(fsr)))
    for k in R[0]['fewshot'][fsr[0]]:
        cells = []
        for n in fsr:
            rr = [r['fewshot'][n][k] for r in R]
            cells.append(f'{np.mean([x["closed"] for x in rr]):.3f} / {np.mean([x["auroc"] for x in rr]):.3f} / '
                         f'{np.mean([x["tuned"]["known_ok"] for x in rr]):.3f}')
        ke = np.mean([r['fewshot'][fsr[0]][k]['k_eff'] for r in R])
        out.append(f'| {k} | {ke:.0f} | ' + ' | '.join(cells) + ' |')
    # ── stages and joint control ──────────────────────────────────────────────
    out.append('\n## Pipeline stages (Part-A centroids)\n')
    out.append('| Stage | Softmax acc. | Centroid acc. | AUROC |')
    out.append('|---|---|---|---|')
    for st in ('phase1', 'phase2', 'phase3'):
        rr = [r['stages'][st] for r in R]
        out.append(f'| {st} | {ms([x["softmax"] for x in rr])} | {ms([x["centroid"] for x in rr])} | {ms([x["auroc"] for x in rr])} |')
    J = {}
    for f in glob.glob(f'res_joint{a.root[3:]}/{tag}/shift*/seed*/joint_eval.json'):
        jd = os.path.dirname(f); au = os.path.join(jd, 'audit.json')
        A = json.load(open(au)) if os.path.exists(au) else {}
        if not A.get('all_pass') or A.get('kind') != 'joint' or os.path.exists(os.path.join(jd, 'AUDIT_FAILED')):
            out.append(f'- joint run {jd} excluded: joint audit missing or failed'); continue
        if not manifest.artifacts_match(jd, A.get('artifacts'), None)[0]:
            out.append(f'- joint run {jd} excluded: files changed after the audit'); continue
        j = json.load(open(f)); rm = next((r for r in R if key(r) == (j['shift'], j['seed'])), None)
        if rm is None or j.get('test_input_sha1') != rm['test_set'].get('input_sha1'):
            out.append(f'- joint run {jd} excluded: no matching main run or different test input'); continue
        # same study as the paired pipeline run, except the intended joint-vs-sequential
        # difference (T2R_JOINT_DAYS); tuned lambda/T/epochs must be the same
        mj = json.load(open(os.path.join(jd, 'manifest_train.json')))
        mr = json.load(open(os.path.join(os.path.dirname(rm['_file']), 'manifest_train.json')))
        pv = ('T2R_SUPCON_LAMBDA', 'T2R_SUPCON_T', 'T2R_P1_EPOCHS')
        if study_key(jd) != study_key(os.path.dirname(rm['_file'])) or \
                any(mj['env'].get(v) != mr['env'].get(v) for v in pv):
            out.append(f'- joint run {jd} excluded: code, versions, configuration, data or Phase-1 settings differ '
                       'from the paired pipeline run'); continue
        J[(j['shift'], j['seed'])] = j
    if J:
        out.append('\nJoint-training control vs sequential Phase 2 (paired, split 0; joint runs have no audit):\n')
        out.append('| metric | joint | Phase 2 | mean diff | wins |')
        out.append('|---|---|---|---|---|')
        for m in ('softmax', 'centroid', 'auroc'):
            pr = [(J[key(r)][m], r['stages']['phase2'][m]) for r in R if key(r) in J]
            if pr:
                x, y = np.array(pr).T
                out.append(f'| {m} | {ms(list(x))} | {ms(list(y))} | {np.mean(x - y):+.3f} | {(x > y).sum()}/{len(x)} |')
    # ── selections ────────────────────────────────────────────────────────────
    out.append('\n## Selected settings over runs\n')
    sel = {'W': Counter(), 'decay': Counter(), 'P': Counter(), 'knn (k, weighted)': Counter(),
           'openmax (dist, tail, alpha)': Counter(), 'probe C': Counter()}
    for f, r in zip(files, R):
        sel['W'][r['sel_days_keep']] += 1; sel['decay'][r['sel_decay']] += 1
        sel['knn (k, weighted)'][(r['knn_k'], r.get('knn_weighted'))] += 1
        sel['openmax (dist, tail, alpha)'][(r.get('openmax_dist'), r['openmax_tail'], r['openmax_alpha'])] += 1
        sel['probe C'][r.get('probe_C')] += 1
        wr = r.get('window_rules', {})
        for n_ in ('static_window', 'knn_window', 'temp_rff_p1'):
            if n_ in wr:
                sel.setdefault(f'{n_} W' + (' (k, weighted)' if n_ == 'knn_window' else ' decay' if n_ == 'temp_rff_p1' else ''),
                               Counter())[(wr[n_]['W'],) + tuple(wr[n_][k] for k in ('k', 'weighted', 'decay') if k in wr[n_])] += 1
        t = open(os.path.join(os.path.dirname(f), 'results_phase3.txt')).read()
        pm = re.findall(r'best_n_cosine_prototypes=(\d+)', t)
        if pm:
            v = int(pm[-1]); sel['P']['all' if v >= 10 ** 9 else v] += 1
    for k_, c in sel.items():
        out.append(f'- {k_}: ' + ', '.join(f'{v}×{n}' for v, n in sorted(c.items(), key=lambda z: str(z[0]))))
    sens = [r.get('selection_sensitivity') for r in R if r.get('selection_sensitivity')]
    if sens:
        ch = sum(1 for s in sens if len({json.dumps(v, sort_keys=True) for v in s.values()}) > 1)
        out.append(f'- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in {ch} of {len(sens)} runs')
    S['selected'] = {k_: {str(v): n for v, n in c.items()} for k_, c in sel.items()}
    open(f'{a.root}/{tag}/SUMMARY.md', 'w').write('\n'.join(out) + '\n')
    json.dump(S, open(f'{a.root}/{tag}/summary.json', 'w'), indent=1, default=str)
    print('\n'.join(out))


for tag in a.tags:
    summarize(tag)
if len(a.tags) > 1:
    name = 'PRIMARY_' + '_'.join(a.tags)
    L = [f'# PRIMARY test, combined family ({FAMILY})\n',
         '| Dataset | Rule | outcome | mean diff | splits won | t p (Holm) | Wilcoxon p (Holm) |', '|---|---|---|---|---|---|---|']
    for t in a.tags:
        st_ = 'complete' if not DATA[t]['missing'] else f"INCOMPLETE, {len(DATA[t]['missing'])} missing"
        for k_ in sorted(k for k in DATA[t]['T'] if is_prim(k)):
            v = DATA[t]['T'][k_]; n_, met = k_.split('|')
            L.append(f"| {t} ({st_}) | {n_} | {met} | {v['diff']:+.3f} | {v['wins']}/{v['n']} | {v['t_p_holm']:.3g} | {v['w_p_holm']:.3g} |")
    open(f'{a.root}/{name}.md', 'w').write('\n'.join(L) + '\n')
    json.dump({'family': FAMILY, **{t: {k_: v for k_, v in DATA[t]['T'].items() if is_prim(k_)} for t in a.tags}},
              open(f'{a.root}/{name}.json', 'w'), indent=1)
    print('\n'.join(L))
