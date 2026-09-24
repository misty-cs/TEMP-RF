#!/usr/bin/env python3
"""
aggregate_tables.py — build the paper's Pycom tables from a set of seed runs.

Reads every seed's results_phase4.txt (and results_phase2.txt for the in-day
figures), aggregates across *pipeline seeds*, and emits LaTeX-ready rows plus
a JSON dump.

Two things this handles that a naive parser gets wrong:

1. results_phase4.txt is opened in append mode, so a directory may contain
   several Phase-4 passes. Only the LAST pass in each file is used; mixing
   passes is what produced the earlier provenance problem where the zero-shot
   table and the enrolment table came from different runs.

2. The enrolment sweep (ablation15) is printed as text lines, not as
   `key = value` entries in the results dict, and its own +- is over five
   enrolment draws at a fixed model. That is a narrower quantity than
   seed-to-seed spread, so this script re-aggregates the per-seed means
   across seeds and reports that instead.

Usage
-----
    python aggregate_tables.py res_out_pycom_capture_seed*
    python aggregate_tables.py --out tables/ res_out_*/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from statistics import mean, stdev

# Table 7.1 rows: (label, key prefix). Softmax is stored under a bare name.
ZS_ROWS = [
    ('Softmax (no rejection)', 'softmax'),
    ('OpenMax',                'openmax'),
    ('Static prototype',       'static'),
    ('Cosine $k$-NN',          'knn'),
    ('T2R (ours)',             'traj'),
]

# Appendix table: T2R variants. These are computed by Phase 4 alongside the
# main methods and are reported in the appendix, not the main comparison.
APPENDIX_ROWS = [
    ('Multi-proto cosine traj.',        'mp_cos_traj'),
    ('Probe/trajectory fusion',         'fusion'),
    ('Traj-reject + $k$-NN classify',   'traj_knn'),
    ('Trajectory through traj-day',     'traj'),
    ('Trajectory, BN-adapted emb.',     'traj_bn'),
]

# An accept rate at or below this means the rule rejects essentially every
# query; its unknown-detection score is then the trivial degenerate solution
# and must not be read as a ranking.
DEGENERATE_ACCEPT = 0.05


def _seed_of(path: str) -> str:
    """Seed number from a directory named seed_42, seed42, or similar."""
    m = re.search(r'(\d+)$', os.path.basename(path.rstrip('/')))
    return m.group(1) if m else os.path.basename(path.rstrip('/'))


def _last_pass(text: str) -> str:
    """Return only the final Phase-4 pass in an appended results file."""
    marks = [m.start() for m in re.finditer(r'^### Phase 4  started', text, re.M)]
    return text[marks[-1]:] if marks else text


def _dict_val(block: str, key: str):
    """Read `  key = value` from the '### Full results dict' section."""
    m = re.search(rf'^  {re.escape(key)} = ([-\d.eE+]+|nan)\s*$', block, re.M)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return None if v != v else v          # drop NaN


def parse_phase4(path: str) -> dict:
    """Extract Table-7.1 metrics and the enrolment sweep from one seed."""
    block = _last_pass(open(path).read())
    out: dict = {'_path': path}

    for _, pre in list(ZS_ROWS) + list(APPENDIX_ROWS):
        acc_key = 'softmax_acc' if pre == 'softmax' else f'{pre}_closed_acc'
        out[f'{pre}_closed'] = _dict_val(block, acc_key)
        for metric in ('auroc', 'unk_det', 'accept_rate', 'open_acc'):
            out[f'{pre}_{metric}'] = _dict_val(block, f'{pre}_{metric}')

    # Enrolment sweep — text lines of the form
    #   [proto] sessions=3  k=200  n1=0.6535+-0.0113  n5=...  n20=...  n50=...
    for m in re.finditer(
        r'^\s*\[(proto|T2R)\]\s+sessions=(\d+)\s+k=\s*(\d+)\s+'
        r'n1=([\d.]+)\+-[\d.]+\s+n5=([\d.]+)\+-[\d.]+\s+'
        r'n20=([\d.]+)\+-[\d.]+\s+n50=([\d.]+)\+-[\d.]+',
        block, re.M,
    ):
        rule, sess, k, n1, n5, n20, n50 = m.groups()
        base = f'div_{rule}_s{sess}_k{k}'
        out[f'{base}_n1']  = float(n1)
        out[f'{base}_n50'] = float(n50)

    z = re.search(r'zero-shot\s+slices/decision=\s*1\s+acc=([\d.]+)', block)
    if z:
        out['div_zeroshot_n1'] = float(z.group(1))
    return out


def parse_phase2(path: str) -> dict:
    """In-day accuracy per fine-tune day."""
    out = {}
    if not os.path.isfile(path):
        return out
    for m in re.finditer(r'^\s*day=(\d+)\s+softmax=([\d.]+)', open(path).read(), re.M):
        out[f'inday_day{m.group(1)}'] = float(m.group(2))
    return out


def agg(values: list[float]) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return float('nan'), float('nan'), 0
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def _paired_test(a: list[float], b: list[float]) -> tuple[str, float] | None:
    """
    Paired comparison of two methods across the same seeds.

    Wilcoxon signed-rank when SciPy is available and there are enough pairs;
    otherwise a paired t-test. Returns (test name, p-value), or None when the
    sample is too small to say anything -- which, with five seeds, is a real
    possibility worth reporting honestly rather than hiding behind a number.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if all(x == y for x, y in pairs):
        return ('identical', 1.0)
    try:
        from scipy import stats
        t_p = float(stats.ttest_rel(xs, ys).pvalue)
        if len(pairs) >= 6:
            return ('wilcoxon', float(stats.wilcoxon(xs, ys).pvalue))
        # Wilcoxon's smallest attainable p-value is 2^-(n-1): 0.0625 at n=5.
        # A method winning on every seed still cannot reach p < 0.05, so the
        # rank test alone would understate the evidence. Report the paired
        # t-test, which is not floored, and note the constraint.
        return ('paired t', t_p)
    except Exception:
        return None


def _fmt(m: float, s: float, n: int, prec: int = 3) -> str:
    if n == 0:
        return '--'
    if n == 1:
        return f'{m:.{prec}f}'
    return f'{m:.{prec}f}$\\pm${s:.{prec}f}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('seed_dirs', nargs='+', help='Seed output directories')
    ap.add_argument('--out', default='.', help='Where to write tables/JSON')
    a = ap.parse_args()

    runs, skipped = [], []
    for d in sorted(a.seed_dirs):
        p4 = os.path.join(d, 'results_phase4.txt')
        if not os.path.isfile(p4):
            skipped.append(d)
            continue
        r = parse_phase4(p4)
        r.update(parse_phase2(os.path.join(d, 'results_phase2.txt')))
        r['_seed_dir'] = d
        runs.append(r)

    if not runs:
        print('No seed directories with results_phase4.txt found.', file=sys.stderr)
        return 1
    for d in skipped:
        print(f'  skipped (no Phase-4 results): {d}')

    n_seeds = len(runs)
    print(f'\nAggregating {n_seeds} seed(s): '
          f'{", ".join(os.path.basename(r["_seed_dir"]) for r in runs)}\n')

    os.makedirs(a.out, exist_ok=True)
    lines: list[str] = []
    sections: dict[str, list[str]] = {}
    _mark = 0

    # ── Table 7.1 — zero-shot ────────────────────────────────────────────
    lines += ['% Table: Pycom zero-shot. Generated by aggregate_tables.py',
              f'% {n_seeds} pipeline seeds; $\\pm$ is seed-to-seed spread.',
              '\\begin{tabular}{lcccc}', '\\toprule',
              'Method & Closed acc. & AUROC & Unk.\\ det. & Acc.\\ rate \\\\',
              '\\midrule']
    print(f'{"method":<24}{"closed":>16}{"AUROC":>16}{"unk det":>16}{"accept":>9}')
    for label, pre in ZS_ROWS:
        c  = agg([r.get(f'{pre}_closed')      for r in runs])
        au = agg([r.get(f'{pre}_auroc')       for r in runs])
        ud = agg([r.get(f'{pre}_unk_det')     for r in runs])
        ar = agg([r.get(f'{pre}_accept_rate') for r in runs])
        oa = agg([r.get(f'{pre}_open_acc')    for r in runs])

        # Flag the degenerate operating point: a rule that accepts (almost)
        # nothing scores a perfect unknown-detection rate by construction.
        degen = (ar[2] and ar[0] <= DEGENERATE_ACCEPT) or (oa[2] and oa[0] == 0.0)
        ud_s = _fmt(*ud) + ('$^{\\dagger}$' if degen else '')

        lines.append(f'{label} & {_fmt(*c)} & {_fmt(*au)} & {ud_s} & {_fmt(*ar)} \\\\')
        print(f'{label:<24}{_fmt(*c):>16}{_fmt(*au):>16}{_fmt(*ud):>16}'
              f'{(f"{ar[0]:.3f}" if ar[2] else "--"):>9}'
              + ('   <- degenerate' if degen else ''))
    lines += ['\\bottomrule', '\\end{tabular}', '']
    sections['zeroshot'] = lines[_mark:]; _mark = len(lines)

    # ── Significance vs T2R ──────────────────────────────────────────────
    # Paired across seeds: each baseline is compared with T2R on the same
    # seeds, so the test controls for seed-to-seed variation rather than
    # treating the two as independent samples.
    t2r_closed = [r.get('traj_closed') for r in runs]
    t2r_auroc  = [r.get('traj_auroc')  for r in runs]
    sig_lines = []
    for label, pre in ZS_ROWS:
        if pre == 'traj':
            continue
        for metric, series in (('closed acc.', 'closed'), ('AUROC', 'auroc')):
            base = [r.get(f'{pre}_{series}') for r in runs]
            ref  = t2r_closed if series == 'closed' else t2r_auroc
            res  = _paired_test(ref, base)
            if res:
                name, pv = res
                sig_lines.append(f'%   T2R vs {label:<24} {metric:<12} '
                                 f'{name}: p = {pv:.4f}')
    if sig_lines:
        n_pairs = sum(1 for r in runs if r.get('traj_closed') is not None)
        lines += ['', '% Paired significance tests across seeds (T2R vs each baseline).',
                  f'% n = {n_pairs} seeds. Paired t-test on the seed-wise differences.',
                  '% NOTE: a Wilcoxon signed-rank test cannot fall below',
                  f'% 2^-(n-1) = {2 ** -(max(n_pairs, 2) - 1):.4f} at this sample size, so a method winning on',
                  '% every seed would still not reach p < 0.05 by that test. The',
                  '% t-test is reported instead; with this few seeds treat either',
                  '% as indicative rather than decisive.']
        lines += sig_lines
        lines += ['']
        print('\nPaired tests vs T2R:')
        for L in sig_lines:
            print('  ' + L.lstrip('% '))
        sections['significance'] = lines[_mark:]
        _mark = len(lines)

    # ── Table 7.2 — enrolment diversity ──────────────────────────────────
    zs = agg([r.get('div_zeroshot_n1') for r in runs])
    lines += ['', '% Table: Pycom enrolment diversity. $\\pm$ is seed-to-seed',
              '% spread (NOT the 5 enrolment draws inside a single run).',
              '\\begin{tabular}{llccc}', '\\toprule',
              'Rule & Sess. & $k{=}5$ & $k{=}50$ & $k{=}200$ \\\\', '\\midrule',
              f'\\multicolumn{{2}}{{l}}{{Zero-shot}} & -- & -- & {_fmt(*zs)} \\\\',
              '\\midrule']
    print(f'\n{"rule":<10}{"sess":>5}{"k=5":>16}{"k=50":>16}{"k=200":>16}')
    for rule, label in (('proto', 'Target-day proto.'), ('T2R', 'T2R (ours)')):
        for sess in (1, 2, 3):
            cells = [agg([r.get(f'div_{rule}_s{sess}_k{k}_n1') for r in runs])
                     for k in (5, 50, 200)]
            if not any(c[2] for c in cells):
                continue
            name = f'\\multirow{{3}}{{*}}{{{label}}}' if sess == 1 else ''
            lines.append(f'{name} & {sess} & ' +
                         ' & '.join(_fmt(*c) for c in cells) + ' \\\\')
            print(f'{rule:<10}{sess:>5}' + ''.join(f'{_fmt(*c):>16}' for c in cells))
        lines.append('\\midrule' if rule == 'proto' else '')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    # Only emit the enrolment table if the sweep actually ran. NEU has no
    # capture groups, so it produces no sweep; writing a header-only table
    # would leave a file that looks valid and renders as an empty tabular.
    have_sweep = any(k.startswith('div_') and not k.startswith('div_zeroshot')
                     for r in runs for k in r)
    if have_sweep:
        sections['enrolment'] = lines[_mark:]
    _mark = len(lines)

    # ── Appendix — T2R variants ──────────────────────────────────────────
    lines += ['', '% Appendix table: T2R variants. Same runs as the main table;',
              '% reported separately because the paper compares six methods.',
              '\\begin{tabular}{lcccc}', '\\toprule',
              'Variant & Closed acc. & AUROC & Unk.\\ det. & Acc.\\ rate \\\\',
              '\\midrule']
    print(f'\n{"variant":<30}{"closed":>16}{"AUROC":>16}')
    for label, pre in APPENDIX_ROWS:
        c  = agg([r.get(f'{pre}_closed')      for r in runs])
        au = agg([r.get(f'{pre}_auroc')       for r in runs])
        ud = agg([r.get(f'{pre}_unk_det')     for r in runs])
        ar = agg([r.get(f'{pre}_accept_rate') for r in runs])
        oa = agg([r.get(f'{pre}_open_acc')    for r in runs])
        if c[2] == 0:
            continue
        degen = (ar[2] and ar[0] <= DEGENERATE_ACCEPT) or (oa[2] and oa[0] == 0.0)
        ud_s = _fmt(*ud) + ('$^{\\dagger}$' if degen else '')
        lines.append(f'{label} & {_fmt(*c)} & {_fmt(*au)} & {ud_s} & {_fmt(*ar)} \\\\')
        print(f'{label:<30}{_fmt(*c):>16}{_fmt(*au):>16}')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    sections['variants'] = lines[_mark:]; _mark = len(lines)

    # ── Appendix — per-seed breakdown ────────────────────────────────────
    # Reviewers occasionally ask to see the individual seeds behind a mean.
    # Cheap to emit and it forecloses the question.
    lines += ['', '% Appendix table: per-seed values behind the main table.',
              '\\begin{tabular}{lcccc}', '\\toprule',
              'Seed & Softmax & T2R closed & T2R AUROC & T2R unk.\\ det. \\\\',
              '\\midrule']
    for r in runs:
        seed = os.path.basename(r['_seed_dir']).replace('seed_', '')
        def _v(k, p_=3):
            v = r.get(k)
            return f'{v:.{p_}f}' if v is not None else '--'
        lines.append(f"{seed} & {_v('softmax_closed')} & {_v('traj_closed')} & "
                     f"{_v('traj_auroc')} & {_v('traj_unk_det')} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '']
    sections['per_seed'] = lines[_mark:]; _mark = len(lines)

    # ── in-day ───────────────────────────────────────────────────────────
    inday = []
    _P = len('inday_day')
    for day in sorted({int(k[_P:]) for r in runs for k in r if k.startswith('inday_day')}):
        m, s, n = agg([r.get(f'inday_day{day}') for r in runs])
        inday.append(f'Day~{day}: ${m:.3f}\\pm{s:.3f}$')
    if inday:
        lines += ['% In-day accuracy: ' + ', '.join(inday), '']
        print('\nin-day: ' + '  '.join(x.replace('$', '').replace('\\pm', '+-')
                                       for x in inday))

    # ── One full Phase-4 table per seed ──────────────────────────────────
    # The aggregated tables report mean+-std; these show the complete method
    # comparison for each individual seed, so any aggregate value can be
    # traced back to the runs behind it.
    per_seed_dir = os.path.join(a.out, 'per_seed_tables')
    os.makedirs(per_seed_dir, exist_ok=True)
    for r in runs:
        seed = _seed_of(r['_seed_dir'])
        body = [f'% Phase-4 results, seed {seed} (single run, no aggregation).',
                f'% source: {r["_seed_dir"]}',
                '\\begin{tabular}{lcccc}', '\\toprule',
                'Method & Closed acc. & AUROC & Unk.\\ det. & Acc.\\ rate \\\\',
                '\\midrule']
        for label, pre in list(ZS_ROWS) + [(None, None)] + list(APPENDIX_ROWS):
            if label is None:
                body.append('\\midrule')
                continue
            c, au = r.get(f'{pre}_closed'), r.get(f'{pre}_auroc')
            ud, ar = r.get(f'{pre}_unk_det'), r.get(f'{pre}_accept_rate')
            oa = r.get(f'{pre}_open_acc')
            if c is None:
                continue
            degen = (ar is not None and ar <= DEGENERATE_ACCEPT) or (oa == 0.0)
            f3 = lambda v: '--' if v is None else f'{v:.3f}'
            ud_s = f3(ud) + ('$^{\\dagger}$' if degen else '')
            body.append(f'{label} & {f3(c)} & {f3(au)} & {ud_s} & {f3(ar)} \\\\')
        body += ['\\bottomrule', '\\end{tabular}', '']
        with open(os.path.join(per_seed_dir, f'seed{seed}_phase4.tex'), 'w') as f:
            f.write('\n'.join(body))
    print(f'\nper-seed Phase-4 tables -> {per_seed_dir}/ '
          f'({len(runs)} file(s))')

    # One file per paper table. A combined dump invites copying the wrong
    # block; a named file per table does not.
    written = []
    for name, body in sections.items():
        if not body:
            continue
        dest = os.path.join(a.out, f'{name}.tex')
        with open(dest, 'w') as f:
            f.write('\n'.join(body).strip() + '\n')
        written.append(os.path.basename(dest))
    if inday:
        dest = os.path.join(a.out, 'inday.tex')
        with open(dest, 'w') as f:
            f.write('% In-day accuracy: ' + ', '.join(inday) + '\n')
        written.append('inday.tex')
    tex = ', '.join(written)

    js = os.path.join(a.out, 'pycom_aggregate.json')
    with open(js, 'w') as f:
        json.dump({'n_seeds': n_seeds,
                   'seed_dirs': [r['_seed_dir'] for r in runs],
                   'per_seed': runs}, f, indent=2)

    print(f'\nwrote {tex}\n      {os.path.basename(js)}')
    if n_seeds < 3:
        print(f'\nNOTE: only {n_seeds} seed(s) — std is unreliable below 3.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
