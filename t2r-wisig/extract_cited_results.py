#!/usr/bin/env python3
"""
extract_cited_results.py — per-seed evidence files for the numbers in the .tex.

For each seed, pulls out only the Phase 1-4 lines that the manuscript actually
cites, and writes one compact text file. The point is traceability: when
someone asks where a number in a table came from, the answer is a named file
with that number in it, not a 3000-line results dump.

Kept deliberately small. Everything Phase 4 computes but the paper does not
report stays in the full results files under all_results/.

What is extracted:
  Phase 1   final validation / test accuracy of the initial model
  Phase 2   per-day in-day accuracy (the "in-day" figures quoted in the text)
  Phase 3   selected trajectory hyperparameters and calibrated threshold
  Phase 4   the six reported methods, the T2R variants used in the appendix,
            and the enrolment sweep (Pycom only)

Usage:
    python extract_cited_results.py <seed_dir> [...] --out <dir> --tag neu
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Phase-4 result-dict keys the manuscript cites, in table order.
CITED_P4 = [
    ('softmax_acc',            'Softmax closed acc.'),
    ('openmax_closed_acc',     'OpenMax closed acc.'),
    ('openmax_auroc',          'OpenMax AUROC'),
    ('openmax_unk_det',        'OpenMax unknown det.'),
    ('static_closed_acc',      'Static prototype closed acc.'),
    ('static_auroc',           'Static prototype AUROC'),
    ('static_unk_det',         'Static prototype unknown det.'),
    ('static_open_acc',        'Static prototype open acc. (0 = degenerate)'),
    ('knn_closed_acc',         'Cosine k-NN closed acc.'),
    ('knn_auroc',              'Cosine k-NN AUROC'),
    ('knn_unk_det',            'Cosine k-NN unknown det.'),
    ('knn_accept_rate',        'Cosine k-NN accept rate'),
    ('traj_closed_acc',        'T2R closed acc.'),
    ('traj_auroc',             'T2R AUROC'),
    ('traj_unk_det',           'T2R unknown det.'),
    ('traj_accept_rate',       'T2R accept rate'),
    # appendix variants
    ('mp_cos_traj_closed_acc', 'Multi-proto traj. closed acc. (appendix)'),
    ('mp_cos_traj_auroc',      'Multi-proto traj. AUROC (appendix)'),
    ('fusion_closed_acc',      'Probe/traj. fusion closed acc. (appendix)'),
    ('fusion_auroc',           'Probe/traj. fusion AUROC (appendix)'),
    ('traj_knn_closed_acc',    'Traj-reject + k-NN closed acc. (appendix)'),
    ('traj_knn_auroc',         'Traj-reject + k-NN AUROC (appendix)'),
    ('traj_bn_closed_acc',     'Traj. BN-adapted closed acc. (appendix)'),
    ('traj_bn_auroc',          'Traj. BN-adapted AUROC (appendix)'),
]

CITED_P3 = [
    ('mp_cos_traj_days_keep',  'trajectory history depth'),
    ('mp_cos_traj_day_decay',  'trajectory day decay'),
    ('traj_threshold',         'calibrated rejection threshold'),
]


def _seed_of(path: str) -> str:
    """Seed number from a directory named seed_42, seed42, or similar."""
    base = os.path.basename(path.rstrip('/'))
    m = re.search(r'(\d+)$', base)
    return m.group(1) if m else base


def _last_pass(text: str) -> str:
    marks = [m.start() for m in re.finditer(r'^### Phase 4  started', text, re.M)]
    return text[marks[-1]:] if marks else text


def _val(block: str, key: str):
    m = re.search(rf'^  {re.escape(key)} = ([-\d.eE+]+|nan)\s*$', block, re.M)
    return m.group(1) if m else None


def _config_of(seed_dir: str) -> list[str]:
    """
    Echo the run's configuration from its sibling log.

    Recorded so each evidence file is self-describing. Tuning runs vary
    slice_len and augmentation; only the main configuration belongs here, and
    stating it in the file removes any doubt about which run a number is from.
    """
    base = os.path.basename(seed_dir.rstrip('/'))
    log = os.path.join(os.path.dirname(seed_dir.rstrip('/')), base + '.log')
    if not os.path.isfile(log):
        return []
    head = open(log, errors='replace').read(8000)
    keys = ('data_root', 'dataset_layout', 'init_day', 'finetune_days',
            'traj_day', 'test_day', 'num_slice', 'ft_n_train', 'slice_len')
    rows = []
    for k in keys:
        m = re.search(rf'^\s+{k}\s*:\s*(.+?)\s*$', head, re.M)
        if m:
            rows.append(f'#   {k:<18} {m.group(1)}')
    return rows


def extract(seed_dir: str) -> str:
    seed = _seed_of(seed_dir)
    out = [f'# Cited results — seed {seed}',
           f'# source: {seed_dir}',
           '#',
           '# Only the values the manuscript cites. The complete Phase-4 output,',
           '# including the ~30 methods the paper does not report, is in the',
           '# original results_phase4.txt under all_results/.',
           '#',
           '# This is a MAIN run. Sensitivity/tuning runs (varied slice length or',
           '# augmentation) are never extracted here — they live under',
           '# all_results/pycom_sensitivity/ and are reported only in the',
           '# appendix sensitivity table.',
           '#']
    cfg = _config_of(seed_dir)
    if cfg:
        out += ['# configuration of this run:'] + cfg
    out.append('')

    # Phase 1
    p1 = os.path.join(seed_dir, 'results_phase1.txt')
    if os.path.isfile(p1):
        t = open(p1).read()
        hits = re.findall(r'^\s*(val_acc|test_acc|final.*acc)\s*[=:]\s*([\d.]+)',
                          t, re.M | re.I)
        if hits:
            out.append('## Phase 1 — initial training')
            for k, v in hits:
                out.append(f'  {k:<44} {v}')
            out.append('')

    # Phase 2 — in-day accuracy per day
    p2 = os.path.join(seed_dir, 'results_phase2.txt')
    if os.path.isfile(p2):
        rows = re.findall(r'^\s*day=(\d+)\s+softmax=([\d.]+)', open(p2).read(), re.M)
        if rows:
            out.append('## Phase 2 — in-day accuracy (quoted in text)')
            for d, v in rows:
                out.append(f'  day {d:<40} {v}')
            out.append('')

    # Phase 4 (and Phase-3 selections, which are echoed there)
    p4 = os.path.join(seed_dir, 'results_phase4.txt')
    if not os.path.isfile(p4):
        out.append('## Phase 4 — MISSING')
        return '\n'.join(out) + '\n'

    block = _last_pass(open(p4).read())

    sel = [(lbl, _val(block, k)) for k, lbl in CITED_P3]
    if any(v for _, v in sel):
        out.append('## Phase 3 — selected hyperparameters')
        for lbl, v in sel:
            if v is not None:
                out.append(f'  {lbl:<44} {v}')
        out.append('')

    out.append('## Phase 4 — cited metrics')
    for key, lbl in CITED_P4:
        v = _val(block, key)
        if v is not None:
            out.append(f'  {lbl:<44} {v}')
    out.append('')

    # Enrolment sweep (Pycom only)
    sweep = re.findall(
        r'^\s*\[(proto|T2R)\]\s+sessions=(\d+)\s+k=\s*(\d+)\s+n1=([\d.]+)',
        block, re.M)
    if sweep:
        out.append('## Phase 4 — enrolment diversity sweep (per-slice)')
        for rule, sess, k, v in sweep:
            out.append(f'  {rule:<8} sessions={sess}  k={k:<5} {v}')
        z = re.search(r'zero-shot\s+slices/decision=\s*1\s+acc=([\d.]+)', block)
        if z:
            out.append(f'  {"zero-shot":<8} (no target-day labels)      {z.group(1)}')
        out.append('')

    return '\n'.join(out) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('seed_dirs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True, help="corpus name, e.g. neu / pycom")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    n = 0
    for d in sorted(a.seed_dirs):
        if not os.path.isdir(d):
            continue
        dest = os.path.join(a.out, f'{a.tag}_seed{_seed_of(d)}.txt')
        with open(dest, 'w') as f:
            f.write(extract(d))
        print(f'  {os.path.basename(dest)}')
        n += 1
    if n == 0:
        print('  (no seed directories found)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
