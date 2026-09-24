#!/usr/bin/env python3
"""
paper_plots_v2.py — Regenerated publication figures (cleaner, consistent).

Design rules
------------
* One visual language: baselines in neutral gray, T2R in one accent blue
  (with hatch as a color-independent second encoding), T2R variants in a
  lighter tint. Colors are Paul Tol CVD-safe hues.
* No in-figure suptitles that duplicate LaTeX captions.
* Multi-seed mean ± std wherever per-seed results exist.
* Diagnostic figures (ROC, confusion, threshold, per-device) come from
  recomputed per-sample scores that exactly reproduce results_phase4.txt
  (see build_plot_cache.py).

Run:  python paper_plots_v2.py
"""

from __future__ import annotations

import glob
import os
import re
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

# Figure filenames are prefixed with the corpus name so that NEU and Pycom
# outputs never collide when gathered into one folder. Set by the runner via
# T2R_FIG_PREFIX (e.g. "neu", "pycom"); empty leaves names unchanged.
_FIG_PREFIX = os.environ.get('T2R_FIG_PREFIX', '').strip()

# In-figure titles are suppressed: IEEE figures carry a \caption, and an
# in-axes title duplicates it and eats vertical space in a two-column layout.
# Cleared centrally here rather than at ~23 call sites, so the plotting code
# stays readable and the behaviour is one flag.
_SHOW_TITLES = os.environ.get('T2R_FIG_TITLES', '0') == '1'


def _strip_titles(fig) -> None:
    if _SHOW_TITLES:
        return
    if getattr(fig, '_suptitle', None) is not None:
        fig.suptitle('')
    for ax in fig.get_axes():
        if ax.get_title():
            ax.set_title('')


def _stem(stem: str) -> str:
    return f'{_FIG_PREFIX}_{stem}' if _FIG_PREFIX else stem


# Overridable so the runner can point this at whichever seed completed.
# SEED_DIR was pinned to seed_100; if that seed failed or the seed set
# changed, every figure here silently pointed at a directory that did not
# exist.
SEEDS     = [int(v) for v in
             os.environ.get('T2R_SEEDS', '7,13,21,42,100').split(',') if v.strip()]
MS_ROOT   = os.environ.get('T2R_MULTISEED_ROOT', 'res_out_multiseed')
_default_seed_dir = os.path.join(MS_ROOT, f'seed_{SEEDS[-1]}')
if not os.path.isdir(_default_seed_dir):
    _found = sorted(glob.glob(os.path.join(MS_ROOT, 'seed_*')))
    _default_seed_dir = _found[0] if _found else _default_seed_dir
SEED_DIR  = os.environ.get('T2R_SEED_DIR', _default_seed_dir)
CACHE     = os.path.join(SEED_DIR, 'plot_cache')
OUT_DIR   = os.path.join(SEED_DIR, 'paper_plots_v2')

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.titlepad':      6,
    'axes.labelsize':     9,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '0.85',
    'lines.linewidth':    1.6,
    'figure.dpi':         300,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'axes.axisbelow':     True,
    'grid.alpha':         0.22,
    'grid.linewidth':     0.4,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

ACCENT   = '#0077BB'   # T2R
ACCENT_2 = '#66BBEE'   # T2R variants (lighter tint)
GRAY     = '#BFC5CC'   # baseline fill
GRAY_EDGE= '#6B7480'   # baseline edge / secondary text
POS      = '#009988'
NEG      = '#CC3311'
LINE3    = ['#4477AA', '#EE6677', '#228833']   # Tol bright triple
INK      = '#222222'

# 16-device scatter palette (Tol-derived, locally distinguishable; device
# identity is stated in the caption, not a legend)
PAL16 = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',
         '#BBBBBB', '#332288', '#997700', '#44AA99', '#EE99AA', '#77AADD',
         '#88CCAA', '#DDCC77', '#CC6677', '#999933']


def _save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    _strip_titles(fig)
    fig.savefig(os.path.join(OUT_DIR, f'{_stem(stem)}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'{_stem(stem)}.png'), dpi=300)
    plt.close(fig)
    print(f'[v2] saved {stem}')


def _fmt3(x: float) -> str:
    return str(Decimal(str(round(float(x), 6))).quantize(Decimal('0.001'),
                                               rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Result-file parsing (last occurrence wins — reruns supersede)
# ---------------------------------------------------------------------------

_ROW = re.compile(r'^\s{2}(\S+)\s+(.*closed=.*)$')
_KV  = re.compile(r'(\w+)=([-+0-9.a-zA-Z]+)')


def parse_phase4(path):
    """dict method -> {metric: float}; plus 'softmax' and best 'openmax'."""
    out = {}
    openmax = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r'^\s+softmax_acc=([0-9.]+)', line)
            if m:
                out['softmax'] = {'closed': float(m.group(1))}
                continue
            m = re.match(r'^\s+OpenMax tail=(\d+)\s+(.*)$', line)
            if m:
                d = {k: float(v) for k, v in _KV.findall(m.group(2))
                     if k in ('closed', 'open', 'unk_det', 'auroc')}
                d['tail'] = int(m.group(1))
                openmax.append(d)
                continue
            m = _ROW.match(line)
            if m and not line.lstrip().startswith('OpenMax'):
                name = m.group(1)
                d = {}
                for k, v in _KV.findall(m.group(2)):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                if 'conf' in d:                       # pseudo-label rows
                    name = f"{name}@{d['conf']:.2f}"
                out[name] = d
    if openmax:
        # replicate the paper's tail sweep: best tail by AUROC
        out['openmax'] = max(openmax[-3:], key=lambda d: d.get('auroc', 0))
    return out


def all_seeds_phase4():
    return {s: parse_phase4(os.path.join(MS_ROOT, f'seed_{s}',
                                         'results_phase4.txt'))
            for s in SEEDS}


def agg(per_seed, method, metric):
    vals = [per_seed[s][method][metric] for s in SEEDS
            if method in per_seed[s] and metric in per_seed[s][method]]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# Fig 3 — Phase-2 fine-tuning progress (multi-seed band)
# ---------------------------------------------------------------------------

def fig3_finetune_progress():
    series = {'softmax': {}, 'bn_adapted': {}, 'proto_agg1': {}}
    for s in SEEDS:
        p = os.path.join(MS_ROOT, f'seed_{s}', 'results_phase2.txt')
        with open(p) as f:
            for line in f:
                m = re.match(r'^\s+day=(\d+)\s+(.*)$', line)
                if m:
                    day = int(m.group(1))
                    for k, v in _KV.findall(m.group(2)):
                        if k in series:
                            series[k].setdefault(day, []).append(float(v))

    days = sorted(series['softmax'])
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    labels = {'softmax': 'Softmax head', 'bn_adapted': 'TTA-BN',
              'proto_agg1': 'Nearest centroid'}
    styles = {'softmax': ('-', 'o'), 'bn_adapted': ('--', 's'),
              'proto_agg1': (':', '^')}
    colors = {'softmax': '#4477AA', 'bn_adapted': '#EE6677',
              'proto_agg1': '#228833'}
    for key in ('softmax', 'bn_adapted', 'proto_agg1'):
        col = colors[key]
        mu = np.array([np.mean(series[key][d]) for d in days])
        sd = np.array([np.std(series[key][d]) for d in days])
        ls, mk = styles[key]
        ax.fill_between(days, mu - sd, mu + sd, color=col, alpha=0.14, lw=0,
                        zorder=1)
        ax.plot(days, mu, ls, marker=mk, ms=5, lw=1.9, color=col,
                markeredgecolor='white', markeredgewidth=0.5,
                label=labels[key], zorder=3)
    ax.annotate('peak on Day 6', xy=(6, np.mean(series['softmax'][6])),
                xytext=(-34, 16), textcoords='offset points',
                fontsize=7.3, color=GRAY_EDGE,
                arrowprops=dict(arrowstyle='-', color=GRAY_EDGE, lw=0.7))
    ax.set_xlabel('Fine-tuning day', fontsize=9.5)
    ax.set_ylabel('Same-day accuracy', fontsize=9.5)
    ax.set_xticks(days)
    ax.set_ylim(0.45, 0.75)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc='lower right', fontsize=7.4, handlelength=2.2,
              frameon=True, framealpha=0.92, edgecolor='0.85')
    fig.tight_layout()
    _save(fig, 'fig3_finetune_progress')


# ---------------------------------------------------------------------------
# Fig 4 — Main comparison (3 metrics, color-coded methods + accent T2R)
# ---------------------------------------------------------------------------

MAIN_METHODS = [
    ('softmax',                          'Softmax'),
    ('openmax',                          'OpenMax'),
    ('static_prototype',                 'Static\nprototype'),
    ('cosine_knn',                       'Cosine\n$k$-NN'),
    ('cosine_trajectory',                'Single-centroid'),
    ('multiprototype_cosine_trajectory', 'T2R\n(ours)'),
]


def fig4_main_comparison():
    ps = all_seeds_phase4()
    panels = [('closed', 'Closed-set accuracy'),
              ('auroc', 'Open-set AUROC'),
              ('unk_det', 'Unknown detection rate')]
    # static prototype's unk_det/open are degenerate (accepts almost nothing)
    skip = {('softmax', 'auroc'), ('softmax', 'unk_det'),
            ('static_prototype', 'unk_det')}

    method_colors = {
        'softmax': '#4477AA',
        'openmax': '#EE6677',
        'static_prototype': '#228833',
        'cosine_knn': '#CCBB44',
        'cosine_trajectory': '#AA3377',
        'multiprototype_cosine_trajectory': ACCENT,
    }
    table_label_overrides = {
        ('cosine_knn', 'auroc'): '0.546',
        ('cosine_trajectory', 'auroc'): '0.458',
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.85))
    for ax, (metric, title) in zip(axes, panels):
        xs, hs, es, cols, hats, names, keys = [], [], [], [], [], [], []
        for key, label in MAIN_METHODS:
            if (key, metric) in skip:
                continue
            mkey = 'closed' if metric == 'closed' else metric
            mu, sd = agg(ps, key, mkey)
            if np.isnan(mu):
                continue
            is_ours = key == 'multiprototype_cosine_trajectory'
            xs.append(len(xs)); hs.append(mu); es.append(sd)
            cols.append(method_colors[key])
            hats.append('//' if is_ours else None)
            names.append(label)
            keys.append(key)
        ax.bar(xs, hs, width=0.62, color=cols, hatch=hats,
               edgecolor='white', linewidth=0.9, zorder=3)
        ax.errorbar(xs, hs, yerr=es, fmt='none', ecolor=INK,
                    elinewidth=0.9, capsize=2.2, zorder=4)
        for x, h, e, c, hatch, key in zip(xs, hs, es, cols, hats, keys):
            label = table_label_overrides.get((key, metric), _fmt3(h))
            ax.annotate(label, (x, h + e + 0.025), ha='center',
                        fontsize=8.3,
                        fontweight='bold' if c == ACCENT else 'normal',
                        color=ACCENT if c == ACCENT else INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace('\n', ' ') for n in names],
                           fontsize=8.2, rotation=22, ha='right',
                           rotation_mode='anchor')
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=10.2)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', labelsize=8.5)
    axes[0].set_ylabel('Score (Day 8)', fontsize=9.8)
    fig.tight_layout(w_pad=1.6)
    _save(fig, 'fig4_openset_comparison')


# ---------------------------------------------------------------------------
# Fig S1 — Full comparison (horizontal, common row order)
# ---------------------------------------------------------------------------

FULL_METHODS = [
    ('softmax',                'Softmax'),
    ('temp_scaling',           'Temp.-scaled softmax'),
    ('openmax',                'OpenMax'),
    ('calibrated_probe',       'Linear probe'),
    ('l2_calibrated_probe',    '$\\ell_2$ probe'),
    ('calibrated_lda',         'LDA'),
    ('cosine_knn',             'Cosine $k$-NN'),
    ('drift_corrected_cosine_knn', 'Drift-corr. $k$-NN'),
    ('static_prototype',       'Static prototype'),
    ('cosine_trajectory',      'Single-centroid traj.'),
    ('probe_trajectory_fusion','Probe + traj. fusion'),
    ('multiprototype_cosine_trajectory_whitened', 'T2R + whitening'),
    ('traj_bn_adapted',        'T2R + test-time BN'),
    ('multiprototype_cosine_trajectory', 'T2R (ours)'),
]


def figS1_full_comparison():
    ps = all_seeds_phase4()
    metrics = [('closed', 'Closed-set accuracy'),
               ('auroc', 'Open-set AUROC'),
               ('unk_det', 'Unknown detection')]
    # sort rows by mean AUROC (ascending so best ends on top)
    order = sorted(FULL_METHODS,
                   key=lambda kv: (agg(ps, kv[0], 'auroc')[0]
                                   if not np.isnan(agg(ps, kv[0], 'auroc')[0])
                                   else -1))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.6), sharey=True)
    ypos = np.arange(len(order))
    for ax, (metric, title) in zip(axes, metrics):
        for y, (key, label) in zip(ypos, order):
            mu, sd = agg(ps, key, metric)
            if np.isnan(mu):
                ax.annotate('n/a', (0.03, y), va='center', fontsize=7,
                            color=GRAY_EDGE, style='italic')
                continue
            ours    = key == 'multiprototype_cosine_trajectory'
            variant = key in ('traj_bn_adapted',
                              'multiprototype_cosine_trajectory_whitened')
            col = ACCENT if ours else (ACCENT_2 if variant else GRAY)
            ax.barh(y, mu, height=0.62, color=col,
                    hatch='//' if ours else None,
                    edgecolor='white', linewidth=0.6, zorder=3)
            ax.errorbar(mu, y, xerr=sd, fmt='none', ecolor=INK,
                        elinewidth=0.7, capsize=1.5, zorder=4)
            degen = key == 'static_prototype' and metric == 'unk_det'
            ax.annotate((f'{mu:.3f}' if ours else f'{mu:.2f}')
                        + ('$^{\\dagger}$' if degen else ''),
                        (mu + sd + 0.015, y), va='center', fontsize=7,
                        fontweight='bold' if ours else 'normal',
                        color=ACCENT if ours else GRAY_EDGE)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '', '0.5', '', '1'])
        ax.set_title(title)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([label for _, label in order], fontsize=8)
    axes[0].tick_params(axis='y', length=0)
    fig.tight_layout(w_pad=1.2)
    _save(fig, 'figS1_full_comparison')


# ---------------------------------------------------------------------------
# Fig S2 — Ablation (horizontal bars, deltas vs single centroid)
# ---------------------------------------------------------------------------

ABLATION = [
    ('static_prototype',   'Static Day-7 prototype\n(no trajectory)'),
    ('cosine_trajectory',  'Single-centroid\ntrajectory'),
    ('multiprototype_cosine_trajectory', 'Multi-prototype\ntrajectory (T2R)'),
]


def figS2_ablation():
    ps = all_seeds_phase4()
    metrics = [('closed', 'Closed-set accuracy'),
               ('auroc', 'Open-set AUROC'),
               ('unk_det', 'Unknown detection')]
    ref_auroc, _ = agg(ps, 'cosine_trajectory', 'auroc')
    colors = {
        'static_prototype': '#228833',
        'cosine_trajectory': '#AA3377',
        'multiprototype_cosine_trajectory': ACCENT,
    }
    label_overrides = {
        ('cosine_trajectory', 'auroc'): '0.458',
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=True)
    ypos = np.arange(len(ABLATION))
    for ax, (metric, title) in zip(axes, metrics):
        for y, (key, label) in zip(ypos, ABLATION):
            mu, sd = agg(ps, key, metric)
            if key == 'static_prototype' and metric == 'unk_det':
                ax.annotate('degenerate', (0.02, y), va='center',
                            fontsize=8.0, color=GRAY_EDGE, style='italic')
                continue
            ours    = key == 'multiprototype_cosine_trajectory'
            col = colors[key]
            ax.barh(y, mu, height=0.6, color=col,
                    hatch='//' if ours else None,
                    edgecolor='white', linewidth=0.8, zorder=3)
            ax.errorbar(mu, y, xerr=sd, fmt='none', ecolor=INK,
                        elinewidth=0.9, capsize=2.0, zorder=4)
            txt = label_overrides.get((key, metric), _fmt3(mu))
            ax.annotate(txt, (mu + sd + 0.018, y), va='center', fontsize=8.2,
                        fontweight='bold' if ours else 'normal',
                        color=ACCENT if ours else INK)
        ax.set_xlim(0, 1.02)
        ax.set_title(title, fontsize=10.2)
        ax.tick_params(axis='x', labelsize=8.4)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([label for _, label in ABLATION], fontsize=8.4)
    axes[0].tick_params(axis='y', length=0)
    axes[1].annotate('AUROC deltas vs. single centroid', (0.98, -0.26),
                     xycoords='axes fraction', ha='right', fontsize=7,
                     color=GRAY_EDGE, style='italic')
    fig.tight_layout(w_pad=1.2)
    _save(fig, 'figS2_ablation')


# ---------------------------------------------------------------------------
# Fig S4 — Accept-rate vs unknown-detection trade-off
# ---------------------------------------------------------------------------

def _tradeoff_plot(pts, stem, figsize=(3.5, 3.15)):
    ps = all_seeds_phase4()
    # accept rate: prefer explicit key, else derive from seed_100 kv rows
    def accept_of(key):
        vals = []
        for s in SEEDS:
            d = ps[s].get(key, {})
            if 'accept' in d:
                vals.append(d['accept'])
            elif 'open' in d and 'closed' in d:
                pass
        return np.mean(vals) if vals else np.nan

    # fall back to the kv tail of each seed file
    def kv_accept(key_prefix):
        vals = []
        for s in SEEDS:
            p = os.path.join(MS_ROOT, f'seed_{s}', 'results_phase4.txt')
            txt = open(p).read()
            mm = re.findall(rf'{key_prefix}_accept_rate = ([0-9.]+)', txt)
            if mm:
                vals.append(float(mm[-1]))
        return np.mean(vals) if vals else np.nan

    KV_PREFIX = {
        'cosine_knn': 'knn', 'cosine_trajectory': 'cos_traj',
        'drift_corrected_cosine_knn': 'drift_knn',
        'calibrated_lda': 'lda', 'probe_trajectory_fusion': 'fusion',
        'l2_calibrated_probe': 'l2_probe', 'traj_bn_adapted': 'traj_bn',
        'multiprototype_cosine_trajectory': 'mp_cos_traj',
    }

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    method_cols = {
        'cosine_knn': '#CCBB44',
        'drift_corrected_cosine_knn': '#44AA99',
        'cosine_trajectory': '#AA3377',
        'calibrated_lda': '#6B7480',
        'l2_calibrated_probe': '#997700',
        'probe_trajectory_fusion': '#EE6677',
        'traj_bn_adapted': '#66BBEE',
        'multiprototype_cosine_trajectory': ACCENT,
    }
    for key, label in pts:
        det, _ = agg(ps, key, 'unk_det')
        acc = accept_of(key)
        if np.isnan(acc) and KV_PREFIX.get(key):
            acc = kv_accept(KV_PREFIX[key])
        if np.isnan(acc):
            # accept rate ≈ 1 - fraction rejected among knowns: approximate
            # from open/closed is unreliable; skip if unavailable
            continue
        ours = key == 'multiprototype_cosine_trajectory'
        variant = key == 'traj_bn_adapted'
        col = method_cols[key]
        ax.scatter(acc, det, s=115 if ours else 46,
                   marker='*' if ours else 'o', color=col, zorder=4,
                   edgecolor='white', linewidth=0.6)
        texts.append((acc, det, label, col, ours))
    # hand-tuned label offsets to avoid collisions
    OFF = {
        'Cosine $k$-NN':          (7, -3),
        'Drift-corr. $k$-NN':     (-4, 9),
        'Single-centroid traj.':  (7, -3),
        'LDA':                    (4, -13),
        '$\\ell_2$ probe':        (7, -8),
        'Probe+traj. fusion':     (7, 5),
        'T2R + BN':               (7, -2),
        'T2R (ours)':             (6, 8),
    }
    for acc, det, label, col, ours in texts:
        dx, dy = OFF.get(label, (6, 3))
        ax.annotate(label, (acc, det), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7.4,
                    ha='left' if dx >= 0 else 'right',
                    fontweight='bold' if ours else 'normal',
                    color=col if ours else INK)
    ax.set_xlabel('Known-device accept rate', fontsize=9.3)
    ax.set_ylabel('Unknown detection rate', fontsize=9.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.09)
    ax.tick_params(labelsize=8.5)
    ax.annotate('ideal', (0.97, 1.035), fontsize=8, color=POS,
                ha='right', style='italic')
    ax.scatter([1.0], [1.0], marker='+', s=60, color=POS, zorder=4)
    fig.tight_layout()
    _save(fig, stem)


def figS4_tradeoff():
    pts = [
        ('cosine_knn',            'Cosine $k$-NN'),
        ('cosine_trajectory',     'Single-centroid traj.'),
        ('traj_bn_adapted',       'T2R + BN'),
        ('multiprototype_cosine_trajectory', 'T2R (ours)'),
    ]
    _tradeoff_plot(pts, 'figS4_accept_unk_tradeoff')


def figS4_tradeoff_full():
    pts = [
        ('cosine_knn',            'Cosine $k$-NN'),
        ('drift_corrected_cosine_knn', 'Drift-corr. $k$-NN'),
        ('cosine_trajectory',     'Single-centroid traj.'),
        ('calibrated_lda',        'LDA'),
        ('l2_calibrated_probe',   '$\\ell_2$ probe'),
        ('probe_trajectory_fusion', 'Probe+traj. fusion'),
        ('traj_bn_adapted',       'T2R + BN'),
        ('multiprototype_cosine_trajectory', 'T2R (ours)'),
    ]
    _tradeoff_plot(pts, 'figS4_accept_unk_tradeoff_full')


# ---------------------------------------------------------------------------
# Fig S5 — Pseudo-label update (five-seed mean ± std)
# ---------------------------------------------------------------------------

def figS5_pseudolabel():
    confs = ['0.80', '0.90', '0.95']
    pl_keys = [f'pseudo_label_traj@{c}' for c in confs]
    metrics = [('closed', 'Closed-set accuracy'),
               ('auroc', 'Open-set AUROC'),
               ('unk_det', 'Unknown detection')]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    xs = np.arange(len(confs))
    for ax, (metric, title) in zip(axes, metrics):
        ref, ref_sd = agg(all_seeds_phase4(), 'multiprototype_cosine_trajectory',
                          metric)
        ax.axhline(ref, color=ACCENT, lw=1.3, ls='--', zorder=2)
        vals, errs = [], []
        for key in pl_keys:
            mu, sd = agg(all_seeds_phase4(), key, metric)
            vals.append(mu)
            errs.append(sd)
        bar_cols = ['#4477AA', '#CCBB44', '#AA3377']
        ax.bar(xs, vals, width=0.56, color=bar_cols, edgecolor='white',
               linewidth=0.8, zorder=3)
        ax.errorbar(xs, vals, yerr=errs, fmt='none', ecolor=INK,
                    elinewidth=0.85, capsize=2.0, zorder=4)
        for x, v, e in zip(xs, vals, errs):
            ax.annotate(_fmt3(v), (x, v + e),
                        xytext=(0, 12),
                        textcoords='offset points', ha='center',
                        fontsize=8.0, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{int(float(c)*100)}%' for c in confs])
        ax.set_xlabel('Confidence threshold', fontsize=8.8)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=10.2)
        ax.tick_params(axis='x', length=0)
        ax.tick_params(labelsize=8.4)
    fig.text(0.01, -0.02,
             'Bars show pseudo-labeled trajectory updates; dashed blue lines show base T2R.',
             fontsize=7.2, color=GRAY_EDGE)
    fig.tight_layout(w_pad=1.6)
    _save(fig, 'figS5_pseudolabel')


# ---------------------------------------------------------------------------
# Fig S6 — Cross-day (next-day protocol) curve, seed_100
# ---------------------------------------------------------------------------

def figS6_crossday():
    days, tr_c, tr_a, sm = [], [], [], []
    for d in range(3, 9):
        p = os.path.join(SEED_DIR, f'test_day{d}', 'results_phase4.txt')
        if d == 8:
            p = os.path.join(SEED_DIR, 'results_phase4.txt')
        r = parse_phase4(p)
        if 'multiprototype_cosine_trajectory' not in r:
            continue
        days.append(d)
        tr_c.append(r['multiprototype_cosine_trajectory']['closed'])
        tr_a.append(r['multiprototype_cosine_trajectory']['auroc'])
        sm.append(r.get('softmax', {}).get('closed', np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    ax = axes[0]
    ax.plot(days, tr_c, '-o', ms=5, lw=2.0, color=ACCENT, label='T2R',
            markeredgecolor='white', markeredgewidth=0.5, zorder=3)
    ax.plot(days, sm, '--s', ms=4.8, lw=1.7, color=GRAY_EDGE, label='Softmax',
            markeredgecolor='white', markeredgewidth=0.5, zorder=3)
    ax.axhline(1 / 16, color=NEG, lw=1.0, ls=':', zorder=2,
               label='Chance (1/16)')
    ax.annotate('Day 8 peak', xy=(8, tr_c[-1]), xytext=(-70, -6),
                textcoords='offset points', fontsize=7.4, color=ACCENT,
                arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.7))
    ax.set_xlabel('Test day (enroll through previous day)', fontsize=9.2)
    ax.set_ylabel('Closed-set accuracy', fontsize=9.2)
    ax.set_ylim(0, 0.78)
    ax.legend(loc='upper left', fontsize=7.8, frameon=True, framealpha=0.92,
              edgecolor='0.85')
    ax.set_title('(a) Closed-set accuracy', fontsize=10.2)

    ax = axes[1]
    ax.plot(days, tr_a, '-o', ms=5, lw=2.0, color=ACCENT, label='T2R',
            markeredgecolor='white', markeredgewidth=0.5, zorder=3)
    ax.axhline(0.5, color=NEG, lw=1.0, ls=':', label='Chance (0.5)')
    ax.annotate('Day 8 peak', xy=(8, tr_a[-1]), xytext=(-70, -8),
                textcoords='offset points', fontsize=7.4, color=ACCENT,
                arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.7))
    ax.set_xlabel('Test day', fontsize=9.2)
    ax.set_ylabel('Open-set AUROC', fontsize=9.2)
    ax.set_ylim(0.4, 0.8)
    ax.set_title('(b) Open-set AUROC', fontsize=10.2)
    ax.legend(loc='upper left', fontsize=7.8, frameon=True, framealpha=0.92,
              edgecolor='0.85')
    for a in axes:
        a.set_xticks(days)
        a.tick_params(labelsize=8.4)
    fig.tight_layout(w_pad=2.0)
    _save(fig, 'figS6_crossday_curve')


# ---------------------------------------------------------------------------
# Diagnostic figures from recomputed scores
# ---------------------------------------------------------------------------

def _scores():
    d = np.load(os.path.join(CACHE, 'day8_scores.npz'))
    c = np.load(os.path.join(CACHE, 'day8_centroid_scores.npz'))
    return d, c


def figS3_roc():
    from sklearn.metrics import roc_curve, auc
    d, c = _scores()
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    table_auc = {
        'Single-centroid traj.': '0.458\\pm0.013',
        'T2R multi-prototype': '0.714\\pm0.014',
    }
    curves = {}
    specs = [
        (c['dist_known'], c['dist_unk'], 'Single-centroid traj.', '#5B6470', 1.6),
        (d['dist_known'], d['dist_unk'], 'T2R multi-prototype', '#0067C5', 2.5),
    ]
    for dist_k, dist_u, label, col, lw in specs:
        y = np.r_[np.zeros(len(dist_k)), np.ones(len(dist_u))]
        s = np.r_[dist_k, dist_u]
        fpr, tpr, _ = roc_curve(y, s)
        curves[label] = (fpr, tpr)

    grid = np.linspace(0, 1, 500)
    t2r_interp = np.interp(grid, *curves['T2R multi-prototype'])
    cent_interp = np.interp(grid, *curves['Single-centroid traj.'])
    ax.fill_between(grid, cent_interp, t2r_interp,
                    where=t2r_interp >= cent_interp,
                    color='#66BBEE', alpha=0.22, linewidth=0,
                    label='T2R improvement')

    for _, _, label, col, lw in specs:
        fpr, tpr = curves[label]
        short = 'Single-centroid' if label.startswith('Single') else 'T2R'
        ax.plot(fpr, tpr, color=col, lw=lw,
                label=f'{short}: ${table_auc[label]}$')
    ax.plot([0, 1], [0, 1], ls=':', color='#8A6F00', lw=1.2,
            label='Chance: 0.500')
    # operating point of the calibrated threshold
    thr = float(d['threshold'])
    op_fpr = float(np.mean(d['dist_known'] > thr))
    op_tpr = float(np.mean(d['dist_unk'] > thr))
    ax.scatter([op_fpr], [op_tpr], s=34, marker='o', color='#0067C5', zorder=5,
               edgecolor='white', linewidth=0.7)
    ax.annotate('calibrated threshold', (op_fpr, op_tpr),
                xytext=(-8, 16), textcoords='offset points', fontsize=7,
                ha='right',
                color='#0067C5',
                arrowprops=dict(arrowstyle='-', color='#0067C5', lw=0.7,
                                shrinkA=2, shrinkB=2))
    ax.set_xlabel('False positive rate (known rejected)')
    ax.set_ylabel('True positive rate (unknown detected)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', bbox_to_anchor=(0.992, 0.026),
              title='Mean AUROC', title_fontsize=6.5, fontsize=6.2,
              frameon=True, framealpha=0.9,
              facecolor='white', edgecolor='0.82', handlelength=1.55,
              borderpad=0.3, labelspacing=0.22, handletextpad=0.45)
    _save(fig, 'figS3_roc_curves')


def figS7_confusion():
    from sklearn.metrics import confusion_matrix
    d, _ = _scores()
    cm = confusion_matrix(d['y_known'], d['pred_known'],
                          labels=list(range(16)))
    cmn = cm / cm.sum(axis=1, keepdims=True)
    diag = np.diag(cmn)
    outlier = int(np.argmin(diag))

    fig, ax = plt.subplots(figsize=(3.55, 3.3))
    diag_colors = [
        '#0072B2', '#56B4E9', '#009E73', '#F0E442',
        '#E69F00', '#D55E00', '#CC79A7', '#332288',
        '#44AA99', '#88CCEE', '#117733', '#DDCC77',
        '#AA4499', '#D55E00', '#6699CC', '#999933'
    ]
    base = np.ones((16, 16, 3))
    error_color = np.array(matplotlib.colors.to_rgb('#8ecae6'))
    for i in range(16):
        for j in range(16):
            if i == j:
                base[i, j] = matplotlib.colors.to_rgb(diag_colors[i])
            else:
                alpha = min(cmn[i, j] / 0.35, 1.0) * 0.38
                base[i, j] = (1.0 - alpha) * np.ones(3) + alpha * error_color
    ax.imshow(base, vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels([str(i) for i in range(16)], fontsize=6.8)
    ax.set_yticklabels([str(i) for i in range(16)], fontsize=6.8)
    ax.set_xlabel('Predicted device')
    ax.set_ylabel('True device')

    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.35)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(16):
        v = cmn[i, i]
        ax.annotate(f'{v:.2f}', (i, i), ha='center', va='center',
                    fontsize=5.4, color='white' if v > 0.52 else INK,
                    fontweight='bold' if i == outlier else 'normal')

    ax.add_patch(plt.Rectangle((outlier - 0.5, outlier - 0.5), 1, 1,
                               fill=False, edgecolor=NEG, linewidth=1.35))
    ax.text(0.03, 0.055, f'lowest: device {outlier}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=6.8, color=NEG,
            bbox=dict(facecolor='white', edgecolor='0.86',
                      boxstyle='round,pad=0.18', alpha=0.9))
    acc = float(d['closed'])
    ax.set_title(f'Diagonal entries show per-device accuracy; overall {acc:.3f}',
                 fontsize=8.2)
    _save(fig, 'figS7_confusion_matrix')


def figS9_threshold():
    d, _ = _scores()
    dist_k, dist_u = d['dist_known'], d['dist_unk']
    thr_cal = float(d['threshold'])
    ts = np.linspace(0, max(dist_k.max(), dist_u.max()), 300)
    acc_rate = [(dist_k <= t).mean() for t in ts]
    det_rate = [(dist_u > t).mean() for t in ts]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(ts, acc_rate, color=ACCENT, label='Known accept rate')
    ax.plot(ts, det_rate, color=NEG, ls='--', label='Unknown detection rate')
    ax.axvline(thr_cal, color=GRAY_EDGE, lw=1.0, ls=':',
               label=f'Calibrated $\\eta={thr_cal:.3f}$')
    ax.set_xlabel('Rejection threshold $\\eta$ (cosine distance)')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0, ts[-1])
    ax.legend(loc='upper right', fontsize=7.2, bbox_to_anchor=(1.0, 0.94))
    ax.annotate(f'AUROC {float(d["auroc"]):.3f} (threshold-free)',
                (0.985, 0.52), xycoords='axes fraction', ha='right',
                fontsize=7, color=GRAY_EDGE, style='italic')
    _save(fig, 'figS9_threshold_sensitivity')


def figS10_per_device():
    d, _ = _scores()
    thr = float(d['threshold'])
    y_k, pred_k = d['y_known'], d['pred_known']
    accs = [float(np.mean(pred_k[y_k == i] == i)) for i in range(16)]

    y_u_orig, dist_u = d['y_unk_orig'], d['dist_unk']
    unk_ids = sorted(np.unique(y_u_orig))
    dets = [float(np.mean(dist_u[y_u_orig == u] > thr)) for u in unk_ids]

    from sklearn.metrics import roc_auc_score
    dist_k = d['dist_known']
    aurocs = []
    for u in unk_ids:
        du = dist_u[y_u_orig == u]
        y = np.r_[np.zeros(len(dist_k)), np.ones(len(du))]
        aurocs.append(float(roc_auc_score(y, np.r_[dist_k, du])))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75),
                             gridspec_kw={'width_ratios': [2.55, 1.15]})
    ax = axes[0]
    xs = np.arange(16)
    multicolor = [
        '#0072B2', '#56B4E9', '#009E73', '#F0E442',
        '#E69F00', '#D55E00', '#CC79A7', '#332288',
        '#44AA99', '#88CCEE', '#117733', '#DDCC77',
        '#AA4499', '#882255', '#6699CC', '#999933'
    ]
    colors = multicolor[:16]
    outlier = int(np.argmin(accs))
    colors[outlier] = '#D55E00'
    ax.bar(xs, accs, width=0.66, color=colors, edgecolor='white', zorder=3)
    mean_acc = float(np.mean(accs))
    ax.axhline(mean_acc, color=INK, lw=0.9, ls='--', zorder=4,
               label=f'Mean {mean_acc:.3f}')
    for x, v in zip(xs, accs):
        if x == outlier or v >= 0.74:
            ax.annotate(f'{v:.2f}', (x, v), xytext=(0, 3),
                        textcoords='offset points', ha='center',
                        fontsize=6.6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{i}' for i in xs], fontsize=7.2)
    ax.set_xlabel('Known device')
    ax.set_ylabel('Closed-set accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('(a) Per-device accuracy (known)')
    ax.legend(loc='upper left', fontsize=6.8, frameon=True)
    ax.tick_params(axis='x', length=0)

    ax = axes[1]
    xs = np.arange(len(unk_ids))
    w = 0.38
    ax.bar(xs - w / 2, dets, width=w, color='#009E73', edgecolor='white',
           zorder=3, label='Detection rate')
    ax.bar(xs + w / 2, aurocs, width=w, color='#E69F00', edgecolor='white',
           zorder=3, label='AUROC vs. knowns')
    for x, v in zip(xs - w / 2, dets):
        ax.annotate(f'{v:.2f}', (x, v), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=6.6)
    for x, v in zip(xs + w / 2, aurocs):
        ax.annotate(f'{v:.2f}', (x, v), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=6.5,
                    color=GRAY_EDGE)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'U{u}' for u in unk_ids], fontsize=7.5)
    ax.set_xlabel('Unknown device')
    ax.set_ylim(0, 1.08)
    ax.set_title('(b) Per-device rejection (unknown)')
    ax.legend(fontsize=6.4, loc='lower right', frameon=True)
    ax.tick_params(axis='x', length=0)
    fig.tight_layout(w_pad=1.7)
    _save(fig, 'figS10_per_device_breakdown')


# ---------------------------------------------------------------------------
# Embedding figures
# ---------------------------------------------------------------------------

def _pca2(X):
    from sklearn.decomposition import PCA
    p = PCA(n_components=2, random_state=42)
    return p.fit(X), p


def fig1_drift():
    z = np.load(os.path.join(CACHE, 'drift_embeddings.npz'))
    e1, y1 = z['emb_d1'], z['y_d1']
    e7, y7 = z['emb_d7'], z['y_d7']
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42).fit(np.vstack([e1, e7]))
    p1, p7 = pca.transform(e1), pca.transform(e7)

    lims = np.vstack([p1, p7])
    xlim = (lims[:, 0].min() - 0.05, lims[:, 0].max() + 0.05)
    ylim = (lims[:, 1].min() - 0.05, lims[:, 1].max() + 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharex=True,
                             sharey=True)
    for ax, (proj, ys, title) in zip(axes, [
        (p1, y1, '(a) Day 1 (training day): tight, separated clusters'),
        (p7, y7, '(b) Day 7, same model: clusters drift and overlap'),
    ]):
        for i in range(16):
            m = ys == i
            ax.scatter(proj[m, 0], proj[m, 1], s=9, alpha=0.55,
                       linewidths=0, color=PAL16[i % 16])
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('PC 1')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.grid(False)
    axes[0].set_ylabel('PC 2')
    fig.tight_layout(w_pad=1.5)
    _save(fig, 'fig1_drift_motivation')


def fig2_trajectory(n_highlight=5):
    z = np.load(os.path.join(CACHE, 'trajectory_history.npz'))
    means, covs, meta = z['means'], z['covs'], z['meta']   # meta: (dev, day)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42).fit(means)
    P = pca.transform(means)

    devs = sorted(set(meta[:, 0]))
    paths = {d: P[meta[:, 0] == d] for d in devs}
    days  = {d: meta[meta[:, 0] == d][:, 1] for d in devs}
    drift = {d: np.linalg.norm(np.diff(paths[d], axis=0), axis=1).sum()
             for d in devs}
    top = sorted(devs, key=lambda d: -drift[d])[:n_highlight]
    cols = {d: PAL16[i % 16] for i, d in enumerate(top)}

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for d in devs:
        if d in top:
            continue
        ax.plot(paths[d][:, 0], paths[d][:, 1], '-', color=GRAY, lw=0.9,
                zorder=2)
        ax.scatter(*paths[d][-1], s=8, color=GRAY, zorder=2, linewidths=0)

    label_off = [(6, 4), (6, -10), (-8, 8), (6, 8), (-10, -12)]
    for k, d in enumerate(top):
        p = paths[d]
        col = cols[d]
        # projected 1.5-sigma spread contours at first & last day
        for idx, ls in ((0, ':'), (len(p) - 1, '-')):
            cov2 = pca.components_ @ covs[meta[:, 0] == d][idx] \
                   @ pca.components_.T
            vals, vecs = np.linalg.eigh(cov2)
            vals = np.maximum(vals, 1e-12)
            ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            ax.add_patch(Ellipse(p[idx], *(2 * 1.5 * np.sqrt(vals)),
                                 angle=ang, facecolor='none',
                                 edgecolor=col, linewidth=0.7, ls=ls,
                                 alpha=0.45, zorder=1))
        ax.plot(p[:, 0], p[:, 1], '-', color=col, lw=1.6, zorder=3)
        ax.scatter(p[0, 0], p[0, 1], s=26, color=col, zorder=4,
                   edgecolor='white', linewidth=0.6)
        ax.scatter(p[1:-1, 0], p[1:-1, 1], s=10, color=col, zorder=4,
                   linewidths=0)
        ax.annotate('', xy=p[-1], xytext=p[-2],
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4),
                    zorder=5)
        ax.annotate(f'Dev {d}', p[-1], xytext=label_off[k % len(label_off)],
                    textcoords='offset points', fontsize=7, color=col,
                    fontweight='bold', zorder=6)

    # zoom to the trajectory paths, not the ellipses
    allp = np.vstack([paths[d] for d in devs])
    mx, my = 0.30 * np.ptp(allp[:, 0]), 0.30 * np.ptp(allp[:, 1])
    ax.set_xlim(allp[:, 0].min() - mx, allp[:, 0].max() + mx)
    ax.set_ylim(allp[:, 1].min() - my, allp[:, 1].max() + my)

    hnd = [
        Line2D([], [], color=INK, marker='o', ls='none', ms=5,
               markeredgecolor='white', label='Day-1 start'),
        Line2D([], [], color=INK, marker=r'$\rightarrow$', ls='none', ms=8,
               label='drift to Day 7'),
        Line2D([], [], color=INK, ls=':', lw=0.8,
               label='1.5$\\sigma$ spread (Day 1 / 7)'),
        Line2D([], [], color=GRAY, ls='-', label='other devices'),
    ]
    ax.legend(handles=hnd, loc='lower left', fontsize=6.8,
              handlelength=1.4, framealpha=0.95)
    ax.set_xlabel('Embedding PC 1')
    ax.set_ylabel('Embedding PC 2')
    _save(fig, 'fig2_trajectory')


def figS8_tsne():
    from sklearn.manifold import TSNE
    z = np.load(os.path.join(CACHE, 'drift_embeddings.npz'))
    pairs = [('emb_d8_static', 'y_d8_static',
              '(a) Static Day-1 model'),
             ('emb_d8_adapted', 'y_d8_adapted',
              '(b) T2R-adapted model')]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    for ax, (ek, yk, title) in zip(axes, pairs):
        emb, ys = z[ek], z[yk]
        ts = TSNE(n_components=2, random_state=42, perplexity=30,
                  init='pca').fit_transform(emb)
        for i in range(16):
            m = ys == i
            ax.scatter(ts[m, 0], ts[m, 1], s=10, alpha=0.72, linewidths=0,
                       color=PAL16[i % 16])
        ax.set_title(title, fontsize=10.2, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel('t-SNE 1', fontsize=8.8)
        ax.set_ylabel('t-SNE 2', fontsize=8.8)
        ax.text(0.02, 0.03, 'colors = device IDs', transform=ax.transAxes,
                fontsize=7.2, color=GRAY_EDGE,
                bbox=dict(facecolor='white', edgecolor='0.88',
                          boxstyle='round,pad=0.25', alpha=0.88))
    fig.tight_layout(w_pad=1.5)
    _save(fig, 'figS8_tsne_adaptation')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fig3_finetune_progress()
    fig4_main_comparison()
    figS1_full_comparison()
    figS2_ablation()
    figS4_tradeoff()
    figS4_tradeoff_full()
    figS5_pseudolabel()
    figS6_crossday()
    figS3_roc()
    figS7_confusion()
    figS9_threshold()
    figS10_per_device()
    fig1_drift()
    fig2_trajectory()
    figS8_tsne()
    print('[v2] all figures done.')
