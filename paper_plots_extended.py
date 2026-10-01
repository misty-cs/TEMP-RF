#!/usr/bin/env python3
"""
paper_plots_extended.py — Supplementary publication figures.

Figures
-------
S1. figS1_full_comparison    — All main methods: closed-acc, open-acc, AUROC, F1-macro
S2. figS2_ablation           — Trajectory ablation: component-by-component breakdown
S3. figS3_roc_curves         — ROC curves (known vs unknown) for key methods
S4. figS4_accept_unk_tradeoff — Scatter: known-accept-rate vs unknown-detection per method
S5. figS5_pseudolabel        — Pseudo-label trajectory update vs base trajectory

All figures → {output_root}/paper_plots_extended/  (PDF + PNG)

Standalone usage:
    python paper_plots_extended.py --output_root res_out --data_root /path/to/neu2
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
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


# ---------------------------------------------------------------------------
# Global style  (mirrors paper_plots.py)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    # IEEE two-column — Times New Roman, readable at print size
    'font.family':           'serif',
    'font.serif':            ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':             11,
    'axes.titlesize':        11,
    'axes.titlepad':         5,
    'axes.labelsize':        11,
    'xtick.labelsize':       10,
    'ytick.labelsize':       10,
    'legend.fontsize':       10,
    'legend.title_fontsize': 10,
    'legend.framealpha':     0.85,
    'legend.edgecolor':      '0.8',
    'lines.linewidth':       1.8,
    'lines.markersize':      7,
    'figure.dpi':            300,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'axes.grid':             True,
    'grid.alpha':            0.3,
    'grid.linewidth':        0.4,
    'savefig.bbox':          'tight',
    'savefig.pad_inches':    0.05,
    'pdf.fonttype':          42,
    'ps.fonttype':           42,
})

_PAL = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
    '#AA3377', '#BBBBBB', '#332288', '#997700', '#44AA99',
    '#EE99AA', '#77AADD', '#88CCAA', '#DDCC77', '#CC6677',
    '#999933', '#882255', '#661100', '#117733', '#0077BB',
]


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _strip_titles(fig)
    fig.savefig(os.path.join(out_dir, f'{_stem(stem)}.pdf'))
    fig.savefig(os.path.join(out_dir, f'{_stem(stem)}.png'), dpi=300)
    plt.close(fig)
    print(f'[paper_plots_extended] saved: {stem}')


def _nan(v):
    return float('nan') if v is None else v


# ---------------------------------------------------------------------------
# Figure S1 — Full method comparison (all baselines)
# ---------------------------------------------------------------------------

def _paired_ttest(a: list, b: list) -> float:
    """Two-sided paired t-test p-value. Returns nan if too few samples."""
    try:
        from scipy.stats import ttest_rel
        if len(a) < 2 or len(a) != len(b):
            return float('nan')
        _, p = ttest_rel(a, b)
        return float(p)
    except Exception:
        return float('nan')


def plot_full_comparison(cfg, phase4_result: dict, out_dir: str,
                         multiseed_summary: dict | None = None) -> None:
    """
    Four-panel bar chart: closed-acc, open-acc, AUROC, F1-macro for every
    method produced by phase4.
    """
    r = phase4_result

    # (display_name, key_prefix, has_open_set)
    METHOD_DEFS = [
        ('Softmax',          'softmax',        False),
        ('TTA-BN',           'softmax_bn',     False),
        ('Static proto',     'static',         True),
        ('OpenMax',          'openmax',        True),
        ('Lin. probe',       'probe',          True),
        ('L2 probe',         'l2_probe',       True),
        ('Temp. softmax',    'ts',             True),
        ('LDA',              'lda',            True),
        ('KNN',              'knn',            True),
        ('Cosine traj',      'cos_traj',       True),
        ('T2R (ours)',       'mp_cos_traj',    True),
        ('Traj+BN',          'traj_bn',        True),
        ('Fusion',           'fusion',         True),
        ('Traj-Mahal.',      'traj',           True),
    ]

    def _get(prefix, suffix, fallback_keys=None):
        # Handle the softmax special case (no prefix pattern)
        specials = {
            ('softmax', 'closed_acc'):    r.get('softmax_acc',    float('nan')),
            ('softmax_bn', 'closed_acc'): r.get('softmax_bn_acc', float('nan')),
        }
        k = (prefix, suffix)
        if k in specials:
            return specials[k]
        return r.get(f'{prefix}_{suffix}', float('nan'))

    def _std(prefix, suffix):
        if multiseed_summary and 'std' in multiseed_summary:
            return multiseed_summary['std'].get(f'{prefix}_{suffix}', 0.0)
        return 0.0

    names        = [m[0] for m in METHOD_DEFS]
    prefixes     = [m[1] for m in METHOD_DEFS]
    has_open     = [m[2] for m in METHOD_DEFS]

    closed  = [_get(p, 'closed_acc') for p in prefixes]
    open_   = [_get(p, 'open_acc') if h else float('nan')
               for p, h in zip(prefixes, has_open)]
    auroc   = [_get(p, 'auroc') if h else float('nan')
               for p, h in zip(prefixes, has_open)]
    f1      = [_get(p, 'f1_macro') if h else float('nan')
               for p, h in zip(prefixes, has_open)]

    closed_err = [_std(p, 'closed_acc') for p in prefixes]
    open_err   = [_std(p, 'open_acc')   for p in prefixes]
    auroc_err  = [_std(p, 'auroc')      for p in prefixes]
    f1_err     = [_std(p, 'f1_macro')   for p in prefixes]

    y   = np.arange(len(names))
    h   = 0.55
    col = [_PAL[i % len(_PAL)] for i in range(len(names))]
    hatches = ['//' if n == 'T2R (ours)' else '' for n in names]

    # Horizontal bar chart: method names on y-axis — no rotation, no crowding
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 4.2))

    panels = [
        (axes[0], closed,  closed_err, 'Accuracy',      'Closed-set\naccuracy'),
        (axes[1], open_,   open_err,   'Open accuracy', 'Open-set\naccuracy'),
        (axes[2], auroc,   auroc_err,  'AUROC',         'AUROC'),
        (axes[3], f1,      f1_err,     'F1-macro',      'F1-macro'),
    ]

    for i, (ax, vals, errs, xlabel, title) in enumerate(panels):
        bars = ax.barh(y, vals, height=h, color=col, edgecolor='white', linewidth=0.5,
                       xerr=[e if not np.isnan(v) else 0 for v, e in zip(vals, errs)],
                       capsize=2, error_kw=dict(lw=0.8, capthick=0.8, ecolor='#333333'))
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        # Annotate only "ours" with value
        for bar, v, hatch in zip(bars, vals, hatches):
            if hatch == '//' and not np.isnan(v):
                ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{v:.3f}', va='center', ha='left',
                        fontsize=8, fontweight='bold')
        # Open-acc panel: annotate accept rate — high open-acc at tiny
        # accept rate is a threshold artifact, not superiority
        if xlabel == 'Open accuracy':
            for bar, p in zip(bars, prefixes):
                ar = r.get(f'{p}_accept_rate', float('nan'))
                if not np.isnan(ar):
                    ax.text(0.02, bar.get_y() + bar.get_height() / 2,
                            f'@{ar:.2f}', va='center', ha='left',
                            fontsize=6, color='white')
        ax.set_yticks(y)
        # Method names only on leftmost panel; others get tick marks only
        if i == 0:
            ax.set_yticklabels(names, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(0, 1.12)
        ax.axvline(1.0, color='gray', lw=0.6, ls=':', alpha=0.4)
        ax.invert_yaxis()  # top = first method

    # Significance annotation: paired t-test between "Traj (ours)" and best baseline
    # on AUROC (axes[2]) using per-seed values from multiseed_summary if available
    if multiseed_summary and 'per_seed' in multiseed_summary:
        per_seed = multiseed_summary['per_seed']
        ours_auroc = [v.get('traj_auroc', float('nan'))
                      for v in per_seed.values() if not np.isnan(v.get('traj_auroc', float('nan')))]
        # Best non-ours method that has per-seed data
        for cmp_key in ('mp_cos_traj_auroc', 'openmax_auroc', 'static_auroc'):
            cmp_vals = [v.get(cmp_key, float('nan'))
                        for v in per_seed.values()
                        if not np.isnan(v.get(cmp_key, float('nan')))]
            if len(cmp_vals) >= 2:
                p = _paired_ttest(ours_auroc, cmp_vals)
                if not np.isnan(p):
                    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
                    axes[2].text(0.98, 0.02,
                                 f'vs. best: {stars}',
                                 transform=axes[2].transAxes,
                                 ha='right', va='bottom', fontsize=8,
                                 color='#555555', style='italic')
                    break

    fig.tight_layout(w_pad=1.5)
    _save(fig, out_dir, 'figS1_full_comparison')


# ---------------------------------------------------------------------------
# Figure S2 — Trajectory ablation
# ---------------------------------------------------------------------------

def plot_ablation(cfg, phase4_result: dict, out_dir: str,
                  multiseed_summary: dict | None = None) -> None:
    """
    Component-by-component ablation of the trajectory method.
    Shows the contribution of each design choice.
    """
    r = phase4_result

    # Ordered ablation steps: (label, closed_acc key, auroc key, unk_det key)
    ABLATION = [
        ('Static\nproto',       'static_closed_acc',       'static_auroc',       'static_unk_det'),
        ('Cosine\ntraj',        'cos_traj_closed_acc',      'cos_traj_auroc',      'cos_traj_unk_det'),
        ('Mp-cos\ntraj',        'mp_cos_traj_closed_acc',   'mp_cos_traj_auroc',   'mp_cos_traj_unk_det'),
        ('Mp-cos\n+TTA-BN',     'traj_bn_closed_acc',       'traj_bn_auroc',       'traj_bn_unk_det'),
        ('Fusion\n(probe+traj)','fusion_closed_acc',        'fusion_auroc',        'fusion_unk_det'),
        ('Traj\n(full, ours)',  'traj_closed_acc',          'traj_auroc',          'traj_unk_det'),
    ]

    names    = [a[0] for a in ABLATION]
    closed   = [r.get(a[1], float('nan')) for a in ABLATION]
    auroc    = [r.get(a[2], float('nan')) for a in ABLATION]
    unk_det  = [r.get(a[3], float('nan')) for a in ABLATION]

    def _std(key):
        if multiseed_summary and 'std' in multiseed_summary:
            return multiseed_summary['std'].get(key, 0.0)
        return 0.0

    closed_err  = [_std(a[1]) for a in ABLATION]
    auroc_err   = [_std(a[2]) for a in ABLATION]
    unk_det_err = [_std(a[3]) for a in ABLATION]

    x   = np.arange(len(names))
    w   = 0.55
    col = [_PAL[i % len(_PAL)] for i in range(len(names))]
    hatches = [''] * (len(names) - 1) + ['//']

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    for ax, vals, errs, ylabel, title in [
        (axes[0], closed,  closed_err,  'Accuracy',       'Closed-set accuracy'),
        (axes[1], auroc,   auroc_err,   'AUROC',          'Open-set AUROC'),
        (axes[2], unk_det, unk_det_err, 'Detection rate', 'Unknown detection rate'),
    ]:
        bars = ax.bar(x, vals, width=w, color=col, edgecolor='white', linewidth=0.6,
                      yerr=[e if not np.isnan(v) else 0 for v, e in zip(vals, errs)],
                      capsize=4, error_kw=dict(lw=1.2, capthick=1.2, ecolor='#333333'))
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for bar, v, hatch in zip(bars, vals, hatches):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.02, f'{v:.2f}',
                        ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold' if hatch == '//' else 'normal')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', rotation_mode='anchor',
                           fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.22)
        ax.axhline(1.0, color='gray', lw=0.6, ls=':', alpha=0.4)
        ax.tick_params(axis='x', pad=2)

    # Annotate progression arrows between bars in AUROC panel
    auroc_vals = [r.get(a[2], float('nan')) for a in ABLATION]
    for i in range(len(auroc_vals) - 1):
        v0, v1 = auroc_vals[i], auroc_vals[i + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        delta = v1 - v0
        if abs(delta) < 0.002:
            continue
        color = '#228833' if delta > 0 else '#EE6677'
        axes[1].annotate(
            f'{delta:+.2f}',
            xy=(i + 1, max(v0, v1) + 0.05),
            ha='center', va='bottom', fontsize=8, color=color,
            fontweight='bold',
        )

    fig.suptitle(
        f'Ablation: trajectory component contributions  —  Day {cfg.test_day} test',
        fontsize=9, y=1.01,
    )
    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'figS2_ablation')


# ---------------------------------------------------------------------------
# Figure S3 — ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(cfg, phase4_result: dict, out_dir: str,
                    roc_scores: dict | None = None) -> None:
    """
    ROC curves (known=negative, unknown=positive) for key methods.

    If roc_scores is provided it should be a dict mapping method name →
    {'scores_known': array, 'scores_unknown': array, 'higher_is_unknown': bool}.
    Otherwise falls back to plotting AUROC scalars as horizontal dashed lines
    (still useful for comparison without raw scores).
    """
    from sklearn.metrics import roc_curve, auc

    KEY_METHODS = [
        ('Static proto',  'static'),
        ('OpenMax',       'openmax'),
        ('KNN',           'knn'),
        ('Cosine traj',   'cos_traj'),
        ('T2R (ours)',    'mp_cos_traj'),
        ('Traj-Mahal.',   'traj'),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random')

    for i, (name, prefix) in enumerate(KEY_METHODS):
        col = _PAL[i % len(_PAL)]
        auroc_scalar = phase4_result.get(f'{prefix}_auroc', float('nan'))

        if roc_scores and prefix in roc_scores:
            sc = roc_scores[prefix]
            scores_known   = np.asarray(sc['scores_known'],   dtype=np.float32)
            scores_unknown = np.asarray(sc['scores_unknown'],  dtype=np.float32)
            if not sc.get('higher_is_unknown', True):
                scores_known   = -scores_known
                scores_unknown = -scores_unknown
            binary = np.concatenate([
                np.zeros(len(scores_known)),
                np.ones(len(scores_unknown)),
            ])
            scores_all = np.concatenate([scores_known, scores_unknown])
            fpr, tpr, _ = roc_curve(binary, scores_all)
            auroc_val   = auc(fpr, tpr)
            lw = 2.0 if prefix == 'traj' else 1.5
            ls = '-'
            ax.plot(fpr, tpr, color=col, lw=lw, ls=ls,
                    label=f'{name}  (AUC={auroc_val:.3f})')
        elif not np.isnan(auroc_scalar):
            # No raw scores — show AUROC as text annotation only
            ax.axhline(auroc_scalar, color=col, lw=1.2, ls=':',
                       alpha=0.7, label=f'{name}  (AUC={auroc_scalar:.3f}, scalar only)')
        else:
            continue

    ax.set_xlabel('False positive rate  (known rejected as unknown)')
    ax.set_ylabel('True positive rate  (unknown detected)')
    ax.set_title('ROC curves: open-set unknown detection')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower right', framealpha=0.90)

    note = '' if roc_scores else '  (dotted = AUROC scalar; raw scores not available)'
    ax.text(0.98, 0.05, note, transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', color='gray', style='italic')

    fig.tight_layout()
    _save(fig, out_dir, 'figS3_roc_curves')


# ---------------------------------------------------------------------------
# Figure S4 — Accept-rate vs Unknown-detection tradeoff
# ---------------------------------------------------------------------------

def plot_accept_unk_tradeoff(cfg, phase4_result: dict, out_dir: str) -> None:
    """
    Scatter: x = known_accept_rate, y = unk_det, one point per method.
    Good methods are top-right (high accept AND high detection).
    """
    r = phase4_result

    METHODS = [
        ('Softmax',        float('nan'), float('nan')),   # no rejection
        ('TTA-BN',         float('nan'), float('nan')),   # no rejection
        ('Static proto',   r.get('static_closed_acc',      float('nan')),
                           r.get('static_unk_det',         float('nan'))),
        ('OpenMax',        r.get('openmax_closed_acc',     float('nan')),
                           r.get('openmax_unk_det',        float('nan'))),
        ('Lin. probe',     r.get('probe_accept_rate',      float('nan')),
                           r.get('probe_unk_det',          float('nan'))),
        ('Temp. softmax',  r.get('ts_accept_rate',         float('nan')),
                           r.get('ts_unk_det',             float('nan'))),
        ('LDA',            r.get('lda_accept_rate',        float('nan')),
                           r.get('lda_unk_det',            float('nan'))),
        ('KNN',            r.get('knn_accept_rate',        float('nan')),
                           r.get('knn_unk_det',            float('nan'))),
        ('Cosine traj',    r.get('cos_traj_accept_rate',   float('nan')),
                           r.get('cos_traj_unk_det',       float('nan'))),
        ('T2R (ours)',     r.get('mp_cos_traj_accept_rate',float('nan')),
                           r.get('mp_cos_traj_unk_det',    float('nan'))),
        ('Traj+BN',        r.get('traj_bn_accept_rate',    float('nan')),
                           r.get('traj_bn_unk_det',        float('nan'))),
        ('Fusion',         r.get('fusion_accept_rate',     float('nan')),
                           r.get('fusion_unk_det',         float('nan'))),
        ('Traj-Mahal.',    r.get('traj_accept_rate',       float('nan')),
                           r.get('traj_unk_det',           float('nan'))),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    for i, (name, accept, unk_det) in enumerate(METHODS):
        if np.isnan(accept) or np.isnan(unk_det):
            continue
        col  = _PAL[i % len(_PAL)]
        ms   = 120 if name == 'T2R (ours)' else 70
        mark = '*' if name == 'T2R (ours)' else 'o'
        ax.scatter(accept, unk_det, s=ms, color=col, marker=mark,
                   edgecolors='white', linewidths=0.8, zorder=4)
        ax.annotate(name, (accept, unk_det),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=8.5, color=col)

    # Ideal corner marker
    ax.axvline(0.95, color='gray', lw=0.8, ls=':', alpha=0.4)
    ax.axhline(0.95, color='gray', lw=0.8, ls=':', alpha=0.4)
    ax.text(0.96, 0.96, 'Ideal region', fontsize=8, color='gray',
            ha='left', va='bottom', transform=ax.transAxes)

    ax.set_xlabel('Known accept rate  (fraction of known samples accepted)')
    ax.set_ylabel('Unknown detection rate  (fraction of unknowns rejected)')
    ax.set_title('Accept-rate vs unknown-detection tradeoff per method')
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.08)

    fig.tight_layout()
    _save(fig, out_dir, 'figS4_accept_unk_tradeoff')


# ---------------------------------------------------------------------------
# Figure S5 — Pseudo-label trajectory update
# ---------------------------------------------------------------------------

def plot_pseudolabel(cfg, phase4_result: dict, out_dir: str) -> None:
    """
    Bar chart comparing base trajectory vs pseudo-label updated trajectory
    at conf thresholds 80 / 90 / 95%.
    """
    r = phase4_result

    confs    = [80, 90, 95]
    pl_keys  = [str(c) for c in confs]

    # Check at least one pseudo-label result exists
    if not any(r.get(f'pl{k}_auroc') is not None for k in pl_keys):
        print('[paper_plots_extended] Fig S5: no pseudo-label results — skipping')
        return

    base_closed = r.get('traj_closed_acc', float('nan'))
    base_auroc  = r.get('traj_auroc',      float('nan'))
    base_unk    = r.get('traj_unk_det',    float('nan'))

    metrics = ['Closed-set acc', 'AUROC', 'Unknown det rate']
    base    = [base_closed, base_auroc, base_unk]

    pl_closed = [r.get(f'pl{k}_closed_acc', float('nan')) for k in pl_keys]
    pl_auroc  = [r.get(f'pl{k}_auroc',      float('nan')) for k in pl_keys]
    pl_unk    = [r.get(f'pl{k}_unk_det',    float('nan')) for k in pl_keys]
    n_labels  = [r.get(f'pl{k}_n_pseudolabels', None) for k in pl_keys]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    all_variants = ['Base\ntraj'] + [f'PL\n{c}%' for c in confs]
    x = np.arange(len(all_variants))
    w = 0.55

    for ax, bval, pl_vals, metric in zip(axes, base, [pl_closed, pl_auroc, pl_unk], metrics):
        vals = [bval] + pl_vals
        cols = [_PAL[0]] + [_PAL[i + 1] for i in range(len(confs))]
        bars = ax.bar(x, vals, width=w, color=cols,
                      edgecolor='white', linewidth=0.6)
        # Baseline reference line
        if not np.isnan(bval):
            ax.axhline(bval, color=_PAL[0], lw=1.2, ls='--', alpha=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.020, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(all_variants)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1.18)

    # Annotate pseudo-label counts
    if any(n is not None for n in n_labels):
        count_str = '  |  '.join(
            f'{c}%: {n} labels' for c, n in zip(confs, n_labels) if n is not None
        )
        fig.text(0.5, -0.04, f'Pseudo-label counts: {count_str}',
                 ha='center', fontsize=9, color='gray')

    fig.suptitle(
        f'Pseudo-label trajectory update  —  Day {cfg.test_day} test\n'
        r'Base traj (no labels) vs PL-updated at confidence $\theta$',
        fontsize=9, y=1.03,
    )
    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'figS5_pseudolabel')


# ---------------------------------------------------------------------------
# Figure S6 — Cross-day performance curve
# ---------------------------------------------------------------------------

def plot_crossday_curve(out_dir: str, results_root: str = 'res_out') -> None:
    """
    Performance vs test day for the main trajectory method and softmax baseline.

    Reads results_phase4.txt from subdirectories named test_dayN/ under
    results_root, or from a single results_root if only one test day was run.
    Skips missing days gracefully.

    To generate data for this plot run the pipeline multiple times with
    different --test_day values and --output_root pointing to test_dayN/
    subdirectories, e.g.:
        for d in 2 3 4 5 6 7 8; do
          python run_experiment.py --test_day $d --output_root res_out/test_day${d}
        done
    """
    import re

    days_found = []
    traj_auroc, traj_closed, soft_acc = [], [], []

    def _parse(rpath):
        res = {}
        if not os.path.exists(rpath):
            return res
        with open(rpath) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    try:
                        k, v = line.split('=', 1)
                        res[k.strip()] = float(v.strip())
                    except Exception:
                        pass
        return res

    # Try structured multi-day result dirs first
    for d in range(2, 10):
        candidate_dirs = [
            os.path.join(results_root, f'test_day{d}'),
            os.path.join(results_root, f'day{d}'),
        ]
        for cdir in candidate_dirs:
            rpath = os.path.join(cdir, 'results_phase4.txt')
            r = _parse(rpath)
            if r:
                days_found.append(d)
                traj_auroc.append(r.get('traj_auroc',    float('nan')))
                traj_closed.append(r.get('traj_closed_acc', float('nan')))
                soft_acc.append(r.get('softmax_acc',     float('nan')))
                break

    # Fallback: single result dir — just plot the one test day
    if not days_found:
        rpath = os.path.join(results_root, 'results_phase4.txt')
        r = _parse(rpath)
        if not r:
            print('[paper_plots_extended] Fig S6: no cross-day results found — skipping')
            return
        # Can't plot a curve with one point — skip
        print('[paper_plots_extended] Fig S6: only one test day found — '
              'run with multiple --test_day values to generate the curve')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    ax1.plot(days_found, traj_closed, 'o-', color=_PAL[0], lw=2.0, ms=7,
             label='T2R (ours)')
    ax1.plot(days_found, soft_acc,    's--', color=_PAL[1], lw=1.8, ms=6,
             label='Softmax baseline')
    ax1.set_xlabel('Test day')
    ax1.set_ylabel('Closed-set accuracy')
    ax1.set_title('(a)  Closed-set accuracy vs test day')
    ax1.set_xticks(days_found)
    ax1.set_ylim(0, 1.08)
    ax1.legend(loc='lower left', framealpha=0.88)

    ax2.plot(days_found, traj_auroc, 'o-', color=_PAL[0], lw=2.0, ms=7,
             label='T2R (ours)')
    ax2.set_xlabel('Test day')
    ax2.set_ylabel('AUROC')
    ax2.set_title('(b)  Open-set AUROC vs test day')
    ax2.set_xticks(days_found)
    ax2.set_ylim(0, 1.08)
    ax2.legend(loc='lower left', framealpha=0.88)

    fig.suptitle(
        'Performance degradation as temporal gap grows  '
        '(later test day = more drift from training)',
        fontsize=9, y=1.02,
    )
    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'figS6_crossday_curve')


# ---------------------------------------------------------------------------
# Shared loader: model + Day-N embeddings (used by S7, S8, S9, S10)
# ---------------------------------------------------------------------------

def _load_model_and_embeddings(cfg, output_root: str, day: int,
                                device_ids: list, include_unknown: bool = False):
    """
    Load the BN-adapted Phase 3 model and extract embeddings for `day`.
    Returns (model, emb_known, y_known, emb_unk, y_unk, norm_mean, norm_std).
    emb_unk / y_unk are empty arrays when include_unknown=False.
    """
    import tensorflow as tf
    import rf_models
    import load_slice_IQ
    from temporal_trajectory import extract_embeddings_from_model, TemporalTrajectory

    model_dir = os.path.join(output_root, 'modelDir')

    # Prefer BN-adapted model; fall back to last fine-tuned
    adapted = os.path.join(model_dir, f'phase3_bn_adapted_d{cfg.traj_day}.keras')
    fallback = os.path.join(model_dir, f'phase2_ft_d{cfg.finetune_days[-1]}.keras')
    model_path = adapted if os.path.exists(adapted) else fallback
    model = tf.keras.models.load_model(
        model_path, custom_objects={'L2Normalize': rf_models.L2Normalize}
    )

    nstats   = np.load(os.path.join(model_dir, 'phase3_norm_stats.npz'))
    nm, ns   = nstats['mean'], nstats['std']

    def _emb(ids):
        from data_utils import load_day_raw
        X, y = load_day_raw(cfg, day_id=day, device_ids=ids)
        X    = load_slice_IQ.apply_normalization(X, nm, ns)
        emb  = extract_embeddings_from_model(model, X, batch_size=128)
        return emb, y

    emb_k, y_k = _emb(device_ids)

    if include_unknown and cfg.unknown_ids:
        emb_u, y_u = _emb(cfg.unknown_ids)
    else:
        emb_u = np.empty((0, emb_k.shape[1]), dtype=np.float32)
        y_u   = np.empty((0,), dtype=np.int32)

    traj_path = os.path.join(model_dir, f'phase3_trajectory_d{cfg.traj_day}.npz')
    traj      = TemporalTrajectory.load(traj_path)

    return model, traj, emb_k, y_k, emb_u, y_u, nm, ns


# ---------------------------------------------------------------------------
# Figure S7 — Confusion matrix (Day 8, known devices, trajectory classifier)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cfg, out_dir: str, output_root: str) -> None:
    """
    16×16 confusion matrix of the trajectory classifier on Day 8 known devices.
    Rows = true device, columns = predicted device.
    """
    from sklearn.metrics import confusion_matrix

    print('[paper_plots_extended] Fig S7: loading model for confusion matrix ...')
    try:
        _, traj, emb_k, y_k, _, _, _, _ = _load_model_and_embeddings(
            cfg, output_root, day=cfg.test_day,
            device_ids=cfg.all_known_ids, include_unknown=False,
        )
    except Exception as e:
        print(f'[paper_plots_extended] Fig S7: data load failed — {e}')
        return

    known_ids = sorted(traj.known_device_ids)

    # No rejection — use inf threshold to get all predictions
    pred, _ = traj.classify_cosine_prototypes(emb_k, threshold=np.inf,
                                               known_ids=known_ids)
    if pred is None or len(pred) == 0:
        print('[paper_plots_extended] Fig S7: no predictions — skipping')
        return

    cm = confusion_matrix(y_k, pred, labels=known_ids)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    n = len(known_ids)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues', aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'D{i}' for i in known_ids], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([f'D{i}' for i in known_ids], fontsize=8)
    ax.set_xlabel('Predicted device')
    ax.set_ylabel('True device')
    ax.set_title(
        f'Confusion matrix — trajectory classifier  (Day {cfg.test_day} known devices)\n'
        f'Row-normalised: diagonal = per-device accuracy',
        fontsize=9, pad=8,
    )

    # Annotate cells with value if large enough to matter
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.05:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5,
                        color='white' if v > 0.6 else 'black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fraction of true-class samples', fontsize=9)

    acc = float(np.mean(np.array(pred) == np.array(y_k)))
    ax.text(0.99, -0.12, f'Overall closed-set accuracy: {acc:.4f}',
            transform=ax.transAxes, ha='right', fontsize=9, color='gray')

    fig.tight_layout()
    _save(fig, out_dir, 'figS7_confusion_matrix')


# ---------------------------------------------------------------------------
# Figure S8 — t-SNE: Day 1 (static) vs Day 8 (adapted)
# ---------------------------------------------------------------------------

def plot_tsne_adaptation(cfg, out_dir: str, output_root: str) -> None:
    """
    Two-panel t-SNE.
    Left : Day 8 embeddings from the static Phase 1 model (no adaptation).
    Right: Day 8 embeddings from the trajectory-adapted model (ours).
    Closes the story started in Fig 1.
    """
    import tensorflow as tf
    import rf_models
    import load_slice_IQ
    from temporal_trajectory import extract_embeddings_from_model
    from sklearn.manifold import TSNE

    print('[paper_plots_extended] Fig S8: computing t-SNE embeddings ...')

    model_dir = os.path.join(output_root, 'modelDir')
    norm_path = os.path.join(model_dir, 'phase3_norm_stats.npz')
    if not os.path.exists(norm_path):
        print('[paper_plots_extended] Fig S8: norm stats not found — skipping')
        return

    nstats = np.load(norm_path)
    nm, ns = nstats['mean'], nstats['std']

    def _get_emb(model_path, day, n_per_dev=120):
        if not os.path.exists(model_path):
            return None, None
        mdl = tf.keras.models.load_model(
            model_path, custom_objects={'L2Normalize': rf_models.L2Normalize}
        )
        from data_utils import load_day_raw
        rng = np.random.default_rng(42)
        X, y = load_day_raw(cfg, day_id=day, device_ids=cfg.all_known_ids)
        X    = load_slice_IQ.apply_normalization(X, nm, ns)
        idx  = []
        for dev in np.unique(y):
            di = np.where(y == dev)[0]
            idx.extend(rng.choice(di, min(n_per_dev, len(di)), replace=False).tolist())
        idx = np.array(idx)
        emb = extract_embeddings_from_model(mdl, X[idx], batch_size=128)
        return emb, y[idx]

    p1_path  = os.path.join(model_dir, f'phase1_df_d{cfg.init_day}.keras')
    ada_path = os.path.join(model_dir, f'phase3_bn_adapted_d{cfg.traj_day}.keras')
    if not os.path.exists(ada_path):
        ada_path = os.path.join(model_dir, f'phase2_ft_d{cfg.finetune_days[-1]}.keras')

    emb_static, y_static = _get_emb(p1_path,  cfg.test_day)
    emb_adapt,  y_adapt  = _get_emb(ada_path, cfg.test_day)

    if emb_static is None or emb_adapt is None:
        print('[paper_plots_extended] Fig S8: model files not found — skipping')
        return

    # Fit t-SNE on combined embeddings for a consistent 2D space
    combined = np.concatenate([emb_static, emb_adapt], axis=0)
    tsne     = TSNE(n_components=2, random_state=42, perplexity=40,
                    max_iter=1000, verbose=0)
    proj     = tsne.fit_transform(combined)
    n_s      = len(emb_static)
    proj_s   = proj[:n_s]
    proj_a   = proj[n_s:]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax, proj_2d, y_arr, title in [
        (axes[0], proj_s, y_static,
         f'(a) Day {cfg.test_day} — Phase 1 model (no adaptation)\nclusters drifted, overlapping'),
        (axes[1], proj_a, y_adapt,
         f'(b) Day {cfg.test_day} — Trajectory-adapted model (ours)\nclusters re-separated'),
    ]:
        for i, dev in enumerate(sorted(cfg.all_known_ids)):
            mask = y_arr == dev
            if not mask.any():
                continue
            ax.scatter(proj_2d[mask, 0], proj_2d[mask, 1],
                       c=_PAL[i % len(_PAL)], s=12, alpha=0.55,
                       linewidths=0, label=f'Dev {dev}' if i < 8 else None)
        ax.set_title(title, fontsize=8, pad=6)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_aspect('equal', 'datalim')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='lower center',
               bbox_to_anchor=(0.5, -0.10), ncol=8,
               framealpha=0.85, handlelength=1.0)

    fig.suptitle(
        f'Embedding space on Day {cfg.test_day}: static model vs trajectory-adapted model',
        fontsize=9, y=1.02,
    )
    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'figS8_tsne_adaptation')


# ---------------------------------------------------------------------------
# Figure S9 — Threshold sensitivity curve
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(cfg, out_dir: str, output_root: str) -> None:
    """
    AUROC, unknown-detection rate, and known-accept rate vs rejection threshold.
    Shows the method is not brittle — performance is stable over a range.
    """
    from sklearn.metrics import roc_auc_score
    from temporal_trajectory import TemporalTrajectory

    print('[paper_plots_extended] Fig S9: computing threshold sensitivity ...')
    try:
        _, traj, emb_k, y_k, emb_u, _, _, _ = _load_model_and_embeddings(
            cfg, output_root, day=cfg.test_day,
            device_ids=cfg.all_known_ids, include_unknown=True,
        )
    except Exception as e:
        print(f'[paper_plots_extended] Fig S9: data load failed — {e}')
        return

    if len(emb_u) == 0:
        print('[paper_plots_extended] Fig S9: no unknown embeddings — skipping')
        return

    known_ids = sorted(traj.known_device_ids)

    # Get cosine distances — use inf threshold so nothing is rejected
    _, dist_k = traj.classify_cosine_prototypes(emb_k, threshold=np.inf,
                                                 known_ids=known_ids)
    _, dist_u = traj.classify_cosine_prototypes(emb_u, threshold=np.inf,
                                                 known_ids=known_ids)
    if dist_k is None or dist_u is None:
        print('[paper_plots_extended] Fig S9: prototype extraction failed — skipping')
        return

    # Read swept threshold from results_phase3.txt to mark it
    swept_thr = None
    p3_path = os.path.join(output_root, 'results_phase3.txt')
    if os.path.exists(p3_path):
        for line in open(p3_path):
            if 'using threshold:' in line:
                try:
                    swept_thr = float(line.split('using threshold:')[1].strip())
                except Exception:
                    pass

    thresholds = np.linspace(
        min(dist_k.min(), dist_u.min()),
        max(dist_k.max(), dist_u.max()),
        200,
    )
    aurocs, unk_dets, accept_rates = [], [], []
    binary = np.concatenate([np.zeros(len(dist_k)), np.ones(len(dist_u))])
    scores = np.concatenate([dist_k, dist_u])

    try:
        auroc_global = float(roc_auc_score(binary, scores))
    except Exception:
        auroc_global = float('nan')

    for thr in thresholds:
        accept = dist_k < thr
        reject = dist_u >= thr
        accept_rates.append(float(accept.mean()))
        unk_dets.append(float(reject.mean()))
        aurocs.append(auroc_global)   # AUROC is threshold-independent

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(thresholds, accept_rates, '-',  color=_PAL[0], lw=2.0, label='Known accept rate')
    ax.plot(thresholds, unk_dets,     '--', color=_PAL[1], lw=2.0, label='Unknown det rate')
    ax.axhline(auroc_global, color=_PAL[2], lw=1.5, ls=':', label=f'AUROC = {auroc_global:.3f}')

    if swept_thr is not None:
        ax.axvline(swept_thr, color='gray', lw=1.2, ls='--', alpha=0.7,
                   label=f'Swept threshold ({swept_thr:.3f})')

    ax.set_xlabel('Rejection threshold (cosine distance)')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.08)
    ax.set_title('Threshold sensitivity — trajectory classifier\n'
                 'Performance is stable over a broad threshold range',
                 fontsize=9, pad=8)
    ax.legend(loc='center right', framealpha=0.88)

    fig.tight_layout()
    _save(fig, out_dir, 'figS9_threshold_sensitivity')


# ---------------------------------------------------------------------------
# Figure S10 — Per-device unknown detection breakdown
# ---------------------------------------------------------------------------

def plot_per_device_breakdown(cfg, out_dir: str, output_root: str) -> None:
    """
    Per-unknown-device detection rate and per-known-device accuracy.
    Shows which devices are hardest to fingerprint / detect.
    """
    from sklearn.metrics import roc_auc_score

    print('[paper_plots_extended] Fig S10: computing per-device breakdown ...')
    try:
        _, traj, emb_k, y_k, emb_u, y_u, _, _ = _load_model_and_embeddings(
            cfg, output_root, day=cfg.test_day,
            device_ids=cfg.all_known_ids, include_unknown=True,
        )
    except Exception as e:
        print(f'[paper_plots_extended] Fig S10: data load failed — {e}')
        return

    known_ids = sorted(traj.known_device_ids)

    pred_k, dist_k = traj.classify_cosine_prototypes(emb_k, threshold=np.inf,
                                                      known_ids=known_ids)
    _,      dist_u = traj.classify_cosine_prototypes(emb_u, threshold=np.inf,
                                                      known_ids=known_ids)
    if dist_k is None or len(dist_k) == 0:
        print('[paper_plots_extended] Fig S10: prototype extraction failed — skipping')
        return

    # Read swept threshold
    swept_thr = 0.5
    p3_path = os.path.join(output_root, 'results_phase3.txt')
    if os.path.exists(p3_path):
        for line in open(p3_path):
            if 'using threshold:' in line:
                try:
                    swept_thr = float(line.split('using threshold:')[1].strip())
                except Exception:
                    pass

    # Per-known-device accuracy
    per_known_acc = []
    for dev in known_ids:
        mask = y_k == dev
        if not mask.any():
            per_known_acc.append(float('nan'))
        else:
            per_known_acc.append(float(np.mean(pred_k[mask] == dev)))

    # Per-unknown-device detection rate
    per_unk_det = []
    per_unk_auroc = []
    for dev in cfg.unknown_ids:
        mask = y_u == dev
        if not mask.any():
            per_unk_det.append(float('nan'))
            per_unk_auroc.append(float('nan'))
            continue
        det = float(np.mean(dist_u[mask] >= swept_thr))
        per_unk_det.append(det)
        try:
            binary = np.concatenate([np.zeros(len(dist_k)), np.ones(mask.sum())])
            scores = np.concatenate([dist_k, dist_u[mask]])
            per_unk_auroc.append(float(roc_auc_score(binary, scores)))
        except Exception:
            per_unk_auroc.append(float('nan'))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # Left: per-known-device accuracy
    x_k = np.arange(len(known_ids))
    col_k = [_PAL[i % len(_PAL)] for i in range(len(known_ids))]
    axes[0].bar(x_k, per_known_acc, color=col_k, edgecolor='white', linewidth=0.6)
    mean_k = np.nanmean(per_known_acc)
    axes[0].axhline(mean_k, color='gray', lw=1.2, ls='--',
                    label=f'Mean = {mean_k:.3f}')
    axes[0].set_xticks(x_k)
    axes[0].set_xticklabels([f'D{i}' for i in known_ids],
                             rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('(a)  Per-device accuracy — known devices')
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(fontsize=9)
    for xi, v in zip(x_k, per_known_acc):
        if not np.isnan(v):
            axes[0].text(xi, v + 0.02, f'{v:.2f}',
                         ha='center', va='bottom', fontsize=8)

    # Right: per-unknown-device detection rate + AUROC
    x_u  = np.arange(len(cfg.unknown_ids))
    w    = 0.35
    col0 = _PAL[0]
    col1 = _PAL[1]
    bars1 = axes[1].bar(x_u - w/2, per_unk_det,   width=w,
                         color=col0, label='Unknown det rate', edgecolor='white')
    bars2 = axes[1].bar(x_u + w/2, per_unk_auroc, width=w,
                         color=col1, label='AUROC', edgecolor='white')
    for bars, vals in [(bars1, per_unk_det), (bars2, per_unk_auroc)]:
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                axes[1].text(bar.get_x() + bar.get_width() / 2,
                             v + 0.02, f'{v:.2f}',
                             ha='center', va='bottom', fontsize=8.5)
    axes[1].set_xticks(x_u)
    axes[1].set_xticklabels([f'Unk {d}' for d in cfg.unknown_ids])
    axes[1].set_ylabel('Rate / AUROC')
    axes[1].set_title('(b)  Per-device detection — unknown devices')
    axes[1].set_ylim(0, 1.18)
    axes[1].legend(fontsize=9)

    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'figS10_per_device_breakdown')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_extended_plots(
    cfg,
    phase4_result:     dict,
    multiseed_summary: dict | None = None,
    roc_scores:        dict | None = None,
) -> None:
    """Generate all supplementary figures into {cfg.output_root}/paper_plots_extended/."""
    out_dir = os.path.join(cfg.output_root, 'paper_plots_extended')
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n[paper_plots_extended] Generating figures → {out_dir}')

    for fn, label in [
        (lambda: plot_full_comparison(cfg, phase4_result, out_dir, multiseed_summary),   'S1'),
        (lambda: plot_ablation(cfg, phase4_result, out_dir, multiseed_summary),          'S2'),
        (lambda: plot_roc_curves(cfg, phase4_result, out_dir, roc_scores),               'S3'),
        (lambda: plot_accept_unk_tradeoff(cfg, phase4_result, out_dir),                  'S4'),
        (lambda: plot_pseudolabel(cfg, phase4_result, out_dir),                          'S5'),
        (lambda: plot_crossday_curve(out_dir, cfg.output_root),                          'S6'),
        (lambda: plot_confusion_matrix(cfg, out_dir, cfg.output_root),                   'S7'),
        (lambda: plot_tsne_adaptation(cfg, out_dir, cfg.output_root),                    'S8'),
        (lambda: plot_threshold_sensitivity(cfg, out_dir, cfg.output_root),              'S9'),
        (lambda: plot_per_device_breakdown(cfg, out_dir, cfg.output_root),               'S10'),
    ]:
        try:
            fn()
        except Exception as e:
            print(f'[paper_plots_extended] Fig {label} failed: {e}')

    print(f'[paper_plots_extended] Done → {out_dir}')


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Generate extended supplementary figures from saved results.'
    )
    parser.add_argument('--output_root',    default='res_out')
    parser.add_argument('--data_root',      default=os.environ.get('T2R_DATA_ROOT', './data/neu'))
    parser.add_argument('--multiseed_dir',  default='res_out_multiseed',
                        help='Directory containing aggregate_results.json')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from experiment_config import ExperimentConfig

    cfg = ExperimentConfig(
        data_root   = args.data_root,
        output_root = args.output_root,
    )

    # Load phase4 results
    def _load_p4(results_dir: str) -> dict:
        res   = {}
        rpath = os.path.join(results_dir, 'results_phase4.txt')
        if not os.path.exists(rpath):
            print(f'[paper_plots_extended] WARNING: {rpath} not found')
            return res
        with open(rpath) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    try:
                        k, v = line.split('=', 1)
                        res[k.strip()] = float(v.strip())
                    except Exception:
                        pass
        # Also try JSON sidecar if present
        jpath = os.path.join(results_dir, 'results_phase4.json')
        if os.path.exists(jpath):
            with open(jpath) as f:
                jdata = json.load(f)
            res.update(jdata)
        return res

    phase4_result = _load_p4(cfg.results_dir)

    # Load multiseed summary
    multiseed_summary = None
    ms_path = os.path.join(args.multiseed_dir, 'aggregate_results.json')
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            multiseed_summary = json.load(f)
        print(f'[paper_plots_extended] Loaded multiseed summary from {ms_path}')

    generate_all_extended_plots(cfg, phase4_result, multiseed_summary)
