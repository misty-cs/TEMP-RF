#!/usr/bin/env python3
"""
paper_plots.py — Publication figures for the temporal RF fingerprinting paper.

Figures
-------
1. fig1_drift_motivation   — Phase 1 model on init_day (tight clusters) vs
                             traj_day (drifted). Motivates temporal adaptation.
2. fig2_trajectory         — EWMA centroid paths across all trajectory days with
                             confidence ellipses at first and last day.
                             Core contribution visualisation.
3. fig3_finetune_progress  — Closed-set accuracy per fine-tuning step (Phase 2).
4. fig4_openset_comparison — Method comparison: closed accuracy and AUROC.
5. fig5_fewshot_curve      — k-shot enrollment vs open accuracy and AUROC,
                             with trajectory (no-label) baseline.

All figures → {cfg.output_root}/paper_plots/  (PDF + PNG)

Called automatically from run_experiment.py after Phase 4, or standalone:
    python paper_plots.py --output_root res_out --data_root /path/to/neu2
"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# Global style
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

# 20-colour palette (perceptually distinct, readable in greyscale)
_PAL = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
    '#AA3377', '#BBBBBB', '#332288', '#997700', '#44AA99',
    '#EE99AA', '#77AADD', '#88CCAA', '#DDCC77', '#CC6677',
    '#999933', '#882255', '#661100', '#117733', '#0077BB',
]


def _out(out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    _strip_titles(fig)
    fig.savefig(_out(out_dir, f'{_stem(stem)}.pdf'))
    fig.savefig(_out(out_dir, f'{_stem(stem)}.png'), dpi=300)
    plt.close(fig)
    print(f'[paper_plots] saved: {stem}')


def _project_pca(X: np.ndarray, n: int = 2):
    """PCA projection. Returns (projected_X, pca_object)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n, random_state=42)
    return pca.fit_transform(X), pca


def _cov_ellipse(ax, mean_2d, cov_2d, n_std=2.0, **kwargs):
    """Draw a confidence ellipse from a 2×2 covariance matrix."""
    vals, vecs = np.linalg.eigh(cov_2d)
    vals = np.maximum(vals, 1e-10)
    angle = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
    w, h  = 2.0 * n_std * np.sqrt(vals)
    ell   = Ellipse(xy=mean_2d, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


# ---------------------------------------------------------------------------
# Figure 1 — Embedding drift motivation
# ---------------------------------------------------------------------------

def plot_drift_motivation(cfg, phase1_result: dict, out_dir: str) -> None:
    """
    Left panel  : Phase 1 model on init_day  → tight, well-separated clusters.
    Right panel : Phase 1 model on traj_day  → drifted clusters (no adaptation).
    Shows the problem that motivates temporal trajectory tracking.
    """
    import tensorflow as tf
    import rf_models
    from data_utils import load_day
    import load_slice_IQ
    from temporal_trajectory import extract_embeddings_from_model

    print('[paper_plots] Fig 1: loading Phase 1 model for drift motivation ...')

    model = tf.keras.models.load_model(
        phase1_result['model_path'],
        custom_objects={'L2Normalize': rf_models.L2Normalize},
    )
    nstats    = np.load(phase1_result['norm_path'])
    norm_mean = nstats['mean']
    norm_std  = nstats['std']
    rng       = np.random.default_rng(42)

    def _get_embs(day_id: int, n_per_dev: int = 150):
        X, y, _, _, _, _, _ = load_day(
            cfg, day_id=day_id, device_ids=cfg.all_known_ids,
            split=False, normalize=False, augment=False,
        )
        X = load_slice_IQ.apply_normalization(X, norm_mean, norm_std)
        # Subsample for speed
        idx  = rng.choice(len(X), min(n_per_dev * len(cfg.all_known_ids), len(X)),
                          replace=False)
        idx  = np.sort(idx)
        emb  = extract_embeddings_from_model(model, X[idx], batch_size=128)
        return emb, y[idx]

    emb_init,  y_init  = _get_embs(cfg.init_day)
    emb_traj,  y_traj  = _get_embs(cfg.traj_day)

    # Fit PCA on init_day, apply to both — consistent projection
    all_emb  = np.concatenate([emb_init, emb_traj], axis=0)
    proj_all, pca = _project_pca(all_emb)
    n_init   = len(emb_init)
    proj_init = proj_all[:n_init]
    proj_traj = proj_all[n_init:]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    panels = [
        (axes[0], proj_init, y_init,
         f'(a) Day {cfg.init_day} — training day\nPhase 1 model: clusters tight'),
        (axes[1], proj_traj, y_traj,
         f'(b) Day {cfg.traj_day} — {cfg.traj_day - cfg.init_day} days later\n'
         f'Same model, no adaptation: clusters drifted'),
    ]
    for ax, proj, y_arr, title in panels:
        ax.set_title(title, fontsize=9, pad=6)
        for i, dev in enumerate(sorted(cfg.all_known_ids)):
            mask = y_arr == dev
            if not mask.any():
                continue
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=_PAL[i % len(_PAL)], s=14, alpha=0.55,
                       linewidths=0, label=f'Dev {dev}')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_aspect('equal', 'datalim')

    # Shared legend — show first N devices only to avoid clutter
    handles, labels = axes[0].get_legend_handles_labels()
    max_legend = min(len(handles), 10)
    fig.legend(handles[:max_legend], labels[:max_legend],
               fontsize=9, loc='lower center',
               bbox_to_anchor=(0.5, -0.12), ncol=max_legend,
               framealpha=0.85, handlelength=1.0, handletextpad=0.4)

    fig.suptitle(
        'RF fingerprint drift: static model degrades under temporal distribution shift',
        fontsize=9, y=1.02,
    )
    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'fig1_drift_motivation')


# ---------------------------------------------------------------------------
# Figure 2 — Temporal trajectory (core contribution)
# ---------------------------------------------------------------------------

def plot_temporal_trajectory(cfg, phase3_result: dict, out_dir: str) -> None:
    """
    EWMA centroid path for each known device across all trajectory days.
    Confidence ellipses drawn at first and last day position.
    Uses only stored trajectory statistics — no data re-loading.
    """
    from sklearn.decomposition import PCA

    traj = phase3_result['trajectory']

    # Collect all stored EWMA means in (device, day) order
    devices = sorted(traj._history.keys())
    all_means, dev_slices = [], {}

    for dev in devices:
        history = traj._history[dev]
        start   = len(all_means)
        all_means.extend(e['mean'].astype(np.float32) for e in history)
        dev_slices[dev] = (start, len(all_means), history)

    all_means = np.array(all_means, dtype=np.float32)

    # Single PCA fit on all means — consistent embedding space
    pca        = PCA(n_components=2, random_state=42)
    proj_means = pca.fit_transform(all_means)            # (total_points, 2)
    P          = pca.components_                         # (2, D)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    legend_lines = []
    n_dev_shown  = min(len(devices), 14)

    for i, dev in enumerate(devices[:n_dev_shown]):
        s, e, history = dev_slices[dev]
        coords = proj_means[s:e]                         # (num_days, 2)
        days   = [ent['day'] for ent in history]
        col    = _PAL[i % len(_PAL)]

        # Draw trajectory path
        ax.plot(coords[:, 0], coords[:, 1],
                color=col, lw=1.0, alpha=0.7, zorder=2)

        # Day markers
        for j, (pt, d) in enumerate(zip(coords, days)):
            ms = 70 if j == 0 else 35
            fc = col if j == 0 else 'white'
            ax.scatter(pt[0], pt[1], s=ms, color=col,
                       facecolors=fc, edgecolors=col,
                       linewidths=1.2, zorder=4)

        # Direction arrow on last segment
        if len(coords) >= 2:
            ax.annotate(
                '', xy=coords[-1], xytext=coords[-2],
                arrowprops=dict(arrowstyle='->', color=col, lw=1.1, alpha=0.8),
                zorder=5,
            )

        # Confidence ellipses: first and last day
        for j_ell, ent in [(0, history[0]), (-1, history[-1])]:
            cov_full = ent['cov'].astype(np.float64)
            cov_2d   = P @ cov_full @ P.T
            pt_2d    = proj_means[s + (j_ell % len(history))]
            alpha_e  = 0.12 if j_ell == 0 else 0.20
            _cov_ellipse(ax, pt_2d, cov_2d, n_std=1.5,
                         facecolor=col, edgecolor=col,
                         linewidth=0.7, alpha=alpha_e, zorder=1)

        legend_lines.append(
            Line2D([0], [0], color=col, lw=1.5, label=f'Dev {dev}')
        )

    # Reference legend elements
    legend_lines += [
        Line2D([0], [0], marker='o', color='gray', ms=7,
               markerfacecolor='gray', ls='none', label='Day 1 centroid'),
        Line2D([0], [0], marker='o', color='gray', ms=6,
               markerfacecolor='white', markeredgecolor='gray',
               markeredgewidth=1.2, ls='none', label='Later days'),
    ]

    day_min = min(min(e['day'] for e in h) for h in traj._history.values())
    day_max = max(max(e['day'] for e in h) for h in traj._history.values())
    ax.set_xlabel('Embedding PC 1  (projected)')
    ax.set_ylabel('Embedding PC 2  (projected)')
    ax.set_title(
        f'Temporal trajectory of device embeddings  (Days {day_min}–{day_max})\n'
        r'$\bullet$ filled = day 1 $\quad\bullet$ open = later days '
        r'$\quad\rightarrow$ drift direction $\quad$ shaded = 1.5$\sigma$ ellipse',
        fontsize=9, pad=10,
    )
    fig.legend(handles=legend_lines, fontsize=9, loc='upper left',
               bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
               framealpha=0.90, ncol=1, handlelength=1.5)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    _save(fig, out_dir, 'fig2_trajectory')


# ---------------------------------------------------------------------------
# Figure 3 — Sequential fine-tuning progress
# ---------------------------------------------------------------------------

def plot_finetune_progress(cfg, phase2_result: dict, out_dir: str) -> None:
    """
    Per-day accuracy during Phase 2 sequential fine-tuning.
    Lines: softmax accuracy, prototype aggregate accuracy.
    """
    step_results = phase2_result.get('step_results', [])
    if not step_results:
        print('[paper_plots] Fig 3: no step_results — skipping')
        return

    days       = [r['day']      for r in step_results]
    softmax    = [r.get('softmax',    float('nan')) for r in step_results]
    proto_agg1 = [r.get('proto_agg1', float('nan')) for r in step_results]
    bn_adapted = [r.get('bn_adapted', float('nan')) for r in step_results]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(days, softmax,    'o-', color=_PAL[0], lw=2.0, ms=6,
            label='Softmax (fixed BN)')
    ax.plot(days, bn_adapted, 's--', color=_PAL[2], lw=1.8, ms=5,
            label='TTA-BN adapted')
    ax.plot(days, proto_agg1, '^-', color=_PAL[1], lw=2.0, ms=6,
            label='Prototype (agg-1)')

    ax.set_xlabel('Fine-tuning day')
    ax.set_ylabel('Closed-set accuracy')
    ax.set_title('Phase 2: sequential fine-tuning accuracy per day')
    ax.set_xticks(days)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, framealpha=0.90)

    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    _save(fig, out_dir, 'fig3_finetune_progress')


# ---------------------------------------------------------------------------
# Figure 4 — Open-set method comparison
# ---------------------------------------------------------------------------

def plot_openset_comparison(cfg, phase4_result: dict, out_dir: str,
                            multiseed_summary: dict | None = None) -> None:
    """
    Grouped bar chart comparing open-set detection methods.
    Includes OpenMax baseline. Error bars drawn when multiseed_summary provided.
    """
    r = phase4_result

    methods = ['Softmax', 'TTA-BN', f'Static\n(Day {cfg.traj_day})',
               'OpenMax', 'T2R (ours)']
    closed  = [
        r.get('softmax_acc',       float('nan')),
        r.get('softmax_bn_acc',    float('nan')),
        r.get('static_closed_acc', float('nan')),
        r.get('openmax_closed_acc', float('nan')),
        r.get('mp_cos_traj_closed_acc',   float('nan')),
    ]
    auroc   = [
        float('nan'),
        float('nan'),
        r.get('static_auroc',      float('nan')),
        r.get('openmax_auroc',     float('nan')),
        r.get('mp_cos_traj_auroc',        float('nan')),
    ]
    unk_det = [
        float('nan'),
        float('nan'),
        r.get('static_unk_det',    float('nan')),
        r.get('openmax_unk_det',   float('nan')),
        r.get('mp_cos_traj_unk_det',      float('nan')),
    ]

    # Error bars from multiseed std if available
    def _std(key):
        if multiseed_summary and 'std' in multiseed_summary:
            return multiseed_summary['std'].get(key, 0.0)
        return 0.0

    closed_err  = [0, 0, _std('static_closed_acc'), _std('openmax_closed_acc'), _std('traj_closed_acc')]
    auroc_err   = [0, 0, _std('static_auroc'),       _std('openmax_auroc'),      _std('traj_auroc')]
    unk_det_err = [0, 0, _std('static_unk_det'),     _std('openmax_unk_det'),    _std('traj_unk_det')]

    x   = np.arange(len(methods))
    w   = 0.52
    col = [_PAL[i % len(_PAL)] for i in range(len(methods))]
    hatches = ['', '', '', '', '//']

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))

    for ax, vals, errs, ylabel, title in [
        (axes[0], closed,  closed_err,  'Accuracy',       'Closed-set accuracy'),
        (axes[1], auroc,   auroc_err,   'AUROC',          'Open-set AUROC'),
        (axes[2], unk_det, unk_det_err, 'Detection rate', 'Unknown detection rate'),
    ]:
        bars = ax.bar(x, vals, width=w, color=col,
                      edgecolor='white', linewidth=0.6,
                      yerr=[e if not np.isnan(v) else 0 for v, e in zip(vals, errs)],
                      capsize=4, error_kw=dict(lw=1.2, capthick=1.2, ecolor='#333333'))
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for bar, v, hatch in zip(bars, vals, hatches):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.02, f'{v:.2f}',
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold' if hatch == '//' else 'normal')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha='right', rotation_mode='anchor')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.20)
        ax.axhline(1.0, color='gray', lw=0.6, ls=':', alpha=0.4)
        ax.tick_params(axis='x', pad=2)

    fig.tight_layout(w_pad=2.5)
    _save(fig, out_dir, 'fig4_openset_comparison')


# ---------------------------------------------------------------------------
# Figure 5 — Few-shot enrollment curve
# ---------------------------------------------------------------------------

def plot_fewshot_curve(cfg, phase4_result: dict, out_dir: str) -> None:
    """
    Performance vs enrollment samples per device (k-shot).
    Y-left : open_known_acc.  Y-right : AUROC.
    Horizontal dashed lines show trajectory 2-pt (zero-label) baseline.
    """
    fewshot = phase4_result.get('fewshot', {})
    if not fewshot:
        print('[paper_plots] Fig 5: no fewshot results — skipping')
        return

    ks         = sorted(fewshot.keys())
    open_accs  = [fewshot[k].get('open_known_acc', float('nan')) for k in ks]
    aurocs     = [fewshot[k].get('auroc',          float('nan')) for k in ks]
    unk_dets   = [fewshot[k].get('open_unk_det',   float('nan')) for k in ks]

    baseline_auroc    = phase4_result.get('traj_auroc',    None)
    baseline_open_acc = phase4_result.get('traj_open_acc', None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=False)

    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('Enrollment samples per device  ($k$)')
        ax.set_ylim(0, 1.08)

    # Left panel: accuracy + unknown detection
    ax1.plot(ks, open_accs, 'o-', color=_PAL[0], lw=2.0, ms=7, label='Open known acc')
    ax1.plot(ks, unk_dets,  's--', color=_PAL[4], lw=1.8, ms=6, label='Unknown det rate')
    if baseline_open_acc is not None:
        ax1.axhline(baseline_open_acc, color=_PAL[0], lw=1.4, ls=':',
                    alpha=0.75, label=f'Traj baseline (no labels)  {baseline_open_acc:.3f}')
    ax1.set_ylabel('Accuracy / Detection rate')
    ax1.set_title('(a)  Open-set accuracy & unknown detection')
    ax1.legend(loc='lower right', framealpha=0.88)

    # Right panel: AUROC
    ax2.plot(ks, aurocs, '^-', color=_PAL[1], lw=2.0, ms=7, label='AUROC')
    if baseline_auroc is not None:
        ax2.axhline(baseline_auroc, color=_PAL[1], lw=1.4, ls=':',
                    alpha=0.75, label=f'Traj baseline (no labels)  {baseline_auroc:.3f}')
    ax2.set_ylabel('AUROC')
    ax2.set_title('(b)  Open-set AUROC')
    ax2.legend(loc='lower right', framealpha=0.88)

    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir, 'fig5_fewshot_curve')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_paper_plots(
    cfg,
    phase1_result: dict,
    phase2_result: dict,
    phase3_result: dict,
    phase4_result: dict,
    multiseed_summary: dict | None = None,
) -> None:
    """Generate all 5 paper figures into {cfg.output_root}/paper_plots/.

    Pass multiseed_summary (loaded from aggregate_results.json) to get
    error bars on Fig 4.
    """
    import matplotlib.ticker

    out_dir = os.path.join(cfg.output_root, 'paper_plots')
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n[paper_plots] Generating all figures → {out_dir}')

    try:
        plot_drift_motivation(cfg, phase1_result, out_dir)
    except Exception as e:
        print(f'[paper_plots] Fig 1 failed: {e}')

    try:
        plot_temporal_trajectory(cfg, phase3_result, out_dir)
    except Exception as e:
        print(f'[paper_plots] Fig 2 failed: {e}')

    try:
        plot_finetune_progress(cfg, phase2_result, out_dir)
    except Exception as e:
        print(f'[paper_plots] Fig 3 failed: {e}')

    try:
        plot_openset_comparison(cfg, phase4_result, out_dir, multiseed_summary)
    except Exception as e:
        print(f'[paper_plots] Fig 4 failed: {e}')

    try:
        plot_fewshot_curve(cfg, phase4_result, out_dir)
    except Exception as e:
        print(f'[paper_plots] Fig 5 failed: {e}')

    print(f'[paper_plots] All figures saved to {out_dir}')


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Regenerate paper figures from saved experiment results.'
    )
    parser.add_argument('--output_root', default='res_out')
    parser.add_argument('--data_root',   default=os.environ.get('T2R_DATA_ROOT', './data/neu'))
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from experiment_config import ExperimentConfig
    from temporal_trajectory import TemporalTrajectory
    import rf_models

    cfg = ExperimentConfig(
        data_root   = args.data_root,
        output_root = args.output_root,
    )

    import tensorflow as tf

    def _load_p1():
        mp = os.path.join(cfg.model_dir, f'phase1_df_d{cfg.init_day}.keras')
        np_ = os.path.join(cfg.model_dir, f'phase1_norm_d{cfg.init_day}.npz')
        return {'model_path': mp, 'norm_path': np_}

    def _load_p2():
        sr_path = os.path.join(cfg.results_dir, 'results_phase2.txt')
        if not os.path.exists(sr_path):
            return {'step_results': []}
        step_results = []
        with open(sr_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('day='):
                    parts = dict(p.split('=') for p in line.split('  ') if '=' in p)
                    row = {}
                    for k, v in parts.items():
                        if k == 'time' and v.endswith('s'):
                            v = v[:-1]
                        row[k] = float(v)
                    step_results.append(row)
        return {'step_results': step_results}

    def _load_p3():
        last_day  = cfg.finetune_days[-1]
        trajectory_days = sorted(set([cfg.init_day] + cfg.finetune_days + [cfg.traj_day]))
        traj_path = os.path.join(
            cfg.model_dir,
            f'phase3_trajectory_d{trajectory_days[0]}_to_d{trajectory_days[-1]}.npz'
        )
        norm_path = os.path.join(cfg.model_dir, 'phase3_norm_stats.npz')
        traj  = TemporalTrajectory.load(traj_path)
        nstat = np.load(norm_path)
        return {
            'trajectory': traj,
            'model_path': os.path.join(cfg.model_dir, f'phase2_ft_d{last_day}.keras'),
            'norm_mean':  nstat['mean'],
            'norm_std':   nstat['std'],
        }

    def _load_p4():
        res = {}
        rpath = os.path.join(cfg.results_dir, 'results_phase4.txt')
        if not os.path.exists(rpath):
            return res
        with open(rpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('softmax_acc') or '=' in line:
                    try:
                        k, v = line.split('=', 1)
                        res[k.strip()] = float(v.strip())
                    except Exception:
                        pass
        return res

    multiseed_summary = None
    ms_path = os.path.join('res_out_multiseed', 'aggregate_results.json')
    if os.path.exists(ms_path):
        import json
        with open(ms_path) as f:
            multiseed_summary = json.load(f)
        print(f'[paper_plots] Loaded multiseed summary from {ms_path}')

    generate_all_paper_plots(cfg, _load_p1(), _load_p2(), _load_p3(), _load_p4(),
                             multiseed_summary)
