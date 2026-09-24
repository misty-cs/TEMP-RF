#!/usr/bin/env python3
"""
Fixed-target enrollment freshness experiment.

Keeps the test day fixed at Day 8 and varies the single enrollment day:
Day 1 -> Day 8, Day 4 -> Day 8, and Day 7 -> Day 8.

This directly tests whether using fresher enrollment data improves transfer to
the same target day. It is intentionally separate from the cumulative cross-day
experiment, which enrolls through all previous days.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def _model_for_day(model_dir: Path, day: int) -> tuple[Path, Path]:
    if day == 1:
        return model_dir / 'phase1_df_d1.keras', model_dir / 'phase1_norm_d1.npz'
    return model_dir / f'phase2_ft_d{day}.keras', model_dir / f'phase2_norm_d{day}.npz'


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _stratified_support_calib(y: np.ndarray, seed: int, calib_frac: float = 0.25):
    rng = np.random.default_rng(seed)
    support_idx, calib_idx = [], []
    for c in sorted(np.unique(y)):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_cal = max(1, int(round(len(idx) * calib_frac)))
        calib_idx.extend(idx[:n_cal])
        support_idx.extend(idx[n_cal:])
    return np.array(support_idx), np.array(calib_idx)


def _fit_multiprotos(emb: np.ndarray, y: np.ndarray, n_proto: int, seed: int):
    from sklearn.cluster import KMeans

    protos, labels = [], []
    for c in sorted(np.unique(y)):
        ec = emb[y == c]
        k = min(n_proto, len(ec))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        centers = km.fit(ec).cluster_centers_
        centers = _unit(centers)
        protos.append(centers)
        labels.extend([int(c)] * k)
    return np.vstack(protos), np.array(labels, dtype=np.int32)


def _score(emb: np.ndarray, protos: np.ndarray, proto_labels: np.ndarray):
    sims = _unit(emb) @ protos.T
    best = np.argmax(sims, axis=1)
    return proto_labels[best], 1.0 - sims[np.arange(len(emb)), best]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--seed_dir', default='res_out_multiseed/seed_42')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_root',
               default=os.environ.get('T2R_DATA_ROOT', './data/neu'))
    p.add_argument('--enroll_days', default='1,4,7')
    p.add_argument('--test_day', type=int, default=8)
    p.add_argument('--n_proto', type=int, default=5)
    p.add_argument('--target_accept', type=float, default=0.50)
    p.add_argument('--out_dir', default='freshness_day8')
    args = p.parse_args()

    import tensorflow as tf
    import rf_models
    import load_slice_IQ
    from data_utils import load_day, split_known_unknown
    from experiment_config import ExperimentConfig
    from temporal_trajectory import extract_embeddings_from_model
    from sklearn.metrics import roc_auc_score

    seed_dir = Path(args.seed_dir)
    model_dir = seed_dir / 'modelDir'
    out_dir = seed_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        data_root=args.data_root,
        output_root=str(seed_dir),
        finetune_days=[2, 3, 4, 5, 6, 7],
        traj_day=7,
        test_day=args.test_day,
        seed=args.seed,
    )

    all_devices = cfg.all_known_ids + cfg.unknown_ids
    rows = []

    for enroll_day in [int(x) for x in args.enroll_days.split(',') if x.strip()]:
        model_path, norm_path = _model_for_day(model_dir, enroll_day)
        if not model_path.exists() or not norm_path.exists():
            raise FileNotFoundError(f'Missing model/norm for Day {enroll_day}: '
                                    f'{model_path}, {norm_path}')

        print(f'[freshness] Day {enroll_day} enrollment -> Day {args.test_day} test')
        model = tf.keras.models.load_model(
            model_path, custom_objects={'L2Normalize': rf_models.L2Normalize})
        norm = np.load(norm_path)
        nm, ns = norm['mean'], norm['std']

        X_e, y_e, *_ = load_day(
            cfg, day_id=enroll_day, device_ids=cfg.all_known_ids,
            split=False, normalize=False, augment=False)
        X_e = load_slice_IQ.apply_normalization(X_e, nm, ns)
        emb_e = extract_embeddings_from_model(model, X_e, batch_size=128)

        support_idx, calib_idx = _stratified_support_calib(
            y_e, seed=args.seed + enroll_day)
        protos, proto_labels = _fit_multiprotos(
            emb_e[support_idx], y_e[support_idx], args.n_proto,
            seed=args.seed + enroll_day)

        _, dist_cal = _score(emb_e[calib_idx], protos, proto_labels)
        threshold = float(np.quantile(dist_cal, args.target_accept))

        X_t, y_t, *_ = load_day(
            cfg, day_id=args.test_day, device_ids=all_devices,
            split=False, normalize=False, augment=False)
        X_t = load_slice_IQ.apply_normalization(X_t, nm, ns)
        id_map_rev = {i: dev for i, dev in enumerate(sorted(all_devices))}
        y_orig = np.array([id_map_rev[int(v)] for v in y_t], dtype=np.int32)
        X_k, y_k_orig, X_u, y_u_orig = split_known_unknown(
            X_t, y_orig, known_ids=cfg.all_known_ids, unknown_ids=cfg.unknown_ids)
        known_map = {dev: i for i, dev in enumerate(sorted(cfg.all_known_ids))}
        y_k = np.array([known_map[int(v)] for v in y_k_orig], dtype=np.int32)

        emb_k = extract_embeddings_from_model(model, X_k, batch_size=128)
        emb_u = extract_embeddings_from_model(model, X_u, batch_size=128)

        pred_k, dist_k = _score(emb_k, protos, proto_labels)
        _, dist_u = _score(emb_u, protos, proto_labels)

        closed = float(np.mean(pred_k == y_k))
        unk_det = float(np.mean(dist_u > threshold))
        known_accept = float(np.mean(dist_k <= threshold))
        binary = np.r_[np.zeros(len(dist_k)), np.ones(len(dist_u))]
        auroc = float(roc_auc_score(binary, np.r_[dist_k, dist_u]))

        row = dict(
            enroll_day=enroll_day,
            test_day=args.test_day,
            closed_acc=closed,
            auroc=auroc,
            unknown_detection=unk_det,
            known_accept=known_accept,
            threshold=threshold,
            n_proto=args.n_proto,
            target_accept=args.target_accept,
        )
        rows.append(row)
        print('  closed={closed_acc:.4f}  auroc={auroc:.4f}  '
              'unk_det={unknown_detection:.4f}  accept={known_accept:.4f}'
              .format(**row))

    csv_path = out_dir / 'day8_enrollment_freshness.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    days = [r['enroll_day'] for r in rows]
    x = np.arange(len(days))
    metrics = [
        ('closed_acc', 'Closed-set accuracy', '#0072B2'),
        ('auroc', 'Open-set AUROC', '#009E73'),
        ('unknown_detection', 'Unknown detection', '#E69F00'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for ax, (key, title, color) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        ax.bar(x, vals, color=color, edgecolor='white', width=0.62)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.025, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Day {d}' for d in days])
        ax.set_xlabel('Enrollment day')
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.22, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig_path = out_dir / 'fig_enrollment_freshness_day8.pdf'
    _pfx = os.environ.get('T2R_FIG_PREFIX', '').strip()
    if _pfx:
        fig_path = fig_path.with_name(f'{_pfx}_{fig_path.name}')
    if os.environ.get('T2R_FIG_TITLES', '0') != '1':
        if getattr(fig, '_suptitle', None) is not None:
            fig.suptitle('')
        for _ax in fig.get_axes():
            if _ax.get_title():
                _ax.set_title('')
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(out_dir / f'{_pfx + "_" if _pfx else ""}fig_enrollment_freshness_day8.png',
                dpi=220, bbox_inches='tight', pad_inches=0.04)
    print(f'[freshness] saved {csv_path}')
    print(f'[freshness] saved {fig_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
