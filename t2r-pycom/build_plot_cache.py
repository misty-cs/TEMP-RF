#!/usr/bin/env python3
"""
build_plot_cache.py — One-time recomputation of everything the paper figures
need, cached to res_out_multiseed/seed_100/plot_cache/.

Replicates the exact Phase-4 decision path (BN-adapted model + phase3 norm
stats + classify_cosine_prototypes with the calibrated hyperparameters), so
cached scores reproduce results_phase4.txt. Also extracts the embeddings used
by the drift-motivation / trajectory / t-SNE figures.

Run with the project venv:
    python build_plot_cache.py
"""

import os
import sys
import numpy as np

# Overridable: seed_100 was hard-coded, which breaks if that seed is
# absent or the seed set changes.
SEED_DIR = os.environ.get('T2R_SEED_DIR', 'res_out_multiseed/seed_100')
CACHE    = os.path.join(SEED_DIR, 'plot_cache')

# Calibrated T2R decision parameters from seed_100 results_phase4.txt
T2R_PARAMS = dict(days_keep=5, day_decay=1.0, align_drift=True)
T2R_THR    = 0.097

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import tensorflow as tf
    import rf_models
    import load_slice_IQ
    from data_utils import load_day, split_known_unknown
    from experiment_config import ExperimentConfig
    from temporal_trajectory import (
        TemporalTrajectory, extract_embeddings_from_model,
    )

    os.makedirs(CACHE, exist_ok=True)
    cfg = ExperimentConfig(
        data_root     = os.environ.get('T2R_DATA_ROOT', './data/neu'),
        output_root   = SEED_DIR,
        finetune_days = [2, 3, 4, 5, 6, 7],
        traj_day      = 7,
        test_day      = 8,
        seed          = 100,
    )
    model_dir = os.path.join(SEED_DIR, 'modelDir')

    def _load(path):
        return tf.keras.models.load_model(
            path, custom_objects={'L2Normalize': rf_models.L2Normalize})

    # ────────────────────────────────────────────────────────────────────
    # A. Day-8 embeddings + T2R scores (Phase-4 replica)
    # ────────────────────────────────────────────────────────────────────
    print('[cache] A: Day-8 scores via BN-adapted model ...')
    model  = _load(os.path.join(model_dir, 'phase3_bn_adapted_d7.keras'))
    nstats = np.load(os.path.join(model_dir, 'phase3_norm_stats.npz'))
    nm, ns = nstats['mean'], nstats['std']

    all_devices = cfg.all_known_ids + cfg.unknown_ids
    X, y, *_ = load_day(cfg, day_id=cfg.test_day, device_ids=all_devices,
                        split=False, normalize=False, augment=False)
    X = load_slice_IQ.apply_normalization(X, nm, ns)
    id_map_rev = {i: dev for i, dev in enumerate(sorted(all_devices))}
    y_orig = np.array([id_map_rev[int(v)] for v in y], dtype=np.int32)
    X_k, y_k_orig, X_u, y_u_orig = split_known_unknown(
        X, y_orig, known_ids=cfg.all_known_ids, unknown_ids=cfg.unknown_ids)
    known_map = {dev: i for i, dev in enumerate(sorted(cfg.all_known_ids))}
    y_k = np.array([known_map[int(v)] for v in y_k_orig], dtype=np.int32)

    emb_k = extract_embeddings_from_model(model, X_k, 128)
    emb_u = extract_embeddings_from_model(model, X_u, 128)

    traj = TemporalTrajectory.load(
        os.path.join(model_dir, f'phase3_trajectory_d{cfg.traj_day}.npz'))
    known_ids_0 = list(range(len(cfg.all_known_ids)))

    pred_k, dist_k = traj.classify_cosine_prototypes(
        emb_k, threshold=np.inf, known_ids=known_ids_0, **T2R_PARAMS)
    pred_u, dist_u = traj.classify_cosine_prototypes(
        emb_u, threshold=np.inf, known_ids=known_ids_0, **T2R_PARAMS)

    closed = float(np.mean(pred_k == y_k))
    accept = dist_k <= T2R_THR
    unk_det = float(np.mean(dist_u > T2R_THR))
    from sklearn.metrics import roc_auc_score
    binary = np.concatenate([np.zeros(len(dist_k)), np.ones(len(dist_u))])
    auroc  = float(roc_auc_score(binary, np.concatenate([dist_k, dist_u])))
    print(f'[cache]   sanity: closed={closed:.4f} (want 0.6786)  '
          f'auroc={auroc:.4f} (want 0.7290)  unk_det={unk_det:.4f} (want 0.7700)')

    np.savez_compressed(
        os.path.join(CACHE, 'day8_scores.npz'),
        y_known=y_k, y_known_orig=y_k_orig, y_unk_orig=y_u_orig,
        pred_known=pred_k, dist_known=dist_k,
        pred_unk=pred_u, dist_unk=dist_u,
        threshold=T2R_THR, closed=closed, auroc=auroc, unk_det=unk_det,
    )
    np.savez_compressed(
        os.path.join(CACHE, 'day8_embeddings.npz'),
        emb_known=emb_k.astype(np.float16), y_known=y_k,
        emb_unk=emb_u.astype(np.float16), y_unk_orig=y_u_orig,
    )

    # Single-centroid trajectory scores (ablation ROC): use days_keep=1 with
    # the EWMA centroid bank reduced to its mean — approximate via
    # classify_open_set_history? Keep it simple: score against per-day EWMA
    # centroid (cosine distance to latest EWMA mean).
    print('[cache] A2: single-centroid cosine scores ...')
    means = np.stack([traj.latest_mean(d) for d in known_ids_0])   # (16, m)
    means = means / np.linalg.norm(means, axis=1, keepdims=True)

    def _cent_scores(emb):
        sims = emb @ means.T
        best = np.argmax(sims, axis=1)
        return best.astype(np.int32), (1.0 - sims[np.arange(len(emb)), best])

    cpred_k, cdist_k = _cent_scores(emb_k)
    cpred_u, cdist_u = _cent_scores(emb_u)
    np.savez_compressed(
        os.path.join(CACHE, 'day8_centroid_scores.npz'),
        pred_known=cpred_k, dist_known=cdist_k,
        pred_unk=cpred_u, dist_unk=cdist_u)

    del X, X_k, X_u
    # ────────────────────────────────────────────────────────────────────
    # B. Phase-1 model embeddings on Day 1 / Day 7 (fig1) and Day 8 (figS8a)
    # ────────────────────────────────────────────────────────────────────
    print('[cache] B: Phase-1 model embeddings for drift figures ...')
    p1_model = _load(os.path.join(model_dir, 'phase1_df_d1.keras'))
    p1_norm  = np.load(os.path.join(model_dir, 'phase1_norm_d1.npz'))
    rng = np.random.default_rng(42)

    def _embed_day(model_, nm_, ns_, day, ids, n_per_dev=200):
        Xd, yd, *_ = load_day(cfg, day_id=day, device_ids=ids,
                              split=False, normalize=False, augment=False)
        Xd = load_slice_IQ.apply_normalization(Xd, nm_, ns_)
        keep = []
        for dev in np.unique(yd):
            idx = np.flatnonzero(yd == dev)
            keep.append(rng.choice(idx, min(n_per_dev, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        emb = extract_embeddings_from_model(model_, Xd[keep], 128)
        return emb.astype(np.float32), yd[keep]

    e1, y1 = _embed_day(p1_model, p1_norm['mean'], p1_norm['std'],
                        cfg.init_day, cfg.all_known_ids)
    e7, y7 = _embed_day(p1_model, p1_norm['mean'], p1_norm['std'],
                        cfg.traj_day, cfg.all_known_ids)
    e8s, y8s = _embed_day(p1_model, p1_norm['mean'], p1_norm['std'],
                          cfg.test_day, cfg.all_known_ids, n_per_dev=120)
    e8a, y8a = _embed_day(model, nm, ns,
                          cfg.test_day, cfg.all_known_ids, n_per_dev=120)
    np.savez_compressed(
        os.path.join(CACHE, 'drift_embeddings.npz'),
        emb_d1=e1, y_d1=y1, emb_d7=e7, y_d7=y7,
        emb_d8_static=e8s, y_d8_static=y8s,
        emb_d8_adapted=e8a, y_d8_adapted=y8a,
    )

    # ────────────────────────────────────────────────────────────────────
    # C. Trajectory history (EWMA means + per-day spread) for fig2
    # ────────────────────────────────────────────────────────────────────
    print('[cache] C: trajectory history ...')
    devs = sorted(traj._history.keys())
    means_list, covs, meta = [], [], []
    for dev in devs:
        for e in traj._history[dev]:
            means_list.append(e['mean'].astype(np.float32))
            covs.append(e['cov'].astype(np.float32))
            meta.append((dev, int(e.get('day', -1))))
    np.savez_compressed(
        os.path.join(CACHE, 'trajectory_history.npz'),
        means=np.stack(means_list),
        covs=np.stack(covs),
        meta=np.array(meta, dtype=np.int32),
    )
    print('[cache] done.')


if __name__ == '__main__':
    main()
