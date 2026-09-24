#!/usr/bin/env python3
"""
phase4_openset_evaluation.py  —  Phase 4: Open-set evaluation on the held-out test day.

Fixes applied
-------------
BUG FIX 1: Static prototype threshold was set as `threshold * 0.3` — an
           unprincipled hack.  The static baseline now sweeps its own
           threshold properly using the same trajectory_threshold_sweep
           function, making it a fair comparison.

BUG FIX 2: The static prototype ablation now uses cfg.traj_day embeddings
           as its support set.

Ablations
---------
1. Softmax baseline         — model's own classifier head, no rejection
2. TTA-BN + softmax         — BN-adapted then softmax, no rejection
3. Static prototype      — single centroid per device, own threshold sweep
4. Calibrated linear probe — Day 7 embedding classifier + rejection
5. L2 calibrated probe     — L2 embeddings + regularisation sweep
6. Class-wise L2 probe     — per-device accept/reject thresholds
7. Temperature L2 probe    — softmax temperature sweep + rejection
8. Cosine KNN              — local embedding classifier + rejection
9. Calibrated LDA probe    — Day 7 shrinkage LDA + rejection
10. L2 calibrated LDA      — L2 embeddings + shrinkage LDA
11. Probe/trajectory fusion — calibrated probe confidence + trajectory distance
12. Cosine trajectory      — trajectory EWMA centroids + cosine rejection
13. Multi-proto cosine traj — recent trajectory prototypes + cosine rejection
14. Trajectory             — using latest (μ, Σ) from cfg.traj_day
15. Few-shot enrollment    — updates trajectory with labeled held-out samples
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from experiment_config import ExperimentConfig
from data_utils import load_day, split_known_unknown
from temporal_trajectory import (
    trajectory_threshold_sweep,
    trajectory_history_threshold_sweep,
    trajectory_cosine_threshold_sweep,
    extract_embeddings_from_model,
)
import load_slice_IQ
import tta_bn
import rf_models


BATCH_SIZE = 128
KNOWN_ONLY_ACCEPT_RATE = 0.95


def _has_calib_unknown(emb_calib_unknown) -> bool:
    return emb_calib_unknown is not None and len(emb_calib_unknown) > 0


def _known_only_distance_threshold(dist: np.ndarray, accept_rate: float = KNOWN_ONLY_ACCEPT_RATE) -> float:
    return float(np.quantile(np.asarray(dist, dtype=np.float32), accept_rate))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path,
        custom_objects={'L2Normalize': rf_models.L2Normalize},
    )


def _softmax_accuracy(model, X, y_int, num_class: int) -> float:
    pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return float(np.mean(np.argmax(pred, axis=1) == np.asarray(y_int, dtype=np.int32)))


def _stratified_support_calib_split(
    labels: np.ndarray,
    seed: int,
    support_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split same-day labeled embeddings into disjoint support/calibration sets."""
    labels = np.asarray(labels, dtype=np.int32)
    rng = np.random.default_rng(seed)
    support_idx, calib_idx = [], []
    for cls in sorted(np.unique(labels)):
        cls_idx = np.where(labels == cls)[0]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        n_support = int(round(len(cls_idx) * support_frac))
        n_support = min(max(1, n_support), len(cls_idx) - 1)
        support_idx.extend(cls_idx[:n_support].tolist())
        calib_idx.extend(cls_idx[n_support:].tolist())
    return np.array(support_idx, dtype=np.int64), np.array(calib_idx, dtype=np.int64)


def _static_prototype_eval(
    emb_support:       np.ndarray,
    y_support:         np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
) -> dict:
    """
    Nearest-centroid (cosine) classifier with its own threshold sweep.
    Returns closed_acc, open_acc, unk_det, auroc, best_threshold.
    """
    from sklearn.metrics import roc_auc_score

    classes   = np.array(sorted(known_ids))
    centroids = np.stack([
        emb_support[y_support == c].mean(axis=0) for c in classes
    ])
    # L2-normalise centroids
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    # L2-normalise queries
    eq = emb_known  / (np.linalg.norm(emb_known,  axis=1, keepdims=True) + 1e-8)
    eu = emb_unknown / (np.linalg.norm(emb_unknown, axis=1, keepdims=True) + 1e-8)

    sim_k  = eq @ centroids.T        # (N_k, K)
    sim_u  = eu @ centroids.T        # (N_u, K)
    dist_k = 1.0 - sim_k.max(axis=1) # cosine distance to nearest centroid
    dist_u = 1.0 - sim_u.max(axis=1)
    pred_k = classes[np.argmax(sim_k, axis=1)]

    # Closed-set accuracy (no rejection)
    closed_acc = float(np.mean(pred_k == y_known))

    # AUROC
    try:
        binary = np.concatenate([
            np.zeros(len(dist_k)), np.ones(len(dist_u))
        ])
        auroc = float(roc_auc_score(binary, np.concatenate([dist_k, dist_u])))
    except Exception:
        auroc = float('nan')

    eck = emb_calib_known / (
        np.linalg.norm(emb_calib_known, axis=1, keepdims=True) + 1e-8
    )
    ecu = emb_calib_unknown / (
        np.linalg.norm(emb_calib_unknown, axis=1, keepdims=True) + 1e-8
    )
    sim_ck = eck @ centroids.T
    sim_cu = ecu @ centroids.T
    dist_ck = 1.0 - sim_ck.max(axis=1)
    dist_cu = 1.0 - sim_cu.max(axis=1)
    pred_ck = classes[np.argmax(sim_ck, axis=1)]

    # Threshold sweep on calibration data only.
    thresholds = np.linspace(
        np.concatenate([dist_ck, dist_cu]).min(),
        np.concatenate([dist_ck, dist_cu]).max(),
        50,
    )
    best_thr, best_score = 0.0, 0.0
    best_open, best_unk  = 0.0, 0.0
    for thr in thresholds:
        accepted = pred_ck[dist_ck <= thr]
        y_acc    = y_calib_known[dist_ck <= thr]
        open_acc = float(np.mean(accepted == y_acc)) if len(accepted) > 0 else 0.0
        unk_det  = float(np.mean(dist_cu > thr))
        score    = 0.5 * (open_acc + unk_det)
        if score > best_score:
            best_score = score
            best_thr   = float(thr)
            best_open  = open_acc
            best_unk   = unk_det

    return {
        'closed_acc':    closed_acc,
        'open_acc':      float(np.mean(pred_k[dist_k <= best_thr] == y_known[dist_k <= best_thr]))
                         if np.any(dist_k <= best_thr) else 0.0,
        'unk_det':       float(np.mean(dist_u > best_thr)),
        'auroc':         auroc,
        'best_threshold': best_thr,
    }


def _calibrated_linear_probe_eval(
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    normalize:         bool = False,
    c_values:          tuple[float, ...] = (1.0,),
    emb_train_known:   np.ndarray | None = None,
    y_train_known:     np.ndarray | None = None,
) -> dict:
    """
    Train a linear classifier on calibration-day embeddings and reject by
    calibrated max class probability. This tests whether the embedding space
    is linearly separable even when Mahalanobis/cosine distances are weak.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def _prep(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return _l2_normalize(x) if normalize else x

    if emb_train_known is None:
        emb_train_known = emb_calib_known
    if y_train_known is None:
        y_train_known = y_calib_known

    scaler = StandardScaler()
    X_train = scaler.fit_transform(_prep(emb_train_known))
    X_cal = scaler.transform(_prep(emb_calib_known))
    X_unk_cal = scaler.transform(_prep(emb_calib_unknown))
    X_known = scaler.transform(_prep(emb_known))
    X_unk = scaler.transform(_prep(emb_unknown))

    best = None
    for c in c_values:
        clf = LogisticRegression(
            C=float(c),
            class_weight='balanced',
            max_iter=2000,
            random_state=0,
            solver='lbfgs',
        )
        clf.fit(X_train, y_train_known)

        prob_cal = clf.predict_proba(X_cal)
        prob_unk_cal = clf.predict_proba(X_unk_cal)
        pred_cal = clf.classes_[np.argmax(prob_cal, axis=1)]
        conf_cal = prob_cal.max(axis=1)
        conf_unk_cal = prob_unk_cal.max(axis=1)

        thresholds = np.linspace(
            np.concatenate([conf_cal, conf_unk_cal]).min(),
            np.concatenate([conf_cal, conf_unk_cal]).max(),
            100,
        )
        best_thr, best_score = 0.0, -np.inf
        for thr in thresholds:
            accept = conf_cal >= thr
            if accept.mean() < 0.35:
                continue
            open_acc = (
                float(np.mean(pred_cal[accept] == y_calib_known[accept]))
                if accept.any() else 0.0
            )
            unk_det = float(np.mean(conf_unk_cal < thr))
            score = 0.5 * (open_acc + unk_det)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        if best is None or best_score > best['score']:
            best = {'clf': clf, 'threshold': best_thr, 'score': best_score, 'C': float(c)}

    clf = best['clf']
    best_thr = best['threshold']
    best_score = best['score']

    prob_known = clf.predict_proba(X_known)
    prob_unk = clf.predict_proba(X_unk)
    pred_known = clf.classes_[np.argmax(prob_known, axis=1)]
    conf_known = prob_known.max(axis=1)
    conf_unk = prob_unk.max(axis=1)

    closed_acc = float(np.mean(pred_known == y_known))
    accept = conf_known >= best_thr
    open_acc = (
        float(np.mean(pred_known[accept] == y_known[accept]))
        if accept.any() else float('nan')
    )
    unk_det = float(np.mean(conf_unk < best_thr))
    try:
        binary = np.concatenate([
            np.zeros(len(conf_known), dtype=np.int32),
            np.ones(len(conf_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, -np.concatenate([conf_known, conf_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': closed_acc,
        'open_acc': open_acc,
        'unk_det': unk_det,
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': best_thr,
        'calib_score': float(best_score),
        'C': best['C'],
    }


def _softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return (exp / (exp.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _l2_probe_classwise_eval(
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
) -> dict:
    """
    L2 logistic probe with one confidence threshold per predicted class.
    A global threshold is often unfair when some devices are naturally less
    confident than others under day shift.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(_l2_normalize(emb_calib_known))
    X_unk_cal = scaler.transform(_l2_normalize(emb_calib_unknown))
    X_known = scaler.transform(_l2_normalize(emb_known))
    X_unk = scaler.transform(_l2_normalize(emb_unknown))

    clf = LogisticRegression(
        C=0.3,
        class_weight='balanced',
        max_iter=2000,
        random_state=0,
        solver='lbfgs',
    )
    clf.fit(X_cal, y_calib_known)

    prob_cal = clf.predict_proba(X_cal)
    prob_unk_cal = clf.predict_proba(X_unk_cal)
    prob_known = clf.predict_proba(X_known)
    prob_unk = clf.predict_proba(X_unk)

    pred_cal = clf.classes_[np.argmax(prob_cal, axis=1)]
    pred_known = clf.classes_[np.argmax(prob_known, axis=1)]
    pred_unk = clf.classes_[np.argmax(prob_unk, axis=1)]
    pred_unk_cal = clf.classes_[np.argmax(prob_unk_cal, axis=1)]
    conf_cal = prob_cal.max(axis=1)
    conf_unk_cal = prob_unk_cal.max(axis=1)
    conf_known = prob_known.max(axis=1)
    conf_unk = prob_unk.max(axis=1)

    thresholds = {}
    for cls in clf.classes_:
        known_mask = pred_cal == cls
        unk_mask = pred_unk_cal == cls
        if not np.any(known_mask) or not np.any(unk_mask):
            thresholds[int(cls)] = float(np.quantile(conf_cal, 0.35))
            continue

        vals = np.concatenate([conf_cal[known_mask], conf_unk_cal[unk_mask]])
        best_thr, best_score = float(vals.min()), -np.inf
        for thr in np.linspace(vals.min(), vals.max(), 80):
            accept = known_mask & (conf_cal >= thr)
            denom_known = max(int(known_mask.sum()), 1)
            cls_open = float(np.sum((pred_cal == y_calib_known) & accept) / denom_known)
            cls_unk = float(np.mean(conf_unk_cal[unk_mask] < thr))
            score = 0.5 * (cls_open + cls_unk)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds[int(cls)] = best_thr

    thr_known = np.array([thresholds[int(c)] for c in pred_known], dtype=np.float32)
    thr_unk = np.array([thresholds[int(c)] for c in pred_unk], dtype=np.float32)
    accept = conf_known >= thr_known

    try:
        binary = np.concatenate([
            np.zeros(len(conf_known), dtype=np.int32),
            np.ones(len(conf_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, -np.concatenate([conf_known, conf_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': float(np.mean(pred_known == y_known)),
        'open_acc': (
            float(np.mean(pred_known[accept] == y_known[accept]))
            if accept.any() else float('nan')
        ),
        'unk_det': float(np.mean(conf_unk < thr_unk)),
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': float(np.mean(list(thresholds.values()))),
        'thresholds': thresholds,
    }


def _temperature_l2_probe_eval(
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
) -> dict:
    """
    L2 logistic probe with a temperature sweep over decision logits before
    confidence thresholding.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(_l2_normalize(emb_calib_known))
    X_unk_cal = scaler.transform(_l2_normalize(emb_calib_unknown))
    X_known = scaler.transform(_l2_normalize(emb_known))
    X_unk = scaler.transform(_l2_normalize(emb_unknown))

    clf = LogisticRegression(
        C=0.3,
        class_weight='balanced',
        max_iter=2000,
        random_state=0,
        solver='lbfgs',
    )
    clf.fit(X_cal, y_calib_known)

    logits_cal = clf.decision_function(X_cal)
    logits_unk_cal = clf.decision_function(X_unk_cal)
    best = {'temperature': 1.0, 'threshold': 0.0, 'score': -np.inf}
    for temp in [0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        prob_cal = _softmax_np(logits_cal, temp)
        prob_unk_cal = _softmax_np(logits_unk_cal, temp)
        pred_cal = clf.classes_[np.argmax(prob_cal, axis=1)]
        conf_cal = prob_cal.max(axis=1)
        conf_unk_cal = prob_unk_cal.max(axis=1)
        vals = np.concatenate([conf_cal, conf_unk_cal])
        for thr in np.linspace(vals.min(), vals.max(), 100):
            accept = conf_cal >= thr
            if accept.mean() < 0.35:
                continue
            open_acc = (
                float(np.mean(pred_cal[accept] == y_calib_known[accept]))
                if accept.any() else 0.0
            )
            unk_det = float(np.mean(conf_unk_cal < thr))
            score = 0.5 * (open_acc + unk_det)
            if score > best['score']:
                best = {
                    'temperature': float(temp),
                    'threshold': float(thr),
                    'score': score,
                }

    prob_known = _softmax_np(clf.decision_function(X_known), best['temperature'])
    prob_unk = _softmax_np(clf.decision_function(X_unk), best['temperature'])
    pred_known = clf.classes_[np.argmax(prob_known, axis=1)]
    conf_known = prob_known.max(axis=1)
    conf_unk = prob_unk.max(axis=1)
    accept = conf_known >= best['threshold']

    try:
        binary = np.concatenate([
            np.zeros(len(conf_known), dtype=np.int32),
            np.ones(len(conf_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, -np.concatenate([conf_known, conf_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': float(np.mean(pred_known == y_known)),
        'open_acc': (
            float(np.mean(pred_known[accept] == y_known[accept]))
            if accept.any() else float('nan')
        ),
        'unk_det': float(np.mean(conf_unk < best['threshold'])),
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': best['threshold'],
        'temperature': best['temperature'],
    }


def _cosine_knn_eval(
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    seed:              int,
    emb_support_known: np.ndarray | None = None,
    y_support_known:   np.ndarray | None = None,
) -> dict:
    """
    Balanced cosine KNN with rejection by mean neighbor distance. Sweeps support
    size, k, and uniform vs distance-weighted voting. Also evaluates per-class
    thresholds for the best KNN configuration.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    def _vote(labels: np.ndarray, dist: np.ndarray, weighted: bool) -> np.ndarray:
        preds = np.empty(len(labels), dtype=np.int32)
        for i, row in enumerate(labels):
            vals = np.unique(row)
            if weighted:
                weights = 1.0 / (dist[i] + 1e-6)
                scores = np.array([
                    weights[row == v].sum() for v in vals
                ])
            else:
                scores = np.array([
                    np.sum(row == v) for v in vals
                ])
            preds[i] = vals[np.argmax(scores)]
        return preds

    def _metrics(
        pred_known: np.ndarray,
        dist_known: np.ndarray,
        pred_unk: np.ndarray,
        dist_unk: np.ndarray,
        threshold,
    ) -> dict:
        if isinstance(threshold, dict):
            thr_known = np.array([threshold[int(c)] for c in pred_known], dtype=np.float32)
            thr_unk = np.array([threshold[int(c)] for c in pred_unk], dtype=np.float32)
            threshold_out = float(np.mean(list(threshold.values())))
        else:
            thr_known = float(threshold)
            thr_unk = float(threshold)
            threshold_out = float(threshold)
        accept = dist_known <= thr_known
        try:
            binary = np.concatenate([
                np.zeros(len(dist_known), dtype=np.int32),
                np.ones(len(dist_unk), dtype=np.int32),
            ])
            auroc = float(roc_auc_score(binary, np.concatenate([dist_known, dist_unk])))
        except Exception:
            auroc = float('nan')
        return {
            'closed_acc': float(np.mean(pred_known == y_known)),
            'open_acc': (
                float(np.mean(pred_known[accept] == y_known[accept]))
                if accept.any() else float('nan')
            ),
            'unk_det': float(np.mean(dist_unk > thr_unk)),
            'auroc': auroc,
            'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
            'known_accept_rate': float(accept.mean()),
            'best_threshold': threshold_out,
        }

    rng = np.random.default_rng(seed)
    X_cal = _l2_normalize(emb_calib_known)
    X_unk_cal = _l2_normalize(emb_calib_unknown)
    X_known = _l2_normalize(emb_known)
    X_unk = _l2_normalize(emb_unknown)
    if emb_support_known is None:
        emb_support_known = emb_calib_known
    if y_support_known is None:
        y_support_known = y_calib_known

    best = None
    best_cache = None
    max_k = 20
    for support_per_class in [25, 50, 100, 200, 500]:
        support_idx = []
        for cls in sorted(np.unique(y_support_known)):
            idx = np.where(y_support_known == cls)[0]
            if len(idx) > support_per_class:
                idx = rng.choice(idx, size=support_per_class, replace=False)
            support_idx.extend(idx.tolist())
        support_idx = np.array(support_idx, dtype=np.int64)

        X_support = _l2_normalize(emb_support_known[support_idx])
        y_support = np.asarray(y_support_known[support_idx], dtype=np.int32)
        n_neighbors = min(max_k, len(X_support))
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1,
        )
        nn.fit(X_support)
        dist_cal_all, ind_cal_all = nn.kneighbors(X_cal, return_distance=True)
        dist_unk_cal_all, ind_unk_cal_all = nn.kneighbors(X_unk_cal, return_distance=True)
        labels_cal_all = y_support[ind_cal_all]
        labels_unk_cal_all = y_support[ind_unk_cal_all]

        for k in [1, 3, 5, 10, 20]:
            if k > n_neighbors:
                continue
            for weighted in [False, True]:
                dist_cal = dist_cal_all[:, :k].mean(axis=1)
                dist_unk_cal = dist_unk_cal_all[:, :k].mean(axis=1)
                pred_cal = _vote(
                    labels_cal_all[:, :k],
                    dist_cal_all[:, :k],
                    weighted,
                )
                vals = np.concatenate([dist_cal, dist_unk_cal])
                best_thr, best_score = float(vals.min()), -np.inf
                for thr in np.linspace(vals.min(), vals.max(), 100):
                    accept = dist_cal <= thr
                    if accept.mean() < 0.25:
                        continue
                    open_acc = (
                        float(np.mean(pred_cal[accept] == y_calib_known[accept]))
                        if accept.any() else 0.0
                    )
                    unk_det = float(np.mean(dist_unk_cal > thr))
                    accept_penalty = min(float(accept.mean()) / 0.30, 1.0)
                    score = 0.5 * (open_acc + unk_det) * accept_penalty
                    if score > best_score:
                        best_score = score
                        best_thr = float(thr)

                if best is None or best_score > best['score']:
                    best = {
                        'k': k,
                        'support_per_class': support_per_class,
                        'weighted': weighted,
                        'threshold': best_thr,
                        'score': best_score,
                    }
                    best_cache = {
                        'nn': nn,
                        'y_support': y_support,
                    }

    nn = best_cache['nn']
    y_support = best_cache['y_support']

    def _predict_final(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist_all, ind_all = nn.kneighbors(x, return_distance=True)
        labels_all = y_support[ind_all]
        k = best['k']
        pred = _vote(labels_all[:, :k], dist_all[:, :k], best['weighted'])
        dist = dist_all[:, :k].mean(axis=1)
        return pred, dist

    pred_known, dist_known = _predict_final(X_known)
    pred_unk, dist_unk = _predict_final(X_unk)
    global_metrics = _metrics(pred_known, dist_known, pred_unk, dist_unk, best['threshold'])

    pred_cal, dist_cal = _predict_final(X_cal)
    pred_unk_cal, dist_unk_cal = _predict_final(X_unk_cal)
    class_thresholds = {}
    for cls in sorted(np.unique(y_calib_known)):
        known_mask = pred_cal == cls
        unk_mask = pred_unk_cal == cls
        if not np.any(known_mask) or not np.any(unk_mask):
            class_thresholds[int(cls)] = best['threshold']
            continue
        vals = np.concatenate([dist_cal[known_mask], dist_unk_cal[unk_mask]])
        best_thr, best_score = best['threshold'], -np.inf
        for thr in np.linspace(vals.min(), vals.max(), 80):
            accept = known_mask & (dist_cal <= thr)
            cls_open = (
                float(np.mean(pred_cal[accept] == y_calib_known[accept]))
                if accept.any() else 0.0
            )
            cls_unk = float(np.mean(dist_unk_cal[unk_mask] > thr))
            score = 0.5 * (cls_open + cls_unk)
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        class_thresholds[int(cls)] = best_thr
    class_metrics = _metrics(
        pred_known, dist_known, pred_unk, dist_unk, class_thresholds
    )

    constrained = {}
    for target_accept in [0.50, 0.60, 0.70]:
        # Choose threshold from calibration known distances only. This makes
        # the target an operating-point constraint, not test-set tuning.
        thr = float(np.quantile(dist_cal, target_accept))
        constrained[target_accept] = _metrics(
            pred_known, dist_known, pred_unk, dist_unk, thr
        )

    global_metrics.update({
        'k': best['k'],
        'support_per_class': best['support_per_class'],
        'weighted': best['weighted'],
        'classwise': class_metrics,
        'accept_constrained': constrained,
    })
    return global_metrics


def _drift_correct_gallery(
    emb: np.ndarray,
    y: np.ndarray,
    traj,
    target_day: int,
) -> np.ndarray:
    """
    Shift each device's gallery embeddings forward to ``target_day`` using the
    trajectory's per-device drift vector (additive, raw embedding space).
    Devices absent from the trajectory are left unchanged.
    """
    out = emb.astype(np.float32).copy()
    known_devs = set(traj.known_device_ids)
    for dev in np.unique(y):
        if int(dev) not in known_devs:
            continue
        shift = traj.drift_vector(int(dev), target_day)
        out[y == dev] += shift
    return out


def _drift_corrected_knn_eval(
    traj,
    target_day:        int,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    seed:              int,
    emb_support_known: np.ndarray,
    y_support_known:   np.ndarray,
) -> dict:
    """
    Trajectory drift-corrected cosine KNN. The day-traj gallery (support +
    calibration known) is shifted forward to ``target_day`` along each device's
    observed drift, then matched against the actual target-day test queries.
    Reuses the full cosine-KNN sweep/threshold machinery unchanged.
    """
    sup = _drift_correct_gallery(emb_support_known, y_support_known, traj, target_day)
    cal = _drift_correct_gallery(emb_calib_known, y_calib_known, traj, target_day)
    return _cosine_knn_eval(
        emb_calib_known   = cal,
        y_calib_known     = y_calib_known,
        emb_calib_unknown = emb_calib_unknown,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unknown,
        seed              = seed,
        emb_support_known = sup,
        y_support_known   = y_support_known,
    )


def _calibrated_lda_probe_eval(
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    normalize:         bool = False,
) -> dict:
    """
    Shrinkage LDA on calibration-day embeddings. Compared with logistic
    regression, this allows a pooled covariance model while shrinkage keeps
    the 64-D estimate stable.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def _prep(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return _l2_normalize(x) if normalize else x

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(_prep(emb_calib_known))
    X_unk_cal = scaler.transform(_prep(emb_calib_unknown))
    X_known = scaler.transform(_prep(emb_known))
    X_unk = scaler.transform(_prep(emb_unknown))

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_cal, y_calib_known)

    prob_cal = clf.predict_proba(X_cal)
    prob_unk_cal = clf.predict_proba(X_unk_cal)
    pred_cal = clf.classes_[np.argmax(prob_cal, axis=1)]
    conf_cal = prob_cal.max(axis=1)
    conf_unk_cal = prob_unk_cal.max(axis=1)

    thresholds = np.linspace(
        np.concatenate([conf_cal, conf_unk_cal]).min(),
        np.concatenate([conf_cal, conf_unk_cal]).max(),
        50,
    )
    best_thr, best_score = 0.0, -np.inf
    for thr in thresholds:
        accept = conf_cal >= thr
        if accept.mean() < 0.50:
            continue
        open_acc = (
            float(np.mean(pred_cal[accept] == y_calib_known[accept]))
            if accept.any() else 0.0
        )
        unk_det = float(np.mean(conf_unk_cal < thr))
        score = 0.5 * (open_acc + unk_det)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    prob_known = clf.predict_proba(X_known)
    prob_unk = clf.predict_proba(X_unk)
    pred_known = clf.classes_[np.argmax(prob_known, axis=1)]
    conf_known = prob_known.max(axis=1)
    conf_unk = prob_unk.max(axis=1)

    closed_acc = float(np.mean(pred_known == y_known))
    accept = conf_known >= best_thr
    open_acc = (
        float(np.mean(pred_known[accept] == y_known[accept]))
        if accept.any() else float('nan')
    )
    unk_det = float(np.mean(conf_unk < best_thr))
    try:
        binary = np.concatenate([
            np.zeros(len(conf_known), dtype=np.int32),
            np.ones(len(conf_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, -np.concatenate([conf_known, conf_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': closed_acc,
        'open_acc': open_acc,
        'unk_det': unk_det,
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': best_thr,
        'calib_score': float(best_score),
    }


def _cosine_trajectory_eval(
    traj,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
) -> dict:
    """
    Use the trajectory's latest EWMA means as cosine prototypes. This keeps
    the temporal smoothing from Phase 3 while removing norm/covariance effects.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    classes = np.array(sorted(known_ids), dtype=np.int32)
    centroids = np.stack([traj.latest_mean(int(c)) for c in classes]).astype(np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    def _predict_dist(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim = emb @ centroids.T
        best = np.argmax(sim, axis=1)
        pred = classes[best]
        dist = 1.0 - sim[np.arange(len(emb)), best]
        return pred, dist

    pred_cal, dist_cal = _predict_dist(emb_calib_known)
    _, dist_unk_cal = _predict_dist(emb_calib_unknown)

    thresholds = np.linspace(
        np.concatenate([dist_cal, dist_unk_cal]).min(),
        np.concatenate([dist_cal, dist_unk_cal]).max(),
        100,
    )
    best_thr, best_score = 0.0, -np.inf
    for thr in thresholds:
        accept = dist_cal <= thr
        if accept.mean() < 0.50:
            continue
        open_acc = (
            float(np.mean(pred_cal[accept] == y_calib_known[accept]))
            if accept.any() else 0.0
        )
        unk_det = float(np.mean(dist_unk_cal > thr))
        score = 0.5 * (open_acc + unk_det)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    pred_known, dist_known = _predict_dist(emb_known)
    _, dist_unk = _predict_dist(emb_unknown)
    closed_acc = float(np.mean(pred_known == y_known))
    accept = dist_known <= best_thr
    open_acc = (
        float(np.mean(pred_known[accept] == y_known[accept]))
        if accept.any() else float('nan')
    )
    unk_det = float(np.mean(dist_unk > best_thr))

    try:
        binary = np.concatenate([
            np.zeros(len(dist_known), dtype=np.int32),
            np.ones(len(dist_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, np.concatenate([dist_known, dist_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': closed_acc,
        'open_acc': open_acc,
        'unk_det': unk_det,
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': best_thr,
        'calib_score': float(best_score),
    }


def _multiprototype_cosine_trajectory_eval(
    traj,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
) -> dict:
    """
    Multi-prototype cosine trajectory using recent per-day anchors from Phase 3.
    Calibrates recent-day span, decay, and rejection threshold on cfg.traj_day.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    candidate_days = [1, 2, 3, 5, None]
    candidate_decays = [0.5, 0.7, 0.9, 1.0]
    best = {
        'score': -np.inf,
        'threshold': 0.0,
        'days_keep': 1,
        'day_decay': 1.0,
        'align_drift': False,
    }

    for align_drift in [False, True]:
        for days_keep in candidate_days:
            for day_decay in candidate_decays:
                pred_cal, dist_cal = traj.classify_cosine_prototypes(
                    emb_calib_known,
                    threshold=np.inf,
                    known_ids=known_ids,
                    days_keep=days_keep,
                    day_decay=day_decay,
                    align_drift=align_drift,
                )
                _, dist_unk_cal = traj.classify_cosine_prototypes(
                    emb_calib_unknown,
                    threshold=np.inf,
                    known_ids=known_ids,
                    days_keep=days_keep,
                    day_decay=day_decay,
                    align_drift=align_drift,
                )
                all_dist = np.concatenate([dist_cal, dist_unk_cal])
                thresholds = np.linspace(all_dist.min(), all_dist.max(), 120)

                for thr in thresholds:
                    accept = dist_cal <= thr
                    if accept.mean() < 0.50:
                        continue
                    open_acc = (
                        float(np.mean(pred_cal[accept] == y_calib_known[accept]))
                        if accept.any() else 0.0
                    )
                    unk_det = float(np.mean(dist_unk_cal > thr))
                    score = 0.5 * (open_acc + unk_det)
                    if score > best['score']:
                        best = {
                            'score': score,
                            'threshold': float(thr),
                            'days_keep': days_keep,
                            'day_decay': float(day_decay),
                            'align_drift': bool(align_drift),
                        }



    # Soft-aggregation tuning (if T2R_SOFT_AGG_TUNING=1)
    if os.environ.get('T2R_SOFT_AGG_TUNING', '').lower() in ('1', 'true', 'yes'):
        from proto_improve import soft_classify_cosine_prototypes
        _log("  [SOFT_AGG_TUNING] testing aggregation methods: hard_max, top_k_mean, weighted_mean, softmax")
        for agg_method in ['hard_max', 'top_k_mean', 'weighted_mean', 'softmax']:
            agg_params = [1] if agg_method in ['hard_max', 'weighted_mean'] else                          [2, 3] if agg_method == 'top_k_mean' else [0.1, 0.2, 1.0]
            for agg_param in agg_params:
                pred_cal, dist_cal = soft_classify_cosine_prototypes(
                    emb_calib_known,
                    traj,
                    threshold=np.inf,
                    known_ids=known_ids,
                    days_keep=best['days_keep'],
                    day_decay=best['day_decay'],
                    align_drift=best['align_drift'],
                    agg_method=agg_method,
                    agg_param=agg_param,
                )
                _, dist_unk_cal = soft_classify_cosine_prototypes(
                    emb_calib_unknown,
                    traj,
                    threshold=np.inf,
                    known_ids=known_ids,
                    days_keep=best['days_keep'],
                    day_decay=best['day_decay'],
                    align_drift=best['align_drift'],
                    agg_method=agg_method,
                    agg_param=agg_param,
                )
                all_dist = np.concatenate([dist_cal, dist_unk_cal])
                thresholds = np.linspace(all_dist.min(), all_dist.max(), 60)
                for thr in thresholds:
                    accept = dist_cal <= thr
                    if accept.mean() < 0.50:
                        continue
                    open_acc = float(np.mean(pred_cal[accept] == y_calib_known[accept])) if accept.any() else 0.0
                    unk_det = float(np.mean(dist_unk_cal > thr))
                    score = 0.5 * (open_acc + unk_det)
                    if score > best['score']:
                        best = {
                            'score': score,
                            'threshold': float(thr),
                            'days_keep': best['days_keep'],
                            'day_decay': best['day_decay'],
                            'align_drift': best['align_drift'],
                            'agg_method': agg_method,
                            'agg_param': agg_param,
                        }

    if best.get('agg_method') and best['agg_method'] != 'hard_max':
        from proto_improve import soft_classify_cosine_prototypes
        pred_known, dist_known = soft_classify_cosine_prototypes(
            emb_known,
            traj,
            threshold=np.inf,
            known_ids=known_ids,
            days_keep=best['days_keep'],
            day_decay=best['day_decay'],
            align_drift=best.get('align_drift', False),
            agg_method=best['agg_method'],
            agg_param=best.get('agg_param', 1),
        )
    else:
        pred_known, dist_known = traj.classify_cosine_prototypes(
            emb_known,
            threshold=np.inf,
        known_ids=known_ids,
        days_keep=best['days_keep'],
        day_decay=best['day_decay'],
        align_drift=best['align_drift'],
    )
    _, dist_unk = traj.classify_cosine_prototypes(
        emb_unknown,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=best['days_keep'],
        day_decay=best['day_decay'],
        align_drift=best['align_drift'],
    )

    accept = dist_known <= best['threshold']
    try:
        binary = np.concatenate([
            np.zeros(len(dist_known), dtype=np.int32),
            np.ones(len(dist_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, np.concatenate([dist_known, dist_unk])))
    except Exception:
        auroc = float('nan')

    # ── DIAGNOSTIC: direct closed-set accuracy for corner configs ─────────
    # The selection above maximises rejection (open+unk), not closed-set, so
    # it cannot "see" a closed-set gain from drift alignment. Measure closed
    # accuracy directly on the test known set for each corner config.
    print('  [mp-traj diagnostic] closed-set acc by config (test known set):')
    for diag_align in [False, True]:
        for diag_days in [1, None]:
            dp, _ = traj.classify_cosine_prototypes(
                emb_known, threshold=np.inf, known_ids=known_ids,
                days_keep=diag_days, day_decay=best['day_decay'],
                align_drift=diag_align,
            )
            print(f'    align_drift={diag_align!s:<5}  days_keep={diag_days}  '
                  f'closed_acc={float(np.mean(dp == y_known)):.4f}')

    return {
        'closed_acc': float(np.mean(pred_known == y_known)),
        'open_acc': (
            float(np.mean(pred_known[accept] == y_known[accept]))
            if accept.any() else float('nan')
        ),
        'unk_det': float(np.mean(dist_unk > best['threshold'])),
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': best['threshold'],
        'calib_score': float(best['score']),
        'days_keep': best['days_keep'],
        'day_decay': best['day_decay'],
        'align_drift': best['align_drift'],
    }


def _classwise_multiprototype_cosine_trajectory_eval(
    traj,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
    days_keep:         int,
    day_decay:         float,
) -> dict:
    """
    Multi-prototype cosine trajectory with one rejection threshold per
    predicted device. Thresholds are calibrated only on cfg.traj_day.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    pred_cal, dist_cal = traj.classify_cosine_prototypes(
        emb_calib_known,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=days_keep,
        day_decay=day_decay,
    )
    pred_unk_cal, dist_unk_cal = traj.classify_cosine_prototypes(
        emb_calib_unknown,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=days_keep,
        day_decay=day_decay,
    )

    all_cal_dist = np.concatenate([dist_cal, dist_unk_cal])
    global_threshold = float(np.quantile(all_cal_dist, 0.35))

    profiles = [
        (0.05, (0.50, 0.45, 0.05)),
        (0.10, (0.50, 0.40, 0.10)),
        (0.20, (0.45, 0.45, 0.10)),
        (0.35, (0.40, 0.40, 0.20)),
        (0.50, (0.35, 0.35, 0.30)),
    ]
    best_profile = {
        'score': -np.inf,
        'thresholds': {},
        'min_accept_rate': 0.20,
        'weights': (0.45, 0.45, 0.10),
    }

    for min_accept_rate, weights in profiles:
        thresholds: dict[int, float] = {}
        w_open, w_unk, w_accept = weights

        for dev in known_ids:
            known_mask = pred_cal == dev
            unk_mask = pred_unk_cal == dev
            local_dist = np.concatenate([dist_cal[known_mask], dist_unk_cal[unk_mask]])
            if len(local_dist) < 10 or not known_mask.any():
                thresholds[int(dev)] = global_threshold
                continue

            candidates = np.linspace(local_dist.min(), local_dist.max(), 100)
            best_score = -np.inf
            best_thr = global_threshold
            for thr in candidates:
                known_accept = known_mask & (dist_cal <= thr)
                accept_rate = float(known_accept.sum() / max(1, known_mask.sum()))
                if accept_rate < min_accept_rate:
                    continue
                open_acc = (
                    float(np.mean(pred_cal[known_accept] == y_calib_known[known_accept]))
                    if known_accept.any() else 0.0
                )
                unk_det = (
                    float(np.mean(dist_unk_cal[unk_mask] > thr))
                    if unk_mask.any() else 1.0
                )
                score = w_open * open_acc + w_unk * unk_det + w_accept * accept_rate
                if score > best_score:
                    best_score = score
                    best_thr = float(thr)
            thresholds[int(dev)] = best_thr

        thr_cal = np.array(
            [thresholds.get(int(p), global_threshold) for p in pred_cal],
            dtype=np.float32,
        )
        thr_unk_cal = np.array(
            [thresholds.get(int(p), global_threshold) for p in pred_unk_cal],
            dtype=np.float32,
        )
        accept_cal = dist_cal <= thr_cal
        open_acc_cal = (
            float(np.mean(pred_cal[accept_cal] == y_calib_known[accept_cal]))
            if accept_cal.any() else 0.0
        )
        unk_det_cal = float(np.mean(dist_unk_cal > thr_unk_cal))
        accept_rate_cal = float(accept_cal.mean())
        profile_score = (
            0.45 * open_acc_cal + 0.45 * unk_det_cal + 0.10 * accept_rate_cal
        )
        if profile_score > best_profile['score']:
            best_profile = {
                'score': profile_score,
                'thresholds': thresholds,
                'min_accept_rate': float(min_accept_rate),
                'weights': tuple(float(w) for w in weights),
                'calib_open_acc': open_acc_cal,
                'calib_unk_det': unk_det_cal,
                'calib_accept_rate': accept_rate_cal,
            }

    thresholds = best_profile['thresholds']

    def _threshold_for(pred: np.ndarray) -> np.ndarray:
        return np.array(
            [thresholds.get(int(p), global_threshold) for p in pred],
            dtype=np.float32,
        )

    pred_known, dist_known = traj.classify_cosine_prototypes(
        emb_known,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=days_keep,
        day_decay=day_decay,
    )
    pred_unk, dist_unk = traj.classify_cosine_prototypes(
        emb_unknown,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=days_keep,
        day_decay=day_decay,
    )
    thr_known = _threshold_for(pred_known)
    thr_unk = _threshold_for(pred_unk)
    accept = dist_known <= thr_known

    try:
        binary = np.concatenate([
            np.zeros(len(dist_known), dtype=np.int32),
            np.ones(len(dist_unk), dtype=np.int32),
        ])
        scores = np.concatenate([
            dist_known / (thr_known + 1e-8),
            dist_unk / (thr_unk + 1e-8),
        ])
        auroc = float(roc_auc_score(binary, scores))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': float(np.mean(pred_known == y_known)),
        'open_acc': (
            float(np.mean(pred_known[accept] == y_known[accept]))
            if accept.any() else float('nan')
        ),
        'unk_det': float(np.mean(dist_unk > thr_unk)),
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': float(np.mean(list(thresholds.values()))),
        'thresholds': thresholds,
        'calib_score': float(best_profile['score']),
        'calib_min_accept_rate': best_profile['min_accept_rate'],
        'calib_weights': best_profile['weights'],
        'calib_open_acc': best_profile.get('calib_open_acc', float('nan')),
        'calib_unk_det': best_profile.get('calib_unk_det', float('nan')),
        'calib_accept_rate': best_profile.get('calib_accept_rate', float('nan')),
        'days_keep': int(days_keep),
        'day_decay': float(day_decay),
    }


def _target_accept_classwise_multiprototype_cosine_trajectory_eval(
    traj,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
    target_accept:     float,
) -> dict:
    """
    Class-wise multi-prototype cosine trajectory at a target known accept rate.
    Sweeps recent-day span and decay on cfg.traj_day only.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    candidate_days = [1, 2, 3, 5]
    candidate_decays = [0.5, 0.7, 0.9, 1.0]
    best = {
        'score': -np.inf,
        'thresholds': {},
        'days_keep': 1,
        'day_decay': 1.0,
        'global_threshold': 0.0,
        'calib_open_acc': 0.0,
        'calib_unk_det': 0.0,
        'calib_accept_rate': 0.0,
    }

    for days_keep in candidate_days:
        for day_decay in candidate_decays:
            pred_cal, dist_cal = traj.classify_cosine_prototypes(
                emb_calib_known,
                threshold=np.inf,
                known_ids=known_ids,
                days_keep=days_keep,
                day_decay=day_decay,
            )
            pred_unk_cal, dist_unk_cal = traj.classify_cosine_prototypes(
                emb_calib_unknown,
                threshold=np.inf,
                known_ids=known_ids,
                days_keep=days_keep,
                day_decay=day_decay,
            )
            global_threshold = float(np.quantile(dist_cal, target_accept))
            thresholds: dict[int, float] = {}
            for dev in known_ids:
                mask = pred_cal == dev
                if mask.sum() < 5:
                    thresholds[int(dev)] = global_threshold
                    continue
                thresholds[int(dev)] = float(
                    np.quantile(dist_cal[mask], target_accept)
                )

            thr_cal = np.array(
                [thresholds.get(int(p), global_threshold) for p in pred_cal],
                dtype=np.float32,
            )
            thr_unk_cal = np.array(
                [thresholds.get(int(p), global_threshold) for p in pred_unk_cal],
                dtype=np.float32,
            )
            accept_cal = dist_cal <= thr_cal
            open_acc_cal = (
                float(np.mean(pred_cal[accept_cal] == y_calib_known[accept_cal]))
                if accept_cal.any() else 0.0
            )
            unk_det_cal = float(np.mean(dist_unk_cal > thr_unk_cal))
            accept_rate_cal = float(accept_cal.mean())
            score = (
                0.40 * open_acc_cal
                + 0.35 * unk_det_cal
                + 0.25 * (1.0 - abs(accept_rate_cal - target_accept))
            )
            if score > best['score']:
                best = {
                    'score': score,
                    'thresholds': thresholds,
                    'days_keep': int(days_keep),
                    'day_decay': float(day_decay),
                    'global_threshold': global_threshold,
                    'calib_open_acc': open_acc_cal,
                    'calib_unk_det': unk_det_cal,
                    'calib_accept_rate': accept_rate_cal,
                }

    thresholds = best['thresholds']
    global_threshold = best['global_threshold']

    pred_known, dist_known = traj.classify_cosine_prototypes(
        emb_known,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=best['days_keep'],
        day_decay=best['day_decay'],
    )
    pred_unk, dist_unk = traj.classify_cosine_prototypes(
        emb_unknown,
        threshold=np.inf,
        known_ids=known_ids,
        days_keep=best['days_keep'],
        day_decay=best['day_decay'],
    )
    thr_known = np.array(
        [thresholds.get(int(p), global_threshold) for p in pred_known],
        dtype=np.float32,
    )
    thr_unk = np.array(
        [thresholds.get(int(p), global_threshold) for p in pred_unk],
        dtype=np.float32,
    )
    accept = dist_known <= thr_known

    try:
        binary = np.concatenate([
            np.zeros(len(dist_known), dtype=np.int32),
            np.ones(len(dist_unk), dtype=np.int32),
        ])
        scores = np.concatenate([
            dist_known / (thr_known + 1e-8),
            dist_unk / (thr_unk + 1e-8),
        ])
        auroc = float(roc_auc_score(binary, scores))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': float(np.mean(pred_known == y_known)),
        'open_acc': (
            float(np.mean(pred_known[accept] == y_known[accept]))
            if accept.any() else float('nan')
        ),
        'unk_det': float(np.mean(dist_unk > thr_unk)),
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': float(np.mean(list(thresholds.values()))),
        'thresholds': thresholds,
        'target_accept': float(target_accept),
        'calib_score': float(best['score']),
        'calib_open_acc': float(best['calib_open_acc']),
        'calib_unk_det': float(best['calib_unk_det']),
        'calib_accept_rate': float(best['calib_accept_rate']),
        'days_keep': int(best['days_keep']),
        'day_decay': float(best['day_decay']),
    }


def _probe_trajectory_fusion_eval(
    traj,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
) -> dict:
    """
    Fuse the strongest classifier signal (regularised L2 logistic probe) with
    trajectory rejection distance. The probe predicts identity; the fused score
    only decides accept/reject.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(_l2_normalize(emb_calib_known))
    X_unk_cal = scaler.transform(_l2_normalize(emb_calib_unknown))
    X_known = scaler.transform(_l2_normalize(emb_known))
    X_unk = scaler.transform(_l2_normalize(emb_unknown))

    clf = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=2000,
        random_state=0,
        solver='lbfgs',
    )
    clf.fit(X_cal, y_calib_known)

    prob_cal = clf.predict_proba(X_cal)
    prob_unk_cal = clf.predict_proba(X_unk_cal)
    prob_known = clf.predict_proba(X_known)
    prob_unk = clf.predict_proba(X_unk)

    pred_cal = clf.classes_[np.argmax(prob_cal, axis=1)]
    pred_known = clf.classes_[np.argmax(prob_known, axis=1)]
    conf_cal = prob_cal.max(axis=1)
    conf_unk_cal = prob_unk_cal.max(axis=1)
    conf_known = prob_known.max(axis=1)
    conf_unk = prob_unk.max(axis=1)

    _, dist_cal = traj.classify_open_set(
        emb_calib_known, threshold=np.inf, known_ids=known_ids
    )
    _, dist_unk_cal = traj.classify_open_set(
        emb_calib_unknown, threshold=np.inf, known_ids=known_ids
    )
    _, dist_known = traj.classify_open_set(
        emb_known, threshold=np.inf, known_ids=known_ids
    )
    _, dist_unk = traj.classify_open_set(
        emb_unknown, threshold=np.inf, known_ids=known_ids
    )

    d_mu = float(np.mean(dist_cal))
    d_std = float(np.std(dist_cal) + 1e-8)
    dist_cal_z = (dist_cal - d_mu) / d_std
    dist_unk_cal_z = (dist_unk_cal - d_mu) / d_std
    dist_known_z = (dist_known - d_mu) / d_std
    dist_unk_z = (dist_unk - d_mu) / d_std

    best = {'lambda': 0.0, 'threshold': 0.0, 'score': -np.inf}
    for lam in [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
        score_cal = (1.0 - conf_cal) + lam * dist_cal_z
        score_unk_cal = (1.0 - conf_unk_cal) + lam * dist_unk_cal_z
        thresholds = np.linspace(
            np.concatenate([score_cal, score_unk_cal]).min(),
            np.concatenate([score_cal, score_unk_cal]).max(),
            100,
        )
        for thr in thresholds:
            accept = score_cal <= thr
            if accept.mean() < 0.35:
                continue
            open_acc = (
                float(np.mean(pred_cal[accept] == y_calib_known[accept]))
                if accept.any() else 0.0
            )
            unk_det = float(np.mean(score_unk_cal > thr))
            score = 0.5 * (open_acc + unk_det)
            if score > best['score']:
                best = {'lambda': float(lam), 'threshold': float(thr), 'score': score}

    score_known = (1.0 - conf_known) + best['lambda'] * dist_known_z
    score_unk = (1.0 - conf_unk) + best['lambda'] * dist_unk_z
    accept = score_known <= best['threshold']

    try:
        binary = np.concatenate([
            np.zeros(len(score_known), dtype=np.int32),
            np.ones(len(score_unk), dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, np.concatenate([score_known, score_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc': float(np.mean(pred_known == y_known)),
        'open_acc': (
            float(np.mean(pred_known[accept] == y_known[accept]))
            if accept.any() else float('nan')
        ),
        'unk_det': float(np.mean(score_unk > best['threshold'])),
        'auroc': auroc,
        'f1_macro': float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold': best['threshold'],
        'calib_score': float(best['score']),
        'lambda': best['lambda'],
    }


def _temperature_scaling_eval(
    model:             tf.keras.Model,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
) -> dict:
    """
    Post-hoc temperature scaling on the classifier head.

    Optimises a single temperature T on the calibration-day logits to
    minimise NLL, then uses calibrated confidence for open-set rejection.
    Logits are reconstructed from embeddings via the Dense('classifier')
    layer weights — no re-inference needed.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from scipy.optimize import minimize_scalar

    try:
        clf_layer = model.get_layer('classifier')
    except ValueError:
        print('[whitening] _temperature_scaling_eval: no "classifier" layer found — skipping')
        return {k: float('nan') for k in (
            'closed_acc', 'open_acc', 'unk_det', 'auroc', 'f1_macro',
            'known_accept_rate', 'best_threshold', 'temperature',
        )}
    W, b = [w.astype(np.float64) for w in clf_layer.get_weights()]

    def get_logits(emb: np.ndarray) -> np.ndarray:
        return (np.asarray(emb, dtype=np.float64) @ W + b).astype(np.float32)

    lg_cal     = get_logits(emb_calib_known)
    lg_cal_unk = get_logits(emb_calib_unknown)
    lg_known   = get_logits(emb_known)
    lg_unk     = get_logits(emb_unknown)

    def nll(log_T: float) -> float:
        T  = float(np.exp(log_T))
        z  = lg_cal.astype(np.float64) / T
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True) + 1e-12)
        return float(-np.mean(lp[np.arange(len(y_calib_known)), y_calib_known]))

    res   = minimize_scalar(nll, bounds=(-2.0, 2.0), method='bounded')
    T_opt = float(np.exp(res.x))

    def scaled_probs(lg: np.ndarray):
        z = lg.astype(np.float64) / T_opt
        z -= z.max(axis=1, keepdims=True)
        p  = np.exp(z)
        p /= p.sum(axis=1, keepdims=True) + 1e-12
        return p.argmax(axis=1).astype(np.int32), p.max(axis=1).astype(np.float32)

    pred_cal, conf_cal     = scaled_probs(lg_cal)
    _, conf_cal_unk        = scaled_probs(lg_cal_unk)
    pred_known, conf_known = scaled_probs(lg_known)
    _, conf_unk            = scaled_probs(lg_unk)

    best_thr, best_score = 0.0, -np.inf
    for thr in np.linspace(
        min(conf_cal.min(), conf_cal_unk.min()),
        max(conf_cal.max(), conf_cal_unk.max()),
        100,
    ):
        accept = conf_cal >= thr
        if accept.mean() < 0.35:
            continue
        oa = float(np.mean(pred_cal[accept] == y_calib_known[accept])) if accept.any() else 0.0
        ud = float(np.mean(conf_cal_unk < thr))
        if 0.5 * (oa + ud) > best_score:
            best_score = 0.5 * (oa + ud)
            best_thr   = float(thr)

    accept = conf_known >= best_thr
    try:
        binary = np.concatenate([
            np.zeros(len(conf_known), dtype=np.int32),
            np.ones(len(conf_unk),   dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, -np.concatenate([conf_known, conf_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc':        float(np.mean(pred_known == y_known)),
        'open_acc':          float(np.mean(pred_known[accept] == y_known[accept])) if accept.any() else float('nan'),
        'unk_det':           float(np.mean(conf_unk < best_thr)),
        'auroc':             auroc,
        'f1_macro':          float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold':    best_thr,
        'temperature':       T_opt,
    }


def _extrapolated_trajectory_eval(
    traj,
    test_day:          int,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
) -> dict:
    """
    Open-set classifier using 2-point linear drift extrapolation.

    Instead of the latest observed EWMA mean (Day 7), projects each device's
    drift direction forward to test_day to predict where the embedding cloud
    will be.  Calibrates the cosine rejection threshold on traj_day data.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    # Calibrate threshold on traj_day calibration embeddings
    pred_cal, dist_cal = traj.classify_extrapolated(
        emb_calib_known, target_day=test_day, threshold=np.inf,
        known_ids=known_ids,
    )
    _, dist_unk_cal = traj.classify_extrapolated(
        emb_calib_unknown, target_day=test_day, threshold=np.inf,
        known_ids=known_ids,
    )

    all_dist = np.concatenate([dist_cal, dist_unk_cal])
    thresholds = np.linspace(all_dist.min(), all_dist.max(), 100)
    best_thr, best_score = 0.0, -np.inf
    for thr in thresholds:
        accept = dist_cal <= thr
        if accept.mean() < 0.50:
            continue
        open_acc = (
            float(np.mean(pred_cal[accept] == y_calib_known[accept]))
            if accept.any() else 0.0
        )
        unk_det = float(np.mean(dist_unk_cal > thr))
        score = 0.5 * (open_acc + unk_det)
        if score > best_score:
            best_score = score
            best_thr   = float(thr)

    # Evaluate on test-day embeddings
    pred_known, dist_known = traj.classify_extrapolated(
        emb_known, target_day=test_day, threshold=np.inf,
        known_ids=known_ids,
    )
    _, dist_unk = traj.classify_extrapolated(
        emb_unknown, target_day=test_day, threshold=np.inf,
        known_ids=known_ids,
    )

    accept = dist_known <= best_thr
    try:
        binary = np.concatenate([
            np.zeros(len(dist_known), dtype=np.int32),
            np.ones(len(dist_unk),   dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, np.concatenate([dist_known, dist_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc':        float(np.mean(pred_known == y_known)),
        'open_acc':          float(np.mean(pred_known[accept] == y_known[accept])) if accept.any() else float('nan'),
        'unk_det':           float(np.mean(dist_unk > best_thr)),
        'auroc':             auroc,
        'f1_macro':          float(f1_score(y_known, pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold':    best_thr,
        'calib_score':       float(best_score),
    }


def _traj_knn_fusion_eval(
    traj,
    emb_calib_known:   np.ndarray,
    y_calib_known:     np.ndarray,
    emb_calib_unknown: np.ndarray,
    emb_known:         np.ndarray,
    y_known:           np.ndarray,
    emb_unknown:       np.ndarray,
    known_ids:         list,
    seed:              int,
    days_keep=None,
    day_decay:         float = 1.0,
) -> dict:
    """
    Trajectory-reject + KNN-classify fusion.

    Reject/accept decision: trajectory cosine prototype distance (sweep on
    calibration data). Classification label: cosine KNN on Day-7 support
    embeddings. The two signals are decoupled — trajectory handles the
    open-set boundary, KNN handles identity within the boundary.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    # ── Step 1: calibrate trajectory rejection threshold ──────────────────
    _, dist_cal = traj.classify_cosine_prototypes(
        emb_calib_known, threshold=np.inf,
        known_ids=known_ids, days_keep=days_keep, day_decay=day_decay,
    )
    _, dist_unk_cal = traj.classify_cosine_prototypes(
        emb_calib_unknown, threshold=np.inf,
        known_ids=known_ids, days_keep=days_keep, day_decay=day_decay,
    )
    all_dist = np.concatenate([dist_cal, dist_unk_cal])
    best_thr, best_score = 0.0, -np.inf
    for thr in np.linspace(all_dist.min(), all_dist.max(), 120):
        accept = dist_cal <= thr
        if accept.mean() < 0.35:
            continue
        # Use KNN labels on calibration set to score
        unk_det = float(np.mean(dist_unk_cal > thr))
        # Penalise thresholds that reject too aggressively
        score = 0.5 * (float(accept.mean()) + unk_det)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    # ── Step 2: build KNN on calibration support ──────────────────────────
    rng = np.random.default_rng(seed)
    X_support = _l2_normalize(emb_calib_known)
    y_support = np.asarray(y_calib_known, dtype=np.int32)
    # Balanced subsample: up to 500/class
    sup_idx = []
    for cls in sorted(np.unique(y_support)):
        idx = np.where(y_support == cls)[0]
        if len(idx) > 500:
            idx = rng.choice(idx, size=500, replace=False)
        sup_idx.extend(idx.tolist())
    sup_idx = np.array(sup_idx, dtype=np.int64)
    X_sup = X_support[sup_idx]
    y_sup = y_support[sup_idx]

    nn = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X_sup)

    # ── Step 3: get trajectory distances and KNN labels on test set ───────
    _, dist_known = traj.classify_cosine_prototypes(
        emb_known, threshold=np.inf,
        known_ids=known_ids, days_keep=days_keep, day_decay=day_decay,
    )
    _, dist_unk = traj.classify_cosine_prototypes(
        emb_unknown, threshold=np.inf,
        known_ids=known_ids, days_keep=days_keep, day_decay=day_decay,
    )

    X_known_l2 = _l2_normalize(emb_known)
    X_unk_l2   = _l2_normalize(emb_unknown)
    _, ind_known = nn.kneighbors(X_known_l2, return_distance=True)
    _, ind_unk   = nn.kneighbors(X_unk_l2,   return_distance=True)
    knn_pred_known = y_sup[ind_known[:, 0]]
    knn_pred_unk   = y_sup[ind_unk[:, 0]]

    # ── Step 4: apply trajectory threshold for accept/reject ─────────────
    accept = dist_known <= best_thr
    closed_acc = float(np.mean(knn_pred_known == y_known))
    open_acc = (
        float(np.mean(knn_pred_known[accept] == y_known[accept]))
        if accept.any() else float('nan')
    )
    unk_det = float(np.mean(dist_unk > best_thr))

    try:
        binary = np.concatenate([
            np.zeros(len(dist_known), dtype=np.int32),
            np.ones(len(dist_unk),   dtype=np.int32),
        ])
        auroc = float(roc_auc_score(binary, np.concatenate([dist_known, dist_unk])))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc':        closed_acc,
        'open_acc':          open_acc,
        'unk_det':           unk_det,
        'auroc':             auroc,
        'f1_macro':          float(f1_score(y_known, knn_pred_known, average='macro', zero_division=0)),
        'known_accept_rate': float(accept.mean()),
        'best_threshold':    best_thr,
        'calib_score':       float(best_score),
        'days_keep':         days_keep,
        'day_decay':         day_decay,
    }


def _pseudo_label_trajectory_update(
    traj,
    model:        'tf.keras.Model',
    emb_known_test: np.ndarray,
    y_known_test:   np.ndarray,
    emb_unk_test:   np.ndarray,
    known_ids:      list,
    test_day:       int,
    confidence_threshold: float = 0.90,
) -> tuple:
    """
    Update trajectory centroids using high-confidence pseudo-labels from
    the test-day data (both known and unknown pools).

    Only predictions where softmax max-confidence >= confidence_threshold
    and the predicted class is in known_ids are used.  Unknown pool is
    included so the model can self-select borderline knowns missed by the
    trajectory threshold.

    Returns
    -------
    traj_pl   : updated deep copy of traj
    n_accepted: number of pseudo-labeled samples used
    pl_acc    : pseudo-label accuracy on the known-device accepted set
                (requires y_known_test for verification — not used at
                 inference time, only for diagnostics)
    """
    import copy
    import tensorflow as tf

    traj_pl = copy.deepcopy(traj)

    # Get classifier logits → softmax confidence
    try:
        clf_layer = model.get_layer('classifier')
    except ValueError:
        print('[pseudo_label] No classifier layer — skipping pseudo-label update.')
        return traj_pl, 0, float('nan')

    W, b = [w.astype(np.float64) for w in clf_layer.get_weights()]

    def _softmax_conf(emb: np.ndarray):
        logits = np.asarray(emb, dtype=np.float64) @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-12
        pred = probs.argmax(axis=1).astype(np.int32)
        conf = probs.max(axis=1).astype(np.float32)
        return pred, conf

    # Pool all test samples (known + unknown) for pseudo-labeling
    emb_all = np.concatenate([emb_known_test, emb_unk_test], axis=0)
    pred_all, conf_all = _softmax_conf(emb_all)

    # Accept only high-confidence predictions of known classes
    mask = conf_all >= confidence_threshold
    emb_accepted  = emb_all[mask]
    pred_accepted = pred_all[mask]

    if len(emb_accepted) == 0:
        print(f'[pseudo_label] No samples above conf={confidence_threshold:.2f} — skipping.')
        return traj_pl, 0, float('nan')

    # Diagnostic: check accuracy on the known portion
    n_known = len(emb_known_test)
    known_mask = mask[:n_known]
    if known_mask.any():
        pl_acc = float(np.mean(pred_all[:n_known][known_mask] == y_known_test[known_mask]))
    else:
        pl_acc = float('nan')

    print(f'[pseudo_label] Accepted {len(emb_accepted)} / {len(emb_all)} samples '
          f'(conf >= {confidence_threshold:.2f})')
    print(f'[pseudo_label] Pseudo-label acc on known accepted: {pl_acc:.4f}')

    traj_pl.update(day_id=test_day, emb=emb_accepted, labels=pred_accepted)
    return traj_pl, int(len(emb_accepted)), pl_acc


# ---------------------------------------------------------------------------
# Ablation 15: burst-level decisions and capture-aware enrolment
# ---------------------------------------------------------------------------

def _burst_accuracy(scores, y_q, g_q, classes, n_slices, rng):
    """
    Decision-level accuracy when each decision pools `n_slices` query slices
    from the same device and the same capture file. n_slices=1 recovers
    per-slice accuracy. Bags never mix devices or captures, so this pools
    evidence rather than adding information.
    """
    correct = total = 0
    for c in classes:
        idx_c = np.where(y_q == c)[0]
        for f in np.unique(g_q[idx_c, 0]):
            idx = idx_c[g_q[idx_c, 0] == f]
            if len(idx) < n_slices:
                continue
            idx = idx[rng.permutation(len(idx))]
            for b in range(len(idx) // n_slices):
                bag = idx[b * n_slices:(b + 1) * n_slices]
                if classes[int(np.argmax(scores[bag].sum(axis=0)))] == c:
                    correct += 1
                total += 1
    return (correct / total if total else float('nan')), total


def _proto_scores(emb_sup, y_sup, emb_qry, classes):
    """Per-class score for each query slice: max cosine to that class's support."""
    S = emb_sup / (np.linalg.norm(emb_sup, axis=1, keepdims=True) + 1e-8)
    Q = emb_qry / (np.linalg.norm(emb_qry, axis=1, keepdims=True) + 1e-8)
    sims = Q @ S.T
    return np.stack([sims[:, y_sup == c].max(axis=1) for c in classes], axis=1)


def _t2r_scores(traj, emb_q, classes, days_keep=None, day_decay=None):
    """
    Per-class score under the T2R decision rule (Eq. 3): the best
    recency-weighted cosine similarity to that device's recent trajectory
    prototype banks. Higher is closer, matching _proto_scores' convention.

    This is the trajectory rule itself, not a single-day prototype match,
    so it retains the multi-day banks and recency weighting.
    """
    E = np.asarray(emb_q, dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S = np.full((len(E), len(classes)), -np.inf, dtype=np.float32)
    for j, dev in enumerate(classes):
        if dev not in traj._history or len(traj._history[dev]) == 0:
            continue
        proto, weight = traj.cosine_prototype_bank(
            dev, days_keep=days_keep, day_decay=day_decay, align_drift=False)
        sim = E @ proto.T
        sim = sim * (0.9 + 0.1 * weight[None, :])
        S[:, j] = sim.max(axis=1)
    return S


def run_burst_and_diversity_ablation(
    emb_known, y_known, groups_known, emb_day4, y_day4,
    ks=(5, 50, 200, 2000), agg_sizes=(1, 5, 20, 50), n_query_files=2,
    n_seeds=5, log=print, traj=None, test_day=None,
    days_keep=None, day_decay=None,
):
    """
    Two additions to the Day-D evaluation.

    Burst decisions: pool several test slices per decision instead of scoring
    each slice independently.

    Capture-aware enrolment: draw the k enrolment labels from a controlled
    number of distinct capture sessions, holding the query captures fixed so
    that only enrolment composition varies. Labelling cost is identical
    across rows -- only the number of sessions changes.

    Zero-shot rows use Day-(D-1) embeddings as support, so no target-day
    labels are involved.
    """
    classes = np.array(sorted(set(y_known.tolist())))
    files   = np.unique(groups_known[:, 0])
    if len(files) < n_query_files + 1:
        log(f'  [ablation15] only {len(files)} capture file(s); skipped')
        return {}

    qry_files = set(files[-n_query_files:].tolist())
    pool      = [f for f in files.tolist() if f not in qry_files]
    q_mask    = np.isin(groups_known[:, 0], list(qry_files))
    emb_q, y_q, g_q = emb_known[q_mask], y_known[q_mask], groups_known[q_mask]

    out = {}
    log(f'\n  [ablation15] enrol pool={pool}  query captures={sorted(qry_files)}'
        f'  query slices={len(y_q)}')

    # -- zero-shot: support is the previous enrolment day, not the test day --
    S0 = _proto_scores(emb_day4, y_day4, emb_q, classes)
    for n in agg_sizes:
        acc, nb = _burst_accuracy(S0, y_q, g_q, classes, n,
                                  np.random.default_rng(0))
        out[f'burst_zeroshot_n{n}'] = acc
        log(f'    zero-shot  slices/decision={n:3d}  acc={acc:.4f}  bags={nb}')

    # -- few-shot with controlled enrolment-session diversity --
    for n_sf in range(1, len(pool) + 1):
        sup_files = pool[:n_sf]
        for k in ks:
            per_n = {n: [] for n in agg_sizes}
            per_n_t2r = {n: [] for n in agg_sizes}
            for sd in range(n_seeds):
                rng = np.random.default_rng(1234 + sd)
                si = []
                for c in classes:
                    idx_c = np.where(y_known == c)[0]
                    per_f = int(np.ceil(k / len(sup_files)))
                    picked = []
                    for f in sup_files:
                        pf = idx_c[groups_known[idx_c, 0] == f]
                        if len(pf) == 0:
                            continue
                        picked.extend(pf[rng.permutation(len(pf))][:per_f].tolist())
                    si.extend(picked[:k])
                si = np.array(si, dtype=np.int64)
                S = _proto_scores(emb_known[si], y_known[si], emb_q, classes)
                for n in agg_sizes:
                    a, _ = _burst_accuracy(S, y_q, g_q, classes, n,
                                           np.random.default_rng(9000 + sd))
                    per_n[n].append(a)

                # Same enrolment, scored by the full T2R trajectory rule:
                # the target-day support is appended to the trajectory and
                # the recency-weighted multi-day banks do the scoring.
                if traj is not None:
                    t2r = copy.deepcopy(traj)
                    t2r.update(day_id=test_day, emb=emb_known[si],
                               labels=y_known[si])
                    St = _t2r_scores(t2r, emb_q, classes,
                                     days_keep=days_keep, day_decay=day_decay)
                    for n in agg_sizes:
                        a, _ = _burst_accuracy(St, y_q, g_q, classes, n,
                                               np.random.default_rng(9000 + sd))
                        per_n_t2r[n].append(a)
            for n in agg_sizes:
                m, sd_ = float(np.mean(per_n[n])), float(np.std(per_n[n]))
                out[f'div{n_sf}_k{k}_n{n}_mean'] = m
                out[f'div{n_sf}_k{k}_n{n}_std']  = sd_
                if per_n_t2r[n]:
                    out[f't2r_div{n_sf}_k{k}_n{n}_mean'] = float(np.mean(per_n_t2r[n]))
                    out[f't2r_div{n_sf}_k{k}_n{n}_std']  = float(np.std(per_n_t2r[n]))
            log(f'    [proto] sessions={n_sf}  k={k:3d}  ' + '  '.join(
                f'n{n}={np.mean(per_n[n]):.4f}+-{np.std(per_n[n]):.4f}'
                for n in agg_sizes))
            if per_n_t2r[agg_sizes[0]]:
                log(f'    [T2R]   sessions={n_sf}  k={k:3d}  ' + '  '.join(
                    f'n{n}={np.mean(per_n_t2r[n]):.4f}+-{np.std(per_n_t2r[n]):.4f}'
                    for n in agg_sizes))
    return out


MINIMAL = os.environ.get('T2R_MINIMAL', '0') == '1'
# Minimal mode keeps the methods reported in the paper and skips the expensive
# diagnostic sweeps.


def run_phase4(cfg: ExperimentConfig, phase3_result: dict) -> dict:
    print('\n' + '=' * 62)
    print(f'  PHASE 4 — Open-set evaluation (Day {cfg.test_day})')
    print(f'  Known devices  : {cfg.all_known_ids}  ({len(cfg.all_known_ids)})')
    print(f'  Unknown devices: {cfg.unknown_ids}  ({len(cfg.unknown_ids)})')
    print('=' * 62)

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    results_path = os.path.join(cfg.results_dir, 'results_phase4.txt')

    def _log(line: str):
        print(line)
        with open(results_path, 'a') as f:
            print(line, file=f, flush=True)

    _log(f'\n### Phase 4  started at {time.ctime()}')

    traj      = phase3_result['trajectory']
    model     = _load_model(phase3_result['model_path'])
    threshold = phase3_result['threshold']
    NUM_CLASS = phase3_result['num_class']
    norm_mean = phase3_result['norm_mean']
    norm_std  = phase3_result['norm_std']

    assert NUM_CLASS == len(cfg.all_known_ids), (
        f"num_class mismatch: phase3 says {NUM_CLASS}, "
        f"config says {len(cfg.all_known_ids)}"
    )

    # ── Load held-out test day: ALL devices ────────────────────────────────
    print(f'\n[Phase 4] Loading Day {cfg.test_day} — all 20 devices...')
    all_devices = cfg.all_known_ids + cfg.unknown_ids

    # Capture provenance is needed for burst-level decisions and for
    # capture-aware enrolment (Ablation 15). Falls back to the plain loader
    # if groups are unavailable for this dataset layout.
    try:
        from data_utils import load_day_with_groups
        X_test_all, y_test_all, groups_test_all = load_day_with_groups(
            cfg, day_id=cfg.test_day, device_ids=all_devices,
        )
    except Exception as _e:
        print(f'  [groups] unavailable ({_e}); burst/diversity ablation skipped')
        groups_test_all = None
        X_test_all, y_test_all, _, _, _, _, _ = load_day(
            cfg,
            day_id     = cfg.test_day,
            device_ids = all_devices,
            split      = False,
            normalize  = False,
            augment    = False,
        )
    # Use Phase 3 norm stats so held-out embeddings live in the same normalized
    # space as the trajectory.
    X_test_all = load_slice_IQ.apply_normalization(X_test_all, norm_mean, norm_std)

    # Recover original device IDs (load_day remaps to 0-based)
    id_map_rev = {i: dev for i, dev in enumerate(sorted(all_devices))}
    y_test_orig  = np.array([id_map_rev[int(v)] for v in y_test_all], dtype=np.int32)

    # Split known vs unknown
    X_known, y_known_orig, X_unk, y_unk_orig = split_known_unknown(
        X_test_all, y_test_orig,
        known_ids   = cfg.all_known_ids,
        unknown_ids = cfg.unknown_ids,
    )

    # Remap known labels to 0-based
    known_id_map = {dev: i for i, dev in enumerate(sorted(cfg.all_known_ids))}
    y_known = np.array([known_id_map[int(v)] for v in y_known_orig], dtype=np.int32)

    # Same mask split_known_unknown applied, so groups line up with X_known.
    groups_known = (groups_test_all[np.isin(y_test_orig, cfg.all_known_ids)]
                    if groups_test_all is not None else None)

    print(f'  Known  : {len(X_known)} samples, {len(np.unique(y_known))} devices')
    print(f'  Unknown: {len(X_unk)} samples, {len(np.unique(y_unk_orig))} devices')

    all_results: dict = {}
    known_ids_0based  = list(range(NUM_CLASS))

    # ── Extract embeddings ────────────────────────────────────────────────
    print('\n[Phase 4] Extracting embeddings...')
    emb_known = extract_embeddings_from_model(model, X_known, BATCH_SIZE)
    emb_unk   = extract_embeddings_from_model(model, X_unk,   BATCH_SIZE)

    # ── Load non-test adaptation/calibration day (known devices only) ────
    # Loading unknown device data from traj_day for BN adaptation or
    # calibration would leak information about the same device IDs tested
    # in Phase 4 — known devices only.
    print(f'\n[Phase 4] Loading Day {cfg.traj_day} calibration/adaptation data (known only)...')
    X_adapt_all, y_adapt_all, _, _, _, _, _ = load_day(
        cfg,
        day_id     = cfg.traj_day,
        device_ids = cfg.all_known_ids,
        split      = False,
        normalize  = False,
        augment    = False,
    )
    X_adapt_all = load_slice_IQ.apply_normalization(
        X_adapt_all, norm_mean, norm_std
    )
    known_id_map_adapt = {dev: i for i, dev in enumerate(sorted(cfg.all_known_ids))}
    y_adapt_orig = np.array(
        [known_id_map_adapt[int(v)] for v in y_adapt_all], dtype=np.int32
    )
    X_adapt_known  = X_adapt_all
    y_adapt_known  = y_adapt_orig
    X_adapt_unk    = np.empty((0, X_adapt_all.shape[1], X_adapt_all.shape[2]),
                              dtype=X_adapt_all.dtype)

    # ── Ablation 1: Softmax baseline ──────────────────────────────────────
    print('\n[Phase 4] Ablation 1: softmax baseline (no rejection)...')
    softmax_acc = _softmax_accuracy(model, X_known, y_known, NUM_CLASS)
    all_results['softmax_acc'] = softmax_acc
    _log(f'\n  softmax_acc={softmax_acc:.4f}')

    # ── Ablation 2: TTA-BN + softmax ─────────────────────────────────────
    print('\n[Phase 4] Ablation 2: TTA-BN + softmax...')
    original_bn = tta_bn.save_bn_statistics(model)
    tta_bn.adapt_bn_statistics(model, X_adapt_all, batch_size=BATCH_SIZE, n_passes=2)
    softmax_bn_acc = _softmax_accuracy(model, X_known, y_known, NUM_CLASS)
    tta_bn.reset_bn_statistics(model, original_bn)
    all_results['softmax_bn_acc'] = softmax_bn_acc
    _log(f'  softmax_bn_acc={softmax_bn_acc:.4f}  '
         f'delta={softmax_bn_acc - softmax_acc:+.4f}')

    # Trajectory was built in Phase 3 without TTA-BN, so keep embeddings
    # from the original model (already extracted above) for all
    # trajectory/prototype evaluations ─ consistent embedding space.

    # Extract embeddings from BN-adapted model for trajectory ablation.
    # These share the same BN statistics as the softmax_bn_acc evaluation.
    print('\n[Phase 4] Extracting BN-adapted embeddings for trajectory ablation...')
    original_bn_2 = tta_bn.save_bn_statistics(model)
    tta_bn.adapt_bn_statistics(model, X_adapt_all, batch_size=BATCH_SIZE, n_passes=2)
    emb_adapt_known_bn = extract_embeddings_from_model(
        model, X_adapt_known, BATCH_SIZE
    )
    emb_adapt_unk_bn = extract_embeddings_from_model(
        model, X_adapt_unk, BATCH_SIZE
    )
    emb_known_bn = extract_embeddings_from_model(model, X_known, BATCH_SIZE)
    emb_unk_bn   = extract_embeddings_from_model(model, X_unk,   BATCH_SIZE)
    tta_bn.reset_bn_statistics(model, original_bn_2)
    print(f'  BN-adapted emb_known={emb_known_bn.shape}  emb_unk={emb_unk_bn.shape}')

    # ── Ablation 3: Static prototype with proper threshold sweep ──────────
    # FIX: was using Day 7 and an unprincipled `threshold * 0.3`.
    # Uses cfg.traj_day and sweeps its own threshold.
    print(f'\n[Phase 4] Ablation 3: static prototype (Day {cfg.traj_day}, threshold sweep)...')

    # Reference embeddings come from cfg.traj_day -- the last day before
    # the test day. Every baseline's support set, calibration split and
    # pseudo-unknowns are drawn from these, so no test-day data reaches
    # threshold selection.
    X_trajday_ref, y_trajday_ref, _, _, _, _, _ = load_day(
        cfg,
        day_id     = cfg.traj_day,
        device_ids = cfg.all_known_ids,
        split      = False,
        normalize  = False,
        augment    = False,
    )
    X_trajday_ref = load_slice_IQ.apply_normalization(X_trajday_ref, norm_mean, norm_std)
    emb_trajday   = extract_embeddings_from_model(model, X_trajday_ref, BATCH_SIZE)

    # Remap traj_day labels to 0-based
    d8_id_map = {dev: i for i, dev in enumerate(sorted(cfg.all_known_ids))}
    y_trajday_0    = np.array([d8_id_map[int(v)] for v in y_trajday_ref], dtype=np.int32)

    # Pseudo-unknown calibration: use the hardest-to-classify known samples
    # (farthest from their device centroid) as a proxy for unknowns.
    # Uses only known-device data — no leakage from cfg.unknown_ids.
    _centroids = np.vstack([
        emb_trajday[y_trajday_0 == c].mean(axis=0) for c in range(len(cfg.all_known_ids))
    ])
    _centroids /= (np.linalg.norm(_centroids, axis=1, keepdims=True) + 1e-8)
    _emb_n = emb_trajday / (np.linalg.norm(emb_trajday, axis=1, keepdims=True) + 1e-8)
    _within_dist = 1.0 - (_emb_n @ _centroids.T).max(axis=1)
    _n_pseudo = max(50, len(_within_dist) // 4)
    _pseudo_idx = np.argsort(_within_dist)[-_n_pseudo:]
    emb_static_unk_calib = emb_trajday[_pseudo_idx]
    print(f'  Pseudo-unknown calib: {emb_static_unk_calib.shape} '
          f'(hardest known, dist>={_within_dist[_pseudo_idx[0]]:.3f})')
    trajday_support_idx, trajday_calib_idx = _stratified_support_calib_split(
        y_trajday_0, cfg.seed + 17, support_frac=0.5
    )
    emb_trajday_support = emb_trajday[trajday_support_idx]
    y_trajday_support = y_trajday_0[trajday_support_idx]
    emb_trajday_calib = emb_trajday[trajday_calib_idx]
    y_d7_calib = y_trajday_0[trajday_calib_idx]
    print(
        f'  Day {cfg.traj_day} support={emb_trajday_support.shape}  '
        f'calibration={emb_trajday_calib.shape}'
    )

    static_res = _static_prototype_eval(
        emb_support = emb_trajday_support,
        y_support   = y_trajday_support,
        emb_known   = emb_known,
        y_known     = y_known,
        emb_unknown = emb_unk,
        known_ids   = known_ids_0based,
        emb_calib_known   = emb_trajday_calib,
        y_calib_known     = y_d7_calib,
        emb_calib_unknown = emb_static_unk_calib,
    )
    all_results['static_closed_acc'] = static_res['closed_acc']
    all_results['static_open_acc']   = static_res['open_acc']
    all_results['static_unk_det']    = static_res['unk_det']
    all_results['static_auroc']      = static_res['auroc']
    all_results['static_threshold']  = static_res['best_threshold']
    _log(f'  static_prototype  closed={static_res["closed_acc"]:.4f}  '
         f'open={static_res["open_acc"]:.4f}  '
         f'unk_det={static_res["unk_det"]:.4f}  '
         f'auroc={static_res["auroc"]:.4f}  '
         f'thr={static_res["best_threshold"]:.3f}')

    # ── Ablation 4: Day-7 calibrated linear probe ───────────────────────
    print(f'\n[Phase 4] Ablation 4: calibrated linear probe (Day {cfg.traj_day})...')

    probe_res = _calibrated_linear_probe_eval(
        emb_calib_known   = emb_trajday_calib,
        y_calib_known     = y_d7_calib,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
        emb_train_known   = emb_trajday_support,
        y_train_known     = y_trajday_support,
    )
    all_results['probe_closed_acc'] = probe_res['closed_acc']
    all_results['probe_open_acc']   = probe_res['open_acc']
    all_results['probe_unk_det']    = probe_res['unk_det']
    all_results['probe_auroc']      = probe_res['auroc']
    all_results['probe_f1_macro']   = probe_res['f1_macro']
    all_results['probe_accept_rate'] = probe_res['known_accept_rate']
    all_results['probe_threshold']  = probe_res['best_threshold']
    _log(f'  calibrated_probe  closed={probe_res["closed_acc"]:.4f}  '
         f'open={probe_res["open_acc"]:.4f}  '
         f'unk_det={probe_res["unk_det"]:.4f}  '
         f'auroc={probe_res["auroc"]:.4f}  '
         f'thr={probe_res["best_threshold"]:.3f}')

    print(f'\n[Phase 4] Ablation 5: L2 calibrated probe (Day {cfg.traj_day})...')

    l2_probe_res = _calibrated_linear_probe_eval(
        emb_calib_known   = emb_trajday_calib,
        y_calib_known     = y_d7_calib,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
        normalize         = True,
        c_values          = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
        emb_train_known   = emb_trajday_support,
        y_train_known     = y_trajday_support,
    )
    all_results['l2_probe_closed_acc'] = l2_probe_res['closed_acc']
    all_results['l2_probe_open_acc']   = l2_probe_res['open_acc']
    all_results['l2_probe_unk_det']    = l2_probe_res['unk_det']
    all_results['l2_probe_auroc']      = l2_probe_res['auroc']
    all_results['l2_probe_f1_macro']   = l2_probe_res['f1_macro']
    all_results['l2_probe_accept_rate'] = l2_probe_res['known_accept_rate']
    all_results['l2_probe_threshold']  = l2_probe_res['best_threshold']
    all_results['l2_probe_C']          = l2_probe_res['C']
    _log(f'  l2_calibrated_probe  closed={l2_probe_res["closed_acc"]:.4f}  '
         f'open={l2_probe_res["open_acc"]:.4f}  '
         f'unk_det={l2_probe_res["unk_det"]:.4f}  '
         f'auroc={l2_probe_res["auroc"]:.4f}  '
         f'thr={l2_probe_res["best_threshold"]:.3f}  '
         f'C={l2_probe_res["C"]:.3g}')

    print(f'\n[Phase 4] Ablation 6: class-wise L2 calibrated probe (Day {cfg.traj_day})...')

    class_l2_probe_res = _l2_probe_classwise_eval(
        emb_calib_known   = emb_trajday,
        y_calib_known     = y_trajday_0,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
    )
    all_results['class_l2_probe_closed_acc'] = class_l2_probe_res['closed_acc']
    all_results['class_l2_probe_open_acc']   = class_l2_probe_res['open_acc']
    all_results['class_l2_probe_unk_det']    = class_l2_probe_res['unk_det']
    all_results['class_l2_probe_auroc']      = class_l2_probe_res['auroc']
    all_results['class_l2_probe_f1_macro']   = class_l2_probe_res['f1_macro']
    all_results['class_l2_probe_accept_rate'] = class_l2_probe_res['known_accept_rate']
    all_results['class_l2_probe_threshold']  = class_l2_probe_res['best_threshold']
    _log(f'  classwise_l2_probe  closed={class_l2_probe_res["closed_acc"]:.4f}  '
         f'open={class_l2_probe_res["open_acc"]:.4f}  '
         f'unk_det={class_l2_probe_res["unk_det"]:.4f}  '
         f'auroc={class_l2_probe_res["auroc"]:.4f}  '
         f'mean_thr={class_l2_probe_res["best_threshold"]:.3f}')

    print(f'\n[Phase 4] Ablation 7: temperature L2 calibrated probe (Day {cfg.traj_day})...')

    temp_l2_probe_res = _temperature_l2_probe_eval(
        emb_calib_known   = emb_trajday,
        y_calib_known     = y_trajday_0,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
    )
    all_results['temp_l2_probe_closed_acc'] = temp_l2_probe_res['closed_acc']
    all_results['temp_l2_probe_open_acc']   = temp_l2_probe_res['open_acc']
    all_results['temp_l2_probe_unk_det']    = temp_l2_probe_res['unk_det']
    all_results['temp_l2_probe_auroc']      = temp_l2_probe_res['auroc']
    all_results['temp_l2_probe_f1_macro']   = temp_l2_probe_res['f1_macro']
    all_results['temp_l2_probe_accept_rate'] = temp_l2_probe_res['known_accept_rate']
    all_results['temp_l2_probe_threshold']  = temp_l2_probe_res['best_threshold']
    all_results['temp_l2_probe_temperature'] = temp_l2_probe_res['temperature']
    _log(f'  temp_l2_probe  closed={temp_l2_probe_res["closed_acc"]:.4f}  '
         f'open={temp_l2_probe_res["open_acc"]:.4f}  '
         f'unk_det={temp_l2_probe_res["unk_det"]:.4f}  '
         f'auroc={temp_l2_probe_res["auroc"]:.4f}  '
         f'thr={temp_l2_probe_res["best_threshold"]:.3f}  '
         f'T={temp_l2_probe_res["temperature"]:.2f}')

    print(f'\n[Phase 4] Ablation 8: cosine KNN probe (Day {cfg.traj_day})...')

    knn_res = _cosine_knn_eval(
        emb_calib_known   = emb_trajday_calib,
        y_calib_known     = y_d7_calib,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
        seed              = cfg.seed,
        emb_support_known = emb_trajday_support,
        y_support_known   = y_trajday_support,
    )
    all_results['knn_closed_acc'] = knn_res['closed_acc']
    all_results['knn_open_acc']   = knn_res['open_acc']
    all_results['knn_unk_det']    = knn_res['unk_det']
    all_results['knn_auroc']      = knn_res['auroc']
    all_results['knn_f1_macro']   = knn_res['f1_macro']
    all_results['knn_accept_rate'] = knn_res['known_accept_rate']
    all_results['knn_threshold']  = knn_res['best_threshold']
    all_results['knn_k']          = knn_res['k']
    all_results['knn_support_per_class'] = knn_res['support_per_class']
    all_results['knn_weighted']    = knn_res['weighted']
    class_knn_res = knn_res['classwise']
    all_results['class_knn_closed_acc'] = class_knn_res['closed_acc']
    all_results['class_knn_open_acc']   = class_knn_res['open_acc']
    all_results['class_knn_unk_det']    = class_knn_res['unk_det']
    all_results['class_knn_auroc']      = class_knn_res['auroc']
    all_results['class_knn_f1_macro']   = class_knn_res['f1_macro']
    all_results['class_knn_accept_rate'] = class_knn_res['known_accept_rate']
    all_results['class_knn_threshold']  = class_knn_res['best_threshold']
    constrained_knn = knn_res['accept_constrained']
    for target_accept, constrained_res in constrained_knn.items():
        pct = int(round(target_accept * 100))
        prefix = f'knn_accept{pct}'
        all_results[f'{prefix}_closed_acc'] = constrained_res['closed_acc']
        all_results[f'{prefix}_open_acc'] = constrained_res['open_acc']
        all_results[f'{prefix}_unk_det'] = constrained_res['unk_det']
        all_results[f'{prefix}_auroc'] = constrained_res['auroc']
        all_results[f'{prefix}_f1_macro'] = constrained_res['f1_macro']
        all_results[f'{prefix}_accept_rate'] = constrained_res['known_accept_rate']
        all_results[f'{prefix}_threshold'] = constrained_res['best_threshold']
    _log(f'  cosine_knn  closed={knn_res["closed_acc"]:.4f}  '
         f'open={knn_res["open_acc"]:.4f}  '
         f'unk_det={knn_res["unk_det"]:.4f}  '
         f'auroc={knn_res["auroc"]:.4f}  '
         f'thr={knn_res["best_threshold"]:.3f}  '
         f'k={knn_res["k"]}  '
         f'support={knn_res["support_per_class"]}/class  '
         f'weighted={knn_res["weighted"]}')
    _log(f'  classwise_cosine_knn  closed={class_knn_res["closed_acc"]:.4f}  '
         f'open={class_knn_res["open_acc"]:.4f}  '
         f'unk_det={class_knn_res["unk_det"]:.4f}  '
         f'auroc={class_knn_res["auroc"]:.4f}  '
         f'mean_thr={class_knn_res["best_threshold"]:.3f}')
    for target_accept, constrained_res in constrained_knn.items():
        _log(f'  cosine_knn_at_{int(round(target_accept * 100))}pct_accept  '
             f'closed={constrained_res["closed_acc"]:.4f}  '
             f'open={constrained_res["open_acc"]:.4f}  '
             f'unk_det={constrained_res["unk_det"]:.4f}  '
             f'auroc={constrained_res["auroc"]:.4f}  '
             f'thr={constrained_res["best_threshold"]:.3f}  '
             f'accept={constrained_res["known_accept_rate"]:.4f}')

    # ── Ablation 8b: Trajectory drift-corrected cosine KNN ────────────────
    # Shift the Day-traj gallery forward to the test day along each device's
    # observed drift, then KNN. Uses the trajectory to fix temporal shift while
    # keeping the full gallery (no prototype averaging).
    print(f'\n[Phase 4] Ablation 8b: drift-corrected cosine KNN '
          f'(Day {cfg.traj_day} → Day {cfg.test_day})...')
    drift_knn_res = _drift_corrected_knn_eval(
        traj              = traj,
        target_day        = cfg.test_day,
        emb_calib_known   = emb_trajday_calib,
        y_calib_known     = y_d7_calib,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
        seed              = cfg.seed,
        emb_support_known = emb_trajday_support,
        y_support_known   = y_trajday_support,
    )
    all_results['drift_knn_closed_acc'] = drift_knn_res['closed_acc']
    all_results['drift_knn_open_acc']   = drift_knn_res['open_acc']
    all_results['drift_knn_unk_det']    = drift_knn_res['unk_det']
    all_results['drift_knn_auroc']      = drift_knn_res['auroc']
    all_results['drift_knn_f1_macro']   = drift_knn_res['f1_macro']
    all_results['drift_knn_accept_rate'] = drift_knn_res['known_accept_rate']
    all_results['drift_knn_threshold']  = drift_knn_res['best_threshold']
    drift_class_knn = drift_knn_res['classwise']
    all_results['drift_class_knn_closed_acc'] = drift_class_knn['closed_acc']
    all_results['drift_class_knn_open_acc']   = drift_class_knn['open_acc']
    all_results['drift_class_knn_unk_det']    = drift_class_knn['unk_det']
    all_results['drift_class_knn_auroc']      = drift_class_knn['auroc']
    _log(f'  drift_corrected_cosine_knn  closed={drift_knn_res["closed_acc"]:.4f}  '
         f'open={drift_knn_res["open_acc"]:.4f}  '
         f'unk_det={drift_knn_res["unk_det"]:.4f}  '
         f'auroc={drift_knn_res["auroc"]:.4f}  '
         f'thr={drift_knn_res["best_threshold"]:.3f}  '
         f'k={drift_knn_res["k"]}  '
         f'support={drift_knn_res["support_per_class"]}/class')
    _log(f'  drift_corrected_classwise_knn  closed={drift_class_knn["closed_acc"]:.4f}  '
         f'open={drift_class_knn["open_acc"]:.4f}  '
         f'unk_det={drift_class_knn["unk_det"]:.4f}  '
         f'auroc={drift_class_knn["auroc"]:.4f}')

    print(f'\n[Phase 4] Ablation 9: calibrated LDA probe (Day {cfg.traj_day})...')

    lda_res = _calibrated_lda_probe_eval(
        emb_calib_known   = emb_trajday,
        y_calib_known     = y_trajday_0,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
    )
    all_results['lda_closed_acc'] = lda_res['closed_acc']
    all_results['lda_open_acc']   = lda_res['open_acc']
    all_results['lda_unk_det']    = lda_res['unk_det']
    all_results['lda_auroc']      = lda_res['auroc']
    all_results['lda_f1_macro']   = lda_res['f1_macro']
    all_results['lda_accept_rate'] = lda_res['known_accept_rate']
    all_results['lda_threshold']  = lda_res['best_threshold']
    _log(f'  calibrated_lda  closed={lda_res["closed_acc"]:.4f}  '
         f'open={lda_res["open_acc"]:.4f}  '
         f'unk_det={lda_res["unk_det"]:.4f}  '
         f'auroc={lda_res["auroc"]:.4f}  '
         f'thr={lda_res["best_threshold"]:.3f}')

    print(f'\n[Phase 4] Ablation 10: L2 calibrated LDA probe (Day {cfg.traj_day})...')

    l2_lda_res = _calibrated_lda_probe_eval(
        emb_calib_known   = emb_trajday,
        y_calib_known     = y_trajday_0,
        emb_calib_unknown = emb_static_unk_calib,
        emb_known         = emb_known,
        y_known           = y_known,
        emb_unknown       = emb_unk,
        normalize         = True,
    )
    all_results['l2_lda_closed_acc'] = l2_lda_res['closed_acc']
    all_results['l2_lda_open_acc']   = l2_lda_res['open_acc']
    all_results['l2_lda_unk_det']    = l2_lda_res['unk_det']
    all_results['l2_lda_auroc']      = l2_lda_res['auroc']
    all_results['l2_lda_f1_macro']   = l2_lda_res['f1_macro']
    all_results['l2_lda_accept_rate'] = l2_lda_res['known_accept_rate']
    all_results['l2_lda_threshold']  = l2_lda_res['best_threshold']
    _log(f'  l2_calibrated_lda  closed={l2_lda_res["closed_acc"]:.4f}  '
         f'open={l2_lda_res["open_acc"]:.4f}  '
         f'unk_det={l2_lda_res["unk_det"]:.4f}  '
         f'auroc={l2_lda_res["auroc"]:.4f}  '
         f'thr={l2_lda_res["best_threshold"]:.3f}')

    print(f'\n[Phase 4] Ablation 11: probe/trajectory fusion through Day {cfg.traj_day}...')

    fusion_res = _probe_trajectory_fusion_eval(
        traj                = traj,
        emb_calib_known     = emb_trajday,
        y_calib_known       = y_trajday_0,
        emb_calib_unknown   = emb_static_unk_calib,
        emb_known           = emb_known,
        y_known             = y_known,
        emb_unknown         = emb_unk,
        known_ids           = known_ids_0based,
    )
    all_results['fusion_closed_acc'] = fusion_res['closed_acc']
    all_results['fusion_open_acc']   = fusion_res['open_acc']
    all_results['fusion_unk_det']    = fusion_res['unk_det']
    all_results['fusion_auroc']      = fusion_res['auroc']
    all_results['fusion_f1_macro']   = fusion_res['f1_macro']
    all_results['fusion_accept_rate'] = fusion_res['known_accept_rate']
    all_results['fusion_threshold']  = fusion_res['best_threshold']
    all_results['fusion_lambda']     = fusion_res['lambda']
    _log(f'  probe_trajectory_fusion  closed={fusion_res["closed_acc"]:.4f}  '
         f'open={fusion_res["open_acc"]:.4f}  '
         f'unk_det={fusion_res["unk_det"]:.4f}  '
         f'auroc={fusion_res["auroc"]:.4f}  '
         f'thr={fusion_res["best_threshold"]:.3f}  '
         f'lambda={fusion_res["lambda"]:.2f}')

    # ── Ablation 11b: Temperature-scaled classifier ──────────────────────
    print(f'\n[Phase 4] Ablation 11b: temperature scaling (Day {cfg.traj_day})...')
    ts_res = _temperature_scaling_eval(
        model              = model,
        emb_calib_known    = emb_trajday,
        y_calib_known      = y_trajday_0,
        emb_calib_unknown  = emb_static_unk_calib,
        emb_known          = emb_known,
        y_known            = y_known,
        emb_unknown        = emb_unk,
    )
    all_results['ts_closed_acc']   = ts_res['closed_acc']
    all_results['ts_open_acc']     = ts_res['open_acc']
    all_results['ts_unk_det']      = ts_res['unk_det']
    all_results['ts_auroc']        = ts_res['auroc']
    all_results['ts_f1_macro']     = ts_res['f1_macro']
    all_results['ts_accept_rate']  = ts_res['known_accept_rate']
    all_results['ts_threshold']    = ts_res['best_threshold']
    all_results['ts_temperature']  = ts_res['temperature']
    _log(f'  temp_scaling  closed={ts_res["closed_acc"]:.4f}  '
         f'open={ts_res["open_acc"]:.4f}  '
         f'unk_det={ts_res["unk_det"]:.4f}  '
         f'auroc={ts_res["auroc"]:.4f}  '
         f'T={ts_res["temperature"]:.3f}  '
         f'thr={ts_res["best_threshold"]:.3f}')

    # ── Ablation 12: Cosine trajectory using latest EWMA means ───────────
    print(f'\n[Phase 4] Ablation 12: cosine trajectory through Day {cfg.traj_day}...')

    cosine_traj_res = _cosine_trajectory_eval(
        traj                = traj,
        emb_calib_known     = emb_trajday,
        y_calib_known       = y_trajday_0,
        emb_calib_unknown   = emb_static_unk_calib,
        emb_known           = emb_known,
        y_known             = y_known,
        emb_unknown         = emb_unk,
        known_ids           = known_ids_0based,
    )
    all_results['cos_traj_closed_acc'] = cosine_traj_res['closed_acc']
    all_results['cos_traj_open_acc']   = cosine_traj_res['open_acc']
    all_results['cos_traj_unk_det']    = cosine_traj_res['unk_det']
    all_results['cos_traj_auroc']      = cosine_traj_res['auroc']
    all_results['cos_traj_f1_macro']   = cosine_traj_res['f1_macro']
    all_results['cos_traj_accept_rate'] = cosine_traj_res['known_accept_rate']
    all_results['cos_traj_threshold']  = cosine_traj_res['best_threshold']
    _log(f'  cosine_trajectory  closed={cosine_traj_res["closed_acc"]:.4f}  '
         f'open={cosine_traj_res["open_acc"]:.4f}  '
         f'unk_det={cosine_traj_res["unk_det"]:.4f}  '
         f'auroc={cosine_traj_res["auroc"]:.4f}  '
         f'thr={cosine_traj_res["best_threshold"]:.3f}')

    # ── Ablation 12b: Multi-prototype cosine trajectory ───────────────────
    print(f'\n[Phase 4] Ablation 12b: multi-prototype cosine trajectory through Day {cfg.traj_day}...')

    mp_cos_traj_res = _multiprototype_cosine_trajectory_eval(
        traj                = traj,
        emb_calib_known     = emb_trajday,
        y_calib_known       = y_trajday_0,
        emb_calib_unknown   = emb_static_unk_calib,
        emb_known           = emb_known,
        y_known             = y_known,
        emb_unknown         = emb_unk,
        known_ids           = known_ids_0based,
    )
    all_results['mp_cos_traj_closed_acc'] = mp_cos_traj_res['closed_acc']
    all_results['mp_cos_traj_open_acc']   = mp_cos_traj_res['open_acc']
    all_results['mp_cos_traj_unk_det']    = mp_cos_traj_res['unk_det']
    all_results['mp_cos_traj_auroc']      = mp_cos_traj_res['auroc']
    all_results['mp_cos_traj_f1_macro']   = mp_cos_traj_res['f1_macro']
    all_results['mp_cos_traj_accept_rate'] = mp_cos_traj_res['known_accept_rate']
    all_results['mp_cos_traj_threshold']  = mp_cos_traj_res['best_threshold']
    all_results['mp_cos_traj_days_keep']  = mp_cos_traj_res['days_keep']
    all_results['mp_cos_traj_day_decay']  = mp_cos_traj_res['day_decay']
    all_results['mp_cos_traj_calib_score'] = mp_cos_traj_res['calib_score']
    _log(f'  multiprototype_cosine_trajectory  closed={mp_cos_traj_res["closed_acc"]:.4f}  '
         f'open={mp_cos_traj_res["open_acc"]:.4f}  '
         f'unk_det={mp_cos_traj_res["unk_det"]:.4f}  '
         f'auroc={mp_cos_traj_res["auroc"]:.4f}  '
         f'thr={mp_cos_traj_res["best_threshold"]:.3f}  '
         f'days={mp_cos_traj_res["days_keep"]}  '
         f'decay={mp_cos_traj_res["day_decay"]:.2f}  '
         f'align_drift={mp_cos_traj_res["align_drift"]}')

    # ── Ablation 12b-w: Multi-prototype cosine trajectory + mean-shift whitening ──
    # Unsupervised test-time adaptation: align the Day-8 embedding cloud to the
    # Day-7 (trajectory) frame by removing the global centroid shift, estimated
    # from UNLABELLED Day-8 features only (no test labels → no leakage). The
    # trajectory prototypes stay in the Day-7 frame; only the test queries move.
    print('\n[Phase 4] Ablation 12b-w: multi-proto cosine trajectory + mean-shift whitening...')
    from embedding_whitening import mean_shift_whitening
    src_d7 = emb_trajday  # Day-7 reference cloud (trajectory frame)
    tgt_d8_pool = np.concatenate([emb_known, emb_unk], axis=0)  # unlabelled Day-8
    shift_w = tgt_d8_pool.mean(axis=0) - src_d7.mean(axis=0)
    emb_known_w = _l2_normalize(emb_known - shift_w)
    emb_unk_w   = _l2_normalize(emb_unk - shift_w)
    mp_cos_traj_w_res = _multiprototype_cosine_trajectory_eval(
        traj                = traj,
        emb_calib_known     = emb_trajday,
        y_calib_known       = y_trajday_0,
        emb_calib_unknown   = emb_static_unk_calib,
        emb_known           = emb_known_w,
        y_known             = y_known,
        emb_unknown         = emb_unk_w,
        known_ids           = known_ids_0based,
    )
    all_results['mp_cos_traj_w_closed_acc'] = mp_cos_traj_w_res['closed_acc']
    all_results['mp_cos_traj_w_open_acc']   = mp_cos_traj_w_res['open_acc']
    all_results['mp_cos_traj_w_unk_det']    = mp_cos_traj_w_res['unk_det']
    all_results['mp_cos_traj_w_auroc']      = mp_cos_traj_w_res['auroc']
    _log(f'  multiprototype_cosine_trajectory_whitened  closed={mp_cos_traj_w_res["closed_acc"]:.4f}  '
         f'open={mp_cos_traj_w_res["open_acc"]:.4f}  '
         f'unk_det={mp_cos_traj_w_res["unk_det"]:.4f}  '
         f'auroc={mp_cos_traj_w_res["auroc"]:.4f}  '
         f'days={mp_cos_traj_w_res["days_keep"]}  '
         f'shift_mag={float(np.linalg.norm(shift_w)):.4f}')

    # ── Ablation 12c: Class-wise multi-prototype cosine trajectory ───────
    print(f'\n[Phase 4] Ablation 12c: class-wise multi-prototype cosine trajectory...')

    class_mp_cos_traj_res = _classwise_multiprototype_cosine_trajectory_eval(
        traj                = traj,
        emb_calib_known     = emb_trajday,
        y_calib_known       = y_trajday_0,
        emb_calib_unknown   = emb_static_unk_calib,
        emb_known           = emb_known,
        y_known             = y_known,
        emb_unknown         = emb_unk,
        known_ids           = known_ids_0based,
        days_keep           = mp_cos_traj_res['days_keep'],
        day_decay           = mp_cos_traj_res['day_decay'],
    )
    all_results['class_mp_cos_traj_closed_acc'] = class_mp_cos_traj_res['closed_acc']
    all_results['class_mp_cos_traj_open_acc']   = class_mp_cos_traj_res['open_acc']
    all_results['class_mp_cos_traj_unk_det']    = class_mp_cos_traj_res['unk_det']
    all_results['class_mp_cos_traj_auroc']      = class_mp_cos_traj_res['auroc']
    all_results['class_mp_cos_traj_f1_macro']   = class_mp_cos_traj_res['f1_macro']
    all_results['class_mp_cos_traj_accept_rate'] = class_mp_cos_traj_res['known_accept_rate']
    all_results['class_mp_cos_traj_threshold']  = class_mp_cos_traj_res['best_threshold']
    all_results['class_mp_cos_traj_days_keep']  = class_mp_cos_traj_res['days_keep']
    all_results['class_mp_cos_traj_day_decay']  = class_mp_cos_traj_res['day_decay']
    all_results['class_mp_cos_traj_calib_score'] = class_mp_cos_traj_res['calib_score']
    all_results['class_mp_cos_traj_calib_min_accept_rate'] = class_mp_cos_traj_res['calib_min_accept_rate']
    all_results['class_mp_cos_traj_calib_weights'] = class_mp_cos_traj_res['calib_weights']
    _log(f'  classwise_multiprototype_cosine_trajectory  '
         f'closed={class_mp_cos_traj_res["closed_acc"]:.4f}  '
         f'open={class_mp_cos_traj_res["open_acc"]:.4f}  '
         f'unk_det={class_mp_cos_traj_res["unk_det"]:.4f}  '
         f'auroc={class_mp_cos_traj_res["auroc"]:.4f}  '
         f'mean_thr={class_mp_cos_traj_res["best_threshold"]:.3f}  '
         f'accept={class_mp_cos_traj_res["known_accept_rate"]:.4f}  '
         f'calib_min_accept={class_mp_cos_traj_res["calib_min_accept_rate"]:.2f}  '
         f'calib_score={class_mp_cos_traj_res["calib_score"]:.4f}')

    # ── Ablation 12d: Target-accept class-wise trajectory ────────────────
    print(f'\n[Phase 4] Ablation 12d: class-wise multi-prototype trajectory '
          f'at target accept rates...')
    for target_accept in [0.20, 0.35, 0.50]:
        target_res = _target_accept_classwise_multiprototype_cosine_trajectory_eval(
            traj                = traj,
            emb_calib_known     = emb_trajday,
            y_calib_known       = y_trajday_0,
            emb_calib_unknown   = emb_static_unk_calib,
            emb_known           = emb_known,
            y_known             = y_known,
            emb_unknown         = emb_unk,
            known_ids           = known_ids_0based,
            target_accept       = target_accept,
        )
        pct = int(round(target_accept * 100))
        prefix = f'target{pct}_class_mp_cos_traj'
        all_results[f'{prefix}_closed_acc'] = target_res['closed_acc']
        all_results[f'{prefix}_open_acc'] = target_res['open_acc']
        all_results[f'{prefix}_unk_det'] = target_res['unk_det']
        all_results[f'{prefix}_auroc'] = target_res['auroc']
        all_results[f'{prefix}_f1_macro'] = target_res['f1_macro']
        all_results[f'{prefix}_accept_rate'] = target_res['known_accept_rate']
        all_results[f'{prefix}_threshold'] = target_res['best_threshold']
        all_results[f'{prefix}_days_keep'] = target_res['days_keep']
        all_results[f'{prefix}_day_decay'] = target_res['day_decay']
        all_results[f'{prefix}_calib_score'] = target_res['calib_score']
        _log(f'  classwise_multiprototype_traj_at_{pct}pct_accept  '
             f'closed={target_res["closed_acc"]:.4f}  '
             f'open={target_res["open_acc"]:.4f}  '
             f'unk_det={target_res["unk_det"]:.4f}  '
             f'auroc={target_res["auroc"]:.4f}  '
             f'thr={target_res["best_threshold"]:.3f}  '
             f'accept={target_res["known_accept_rate"]:.4f}  '
             f'days={target_res["days_keep"]}  '
             f'decay={target_res["day_decay"]:.2f}')

    # ── Ablation 13: Full trajectory through cfg.traj_day ────────────────
    print(f'\n[Phase 4] Ablation 13: trajectory through Day {cfg.traj_day} '
          f'(threshold={threshold:.3f})...')

    if phase3_result.get('trajectory_classifier') == 'cosine_prototypes':
        traj_results = traj.evaluate_cosine_prototypes(
            emb_known   = emb_known,
            y_known     = y_known,
            known_ids   = known_ids_0based,
            emb_unknown = emb_unk,
            threshold   = threshold,
            days_keep   = phase3_result.get('history_days_keep'),
            day_decay   = phase3_result.get('history_day_decay', 1.0),
            verbose     = True,
        )
    elif phase3_result.get('trajectory_classifier') == 'history_mahalanobis':
        traj_results = traj.evaluate_history(
            emb_known   = emb_known,
            y_known     = y_known,
            known_ids   = known_ids_0based,
            emb_unknown = emb_unk,
            threshold   = threshold,
            days_keep   = phase3_result.get('history_days_keep'),
            day_decay   = phase3_result.get('history_day_decay', 1.0),
            verbose     = True,
        )
    else:
        traj_results = traj.evaluate(
            emb_known   = emb_known,
            y_known     = y_known,
            known_ids   = known_ids_0based,
            emb_unknown = emb_unk,
            threshold   = threshold,
            verbose     = True,
        )

    all_results.update({
        'traj_closed_acc':  traj_results['closed_acc'],
        'traj_open_acc':    traj_results['open_known_acc'],
        'traj_unk_det':     traj_results['open_unk_det'],
        'traj_auroc':       traj_results['auroc'],
        'traj_f1_macro':    traj_results['f1_macro'],
        'traj_accept_rate': traj_results['known_accept_rate'],
        'traj_threshold':   threshold,
    })

    # ── Ablation 13b: Trajectory with BN-adapted embeddings ───────────────
    print(f'\n[Phase 4] Ablation 13b: trajectory with BN-adapted embeddings...')
    _unk_bn = emb_adapt_unk_bn if _has_calib_unknown(emb_adapt_unk_bn) else None
    if phase3_result.get('trajectory_classifier') == 'cosine_prototypes':
        sweep_bn = trajectory_cosine_threshold_sweep(
            traj        = traj,
            emb_known   = emb_adapt_known_bn,
            y_known     = y_adapt_known,
            emb_unknown = _unk_bn,
            known_ids   = known_ids_0based,
            days_keep   = phase3_result.get('history_days_keep'),
            day_decay   = phase3_result.get('history_day_decay', 1.0),
        )
    elif phase3_result.get('trajectory_classifier') == 'history_mahalanobis':
        sweep_bn = trajectory_history_threshold_sweep(
            traj        = traj,
            emb_known   = emb_adapt_known_bn,
            y_known     = y_adapt_known,
            emb_unknown = _unk_bn,
            known_ids   = known_ids_0based,
            days_keep   = phase3_result.get('history_days_keep'),
            day_decay   = phase3_result.get('history_day_decay', 1.0),
        )
    else:
        sweep_bn = trajectory_threshold_sweep(
            traj        = traj,
            emb_known   = emb_adapt_known_bn,
            y_known     = y_adapt_known,
            emb_unknown = _unk_bn,
            known_ids   = known_ids_0based,
        )
    thr_bn = sweep_bn['best_threshold']
    if phase3_result.get('trajectory_classifier') == 'cosine_prototypes':
        traj_bn_res = traj.evaluate_cosine_prototypes(
            emb_known   = emb_known_bn,
            y_known     = y_known,
            known_ids   = known_ids_0based,
            emb_unknown = emb_unk_bn,
            threshold   = thr_bn,
            days_keep   = phase3_result.get('history_days_keep'),
            day_decay   = phase3_result.get('history_day_decay', 1.0),
            verbose     = False,
        )
    elif phase3_result.get('trajectory_classifier') == 'history_mahalanobis':
        traj_bn_res = traj.evaluate_history(
            emb_known   = emb_known_bn,
            y_known     = y_known,
            known_ids   = known_ids_0based,
            emb_unknown = emb_unk_bn,
            threshold   = thr_bn,
            days_keep   = phase3_result.get('history_days_keep'),
            day_decay   = phase3_result.get('history_day_decay', 1.0),
            verbose     = False,
        )
    else:
        traj_bn_res = traj.evaluate(
            emb_known   = emb_known_bn,
            y_known     = y_known,
            known_ids   = known_ids_0based,
            emb_unknown = emb_unk_bn,
            threshold   = thr_bn,
            verbose     = False,
        )
    all_results.update({
        'traj_bn_closed_acc':  traj_bn_res['closed_acc'],
        'traj_bn_open_acc':    traj_bn_res['open_known_acc'],
        'traj_bn_unk_det':     traj_bn_res['open_unk_det'],
        'traj_bn_auroc':       traj_bn_res['auroc'],
        'traj_bn_f1_macro':    traj_bn_res['f1_macro'],
        'traj_bn_accept_rate': traj_bn_res['known_accept_rate'],
        'traj_bn_threshold':   thr_bn,
    })
    _log(f'  traj_bn_adapted  closed={traj_bn_res["closed_acc"]:.4f}  '
         f'open={traj_bn_res["open_known_acc"]:.4f}  '
         f'unk_det={traj_bn_res["open_unk_det"]:.4f}  '
         f'auroc={traj_bn_res["auroc"]:.4f}  '
         f'thr={thr_bn:.3f}')

    # ── Ablation 13c: 2-point drift extrapolation to test_day ─────────────
    test_day_for_extrap = phase3_result.get('test_day', cfg.test_day)
    print(f'\n[Phase 4] Ablation 13c: 2-point trajectory extrapolation to Day {test_day_for_extrap}...')
    extrap_res = _extrapolated_trajectory_eval(
        traj               = traj,
        test_day           = test_day_for_extrap,
        emb_calib_known    = emb_trajday,
        y_calib_known      = y_trajday_0,
        emb_calib_unknown  = emb_static_unk_calib,
        emb_known          = emb_known,
        y_known            = y_known,
        emb_unknown        = emb_unk,
        known_ids          = known_ids_0based,
    )
    all_results.update({
        'extrap_closed_acc':  extrap_res['closed_acc'],
        'extrap_open_acc':    extrap_res['open_acc'],
        'extrap_unk_det':     extrap_res['unk_det'],
        'extrap_auroc':       extrap_res['auroc'],
        'extrap_f1_macro':    extrap_res['f1_macro'],
        'extrap_accept_rate': extrap_res['known_accept_rate'],
        'extrap_threshold':   extrap_res['best_threshold'],
    })
    _log(f'  drift_extrapolation  closed={extrap_res["closed_acc"]:.4f}  '
         f'open={extrap_res["open_acc"]:.4f}  '
         f'unk_det={extrap_res["unk_det"]:.4f}  '
         f'auroc={extrap_res["auroc"]:.4f}  '
         f'thr={extrap_res["best_threshold"]:.3f}')

    # ── Ablation 13d: Trajectory-reject + KNN-classify fusion ────────────
    print(f'\n[Phase 4] Ablation 13d: trajectory-reject + KNN-classify fusion...')

    traj_knn_res = _traj_knn_fusion_eval(
        traj               = traj,
        emb_calib_known    = emb_trajday,
        y_calib_known      = y_trajday_0,
        emb_calib_unknown  = emb_static_unk_calib,
        emb_known          = emb_known,
        y_known            = y_known,
        emb_unknown        = emb_unk,
        known_ids          = known_ids_0based,
        seed               = cfg.seed,
        days_keep          = phase3_result.get('history_days_keep'),
        day_decay          = phase3_result.get('history_day_decay', 1.0),
    )
    all_results['traj_knn_closed_acc']  = traj_knn_res['closed_acc']
    all_results['traj_knn_open_acc']    = traj_knn_res['open_acc']
    all_results['traj_knn_unk_det']     = traj_knn_res['unk_det']
    all_results['traj_knn_auroc']       = traj_knn_res['auroc']
    all_results['traj_knn_f1_macro']    = traj_knn_res['f1_macro']
    all_results['traj_knn_accept_rate'] = traj_knn_res['known_accept_rate']
    all_results['traj_knn_threshold']   = traj_knn_res['best_threshold']
    _log(f'  traj_reject_knn_classify  '
         f'closed={traj_knn_res["closed_acc"]:.4f}  '
         f'open={traj_knn_res["open_acc"]:.4f}  '
         f'unk_det={traj_knn_res["unk_det"]:.4f}  '
         f'auroc={traj_knn_res["auroc"]:.4f}  '
         f'thr={traj_knn_res["best_threshold"]:.3f}  '
         f'accept={traj_knn_res["known_accept_rate"]:.4f}')

    # ── Ablation 13e: Pseudo-label trajectory update ──────────────────────
    print(f'\n[Phase 4] Ablation 13e: pseudo-label trajectory update '
          f'(conf thresholds: 0.80, 0.90, 0.95)...')

    import copy as _copy
    for pl_conf in [0.80, 0.90, 0.95]:
        traj_pl, n_pl, pl_acc = _pseudo_label_trajectory_update(
            traj               = traj,
            model              = model,
            emb_known_test     = emb_known,
            y_known_test       = y_known,
            emb_unk_test       = emb_unk,
            known_ids          = known_ids_0based,
            test_day           = cfg.test_day,
            confidence_threshold = pl_conf,
        )
        if n_pl == 0:
            print(f'  [pl conf={pl_conf}] skipped — no pseudo-labels.')
            continue

        # Re-evaluate trajectory with pseudo-label update
        if phase3_result.get('trajectory_classifier') == 'cosine_prototypes':
            # Sweep threshold on Day-7 calibration data with updated trajectory
            sweep_pl = trajectory_cosine_threshold_sweep(
                traj        = traj_pl,
                emb_known   = emb_trajday,
                y_known     = y_trajday_0,
                emb_unknown = emb_static_unk_calib,
                known_ids   = known_ids_0based,
                days_keep   = phase3_result.get('history_days_keep'),
                day_decay   = phase3_result.get('history_day_decay', 1.0),
            )
            thr_pl = sweep_pl['best_threshold']
            pl_res = traj_pl.evaluate_cosine_prototypes(
                emb_known   = emb_known,
                y_known     = y_known,
                known_ids   = known_ids_0based,
                emb_unknown = emb_unk,
                threshold   = thr_pl,
                days_keep   = phase3_result.get('history_days_keep'),
                day_decay   = phase3_result.get('history_day_decay', 1.0),
                verbose     = False,
            )
        else:
            sweep_pl = trajectory_threshold_sweep(
                traj        = traj_pl,
                emb_known   = emb_trajday,
                y_known     = y_trajday_0,
                emb_unknown = emb_static_unk_calib,
                known_ids   = known_ids_0based,
            )
            thr_pl = sweep_pl['best_threshold']
            pl_res = traj_pl.evaluate(
                emb_known   = emb_known,
                y_known     = y_known,
                known_ids   = known_ids_0based,
                emb_unknown = emb_unk,
                threshold   = thr_pl,
                verbose     = False,
            )

        conf_key = str(int(pl_conf * 100))
        all_results[f'pl{conf_key}_closed_acc']  = pl_res['closed_acc']
        all_results[f'pl{conf_key}_open_acc']    = pl_res['open_known_acc']
        all_results[f'pl{conf_key}_unk_det']     = pl_res['open_unk_det']
        all_results[f'pl{conf_key}_auroc']       = pl_res['auroc']
        all_results[f'pl{conf_key}_f1_macro']    = pl_res['f1_macro']
        all_results[f'pl{conf_key}_accept_rate'] = pl_res['known_accept_rate']
        all_results[f'pl{conf_key}_threshold']   = thr_pl
        all_results[f'pl{conf_key}_n_pseudolabels'] = n_pl
        all_results[f'pl{conf_key}_pl_acc']      = pl_acc
        _log(f'  pseudo_label_traj conf={pl_conf:.2f}  '
             f'n_pl={n_pl}  pl_acc={pl_acc:.4f}  '
             f'closed={pl_res["closed_acc"]:.4f}  '
             f'open={pl_res["open_known_acc"]:.4f}  '
             f'unk_det={pl_res["open_unk_det"]:.4f}  '
             f'auroc={pl_res["auroc"]:.4f}  '
             f'thr={thr_pl:.3f}')

        # Also evaluate traj-reject + KNN-classify after pseudo-label update
        traj_knn_pl_res = _traj_knn_fusion_eval(
            traj               = traj_pl,
            emb_calib_known    = emb_trajday,
            y_calib_known      = y_trajday_0,
            emb_calib_unknown  = emb_static_unk_calib,
            emb_known          = emb_known,
            y_known            = y_known,
            emb_unknown        = emb_unk,
            known_ids          = known_ids_0based,
            seed               = cfg.seed,
            days_keep          = phase3_result.get('history_days_keep'),
            day_decay          = phase3_result.get('history_day_decay', 1.0),
        )
        all_results[f'pl{conf_key}_traj_knn_closed_acc']  = traj_knn_pl_res['closed_acc']
        all_results[f'pl{conf_key}_traj_knn_open_acc']    = traj_knn_pl_res['open_acc']
        all_results[f'pl{conf_key}_traj_knn_unk_det']     = traj_knn_pl_res['unk_det']
        all_results[f'pl{conf_key}_traj_knn_auroc']       = traj_knn_pl_res['auroc']
        all_results[f'pl{conf_key}_traj_knn_threshold']   = traj_knn_pl_res['best_threshold']
        _log(f'  pseudo_label_traj_knn conf={pl_conf:.2f}  '
             f'closed={traj_knn_pl_res["closed_acc"]:.4f}  '
             f'open={traj_knn_pl_res["open_acc"]:.4f}  '
             f'unk_det={traj_knn_pl_res["unk_det"]:.4f}  '
             f'auroc={traj_knn_pl_res["auroc"]:.4f}  '
             f'thr={traj_knn_pl_res["best_threshold"]:.3f}')

    # ── Ablation 15: OpenMax (Bendale & Boult, CVPR 2016) ────────────────
    print(f'\n[Phase 4] Ablation 15: OpenMax baseline...')
    try:
        import openmax as _openmax

        def _get_logits(X: np.ndarray) -> np.ndarray:
            return _openmax.extract_logits(model, X, batch_size=BATCH_SIZE)

        logits_calib_k = _get_logits(X_adapt_known)   # Day 7 known (all)
        logits_known_t = _get_logits(X_known)           # Day 8 known (test)
        logits_unk_t   = _get_logits(X_unk)             # Day 8 unknown (test)
        logits_calib_u = np.empty((0, logits_calib_k.shape[1]), dtype=np.float32)

        # Remap calibration labels to 0-based
        logit_id_map = {dev: i for i, dev in enumerate(sorted(cfg.all_known_ids))}
        y_adapt_0 = np.array([logit_id_map[int(v)] for v in y_adapt_all], dtype=np.int32)

        for tail in [10, 20, 50]:
            om_res = _openmax.openmax_eval(
                logits_calib_known   = logits_calib_k,
                y_calib_known        = y_adapt_0,
                logits_calib_unknown = logits_calib_u,
                logits_known         = logits_known_t,
                y_known              = y_known,
                logits_unknown       = logits_unk_t,
                n_classes            = NUM_CLASS,
                tail_size            = tail,
            )
            tag = f'openmax_tail{tail}'
            all_results[f'{tag}_closed_acc'] = om_res['closed_acc']
            all_results[f'{tag}_open_acc']   = om_res['open_acc']
            all_results[f'{tag}_unk_det']    = om_res['unk_det']
            all_results[f'{tag}_auroc']      = om_res['auroc']
            _log(f'  OpenMax tail={tail}  '
                 f'closed={om_res["closed_acc"]:.4f}  '
                 f'open={om_res["open_acc"]:.4f}  '
                 f'unk_det={om_res["unk_det"]:.4f}  '
                 f'auroc={om_res["auroc"]:.4f}')

        # Best tail size by AUROC
        best_tail = max([10, 20, 50],
                        key=lambda t: all_results.get(f'openmax_tail{t}_auroc', 0))
        all_results['openmax_closed_acc'] = all_results[f'openmax_tail{best_tail}_closed_acc']
        all_results['openmax_open_acc']   = all_results[f'openmax_tail{best_tail}_open_acc']
        all_results['openmax_unk_det']    = all_results[f'openmax_tail{best_tail}_unk_det']
        all_results['openmax_auroc']      = all_results[f'openmax_tail{best_tail}_auroc']
        _log(f'  OpenMax best (tail={best_tail}): '
             f'closed={all_results["openmax_closed_acc"]:.4f}  '
             f'auroc={all_results["openmax_auroc"]:.4f}')

    except Exception as _e:
        print(f'  [OpenMax] skipped: {_e}')

    # ── Ablation 14: Few-shot enrollment curve (multi-seed) ──────────────
    import copy

    _N_FS_SEEDS = 0 if MINIMAL else 3
    print(f'\n[Phase 4] Ablation 14: few-shot enrollment curve ({_N_FS_SEEDS} seeds)...')

    fewshot_results: dict = {}

    for k in ([] if MINIMAL else [1, 5, 10, 20, 50, 100, 200, 500]):
        seed_runs: list = []

        for seed_idx in range(_N_FS_SEEDS):
            rng_fs = np.random.default_rng(cfg.seed + seed_idx * 1000)

            traj_fs  = copy.deepcopy(traj)
            sup_idx, qry_idx = [], []
            for dev in known_ids_0based:
                idx = np.where(y_known == dev)[0]
                if len(idx) == 0:
                    continue
                if groups_known is not None:
                    # Group-disjoint: support comes from half of this device's
                    # capture files / bursts, queries from the other half, so
                    # no query shares a recording with its own support. A
                    # random slice split here would reproduce the correlated
                    # split effect measured in Section IV.
                    g_dev = (groups_known[idx, 0] if groups_known.ndim == 2
                             else groups_known[idx])
                    ug = np.unique(g_dev); ug = ug[rng_fs.permutation(len(ug))]
                    n_sup_g = max(1, min(len(ug) - 1, int(round(0.5 * len(ug)))))
                    sup_g = np.isin(g_dev, ug[:n_sup_g])
                    cand = idx[sup_g]; cand = cand[rng_fs.permutation(len(cand))]
                    sup_idx.extend(cand[:min(k, len(cand))].tolist())
                    qry_idx.extend(idx[~sup_g].tolist())
                else:
                    perm_dev = rng_fs.permutation(len(idx))
                    n_sup    = min(k, len(idx))
                    sup_idx.extend(idx[perm_dev[:n_sup]].tolist())
                    qry_idx.extend(idx[perm_dev[n_sup:]].tolist())

            sup_idx = np.array(sup_idx, dtype=np.int64)
            qry_idx = np.array(qry_idx, dtype=np.int64)
            if len(qry_idx) == 0:
                break

            emb_sup = emb_known[sup_idx]
            y_sup   = y_known[sup_idx]
            emb_qry = emb_known[qry_idx]
            y_qry   = y_known[qry_idx]

            traj_fs.update(day_id=cfg.test_day, emb=emb_sup, labels=y_sup)

            sweep_fs = trajectory_threshold_sweep(
                traj        = traj_fs,
                emb_known   = emb_trajday_calib,
                y_known     = y_d7_calib,
                emb_unknown = emb_static_unk_calib,
                known_ids   = known_ids_0based,
            )
            thr_fs = sweep_fs['best_threshold']

            res_fs = traj_fs.evaluate(
                emb_known   = emb_qry,
                y_known     = y_qry,
                known_ids   = known_ids_0based,
                emb_unknown = emb_unk,
                threshold   = thr_fs,
                verbose     = False,
            )
            res_fs['threshold'] = thr_fs
            seed_runs.append(res_fs)

        if not seed_runs:
            print(f'  k={k:3d}/dev skipped — no held-out query samples remain.')
            continue

        # Average metrics over seeds; add std for key metrics
        avg: dict = {}
        for key in seed_runs[0]:
            vals = [r[key] for r in seed_runs if not np.isnan(float(r.get(key, float('nan'))))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        avg['std_closed_acc'] = float(np.std([r.get('closed_acc', float('nan')) for r in seed_runs]))
        avg['std_auroc']      = float(np.std([r.get('auroc',      float('nan')) for r in seed_runs]))

        fewshot_results[k] = avg
        total_labels = k * len(known_ids_0based)
        print(f'  k={k:3d}/dev ({total_labels:5d} labels)  '
              f'closed={avg["closed_acc"]:.4f}±{avg["std_closed_acc"]:.4f}  '
              f'open={avg["open_known_acc"]:.4f}  '
              f'auroc={avg["auroc"]:.4f}±{avg["std_auroc"]:.4f}  '
              f'unk_det={avg["open_unk_det"]:.4f}')

    all_results['fewshot'] = fewshot_results
    _log('  few_shot  ' + '  '.join(
        f'k={k}:{fewshot_results[k]["closed_acc"]:.4f}'
        for k in [1, 5, 10, 20, 50, 100, 200, 500]
        if k in fewshot_results
    ))

    # ── Summary table ─────────────────────────────────────────────────────
    sep  = '─' * 70
    head = f"  {'Method':<38} {'Closed':>7}  {'Open':>7}  {'UnkDet':>7}  {'AUROC':>7}"
    ts_label = (
        f"Temp scaling (Day {cfg.traj_day}, "
        f"T={all_results['ts_temperature']:.2f})"
    )
    rows = [
        f"  {'Softmax (no rejection)':<38} {all_results['softmax_acc']:>7.4f}  {'—':>7}  {'—':>7}  {'—':>7}",
        f"  {'TTA-BN + softmax':<38} {all_results['softmax_bn_acc']:>7.4f}  {'—':>7}  {'—':>7}  {'—':>7}",
        f"  {f'Static prototype (Day {cfg.traj_day})':<38} {all_results['static_closed_acc']:>7.4f}  "
        f"{all_results['static_open_acc']:>7.4f}  "
        f"{all_results['static_unk_det']:>7.4f}  "
        f"{all_results['static_auroc']:>7.4f}",
        f"  {f'Calibrated probe (Day {cfg.traj_day})':<38} {all_results['probe_closed_acc']:>7.4f}  "
        f"{all_results['probe_open_acc']:>7.4f}  "
        f"{all_results['probe_unk_det']:>7.4f}  "
        f"{all_results['probe_auroc']:>7.4f}",
        f"  {f'L2 calibrated probe (Day {cfg.traj_day})':<38} {all_results['l2_probe_closed_acc']:>7.4f}  "
        f"{all_results['l2_probe_open_acc']:>7.4f}  "
        f"{all_results['l2_probe_unk_det']:>7.4f}  "
        f"{all_results['l2_probe_auroc']:>7.4f}",
        f"  {f'Class-wise L2 probe (Day {cfg.traj_day})':<38} {all_results['class_l2_probe_closed_acc']:>7.4f}  "
        f"{all_results['class_l2_probe_open_acc']:>7.4f}  "
        f"{all_results['class_l2_probe_unk_det']:>7.4f}  "
        f"{all_results['class_l2_probe_auroc']:>7.4f}",
        f"  {f'Temperature L2 probe (Day {cfg.traj_day})':<38} {all_results['temp_l2_probe_closed_acc']:>7.4f}  "
        f"{all_results['temp_l2_probe_open_acc']:>7.4f}  "
        f"{all_results['temp_l2_probe_unk_det']:>7.4f}  "
        f"{all_results['temp_l2_probe_auroc']:>7.4f}",
        f"  {f'Cosine KNN (Day {cfg.traj_day})':<38} {all_results['knn_closed_acc']:>7.4f}  "
        f"{all_results['knn_open_acc']:>7.4f}  "
        f"{all_results['knn_unk_det']:>7.4f}  "
        f"{all_results['knn_auroc']:>7.4f}",
        f"  {f'Class-wise cosine KNN (Day {cfg.traj_day})':<38} {all_results['class_knn_closed_acc']:>7.4f}  "
        f"{all_results['class_knn_open_acc']:>7.4f}  "
        f"{all_results['class_knn_unk_det']:>7.4f}  "
        f"{all_results['class_knn_auroc']:>7.4f}",
        f"  {f'Cosine KNN @50% accept':<38} {all_results['knn_accept50_closed_acc']:>7.4f}  "
        f"{all_results['knn_accept50_open_acc']:>7.4f}  "
        f"{all_results['knn_accept50_unk_det']:>7.4f}  "
        f"{all_results['knn_accept50_auroc']:>7.4f}",
        f"  {f'Cosine KNN @60% accept':<38} {all_results['knn_accept60_closed_acc']:>7.4f}  "
        f"{all_results['knn_accept60_open_acc']:>7.4f}  "
        f"{all_results['knn_accept60_unk_det']:>7.4f}  "
        f"{all_results['knn_accept60_auroc']:>7.4f}",
        f"  {f'Cosine KNN @70% accept':<38} {all_results['knn_accept70_closed_acc']:>7.4f}  "
        f"{all_results['knn_accept70_open_acc']:>7.4f}  "
        f"{all_results['knn_accept70_unk_det']:>7.4f}  "
        f"{all_results['knn_accept70_auroc']:>7.4f}",
        f"  {f'Drift-corrected KNN (→Day {cfg.test_day})':<38} {all_results['drift_knn_closed_acc']:>7.4f}  "
        f"{all_results['drift_knn_open_acc']:>7.4f}  "
        f"{all_results['drift_knn_unk_det']:>7.4f}  "
        f"{all_results['drift_knn_auroc']:>7.4f}",
        f"  {f'Drift-corrected class-wise KNN':<38} {all_results['drift_class_knn_closed_acc']:>7.4f}  "
        f"{all_results['drift_class_knn_open_acc']:>7.4f}  "
        f"{all_results['drift_class_knn_unk_det']:>7.4f}  "
        f"{all_results['drift_class_knn_auroc']:>7.4f}",
        f"  {f'Calibrated LDA (Day {cfg.traj_day})':<38} {all_results['lda_closed_acc']:>7.4f}  "
        f"{all_results['lda_open_acc']:>7.4f}  "
        f"{all_results['lda_unk_det']:>7.4f}  "
        f"{all_results['lda_auroc']:>7.4f}",
        f"  {f'L2 calibrated LDA (Day {cfg.traj_day})':<38} {all_results['l2_lda_closed_acc']:>7.4f}  "
        f"{all_results['l2_lda_open_acc']:>7.4f}  "
        f"{all_results['l2_lda_unk_det']:>7.4f}  "
        f"{all_results['l2_lda_auroc']:>7.4f}",
        f"  {f'Probe/trajectory fusion (Day {cfg.traj_day})':<38} {all_results['fusion_closed_acc']:>7.4f}  "
        f"{all_results['fusion_open_acc']:>7.4f}  "
        f"{all_results['fusion_unk_det']:>7.4f}  "
        f"{all_results['fusion_auroc']:>7.4f}",
        f"  {f'Cosine trajectory (Day {cfg.traj_day})':<38} {all_results['cos_traj_closed_acc']:>7.4f}  "
        f"{all_results['cos_traj_open_acc']:>7.4f}  "
        f"{all_results['cos_traj_unk_det']:>7.4f}  "
        f"{all_results['cos_traj_auroc']:>7.4f}",
        f"  {f'Multi-proto cosine traj (Day {cfg.traj_day})':<38} {all_results['mp_cos_traj_closed_acc']:>7.4f}  "
        f"{all_results['mp_cos_traj_open_acc']:>7.4f}  "
        f"{all_results['mp_cos_traj_unk_det']:>7.4f}  "
        f"{all_results['mp_cos_traj_auroc']:>7.4f}",
        f"  {f'Class-wise multi-proto traj (Day {cfg.traj_day})':<38} {all_results['class_mp_cos_traj_closed_acc']:>7.4f}  "
        f"{all_results['class_mp_cos_traj_open_acc']:>7.4f}  "
        f"{all_results['class_mp_cos_traj_unk_det']:>7.4f}  "
        f"{all_results['class_mp_cos_traj_auroc']:>7.4f}",
        f"  {f'Trajectory through Day {cfg.traj_day}':<38} {all_results['traj_closed_acc']:>7.4f}  "
        f"{all_results['traj_open_acc']:>7.4f}  "
        f"{all_results['traj_unk_det']:>7.4f}  "
        f"{all_results['traj_auroc']:>7.4f}",
        f"  {f'Trajectory (BN-adapted emb, Day {cfg.traj_day})':<38} {all_results['traj_bn_closed_acc']:>7.4f}  "
        f"{all_results['traj_bn_open_acc']:>7.4f}  "
        f"{all_results['traj_bn_unk_det']:>7.4f}  "
        f"{all_results['traj_bn_auroc']:>7.4f}",
        f"  {f'Drift extrapolation → Day {cfg.test_day}':<38} {all_results['extrap_closed_acc']:>7.4f}  "
        f"{all_results['extrap_open_acc']:>7.4f}  "
        f"{all_results['extrap_unk_det']:>7.4f}  "
        f"{all_results['extrap_auroc']:>7.4f}",
        f"  {ts_label:<38} {all_results['ts_closed_acc']:>7.4f}  "
        f"{all_results['ts_open_acc']:>7.4f}  "
        f"{all_results['ts_unk_det']:>7.4f}  "
        f"{all_results['ts_auroc']:>7.4f}",
        f"  {'Traj-reject + KNN-classify':<38} {all_results.get('traj_knn_closed_acc', float('nan')):>7.4f}  "
        f"{all_results.get('traj_knn_open_acc', float('nan')):>7.4f}  "
        f"{all_results.get('traj_knn_unk_det', float('nan')):>7.4f}  "
        f"{all_results.get('traj_knn_auroc', float('nan')):>7.4f}",
    ]
    for pct in [20, 35, 50]:
        prefix = f'target{pct}_class_mp_cos_traj'
        if f'{prefix}_closed_acc' in all_results:
            rows.append(
                f"  {f'Class-wise multi-proto traj @{pct}% accept':<38} "
                f"{all_results[f'{prefix}_closed_acc']:>7.4f}  "
                f"{all_results[f'{prefix}_open_acc']:>7.4f}  "
                f"{all_results[f'{prefix}_unk_det']:>7.4f}  "
                f"{all_results[f'{prefix}_auroc']:>7.4f}"
            )
    # Pseudo-label trajectory rows (one per confidence level attempted)
    for pl_conf in [0.80, 0.90, 0.95]:
        conf_key = str(int(pl_conf * 100))
        if f'pl{conf_key}_closed_acc' in all_results:
            n_pl = all_results.get(f'pl{conf_key}_n_pseudolabels', 0)
            rows.append(
                f"  {f'PL-traj conf={pl_conf} (n={n_pl})':<38} "
                f"{all_results[f'pl{conf_key}_closed_acc']:>7.4f}  "
                f"{all_results[f'pl{conf_key}_open_acc']:>7.4f}  "
                f"{all_results[f'pl{conf_key}_unk_det']:>7.4f}  "
                f"{all_results[f'pl{conf_key}_auroc']:>7.4f}"
            )
            rows.append(
                f"  {f'PL-traj+KNN conf={pl_conf}':<38} "
                f"{all_results.get(f'pl{conf_key}_traj_knn_closed_acc', float('nan')):>7.4f}  "
                f"{all_results.get(f'pl{conf_key}_traj_knn_open_acc', float('nan')):>7.4f}  "
                f"{all_results.get(f'pl{conf_key}_traj_knn_unk_det', float('nan')):>7.4f}  "
                f"{all_results.get(f'pl{conf_key}_traj_knn_auroc', float('nan')):>7.4f}"
            )
    if 'openmax_closed_acc' in all_results:
        rows.append(
            f"  {'OpenMax (Bendale & Boult 2016)':<38} "
            f"{all_results['openmax_closed_acc']:>7.4f}  "
            f"{all_results.get('openmax_open_acc', float('nan')):>7.4f}  "
            f"{all_results['openmax_unk_det']:>7.4f}  "
            f"{all_results['openmax_auroc']:>7.4f}"
        )
    for k in [1, 5, 10, 20, 50, 100, 200, 500]:
        if k in fewshot_results:
            fr = fewshot_results[k]
            rows.append(
                f"  {f'Few-shot enroll k={k}/dev ({k*16} labels)':<38} {fr['closed_acc']:>7.4f}  "
                f"{fr['open_known_acc']:>7.4f}  "
                f"{fr['open_unk_det']:>7.4f}  "
                f"{fr['auroc']:>7.4f}"
            )
    summary = '\n'.join(['', '=' * 70, '  PHASE 4 RESULTS SUMMARY',
                         sep, head, sep] + rows + ['=' * 70])
    _log(summary)

    with open(results_path, 'a') as f:
        print('\n### Full results dict', file=f)
        for k, v in all_results.items():
            print(f'  {k} = {v}', file=f)

    def _fmt(value) -> str:
        if value is None:
            return '-'
        try:
            if np.isnan(value):
                return 'nan'
        except TypeError:
            pass
        if isinstance(value, (float, np.floating)):
            return f'{float(value):.4f}'
        return str(value)

    final_rows = [
        ('Softmax', all_results['softmax_acc'], None, None, None, None, None),
        ('TTA-BN + softmax', all_results['softmax_bn_acc'], None, None, None, None, None),
        (
            f'Static prototype Day {cfg.traj_day}',
            all_results['static_closed_acc'],
            all_results['static_open_acc'],
            all_results['static_unk_det'],
            all_results['static_auroc'],
            all_results['static_threshold'],
            None,
        ),
        (
            f'Calibrated probe Day {cfg.traj_day}',
            all_results['probe_closed_acc'],
            all_results['probe_open_acc'],
            all_results['probe_unk_det'],
            all_results['probe_auroc'],
            all_results['probe_threshold'],
            all_results['probe_accept_rate'],
        ),
        (
            f'L2 calibrated probe Day {cfg.traj_day}',
            all_results['l2_probe_closed_acc'],
            all_results['l2_probe_open_acc'],
            all_results['l2_probe_unk_det'],
            all_results['l2_probe_auroc'],
            all_results['l2_probe_threshold'],
            all_results['l2_probe_accept_rate'],
        ),
        (
            f'Class-wise L2 probe Day {cfg.traj_day}',
            all_results['class_l2_probe_closed_acc'],
            all_results['class_l2_probe_open_acc'],
            all_results['class_l2_probe_unk_det'],
            all_results['class_l2_probe_auroc'],
            all_results['class_l2_probe_threshold'],
            all_results['class_l2_probe_accept_rate'],
        ),
        (
            f'Temperature L2 probe Day {cfg.traj_day}',
            all_results['temp_l2_probe_closed_acc'],
            all_results['temp_l2_probe_open_acc'],
            all_results['temp_l2_probe_unk_det'],
            all_results['temp_l2_probe_auroc'],
            all_results['temp_l2_probe_threshold'],
            all_results['temp_l2_probe_accept_rate'],
        ),
        (
            f'Cosine KNN Day {cfg.traj_day}',
            all_results['knn_closed_acc'],
            all_results['knn_open_acc'],
            all_results['knn_unk_det'],
            all_results['knn_auroc'],
            all_results['knn_threshold'],
            all_results['knn_accept_rate'],
        ),
        (
            f'Class-wise cosine KNN Day {cfg.traj_day}',
            all_results['class_knn_closed_acc'],
            all_results['class_knn_open_acc'],
            all_results['class_knn_unk_det'],
            all_results['class_knn_auroc'],
            all_results['class_knn_threshold'],
            all_results['class_knn_accept_rate'],
        ),
        (
            'Cosine KNN @50% accept',
            all_results['knn_accept50_closed_acc'],
            all_results['knn_accept50_open_acc'],
            all_results['knn_accept50_unk_det'],
            all_results['knn_accept50_auroc'],
            all_results['knn_accept50_threshold'],
            all_results['knn_accept50_accept_rate'],
        ),
        (
            'Cosine KNN @60% accept',
            all_results['knn_accept60_closed_acc'],
            all_results['knn_accept60_open_acc'],
            all_results['knn_accept60_unk_det'],
            all_results['knn_accept60_auroc'],
            all_results['knn_accept60_threshold'],
            all_results['knn_accept60_accept_rate'],
        ),
        (
            'Cosine KNN @70% accept',
            all_results['knn_accept70_closed_acc'],
            all_results['knn_accept70_open_acc'],
            all_results['knn_accept70_unk_det'],
            all_results['knn_accept70_auroc'],
            all_results['knn_accept70_threshold'],
            all_results['knn_accept70_accept_rate'],
        ),
        (
            f'Calibrated LDA Day {cfg.traj_day}',
            all_results['lda_closed_acc'],
            all_results['lda_open_acc'],
            all_results['lda_unk_det'],
            all_results['lda_auroc'],
            all_results['lda_threshold'],
            all_results['lda_accept_rate'],
        ),
        (
            f'L2 calibrated LDA Day {cfg.traj_day}',
            all_results['l2_lda_closed_acc'],
            all_results['l2_lda_open_acc'],
            all_results['l2_lda_unk_det'],
            all_results['l2_lda_auroc'],
            all_results['l2_lda_threshold'],
            all_results['l2_lda_accept_rate'],
        ),
        (
            f'Probe/trajectory fusion Day {cfg.traj_day}',
            all_results['fusion_closed_acc'],
            all_results['fusion_open_acc'],
            all_results['fusion_unk_det'],
            all_results['fusion_auroc'],
            all_results['fusion_threshold'],
            all_results['fusion_accept_rate'],
        ),
        (
            f'Cosine trajectory Day {cfg.traj_day}',
            all_results['cos_traj_closed_acc'],
            all_results['cos_traj_open_acc'],
            all_results['cos_traj_unk_det'],
            all_results['cos_traj_auroc'],
            all_results['cos_traj_threshold'],
            all_results['cos_traj_accept_rate'],
        ),
        (
            f'Multi-proto cosine traj Day {cfg.traj_day}',
            all_results['mp_cos_traj_closed_acc'],
            all_results['mp_cos_traj_open_acc'],
            all_results['mp_cos_traj_unk_det'],
            all_results['mp_cos_traj_auroc'],
            all_results['mp_cos_traj_threshold'],
            all_results['mp_cos_traj_accept_rate'],
        ),
        (
            f'Class-wise multi-proto traj Day {cfg.traj_day}',
            all_results['class_mp_cos_traj_closed_acc'],
            all_results['class_mp_cos_traj_open_acc'],
            all_results['class_mp_cos_traj_unk_det'],
            all_results['class_mp_cos_traj_auroc'],
            all_results['class_mp_cos_traj_threshold'],
            all_results['class_mp_cos_traj_accept_rate'],
        ),
        (
            f'Trajectory through Day {cfg.traj_day}',
            all_results['traj_closed_acc'],
            all_results['traj_open_acc'],
            all_results['traj_unk_det'],
            all_results['traj_auroc'],
            all_results['traj_threshold'],
            all_results['traj_accept_rate'],
        ),
        (
            f'Trajectory BN-adapted Day {cfg.traj_day}',
            all_results['traj_bn_closed_acc'],
            all_results['traj_bn_open_acc'],
            all_results['traj_bn_unk_det'],
            all_results['traj_bn_auroc'],
            all_results['traj_bn_threshold'],
            all_results['traj_bn_accept_rate'],
        ),
        (
            f'Drift extrapolation → Day {cfg.test_day}',
            all_results['extrap_closed_acc'],
            all_results['extrap_open_acc'],
            all_results['extrap_unk_det'],
            all_results['extrap_auroc'],
            all_results['extrap_threshold'],
            all_results['extrap_accept_rate'],
        ),
    ]
    for pct in [20, 35, 50]:
        prefix = f'target{pct}_class_mp_cos_traj'
        if f'{prefix}_closed_acc' in all_results:
            final_rows.append((
                f'Class-wise multi-proto traj @{pct}% accept',
                all_results[f'{prefix}_closed_acc'],
                all_results[f'{prefix}_open_acc'],
                all_results[f'{prefix}_unk_det'],
                all_results[f'{prefix}_auroc'],
                all_results[f'{prefix}_threshold'],
                all_results[f'{prefix}_accept_rate'],
            ))
    for k in [1, 5, 10, 20, 50, 100, 200, 500]:
        if k not in fewshot_results:
            continue
        fr = fewshot_results[k]
        final_rows.append((
            f'Few-shot k={k}/dev ({k * len(known_ids_0based)} labels)',
            fr['closed_acc'],
            fr['open_known_acc'],
            fr['open_unk_det'],
            fr['auroc'],
            fr['threshold'],
            fr['known_accept_rate'],
        ))

    if 'openmax_closed_acc' in all_results:
        final_rows.append((
            'OpenMax (Bendale & Boult, CVPR 2016)',
            all_results['openmax_closed_acc'],
            all_results.get('openmax_open_acc'),
            all_results['openmax_unk_det'],
            all_results['openmax_auroc'],
            None,
            None,
        ))

    # New methods
    if 'drift_knn_closed_acc' in all_results:
        final_rows.append((
            f'Drift-corrected KNN (→Day {cfg.test_day})',
            all_results['drift_knn_closed_acc'],
            all_results['drift_knn_open_acc'],
            all_results['drift_knn_unk_det'],
            all_results['drift_knn_auroc'],
            all_results['drift_knn_threshold'],
            all_results['drift_knn_accept_rate'],
        ))
    if 'traj_knn_closed_acc' in all_results:
        final_rows.append((
            'Traj-reject + KNN-classify',
            all_results['traj_knn_closed_acc'],
            all_results['traj_knn_open_acc'],
            all_results['traj_knn_unk_det'],
            all_results['traj_knn_auroc'],
            all_results['traj_knn_threshold'],
            all_results['traj_knn_accept_rate'],
        ))
    for pl_conf in [0.80, 0.90, 0.95]:
        conf_key = str(int(pl_conf * 100))
        if f'pl{conf_key}_closed_acc' in all_results:
            n_pl = all_results.get(f'pl{conf_key}_n_pseudolabels', 0)
            final_rows.append((
                f'PL-traj conf={pl_conf} (n={n_pl})',
                all_results[f'pl{conf_key}_closed_acc'],
                all_results[f'pl{conf_key}_open_acc'],
                all_results[f'pl{conf_key}_unk_det'],
                all_results[f'pl{conf_key}_auroc'],
                all_results[f'pl{conf_key}_threshold'],
                all_results[f'pl{conf_key}_accept_rate'],
            ))
            if f'pl{conf_key}_traj_knn_closed_acc' in all_results:
                final_rows.append((
                    f'PL-traj+KNN conf={pl_conf}',
                    all_results[f'pl{conf_key}_traj_knn_closed_acc'],
                    all_results[f'pl{conf_key}_traj_knn_open_acc'],
                    all_results[f'pl{conf_key}_traj_knn_unk_det'],
                    all_results[f'pl{conf_key}_traj_knn_auroc'],
                    all_results[f'pl{conf_key}_traj_knn_threshold'],
                    None,
                ))

    final_table = [
        '',
        '### Phase 4 final table',
        '| Method | Closed | Open | UnkDet | AUROC | Threshold | AcceptRate |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    final_table.extend(
        f'| {method} | {_fmt(closed)} | {_fmt(open_acc)} | {_fmt(unk_det)} | '
        f'{_fmt(auroc)} | {_fmt(threshold)} | {_fmt(accept_rate)} |'
        for method, closed, open_acc, unk_det, auroc, threshold, accept_rate in final_rows
    )
    _log('\n'.join(final_table))

    headline_rows = [
        (
            f'Class-wise cosine KNN Day {cfg.traj_day}',
            all_results['class_knn_closed_acc'],
            all_results['class_knn_open_acc'],
            all_results['class_knn_unk_det'],
            all_results['class_knn_auroc'],
            all_results['class_knn_accept_rate'],
        ),
        (
            f'Cosine KNN Day {cfg.traj_day}',
            all_results['knn_closed_acc'],
            all_results['knn_open_acc'],
            all_results['knn_unk_det'],
            all_results['knn_auroc'],
            all_results['knn_accept_rate'],
        ),
        (
            'Traj-reject + KNN-classify',
            all_results.get('traj_knn_closed_acc', float('nan')),
            all_results.get('traj_knn_open_acc', float('nan')),
            all_results.get('traj_knn_unk_det', float('nan')),
            all_results.get('traj_knn_auroc', float('nan')),
            all_results.get('traj_knn_accept_rate', float('nan')),
        ),
        (
            f'Class-wise multi-proto traj Day {cfg.traj_day}',
            all_results['class_mp_cos_traj_closed_acc'],
            all_results['class_mp_cos_traj_open_acc'],
            all_results['class_mp_cos_traj_unk_det'],
            all_results['class_mp_cos_traj_auroc'],
            all_results['class_mp_cos_traj_accept_rate'],
        ),
        (
            f'Multi-proto cosine traj Day {cfg.traj_day}',
            all_results['mp_cos_traj_closed_acc'],
            all_results['mp_cos_traj_open_acc'],
            all_results['mp_cos_traj_unk_det'],
            all_results['mp_cos_traj_auroc'],
            all_results['mp_cos_traj_accept_rate'],
        ),
        (
            f'Trajectory BN-adapted Day {cfg.traj_day}',
            all_results['traj_bn_closed_acc'],
            all_results['traj_bn_open_acc'],
            all_results['traj_bn_unk_det'],
            all_results['traj_bn_auroc'],
            all_results['traj_bn_accept_rate'],
        ),
        (
            f'Trajectory through Day {cfg.traj_day}',
            all_results['traj_closed_acc'],
            all_results['traj_open_acc'],
            all_results['traj_unk_det'],
            all_results['traj_auroc'],
            all_results['traj_accept_rate'],
        ),
        ('Softmax', all_results['softmax_acc'], None, None, None, None),
    ]
    for pct in [50, 35, 20]:
        prefix = f'target{pct}_class_mp_cos_traj'
        if f'{prefix}_closed_acc' in all_results:
            headline_rows.insert(4, (
                f'Class-wise multi-proto traj @{pct}% accept',
                all_results[f'{prefix}_closed_acc'],
                all_results[f'{prefix}_open_acc'],
                all_results[f'{prefix}_unk_det'],
                all_results[f'{prefix}_auroc'],
                all_results[f'{prefix}_accept_rate'],
            ))
    headline_table = [
        '',
        '### Phase 4 headline table',
        '| Method | Closed | Open | UnkDet | AUROC | AcceptRate |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    headline_table.extend(
        f'| {method} | {_fmt(closed)} | {_fmt(open_acc)} | {_fmt(unk_det)} | '
        f'{_fmt(auroc)} | {_fmt(accept_rate)} |'
        for method, closed, open_acc, unk_det, auroc, accept_rate in headline_rows
    )

    trajectory_comparison = [
        '',
        '### Trajectory improvement table',
        '| Trajectory Method | Closed | Open | UnkDet | AUROC |',
        '|---|---:|---:|---:|---:|',
        (
            f"| Old multi-proto trajectory | {_fmt(all_results['mp_cos_traj_closed_acc'])} | "
            f"{_fmt(all_results['mp_cos_traj_open_acc'])} | "
            f"{_fmt(all_results['mp_cos_traj_unk_det'])} | "
            f"{_fmt(all_results['mp_cos_traj_auroc'])} |"
        ),
        (
            f"| New class-wise multi-proto trajectory | {_fmt(all_results['class_mp_cos_traj_closed_acc'])} | "
            f"{_fmt(all_results['class_mp_cos_traj_open_acc'])} | "
            f"{_fmt(all_results['class_mp_cos_traj_unk_det'])} | "
            f"{_fmt(all_results['class_mp_cos_traj_auroc'])} |"
        ),
    ]
    for pct in [20, 35, 50]:
        prefix = f'target{pct}_class_mp_cos_traj'
        if f'{prefix}_closed_acc' in all_results:
            trajectory_comparison.append(
                f"| Target {pct}% accept class-wise trajectory | "
                f"{_fmt(all_results[f'{prefix}_closed_acc'])} | "
                f"{_fmt(all_results[f'{prefix}_open_acc'])} | "
                f"{_fmt(all_results[f'{prefix}_unk_det'])} | "
                f"{_fmt(all_results[f'{prefix}_auroc'])} |"
            )
    _log('\n'.join(headline_table + trajectory_comparison))

    # ── Ablation 15: burst decisions + capture-aware enrolment ────────────
    if groups_known is not None:
        print(f'\n[Phase 4] Ablation 15: burst decisions and enrolment '
              f'diversity...')
        try:
            abl15 = run_burst_and_diversity_ablation(
                emb_known    = emb_known,
                y_known      = y_known,
                groups_known = groups_known,
                emb_day4     = emb_trajday,
                y_day4       = y_trajday_0,
                log          = _log,
                traj         = traj,
                test_day     = cfg.test_day,
                days_keep    = phase3_result.get('history_days_keep'),
                day_decay    = phase3_result.get('history_day_decay'),
            )
            all_results.update(abl15)

            if abl15:
                ks       = [5, 50, 200, 2000]
                aggs     = [1, 5, 20, 50]
                n_sess   = sorted({int(k.split('_')[0][3:]) for k in abl15
                                   if k.startswith('div')})
                tbl = ['', '### Burst decisions and enrolment diversity',
                       '| Enrol sessions | Labels/dev | ' +
                       ' | '.join(f'{n} slice/dec' for n in aggs) + ' |',
                       '|---|---:|' + '---:|' * len(aggs)]
                tbl.append('| 0 (zero-shot) | 0 | ' + ' | '.join(
                    _fmt(abl15.get(f'burst_zeroshot_n{n}')) for n in aggs) + ' |')
                for pfx, tag in (('', 'proto'), ('t2r_', 'T2R rule')):
                    if not any(kk.startswith(pfx + 'div') for kk in abl15):
                        continue
                    tbl.append(f'| **{tag}** | | | | | |')
                    for ns in n_sess:
                        for k in ks:
                            cells = []
                            for n in aggs:
                                m = abl15.get(f'{pfx}div{ns}_k{k}_n{n}_mean')
                                sd = abl15.get(f'{pfx}div{ns}_k{k}_n{n}_std')
                                cells.append('-' if m is None
                                             else f'{m:.4f}±{sd:.4f}')
                            tbl.append(f'| {ns} | {k} | ' + ' | '.join(cells) + ' |')
                _log('\n'.join(tbl))
        except Exception as e:
            print(f'[Phase 4] Ablation 15 skipped: {e}')

    return all_results
