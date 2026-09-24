#!/usr/bin/env python3
"""
openmax.py — OpenMax open-set recognition baseline.

Reference:
    Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016.

Algorithm:
    1. Calibration: for each known class, compute the Mean Activation Vector
       (MAV) from correctly-classified calibration samples. Fit a Weibull
       distribution to the tail of Euclidean distances from the MAV.
    2. Inference: for a test sample, use the Weibull CDFs to estimate how
       "extreme" each activation is, redistribute that probability mass to
       an "unknown" class, and softmax over [known_1, ..., known_C, unknown].
       Reject if unknown class wins or its probability exceeds a threshold.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import weibull_min
from sklearn.metrics import roc_auc_score
import tensorflow as tf


# ── Logit extraction ──────────────────────────────────────────────────────────

def extract_logits(model: tf.keras.Model, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """
    Extract pre-softmax logits from the model's 'classifier' Dense layer.
    Avoids softmax so distances in logit space are meaningful.
    """
    try:
        clf_layer = model.get_layer('classifier')
    except ValueError:
        # Fallback: use last Dense layer
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        if not dense_layers:
            raise RuntimeError('No Dense layer found for logit extraction')
        clf_layer = dense_layers[-1]

    # Keras 3 forbids tf.matmul on symbolic KerasTensors, so avoid graph
    # surgery: predict the classifier's input embedding with a sub-model,
    # then compute logits = emb @ W + b in NumPy.
    W, b = clf_layer.get_weights()        # (emb_size, n_classes), (n_classes,)

    X_arr = np.asarray(X, dtype=np.float32)
    if len(X_arr) == 0:
        return np.empty((0, int(b.shape[0])), dtype=np.float32)

    # Find the layer feeding the classifier (L2-normalised embedding)
    emb_layer = None
    for name in ('emb_l2norm', 'embedding'):
        try:
            emb_layer = model.get_layer(name)
            break
        except ValueError:
            continue
    if emb_layer is None:
        raise RuntimeError('No embedding layer found for logit extraction')

    emb_model = tf.keras.Model(inputs=model.input, outputs=emb_layer.output)
    emb = emb_model.predict(X_arr, batch_size=batch_size, verbose=0)
    return (emb @ W + b).astype(np.float32)


# ── Weibull fitting ───────────────────────────────────────────────────────────

def _fit_weibull(distances: np.ndarray, tail_size: int) -> tuple | None:
    tail_size = min(tail_size, len(distances))
    if tail_size < 3:
        return None
    tail = np.sort(distances)[-tail_size:]
    try:
        shape, loc, scale = weibull_min.fit(tail, floc=0)
        return float(shape), float(loc), float(scale)
    except Exception:
        return None


def _weibull_cdf(dist: float, params: tuple) -> float:
    shape, loc, scale = params
    return float(weibull_min.cdf(dist, shape, loc, scale))


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    logits:    np.ndarray,   # (N, C) calibration logits
    labels:    np.ndarray,   # (N,)  0-based class labels
    n_classes: int,
    tail_size: int = 20,
) -> dict:
    """
    Compute per-class MAVs and Weibull params from correctly-classified
    calibration samples.
    """
    pred = np.argmax(logits, axis=1)
    mavs     = np.zeros((n_classes, logits.shape[1]), dtype=np.float32)
    weibulls = []

    for c in range(n_classes):
        correct_mask = (labels == c) & (pred == c)
        if correct_mask.sum() < 3:
            correct_mask = (labels == c)        # fallback: all samples
        acts = logits[correct_mask].astype(np.float32)
        mav  = acts.mean(axis=0)
        mavs[c] = mav
        dists = np.linalg.norm(acts - mav, axis=1)
        weibulls.append(_fit_weibull(dists, tail_size))

    return {'mavs': mavs, 'weibulls': weibulls, 'n_classes': n_classes}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(
    logits: np.ndarray,   # (N, C)
    calib:  dict,
    alpha:  int | None = None,   # top-α classes to revise (None = all)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OpenMax probabilities.

    Returns
    -------
    probs       : (N, C+1)  — C known classes + 1 unknown
    unknown_prob: (N,)      — probability mass on the unknown class
    """
    mavs      = calib['mavs']
    weibulls  = calib['weibulls']
    n_classes = calib['n_classes']
    if alpha is None:
        alpha = n_classes

    N = len(logits)
    probs = np.zeros((N, n_classes + 1), dtype=np.float32)

    for i, logit in enumerate(logits):
        dists     = np.linalg.norm(logit - mavs, axis=1)         # (C,)
        top_alpha = np.argsort(logit)[::-1][:alpha]               # top-α by activation

        # Weibull revision weights
        w = np.zeros(n_classes, dtype=np.float32)
        for c in top_alpha:
            if weibulls[c] is not None:
                w[c] = _weibull_cdf(float(dists[c]), weibulls[c])

        revised     = logit * (1.0 - w)
        unknown_act = float(np.sum(logit * w))
        all_acts    = np.append(revised, unknown_act)

        # Numerically stable softmax
        all_acts   -= all_acts.max()
        exp_a       = np.exp(all_acts)
        probs[i]    = exp_a / (exp_a.sum() + 1e-12)

    return probs, probs[:, -1]


# ── Full evaluation ───────────────────────────────────────────────────────────

def openmax_eval(
    logits_calib_known:   np.ndarray,   # (N_cal, C)  traj_day known logits
    y_calib_known:        np.ndarray,   # (N_cal,)
    logits_calib_unknown: np.ndarray,   # (N_cal_u, C) or empty
    logits_known:         np.ndarray,   # (N_k, C)  test_day known
    y_known:              np.ndarray,   # (N_k,)
    logits_unknown:       np.ndarray,   # (N_u, C)  test_day unknown
    n_classes:            int,
    tail_size:            int = 20,
    alpha:                int | None = None,
) -> dict:
    """
    Calibrate OpenMax on traj_day data, evaluate on test_day.

    Threshold is swept on calibration data. If no calibration unknowns
    are provided, falls back to 95th-percentile of known unknown-prob.
    """
    if alpha is None:
        alpha = n_classes

    calib = calibrate(logits_calib_known, y_calib_known,
                      n_classes=n_classes, tail_size=tail_size)

    has_calib_unk = (logits_calib_unknown is not None
                     and len(logits_calib_unknown) > 0)

    # ── Threshold sweep on calibration data ──────────────────────────────
    probs_cal_k, unk_cal_k = score(logits_calib_known, calib, alpha)
    pred_cal_k = np.argmax(probs_cal_k[:, :n_classes], axis=1)

    if has_calib_unk:
        _, unk_cal_u = score(logits_calib_unknown, calib, alpha)
        all_unk_cal  = np.concatenate([unk_cal_k, unk_cal_u])
    else:
        unk_cal_u   = np.array([], dtype=np.float32)
        all_unk_cal = unk_cal_k

    thresholds = np.linspace(all_unk_cal.min(), all_unk_cal.max(), 100)
    best_thr, best_score = 0.5, -np.inf
    for thr in thresholds:
        accept = unk_cal_k < thr
        if accept.mean() < 0.30:
            continue
        open_acc = (
            float(np.mean(pred_cal_k[accept] == y_calib_known[accept]))
            if accept.any() else 0.0
        )
        unk_det = (
            float(np.mean(unk_cal_u >= thr))
            if has_calib_unk else 0.0
        )
        score_val = (
            0.5 * (open_acc + unk_det) if has_calib_unk
            else open_acc * float(accept.mean())
        )
        if score_val > best_score:
            best_score = score_val
            best_thr   = float(thr)

    if not has_calib_unk:
        best_thr = float(np.quantile(unk_cal_k, 0.95))

    # ── Test evaluation ──────────────────────────────────────────────────
    probs_k, unk_k = score(logits_known,  calib, alpha)
    _,       unk_u = score(logits_unknown, calib, alpha)
    pred_k = np.argmax(probs_k[:, :n_classes], axis=1)

    closed_acc = float(np.mean(pred_k == y_known))
    accept     = unk_k < best_thr
    open_acc   = (
        float(np.mean(pred_k[accept] == y_known[accept]))
        if accept.any() else float('nan')
    )
    unk_det = float(np.mean(unk_u >= best_thr))

    try:
        binary = np.concatenate([
            np.zeros(len(unk_k), dtype=np.int32),
            np.ones(len(unk_u),  dtype=np.int32),
        ])
        auroc = float(roc_auc_score(
            binary, np.concatenate([unk_k, unk_u])
        ))
    except Exception:
        auroc = float('nan')

    return {
        'closed_acc':        closed_acc,
        'open_acc':          open_acc,
        'unk_det':           unk_det,
        'auroc':             auroc,
        'known_accept_rate': float(accept.mean()),
        'best_threshold':    best_thr,
        'tail_size':         tail_size,
        'alpha':             alpha,
    }
