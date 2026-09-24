#!/usr/bin/env python3
"""
tta_bn.py  —  Test-Time Adaptation via BatchNorm statistics update.

When a model trained on Day N is applied to Day N+1 data, the BatchNorm
running statistics (mean / variance) computed during training no longer
match the target-day distribution.  A single forward pass with
training=True over the adaptation set re-estimates these statistics at
zero label cost, recovering several percentage points of accuracy.

Public API
----------
save_bn_statistics(model)                → dict of {name: {mean, var}}
reset_bn_statistics(model, saved_stats)  → None  (in-place restore)
adapt_bn_statistics(model, x_adapt, ...) → None  (in-place update)
evaluate_with_bn_adapt(model, ...)       → float (accuracy)
compare_baseline_vs_adapted(model, ...)  → (baseline, adapted, delta)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bn_layers(model: tf.keras.Model) -> list[tf.keras.layers.BatchNormalization]:
    """Return all BatchNormalization layers in the model."""
    return [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]


# ---------------------------------------------------------------------------
# Save / restore BN statistics
# ---------------------------------------------------------------------------

def save_bn_statistics(model: tf.keras.Model) -> dict:
    """
    Snapshot BN running mean and variance for every BN layer.

    Returns
    -------
    dict  {layer_name: {'mean': np.ndarray, 'var': np.ndarray}}
    """
    return {
        layer.name: {
            'mean': layer.moving_mean.numpy().copy(),
            'var':  layer.moving_variance.numpy().copy(),
        }
        for layer in _bn_layers(model)
    }


def reset_bn_statistics(model: tf.keras.Model, saved_stats: dict) -> None:
    """
    Restore BN running mean/variance from a snapshot produced by
    save_bn_statistics().  Operates in-place.
    """
    for layer in _bn_layers(model):
        if layer.name in saved_stats:
            layer.moving_mean.assign(saved_stats[layer.name]['mean'])
            layer.moving_variance.assign(saved_stats[layer.name]['var'])


# ---------------------------------------------------------------------------
# Core adaptation
# ---------------------------------------------------------------------------

def adapt_bn_statistics(
    model:      tf.keras.Model,
    x_adapt:    np.ndarray,
    batch_size: int = 128,
    n_passes:   int = 1,
) -> None:
    """
    Re-estimate BN running statistics using unlabelled target-domain data.

    Each mini-batch forward pass with training=True causes Keras to update
    the BN moving_mean / moving_variance via exponential moving average.
    No gradients are computed and no weights change — only the BN
    non-trainable statistics are updated.

    Parameters
    ----------
    model      : trained tf.keras.Model (modified in-place)
    x_adapt    : (N, slice_len, 2) adaptation samples — labels not needed
    batch_size : mini-batch size for the EMA update (default 128)
    n_passes   : number of full passes over x_adapt (default 1; 2-3 helps
                 when N is small relative to the number of BN layers)
    """
    x_adapt = np.asarray(x_adapt, dtype=np.float32)
    n       = len(x_adapt)

    print(
        f"[tta_bn] Adapting BN stats on {n} samples "
        f"({n_passes} pass{'es' if n_passes > 1 else ''}) ..."
    )

    for _ in range(n_passes):
        perm   = np.random.permutation(n)
        x_shuf = x_adapt[perm]
        for start in range(0, n, batch_size):
            batch = tf.constant(x_shuf[start: start + batch_size])
            model(batch, training=True)   # updates moving stats, no grad

    print("[tta_bn] BN adaptation complete.")


# ---------------------------------------------------------------------------
# Evaluate with adaptation
# ---------------------------------------------------------------------------

def evaluate_with_bn_adapt(
    model:        tf.keras.Model,
    x_adapt:      np.ndarray,
    x_eval:       np.ndarray,
    y_eval_cat:   np.ndarray,
    batch_size:   int  = 128,
    n_passes:     int  = 1,
    reset_after:  bool = False,
) -> float:
    """
    Adapt BN statistics to x_adapt, then evaluate on (x_eval, y_eval_cat).

    Parameters
    ----------
    model       : tf.keras.Model  trained classification model
    x_adapt     : adaptation data (typically the same as x_eval)
    x_eval      : data to evaluate on
    y_eval_cat  : one-hot labels for x_eval
    batch_size  : inference + adaptation batch size
    n_passes    : passes over x_adapt for BN update (default 1)
    reset_after : if True, restore original BN stats after evaluation
                  (useful when calling inside a loop and the original model
                  must remain untouched for the next iteration)

    Returns
    -------
    accuracy : float
    """
    original_stats = save_bn_statistics(model)
    adapt_bn_statistics(model, x_adapt, batch_size=batch_size, n_passes=n_passes)

    _, acc = model.evaluate(x_eval, y_eval_cat,
                            batch_size=batch_size, verbose=1)
    print(f"[tta_bn] Cross-day accuracy (BN-adapted): {acc:.4f}")

    if reset_after:
        reset_bn_statistics(model, original_stats)
        print("[tta_bn] BN statistics restored to training-day values.")

    return float(acc)


# ---------------------------------------------------------------------------
# Convenience comparison helper
# ---------------------------------------------------------------------------

def compare_baseline_vs_adapted(
    model:        tf.keras.Model,
    x_cross:      np.ndarray,
    y_cross_cat:  np.ndarray,
    batch_size:   int = 128,
    n_passes:     int = 1,
) -> tuple[float, float, float]:
    """
    Evaluate cross-day accuracy *before* and *after* BN adaptation and
    print a comparison.  Always restores original BN stats afterwards.

    Parameters
    ----------
    model       : tf.keras.Model
    x_cross     : cross-day evaluation data  (N, slice_len, 2)
    y_cross_cat : one-hot labels              (N, num_class)
    batch_size  : int
    n_passes    : BN adaptation passes (default 1)

    Returns
    -------
    baseline_acc : float
    adapted_acc  : float
    delta        : float  (adapted - baseline)
    """
    print("\n[tta_bn] ── Baseline (no adaptation) ──")
    _, baseline_acc = model.evaluate(
        x_cross, y_cross_cat, batch_size=batch_size, verbose=1
    )

    print("\n[tta_bn] ── After BN adaptation ──")
    adapted_acc = evaluate_with_bn_adapt(
        model       = model,
        x_adapt     = x_cross,
        x_eval      = x_cross,
        y_eval_cat  = y_cross_cat,
        batch_size  = batch_size,
        n_passes    = n_passes,
        reset_after = True,
    )

    delta = adapted_acc - baseline_acc
    sign  = '+' if delta >= 0 else ''
    print(
        f"\n[tta_bn] Baseline={baseline_acc:.4f}  "
        f"Adapted={adapted_acc:.4f}  Δ={sign}{delta:.4f}"
    )
    return float(baseline_acc), float(adapted_acc), float(delta)
