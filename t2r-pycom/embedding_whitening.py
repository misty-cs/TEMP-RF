#!/usr/bin/env python3
"""
embedding_whitening.py  —  Test-time embedding whitening for cross-day RF fingerprinting.

Background
----------
Even after per-slice CFO removal and RMS normalisation, cross-day channel
variation leaves a residual shift in the embedding space.  Because L2
normalisation is applied to every embedding vector (projecting them onto the
unit hypersphere), this shift appears as a *rotation* rather than a
translation — the whole cloud of Day-N+1 embeddings is tilted relative to
the Day-N cloud.

Three complementary whitening strategies are provided, in increasing power:

  1. mean_shift_whitening   — subtract the centroid difference between
                              source and target clouds (fastest, no labels).

  2. per_class_whitening    — align each device's Day-N+1 centroid to its
                              Day-N centroid individually (needs labels on a
                              small adaptation set).

  3. affine_whitening       — fit a full affine (rotation + scale) mapping
                              from source to target using matched centroids
                              (most powerful, still needs only a handful of
                              labelled samples per device).

All three re-normalise to the unit sphere after correcting, so the output
is always compatible with the existing L2-normalised cosine-distance
classifier or kNN.

Public API
----------
extract_embeddings(model, X, batch_size)     → np.ndarray  (N, emb_dim)
compute_source_centroids(embs, labels)        → dict {cls: centroid}
mean_shift_whitening(emb_src, emb_tgt)        → corrected emb_tgt
per_class_whitening(emb_tgt, y_tgt,
                    src_centroids)            → corrected emb_tgt
affine_whitening(emb_tgt, y_tgt,
                 src_centroids)              → corrected emb_tgt
evaluate_whitening(model, x_src, y_src,
                   x_tgt, y_tgt, ...)        → dict of accuracy results
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model:      tf.keras.Model,
    X:          np.ndarray,
    batch_size: int = 128,
    layer_name: str = 'emb_l2norm',
) -> np.ndarray:
    """
    Extract L2-normalised embeddings from a trained model.

    Parameters
    ----------
    model      : trained tf.keras.Model (DeepFingerprinting / DANN backbone)
    X          : (N, slice_len, 2)  preprocessed IQ data
    batch_size : inference batch size
    layer_name : name of the embedding layer to tap.
                 Default 'emb_l2norm' (post-L2-norm).
                 Use 'embedding' to get raw pre-norm embeddings.

    Returns
    -------
    (N, emb_dim) float32  — unit-norm rows when layer_name='emb_l2norm'
    """
    X_arr = np.asarray(X, dtype=np.float32)

    try:
        emb_model = tf.keras.Model(
            inputs  = model.input,
            outputs = model.get_layer(layer_name).output,
        )
    except ValueError:
        # Fallback: DANN backbone already outputs embeddings directly
        emb_model = model

    if len(X_arr) == 0:
        out_shape = emb_model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        emb_dim = out_shape[-1]
        if emb_dim is None:
            raise ValueError(
                "Cannot infer embedding dimension for empty input. "
                "Pass at least one sample or use a model with a fixed output shape."
            )
        return np.empty((0, int(emb_dim)), dtype=np.float32)

    return emb_model.predict(
        X_arr,
        batch_size = batch_size,
        verbose    = 0,
    )


def _l2_renorm(emb: np.ndarray) -> np.ndarray:
    """Re-project embeddings onto the unit hypersphere after correction."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return (emb / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Source centroid computation  (called once on training / Day-N data)
# ---------------------------------------------------------------------------

def compute_source_centroids(
    emb:    np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute per-class L2-normalised centroid vectors from source embeddings.

    These centroids represent the "canonical" location of each device in
    the embedding space on the training day.  Save them alongside the model
    weights so they can be loaded at test time without re-running inference
    on Day-N data.

    Parameters
    ----------
    emb    : (N, emb_dim)  source-day embeddings (already L2-normalised)
    labels : (N,)          integer class labels

    Returns
    -------
    dict  {class_id (int): centroid (emb_dim,) float32}
    """
    classes   = np.unique(labels)
    centroids = {}
    for cls in classes:
        mask        = labels == cls
        raw         = emb[mask].mean(axis=0)
        centroids[int(cls)] = (raw / (np.linalg.norm(raw) + 1e-8)).astype(np.float32)
    return centroids


def save_centroids(centroids: dict[int, np.ndarray], path: str) -> None:
    """Save centroids dict to a .npz file for later reuse."""
    np.savez(path, **{str(k): v for k, v in centroids.items()})
    print(f"[whitening] Centroids saved → {path}")


def load_centroids(path: str) -> dict[int, np.ndarray]:
    """Load centroids previously saved with save_centroids()."""
    data = np.load(path)
    centroids = {int(k): data[k].astype(np.float32) for k in data.files}
    print(f"[whitening] Centroids loaded from {path}  ({len(centroids)} classes)")
    return centroids


# ---------------------------------------------------------------------------
# Strategy 1: Mean-shift whitening  (no labels needed on target day)
# ---------------------------------------------------------------------------

def mean_shift_whitening(
    emb_src: np.ndarray,
    emb_tgt: np.ndarray,
) -> np.ndarray:
    """
    Remove the global centroid shift between source and target embedding clouds.

    This handles the case where all device embeddings shift together as a
    rigid block — the most common cross-day failure mode.  No labels on the
    target day are required; only a small set of unlabelled target samples
    is needed to estimate the shift.

    The shift is estimated as:
        shift = mean(emb_tgt) - mean(emb_src)

    Then subtracted from every target embedding before re-normalising to
    the unit sphere.

    Parameters
    ----------
    emb_src : (N, emb_dim)  source-day embeddings  (e.g. Day 8 test set)
    emb_tgt : (M, emb_dim)  target-day embeddings  (e.g. held-out test day)

    Returns
    -------
    (M, emb_dim) float32  — shift-corrected, re-normalised embeddings
    """
    shift       = emb_tgt.mean(axis=0) - emb_src.mean(axis=0)
    emb_corrected = emb_tgt - shift
    result      = _l2_renorm(emb_corrected)

    shift_mag = float(np.linalg.norm(shift))
    print(f"[whitening] Mean-shift magnitude: {shift_mag:.5f}")
    return result


# ---------------------------------------------------------------------------
# Strategy 2: Per-class whitening  (needs labels on a small adaptation set)
# ---------------------------------------------------------------------------

def per_class_whitening(
    emb_tgt:       np.ndarray,
    y_tgt:         np.ndarray,
    src_centroids: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Align each device's target-day centroid to its source-day centroid.

    More powerful than mean_shift_whitening when different devices shift
    by different amounts (e.g. different AGC behaviour per device).
    Requires labels for a small adaptation set on the target day — even
    1-5 samples per class is enough to estimate per-device shift.

    For each class c:
        shift_c = centroid(emb_tgt[y==c]) - src_centroids[c]
    Then subtract shift_c from every embedding of class c.

    Parameters
    ----------
    emb_tgt       : (M, emb_dim)  target-day embeddings
    y_tgt         : (M,)          integer class labels for target embeddings
    src_centroids : dict {cls: (emb_dim,)}  from compute_source_centroids()

    Returns
    -------
    (M, emb_dim) float32  — per-class shift corrected, re-normalised
    """
    emb_corrected = emb_tgt.copy()
    classes       = np.unique(y_tgt)
    total_shift   = 0.0

    for cls in classes:
        if int(cls) not in src_centroids:
            print(f"  [whitening] WARNING: class {cls} not in src_centroids — skipping")
            continue
        mask     = y_tgt == cls
        tgt_cent = emb_tgt[mask].mean(axis=0)
        shift    = tgt_cent - src_centroids[int(cls)]
        emb_corrected[mask] -= shift
        total_shift += float(np.linalg.norm(shift))

    mean_shift = total_shift / max(len(classes), 1)
    print(f"[whitening] Per-class mean shift magnitude: {mean_shift:.5f}  "
          f"(over {len(classes)} classes)")

    return _l2_renorm(emb_corrected)


# ---------------------------------------------------------------------------
# Strategy 3: Affine whitening  (rotation + scale alignment)
# ---------------------------------------------------------------------------

def affine_whitening(
    emb_tgt:       np.ndarray,
    y_tgt:         np.ndarray,
    src_centroids: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Fit an affine (linear + bias) mapping from target centroids to source
    centroids and apply it to all target embeddings.

    This handles not just translation but also rotation and per-dimension
    scale — corrects more complex cross-day distributional shift than
    mean_shift_whitening or per_class_whitening can.

    The mapping W, b is fitted by least squares:
        min ||C_tgt @ W.T + b - C_src||_F

    where C_tgt / C_src are matrices of matched centroids.  At least
    num_classes > emb_dim samples are needed for a well-determined system;
    in practice having ≥2 samples per class is sufficient to estimate stable
    centroids.

    Parameters
    ----------
    emb_tgt       : (M, emb_dim)  target-day embeddings
    y_tgt         : (M,)          integer class labels
    src_centroids : dict {cls: (emb_dim,)}  from compute_source_centroids()

    Returns
    -------
    (M, emb_dim) float32  — affine-corrected, re-normalised embeddings

    Notes
    -----
    Falls back to per_class_whitening if fewer than 2 matched classes exist,
    or if the least-squares system is degenerate (rank-deficient centroids).
    """
    classes = sorted(
        c for c in np.unique(y_tgt) if int(c) in src_centroids
    )

    if len(classes) < 2:
        print("[whitening] Not enough classes for affine fit — "
              "falling back to per_class_whitening")
        return per_class_whitening(emb_tgt, y_tgt, src_centroids)

    # Build matched centroid matrices
    C_src = np.stack([src_centroids[int(c)] for c in classes])          # (K, D)
    C_tgt = np.stack([emb_tgt[y_tgt == c].mean(axis=0) for c in classes])  # (K, D)

    # Augment with bias column: solve [C_tgt | 1] @ [W; b] = C_src
    ones    = np.ones((len(classes), 1), dtype=np.float32)
    C_tgt_a = np.concatenate([C_tgt, ones], axis=1)                    # (K, D+1)

    try:
        # Least-squares solution: shape (D+1, D)
        sol, residuals, rank, sv = np.linalg.lstsq(C_tgt_a, C_src, rcond=None)
        W = sol[:-1].T   # (D, D)
        b = sol[-1]      # (D,)

        # Condition number check — if very ill-conditioned, fall back
        cond = float(sv[0]) / (float(sv[-1]) + 1e-12) if len(sv) > 0 else np.inf
        if cond > 1e6:
            print(f"[whitening] Affine system ill-conditioned (cond={cond:.1e}) — "
                  "falling back to per_class_whitening")
            return per_class_whitening(emb_tgt, y_tgt, src_centroids)

        emb_corrected = (emb_tgt @ W.T + b).astype(np.float32)
        print(f"[whitening] Affine fit: rank={rank}  cond={cond:.1e}  "
              f"classes={len(classes)}")

    except np.linalg.LinAlgError as e:
        print(f"[whitening] Affine lstsq failed ({e}) — "
              "falling back to per_class_whitening")
        return per_class_whitening(emb_tgt, y_tgt, src_centroids)

    return _l2_renorm(emb_corrected)


# ---------------------------------------------------------------------------
# KNN classifier on embeddings  (used by evaluate_whitening)
# ---------------------------------------------------------------------------

def _knn_accuracy(
    emb_src:  np.ndarray,
    y_src:    np.ndarray,
    emb_tgt:  np.ndarray,
    y_tgt:    np.ndarray,
    k:        int = 1,
) -> float:
    """
    Cosine-distance kNN accuracy: train on src embeddings, evaluate on tgt.
    Since embeddings are L2-normalised, cosine similarity = dot product.
    """
    # (M, N) dot product matrix — each row is one target sample vs all source
    sim   = emb_tgt @ emb_src.T                     # (M, N)
    top_k = np.argsort(sim, axis=1)[:, -k:]         # (M, k) indices
    votes = y_src[top_k]                             # (M, k) labels

    if k == 1:
        preds = votes[:, 0]
    else:
        # Majority vote
        preds = np.array([
            np.bincount(row, minlength=int(y_src.max()) + 1).argmax()
            for row in votes
        ])

    return float(np.mean(preds == y_tgt))


# ---------------------------------------------------------------------------
# Convenience: run all strategies and compare
# ---------------------------------------------------------------------------

def evaluate_whitening(
    model:         tf.keras.Model,
    x_src:         np.ndarray,
    y_src:         np.ndarray,
    x_tgt:         np.ndarray,
    y_tgt:         np.ndarray,
    x_tgt_adapt:   np.ndarray | None = None,
    y_tgt_adapt:   np.ndarray | None = None,
    batch_size:    int = 128,
    knn_k:         int = 1,
    layer_name:    str = 'emb_l2norm',
) -> dict[str, float]:
    """
    Extract embeddings, apply label-free whitening, and optionally compare
    labelled-adaptation whitening when a separate adaptation set is supplied.
    Also reports the softmax classifier accuracy for reference.

    Parameters
    ----------
    model         : trained tf.keras.Model
    x_src         : (N, slice_len, 2)  source-day (e.g. Day 8) IQ data,
                    preprocessed and z-score normalised
    y_src         : (N,)  integer labels for x_src
    x_tgt         : (M, slice_len, 2)  target-day IQ data,
                    preprocessed and z-score normalised
    y_tgt         : (M,)  integer labels for x_tgt
    x_tgt_adapt   : optional small labelled adaptation set on the target day.
                    Required for per_class and affine whitening. If omitted,
                    those metrics are reported as NaN instead of using target
                    evaluation labels.
    y_tgt_adapt   : labels for x_tgt_adapt
    batch_size    : inference batch size
    knn_k         : k for kNN classifier (default 1)
    layer_name    : embedding layer to tap

    Returns
    -------
    dict {
        'baseline_softmax': float,   # model's own softmax on target day
        'baseline_knn':     float,   # 1-NN before whitening
        'mean_shift':       float,   # after mean-shift whitening
        'per_class':        float,   # after per-class whitening (needs labels)
        'affine':           float,   # after affine whitening (needs labels)
    }
    """
    print("\n[whitening] Extracting source embeddings ...")
    emb_src = extract_embeddings(model, x_src, batch_size, layer_name)

    print("[whitening] Extracting target embeddings ...")
    emb_tgt = extract_embeddings(model, x_tgt, batch_size, layer_name)

    # Separate adaptation set for per-class / affine. Never fall back to the
    # evaluation set because that leaks target labels into the transform.
    has_adapt = x_tgt_adapt is not None and y_tgt_adapt is not None
    if has_adapt:
        print("[whitening] Extracting adaptation embeddings ...")
        emb_adapt = extract_embeddings(model, x_tgt_adapt, batch_size, layer_name)
        y_adapt   = np.asarray(y_tgt_adapt)
        if len(emb_adapt) != len(y_adapt):
            raise ValueError(
                f"x_tgt_adapt/y_tgt_adapt length mismatch: "
                f"{len(emb_adapt)} vs {len(y_adapt)}"
            )
    else:
        emb_adapt = None
        y_adapt   = None

    y_src_arr = np.asarray(y_src)
    y_tgt_arr = np.asarray(y_tgt)

    # Source centroids
    src_centroids = compute_source_centroids(emb_src, y_src_arr)

    # Softmax baseline (model's own head on target day)
    print("\n[whitening] Softmax baseline (no whitening) ...")
    softmax_preds = model.predict(
        np.asarray(x_tgt, dtype=np.float32),
        batch_size=batch_size, verbose=0,
    )
    # Handle DANN multi-output
    if isinstance(softmax_preds, (list, tuple)):
        softmax_preds = softmax_preds[0]
    baseline_softmax = float(
        np.mean(np.argmax(softmax_preds, axis=1) == y_tgt_arr)
    )

    # kNN baseline (no whitening)
    baseline_knn = _knn_accuracy(emb_src, y_src_arr, emb_tgt, y_tgt_arr, knn_k)

    # Strategy 1: mean-shift (no labels)
    print("\n[whitening] Strategy 1 — mean-shift ...")
    emb_ms   = mean_shift_whitening(emb_src, emb_tgt)
    acc_ms   = _knn_accuracy(emb_src, y_src_arr, emb_ms, y_tgt_arr, knn_k)

    acc_pc = float('nan')
    acc_aff = float('nan')

    classes = np.array(sorted(src_centroids))
    centroid_matrix = np.stack([src_centroids[int(c)] for c in classes])

    if has_adapt:
        # Strategy 2: per-class shifts learned from the adaptation set. Eval
        # samples are assigned by nearest source centroid, not by true labels.
        print("\n[whitening] Strategy 2 — per-class adaptation ...")
        shifts: dict[int, np.ndarray] = {}
        for cls in np.unique(y_adapt):
            cls_int = int(cls)
            if cls_int not in src_centroids:
                continue
            mask = y_adapt == cls
            shifts[cls_int] = emb_adapt[mask].mean(axis=0) - src_centroids[cls_int]

        pred_cls = classes[np.argmax(emb_tgt @ centroid_matrix.T, axis=1)]
        emb_pc_eval = emb_tgt.copy()
        for cls, shift in shifts.items():
            emb_pc_eval[pred_cls == cls] -= shift
        emb_pc_eval = _l2_renorm(emb_pc_eval)
        acc_pc = _knn_accuracy(emb_src, y_src_arr, emb_pc_eval, y_tgt_arr, knn_k)

        # Strategy 3: affine mapping fitted on adaptation centroids and applied
        # globally to eval embeddings, with no eval labels involved.
        print("\n[whitening] Strategy 3 — affine adaptation ...")
        matched = sorted(c for c in np.unique(y_adapt) if int(c) in src_centroids)
        if len(matched) >= 2:
            C_src = np.stack([src_centroids[int(c)] for c in matched])
            C_adapt = np.stack([emb_adapt[y_adapt == c].mean(axis=0) for c in matched])
            C_adapt_a = np.concatenate(
                [C_adapt, np.ones((len(matched), 1), dtype=np.float32)],
                axis=1,
            )
            try:
                sol, _, _, sv = np.linalg.lstsq(C_adapt_a, C_src, rcond=None)
                cond = float(sv[0]) / (float(sv[-1]) + 1e-12) if len(sv) else np.inf
                if cond <= 1e6:
                    W = sol[:-1].T
                    b = sol[-1]
                    emb_aff_eval = _l2_renorm((emb_tgt @ W.T + b).astype(np.float32))
                    acc_aff = _knn_accuracy(
                        emb_src, y_src_arr, emb_aff_eval, y_tgt_arr, knn_k
                    )
                else:
                    print(f"[whitening] Affine skipped: ill-conditioned cond={cond:.1e}")
            except np.linalg.LinAlgError as e:
                print(f"[whitening] Affine skipped: {e}")
        else:
            print("[whitening] Affine skipped: fewer than 2 adapted classes.")
    else:
        print("\n[whitening] Labelled whitening skipped: no separate adaptation set.")

    results = {
        'baseline_softmax': baseline_softmax,
        'baseline_knn':     baseline_knn,
        'mean_shift':       acc_ms,
        'per_class':        acc_pc,
        'affine':           acc_aff,
    }

    # Pretty print
    print(f"\n{'─'*52}")
    print(f"  {'Method':<28} {'kNN Acc':>10}")
    print(f"{'─'*52}")
    print(f"  {'Softmax (no whitening)':<28} {baseline_softmax:>10.4f}")
    print(f"  {'kNN baseline (no whitening)':<28} {baseline_knn:>10.4f}")
    print(f"  {'Mean-shift (no labels)':<28} {acc_ms:>10.4f}"
          f"  Δ={acc_ms - baseline_knn:+.4f}")
    print(f"  {'Per-class (few labels)':<28} {acc_pc:>10.4f}"
          f"  Δ={acc_pc - baseline_knn:+.4f}")
    print(f"  {'Affine (few labels)':<28} {acc_aff:>10.4f}"
          f"  Δ={acc_aff - baseline_knn:+.4f}")
    print(f"{'─'*52}\n")

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tensorflow as tf
    import sys
    sys.path.insert(0, '.')

    print("=== Synthetic self-test ===\n")
    rng  = np.random.default_rng(0)
    D    = 64
    N    = 200
    K    = 10   # classes

    # Simulate source embeddings: K tight clusters on unit sphere
    centres_src = _l2_renorm(rng.standard_normal((K, D)).astype(np.float32))
    y_src       = np.repeat(np.arange(K), N // K)
    emb_src     = _l2_renorm(
        centres_src[y_src] + rng.standard_normal((N, D)).astype(np.float32) * 0.05
    )

    # Simulate target embeddings: same clusters but with a global rotation-shift
    global_shift = rng.standard_normal(D).astype(np.float32) * 0.3
    y_tgt        = y_src.copy()
    emb_tgt      = _l2_renorm(
        centres_src[y_tgt]
        + rng.standard_normal((N, D)).astype(np.float32) * 0.05
        + global_shift
    )

    src_centroids = compute_source_centroids(emb_src, y_src)

    print("Before whitening:")
    acc_before = _knn_accuracy(emb_src, y_src, emb_tgt, y_tgt, k=1)
    print(f"  kNN accuracy: {acc_before:.4f}")

    print("\nMean-shift whitening:")
    emb_ms = mean_shift_whitening(emb_src, emb_tgt)
    acc_ms = _knn_accuracy(emb_src, y_src, emb_ms, y_tgt, k=1)
    print(f"  kNN accuracy: {acc_ms:.4f}")
    assert acc_ms >= acc_before - 0.01, "Mean-shift should not hurt accuracy"

    print("\nPer-class whitening:")
    emb_pc = per_class_whitening(emb_tgt, y_tgt, src_centroids)
    acc_pc = _knn_accuracy(emb_src, y_src, emb_pc, y_tgt, k=1)
    print(f"  kNN accuracy: {acc_pc:.4f}")

    print("\nAffine whitening:")
    emb_af = affine_whitening(emb_tgt, y_tgt, src_centroids)
    acc_af = _knn_accuracy(emb_src, y_src, emb_af, y_tgt, k=1)
    print(f"  kNN accuracy: {acc_af:.4f}")

    assert acc_af >= acc_before - 0.01, "Affine whitening should not hurt accuracy"
    print("\nSelf-test passed.")
