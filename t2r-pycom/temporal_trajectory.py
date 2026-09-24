#!/usr/bin/env python3
"""
temporal_trajectory.py  —  Multi-day temporal trajectory classifier for RF fingerprinting.

Motivation
----------
A static prototype (single mean embedding per device) cannot account for
natural drift: temperature changes, hardware aging, and channel variation
shift each device's embedding cloud day-by-day.  By modelling each device
as a trajectory of (mean, covariance) pairs — one per observed day — the
classifier follows the device fingerprint over time instead of assuming it
is fixed.

Architecture
------------
For each known device d and observed day t:
    μ_d,t  = mean of embeddings from device d on day t
    Σ_d,t  = covariance of those embeddings

At test time we use the most recent (μ, Σ) pair to compute:
    D_M(x, d) = sqrt( (x - μ_d)^T  Σ_d^{-1}  (x - μ_d) )

Decision rule:
    d* = argmin_d  D_M(x, d)          (closest known device)
    if D_M(x, d*) > threshold  →  UNKNOWN  (open-set rejection)

Integration points
------------------
phase3_trajectory_building.py builds trajectories from calibration-day and
fine-tune-day embeddings. phase4_openset_evaluation.py evaluates them on
the held-out test day.

finetune.py        —  CNN.tune() already has X_src_ppz (Day N) and
                       X_tr_ppz (Day N+1).  Both feed update() before
                       the existing whitening / proto evaluation.

Public API
----------
TemporalTrajectory          — main class
  .update(day_id, emb, y)   — add a day's embeddings to the trajectory
  .mahalanobis(x, device)   — scalar distance for one sample
  .classify_open_set(emb, threshold, top_k)
                            — (N,) int array, -1 = unknown
  .evaluate(emb, y_true, known_ids, emb_unk)
                            — dict of metrics
  .save(path) / .load(path) — persist/restore across sessions

trajectory_threshold_sweep(traj, emb, y, emb_unk)
                            — AUROC curve + optimal threshold

"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, roc_curve,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------

_MIN_SAMPLES_FOR_COV = 3   # need at least 3 samples to estimate covariance
_REG_EPSILON         = 1e-4  # Tikhonov regularisation added to diagonal
_KNOWN_ONLY_ACCEPT_RATE = 0.70  # fraction of known samples accepted when no unknowns available


def _has_calib_unknown(emb_unknown: Optional[np.ndarray]) -> bool:
    return emb_unknown is not None and len(emb_unknown) > 0


def _known_only_distance_threshold(dists_known: np.ndarray) -> float:
    """Set threshold at the _KNOWN_ONLY_ACCEPT_RATE quantile of known distances."""
    return float(np.quantile(dists_known, _KNOWN_ONLY_ACCEPT_RATE))


def _regularised_cov(emb: np.ndarray, eps: float = _REG_EPSILON) -> np.ndarray:
    """
    Covariance matrix for Mahalanobis distance on L2-normalised embeddings.

    Uses Ledoit-Wolf shrinkage when n > D, capturing off-diagonal correlations
    that are significant for hypersphere embeddings trained with SupCon.
    Falls back to diagonal covariance when sample count is insufficient or
    sklearn is unavailable.

    emb : (N, D) float32  — already L2-normalised embeddings
    eps : float           — regularisation floor added to diagonal
    """
    emb = emb.astype(np.float64)
    D   = emb.shape[1]
    n   = len(emb)

    if n > D:
        try:
            from sklearn.covariance import LedoitWolf
            lw  = LedoitWolf(assume_centered=False)
            lw.fit(emb)
            cov = lw.covariance_
            cov += np.eye(D, dtype=np.float64) * eps
            return cov
        except Exception:
            pass

    # Fallback: diagonal covariance
    var = np.var(emb, axis=0)
    return np.diag(var + eps)


def _safe_inv(cov: np.ndarray) -> np.ndarray:
    """
    Invert a covariance matrix, falling back to the pseudo-inverse if
    the matrix is singular or near-singular.
    """
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class TemporalTrajectory:
    """
    Multi-day trajectory classifier using Mahalanobis distance.

    Each call to update() appends a new (mean, covariance) pair for that
    day.  At inference time the most recent statistics are used.

    Parameters
    ----------
    n_days_keep : int
        Maximum number of historical days to retain.  Older entries are
        dropped.  Default 10.  Pass None to keep all.
    reg_epsilon : float
        Tikhonov regularisation on the covariance diagonal.  Default 1e-4.
    ewma_alpha : float | None
        If set (0 < alpha <= 1), the running mean is updated as an
        exponentially weighted moving average instead of computing a fresh
        mean per day.  This smooths the trajectory.
        alpha=1.0 → no smoothing (same as default), alpha=0.1 → heavy.
    """

    def __init__(
        self,
        n_days_keep: int  = 10,
        reg_epsilon: float = _REG_EPSILON,
        ewma_alpha:  Optional[float] = None,
        n_cosine_prototypes: int = 5,
        cosine_days_keep: int = 3,
        cosine_day_decay: float = 0.6,
    ):
        self.n_days_keep = n_days_keep
        self.reg_epsilon = reg_epsilon
        self.ewma_alpha  = ewma_alpha
        self.n_cosine_prototypes = n_cosine_prototypes
        self.cosine_days_keep = cosine_days_keep
        self.cosine_day_decay = cosine_day_decay

        # {device_id: list of {'day': int, 'mean': (D,), 'cov': (D,D),
        #                       'cov_inv': (D,D), 'n': int,
        #                       'cos_proto': (P,D), 'cos_weight': (P,)}}
        self._history: dict[int, list[dict]] = {}
        self._day_ids: list[int] = []   # ordered list of observed day IDs

    def _cosine_prototypes(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a small set of L2-normalised prototypes for one device/day.

        Multiple prototypes preserve multimodal device clouds that a single
        mean would collapse. KMeans is preferred; deterministic quantile
        chunks are used as a dependency-safe fallback.
        """
        emb = np.asarray(emb, dtype=np.float32)
        emb_l2 = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        n = len(emb_l2)
        k = max(1, min(int(self.n_cosine_prototypes), n))

        if k == 1:
            proto = emb_l2.mean(axis=0, keepdims=True)
            weight = np.array([float(n)], dtype=np.float32)
        else:
            try:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=0,
                    batch_size=min(1024, max(64, n)),
                    n_init=3,
                )
                labels = km.fit_predict(emb_l2)
                proto = km.cluster_centers_.astype(np.float32)
                weight = np.bincount(labels, minlength=k).astype(np.float32)
            except Exception:
                order = np.argsort(emb_l2[:, 0])
                chunks = np.array_split(order, k)
                proto = np.stack([emb_l2[idx].mean(axis=0) for idx in chunks])
                weight = np.array([len(idx) for idx in chunks], dtype=np.float32)

        proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)
        weight = weight / (weight.sum() + 1e-8)
        return proto.astype(np.float32), weight.astype(np.float32)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        day_id:   int,
        emb:      np.ndarray,
        labels:   np.ndarray,
    ) -> None:
        """
        Add embeddings from one day to the trajectory.

        Parameters
        ----------
        day_id : int
            Identifier for this day (e.g. 1, 2, 3).  Must be strictly
            greater than any previously observed day_id.
        emb    : (N, D) float32  L2-normalised embeddings
        labels : (N,)  int       device class labels
        """
        emb    = np.asarray(emb,    dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        if day_id not in self._day_ids:
            self._day_ids.append(day_id)

        devices = np.unique(labels)
        for dev in devices:
            mask = labels == dev
            dev_emb = emb[mask]
            n       = len(dev_emb)

            mean_new = dev_emb.mean(axis=0)

            # EWMA smoothing on the mean if enabled
            if self.ewma_alpha is not None and dev in self._history \
                    and len(self._history[dev]) > 0:
                prev_mean = self._history[dev][-1]['mean']
                alpha     = self.ewma_alpha
                mean_new  = alpha * mean_new + (1.0 - alpha) * prev_mean

            if n >= _MIN_SAMPLES_FOR_COV:
                cov     = _regularised_cov(dev_emb, self.reg_epsilon)
                cov_inv = _safe_inv(cov)
            else:
                # Fallback: identity-scaled covariance from previous day
                D       = emb.shape[1]
                if dev in self._history and len(self._history[dev]) > 0:
                    cov     = self._history[dev][-1]['cov'].copy()
                    cov_inv = self._history[dev][-1]['cov_inv'].copy()
                else:
                    cov     = np.eye(D) * self.reg_epsilon
                    cov_inv = np.eye(D) / self.reg_epsilon

            cos_proto, cos_weight = self._cosine_prototypes(dev_emb)

            entry = {
                'day':     day_id,
                'mean':    mean_new.astype(np.float32),
                'cov':     cov.astype(np.float32),
                'cov_inv': cov_inv.astype(np.float32),
                'n':       n,
                'cos_proto': cos_proto,
                'cos_weight': cos_weight,
            }

            if dev not in self._history:
                self._history[dev] = []
            self._history[dev].append(entry)

            # Trim to n_days_keep
            if self.n_days_keep is not None:
                self._history[dev] = self._history[dev][-self.n_days_keep:]

        print(
            f"[trajectory] day={day_id}  devices updated={len(devices)}  "
            f"total_known={len(self._history)}"
        )

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def mahalanobis(
        self,
        x:         np.ndarray,
        device_id: int,
        day_offset: int = -1,
    ) -> float:
        """
        Mahalanobis distance from sample x to device_id's trajectory point.

        Parameters
        ----------
        x          : (D,) embedding vector
        device_id  : which device's statistics to use
        day_offset : which historical entry to use.  -1 = most recent.

        Returns
        -------
        float  distance (0 = identical to prototype)
        """
        x = np.asarray(x, dtype=np.float64)
        entry = self._history[device_id][day_offset]
        diff  = x - entry['mean'].astype(np.float64)
        cov_i = entry['cov_inv'].astype(np.float64)
        dist2 = float(diff @ cov_i @ diff)
        return float(np.sqrt(max(dist2, 0.0)))

    def mahalanobis_batch(
        self,
        emb:        np.ndarray,
        device_id:  int,
        day_offset: int = -1,
    ) -> np.ndarray:
        """
        Mahalanobis distances for a batch of embeddings to one device.

        Parameters
        ----------
        emb        : (N, D)
        device_id  : device whose (μ, Σ⁻¹) to use
        day_offset : trajectory index (-1 = latest)

        Returns
        -------
        (N,) float32  distances
        """
        emb   = np.asarray(emb, dtype=np.float64)
        entry = self._history[device_id][day_offset]
        mu    = entry['mean'].astype(np.float64)
        cov_i = entry['cov_inv'].astype(np.float64)
        diff  = emb - mu                          # (N, D)
        # (N,D) @ (D,D) → (N,D) → element-wise * diff → (N,D) → sum → (N,)
        dist2 = np.sum((diff @ cov_i) * diff, axis=1)
        return np.sqrt(np.maximum(dist2, 0.0)).astype(np.float32)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_open_set(
        self,
        emb:       np.ndarray,
        threshold: float,
        known_ids: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify embeddings with open-set rejection.

        For each sample:
            d* = argmin_{d in known_ids} D_M(x, d)
            if D_M(x, d*) > threshold  →  label = -1  (UNKNOWN)

        Parameters
        ----------
        emb       : (N, D) float32  L2-normalised embeddings
        threshold : float  rejection threshold on Mahalanobis distance
        known_ids : list of device IDs to consider.
                    None → use all devices in the trajectory.

        Returns
        -------
        preds      : (N,) int32   predicted device IDs, -1 = unknown
        min_dists  : (N,) float32 Mahalanobis distance to nearest device
        """
        emb = np.asarray(emb, dtype=np.float32)
        if known_ids is None:
            known_ids = sorted(self._history.keys())

        N            = len(emb)
        dist_matrix  = np.zeros((N, len(known_ids)), dtype=np.float32)

        for j, dev in enumerate(known_ids):
            if dev not in self._history or len(self._history[dev]) == 0:
                dist_matrix[:, j] = np.inf
            else:
                dist_matrix[:, j] = self.mahalanobis_batch(emb, dev)

        best_j    = np.argmin(dist_matrix, axis=1)    # (N,)
        min_dists = dist_matrix[np.arange(N), best_j] # (N,)
        preds     = np.array(
            [known_ids[j] for j in best_j], dtype=np.int32
        )
        preds[min_dists > threshold] = -1

        return preds, min_dists

    def classify_open_set_history(
        self,
        emb:       np.ndarray,
        threshold: float,
        known_ids: Optional[list[int]] = None,
        days_keep: Optional[int] = None,
        day_decay: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify using the best Mahalanobis match across recent trajectory days.

        The original classifier uses only the latest trajectory point. That is
        brittle when drift is non-monotonic or Day 8 looks more like an older
        pre-test day for some devices. This variant keeps the trajectory nature
        while letting each device match against its recent history.
        """
        emb = np.asarray(emb, dtype=np.float32)
        if known_ids is None:
            known_ids = sorted(self._history.keys())

        dist_matrix = np.zeros((len(emb), len(known_ids)), dtype=np.float32)
        for j, dev in enumerate(known_ids):
            if dev not in self._history or len(self._history[dev]) == 0:
                dist_matrix[:, j] = np.inf
                continue

            entries = self._history[dev]
            if days_keep is not None:
                entries = entries[-days_keep:]

            per_day = []
            n_entries = len(entries)
            for i, _ in enumerate(entries):
                hist_offset = -n_entries + i
                d = self.mahalanobis_batch(emb, dev, day_offset=hist_offset)
                if day_decay < 1.0:
                    recency_power = n_entries - i - 1
                    d = d / (float(day_decay) ** recency_power + 1e-8)
                per_day.append(d)
            dist_matrix[:, j] = np.min(np.stack(per_day, axis=1), axis=1)

        best_j = np.argmin(dist_matrix, axis=1)
        min_dists = dist_matrix[np.arange(len(emb)), best_j]
        preds = np.array([known_ids[j] for j in best_j], dtype=np.int32)
        preds[min_dists > threshold] = -1
        return preds, min_dists

    def cosine_prototype_bank(
        self,
        device_id: int,
        days_keep: Optional[int] = None,
        day_decay: Optional[float] = None,
        align_drift: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return recent multi-prototype cosine anchors and weights for a device.
        More recent days receive larger weights.

        When ``align_drift`` is True, each day's prototype cloud is re-centred
        onto the most recent day's location using the known per-day device
        means (drift = ref_mean - day_mean). This removes inter-day drift so
        prototypes from older days line up with the latest frame and can
        contribute instead of adding misaligned noise — letting multi-day
        history help rather than collapsing to days_keep=1.
        """
        entries = self._history[device_id]
        if days_keep is None:
            days_keep = self.cosine_days_keep
        if day_decay is None:
            day_decay = self.cosine_day_decay
        if days_keep is not None:
            entries = entries[-days_keep:]

        ref_n = None
        if align_drift and entries:
            ref_mean = entries[-1]['mean'].astype(np.float32)
            ref_n = ref_mean / (np.linalg.norm(ref_mean) + 1e-8)

        protos = []
        weights = []
        n_entries = len(entries)
        for i, entry in enumerate(entries):
            if 'cos_proto' in entry:
                p = entry['cos_proto'].astype(np.float32)
                w = entry['cos_weight'].astype(np.float32)
            else:
                p = entry['mean'].astype(np.float32)[None, :]
                p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
                w = np.ones(1, dtype=np.float32)
            if ref_n is not None:
                day_mean = entry['mean'].astype(np.float32)
                day_n = day_mean / (np.linalg.norm(day_mean) + 1e-8)
                p = p + (ref_n - day_n)[None, :]
                p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
            recency_power = n_entries - i - 1
            recency_weight = float(day_decay) ** recency_power
            protos.append(p)
            weights.append(w * recency_weight)

        proto = np.concatenate(protos, axis=0)
        weight = np.concatenate(weights, axis=0)
        weight = weight / (weight.max() + 1e-8)
        return proto.astype(np.float32), weight.astype(np.float32)

    def classify_cosine_prototypes(
        self,
        emb: np.ndarray,
        threshold: float,
        known_ids: Optional[list[int]] = None,
        days_keep: Optional[int] = None,
        day_decay: Optional[float] = None,
        align_drift: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Open-set classifier using recent multi-prototype cosine trajectory.
        Distance is 1 minus the best weighted cosine similarity per device.
        """
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        if known_ids is None:
            known_ids = sorted(self._history.keys())

        dist_matrix = np.zeros((len(emb), len(known_ids)), dtype=np.float32)
        for j, dev in enumerate(known_ids):
            if dev not in self._history or len(self._history[dev]) == 0:
                dist_matrix[:, j] = np.inf
                continue
            proto, weight = self.cosine_prototype_bank(
                dev, days_keep=days_keep, day_decay=day_decay,
                align_drift=align_drift,
            )
            sim = emb @ proto.T
            sim = sim * (0.9 + 0.1 * weight[None, :])
            dist_matrix[:, j] = 1.0 - sim.max(axis=1)

        best_j = np.argmin(dist_matrix, axis=1)
        min_dists = dist_matrix[np.arange(len(emb)), best_j]
        preds = np.array([known_ids[j] for j in best_j], dtype=np.int32)
        preds[min_dists > threshold] = -1
        return preds, min_dists

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        emb_known:   np.ndarray,
        y_known:     np.ndarray,
        known_ids:   list[int],
        emb_unknown: Optional[np.ndarray] = None,
        threshold:   float = 5.0,
        verbose:     bool  = True,
    ) -> dict:
        """
        Full evaluation: closed-set accuracy + open-set metrics.

        Parameters
        ----------
        emb_known   : (N, D)  embeddings of known-device test samples
        y_known     : (N,)    true device labels
        known_ids   : list    device IDs considered known
        emb_unknown : (M, D)  embeddings of unknown-device test samples.
                      If None, open-set metrics are skipped.
        threshold   : Mahalanobis rejection threshold (default 5.0)
        verbose     : print summary table

        Returns
        -------
        dict with keys:
            closed_acc   : float  accuracy on known devices (no rejection)
            open_known_acc : float  accuracy on known devices (with rejection)
            open_unk_det : float  fraction of unknowns correctly rejected
            auroc        : float  (requires emb_unknown)
            f1_macro     : float  macro F1 over known classes
            threshold    : float  threshold used
        """
        emb_known = np.asarray(emb_known, dtype=np.float32)
        y_known   = np.asarray(y_known,   dtype=np.int32)

        results: dict = {'threshold': threshold}

        # ── Closed-set: nearest Mahalanobis, no rejection ───────────────
        preds_closed, dists_known = self.classify_open_set(
            emb_known, threshold=np.inf, known_ids=known_ids
        )
        results['closed_acc'] = float(
            np.mean(preds_closed == y_known)
        )

        # ── Open-set on known devices: same but with rejection ──────────
        preds_open, _ = self.classify_open_set(
            emb_known, threshold=threshold, known_ids=known_ids
        )
        # accuracy only on samples not rejected
        accepted_mask   = preds_open != -1
        if accepted_mask.sum() > 0:
            results['open_known_acc'] = float(
                np.mean(preds_open[accepted_mask] == y_known[accepted_mask])
            )
        else:
            results['open_known_acc'] = float('nan')
        results['known_accept_rate'] = float(accepted_mask.mean())

        # ── F1 on known classes ─────────────────────────────────────────
        if _HAS_SKLEARN:
            results['f1_macro'] = float(
                f1_score(y_known, preds_closed, average='macro',
                         zero_division=0)
            )
        else:
            results['f1_macro'] = float('nan')

        # ── Unknown detection + AUROC ───────────────────────────────────
        if emb_unknown is not None:
            emb_unknown = np.asarray(emb_unknown, dtype=np.float32)
            _, dists_unk = self.classify_open_set(
                emb_unknown, threshold=np.inf, known_ids=known_ids
            )

            # Fraction of unknowns correctly rejected at this threshold
            results['open_unk_det'] = float(
                np.mean(dists_unk > threshold)
            )

            if _HAS_SKLEARN:
                # Binary AUROC: known=0, unknown=1
                # Higher Mahalanobis distance → more likely unknown → use
                # distance directly as the score for the "unknown" class.
                scores = np.concatenate([dists_known, dists_unk])
                binary = np.concatenate([
                    np.zeros(len(dists_known), dtype=np.int32),
                    np.ones( len(dists_unk),   dtype=np.int32),
                ])
                try:
                    results['auroc'] = float(roc_auc_score(binary, scores))
                except ValueError:
                    results['auroc'] = float('nan')
            else:
                results['auroc'] = float('nan')
        else:
            results['open_unk_det'] = float('nan')
            results['auroc']        = float('nan')

        if verbose:
            sep = '─' * 52
            print(f"\n{sep}")
            print(f"  Temporal Trajectory Evaluation  (threshold={threshold:.2f})")
            print(sep)
            print(f"  {'Closed-set accuracy':<28} {results['closed_acc']:>8.4f}")
            print(f"  {'Open known-set accuracy':<28} {results['open_known_acc']:>8.4f}")
            print(f"  {'Known accept rate':<28} {results['known_accept_rate']:>8.4f}")
            print(f"  {'Unknown detection rate':<28} {results['open_unk_det']:>8.4f}")
            print(f"  {'AUROC':<28} {results['auroc']:>8.4f}")
            print(f"  {'F1 macro':<28} {results['f1_macro']:>8.4f}")
            print(sep)

        return results

    def evaluate_history(
        self,
        emb_known:   np.ndarray,
        y_known:     np.ndarray,
        known_ids:   list[int],
        emb_unknown: Optional[np.ndarray] = None,
        threshold:   float = 5.0,
        days_keep:   Optional[int] = None,
        day_decay:   float = 1.0,
        verbose:     bool  = True,
    ) -> dict:
        """Evaluate the multi-day history Mahalanobis classifier."""
        emb_known = np.asarray(emb_known, dtype=np.float32)
        y_known   = np.asarray(y_known,   dtype=np.int32)

        preds_closed, dists_known = self.classify_open_set_history(
            emb_known, threshold=np.inf, known_ids=known_ids,
            days_keep=days_keep, day_decay=day_decay,
        )
        preds_open, _ = self.classify_open_set_history(
            emb_known, threshold=threshold, known_ids=known_ids,
            days_keep=days_keep, day_decay=day_decay,
        )
        accepted_mask = preds_open != -1

        results = {
            'threshold': threshold,
            'closed_acc': float(np.mean(preds_closed == y_known)),
            'open_known_acc': (
                float(np.mean(preds_open[accepted_mask] == y_known[accepted_mask]))
                if accepted_mask.any() else float('nan')
            ),
            'known_accept_rate': float(accepted_mask.mean()),
            'f1_macro': (
                float(f1_score(y_known, preds_closed, average='macro', zero_division=0))
                if _HAS_SKLEARN else float('nan')
            ),
        }

        if emb_unknown is not None:
            _, dists_unk = self.classify_open_set_history(
                emb_unknown, threshold=np.inf, known_ids=known_ids,
                days_keep=days_keep, day_decay=day_decay,
            )
            results['open_unk_det'] = float(np.mean(dists_unk > threshold))
            if _HAS_SKLEARN:
                binary = np.concatenate([
                    np.zeros(len(dists_known), dtype=np.int32),
                    np.ones(len(dists_unk), dtype=np.int32),
                ])
                try:
                    results['auroc'] = float(
                        roc_auc_score(binary, np.concatenate([dists_known, dists_unk]))
                    )
                except ValueError:
                    results['auroc'] = float('nan')
            else:
                results['auroc'] = float('nan')
        else:
            results['open_unk_det'] = float('nan')
            results['auroc'] = float('nan')

        if verbose:
            sep = '─' * 52
            print(f"\n{sep}")
            print(
                f"  History Trajectory Evaluation  "
                f"(threshold={threshold:.2f}, days_keep={days_keep})"
            )
            print(sep)
            print(f"  {'Closed-set accuracy':<28} {results['closed_acc']:>8.4f}")
            print(f"  {'Open known-set accuracy':<28} {results['open_known_acc']:>8.4f}")
            print(f"  {'Known accept rate':<28} {results['known_accept_rate']:>8.4f}")
            print(f"  {'Unknown detection rate':<28} {results['open_unk_det']:>8.4f}")
            print(f"  {'AUROC':<28} {results['auroc']:>8.4f}")
            print(f"  {'F1 macro':<28} {results['f1_macro']:>8.4f}")
            print(sep)

        return results

    def evaluate_cosine_prototypes(
        self,
        emb_known:   np.ndarray,
        y_known:     np.ndarray,
        known_ids:   list[int],
        emb_unknown: Optional[np.ndarray] = None,
        threshold:   float = 0.1,
        days_keep:   Optional[int] = None,
        day_decay:   float = 1.0,
        verbose:     bool  = True,
    ) -> dict:
        """Evaluate the multi-day cosine prototype trajectory classifier."""
        emb_known = np.asarray(emb_known, dtype=np.float32)
        y_known = np.asarray(y_known, dtype=np.int32)

        preds_closed, dists_known = self.classify_cosine_prototypes(
            emb_known, threshold=np.inf, known_ids=known_ids,
            days_keep=days_keep, day_decay=day_decay,
        )
        preds_open, _ = self.classify_cosine_prototypes(
            emb_known, threshold=threshold, known_ids=known_ids,
            days_keep=days_keep, day_decay=day_decay,
        )
        accepted_mask = preds_open != -1

        results = {
            'threshold': threshold,
            'closed_acc': float(np.mean(preds_closed == y_known)),
            'open_known_acc': (
                float(np.mean(preds_open[accepted_mask] == y_known[accepted_mask]))
                if accepted_mask.any() else float('nan')
            ),
            'known_accept_rate': float(accepted_mask.mean()),
            'f1_macro': (
                float(f1_score(y_known, preds_closed, average='macro', zero_division=0))
                if _HAS_SKLEARN else float('nan')
            ),
        }

        if emb_unknown is not None:
            _, dists_unk = self.classify_cosine_prototypes(
                emb_unknown, threshold=np.inf, known_ids=known_ids,
                days_keep=days_keep, day_decay=day_decay,
            )
            results['open_unk_det'] = float(np.mean(dists_unk > threshold))
            if _HAS_SKLEARN:
                binary = np.concatenate([
                    np.zeros(len(dists_known), dtype=np.int32),
                    np.ones(len(dists_unk), dtype=np.int32),
                ])
                try:
                    results['auroc'] = float(
                        roc_auc_score(binary, np.concatenate([dists_known, dists_unk]))
                    )
                except ValueError:
                    results['auroc'] = float('nan')
            else:
                results['auroc'] = float('nan')
        else:
            results['open_unk_det'] = float('nan')
            results['auroc'] = float('nan')

        if verbose:
            sep = '─' * 52
            print(f"\n{sep}")
            print(
                f"  Cosine Prototype Trajectory Evaluation  "
                f"(threshold={threshold:.3f}, days_keep={days_keep})"
            )
            print(sep)
            print(f"  {'Closed-set accuracy':<28} {results['closed_acc']:>8.4f}")
            print(f"  {'Open known-set accuracy':<28} {results['open_known_acc']:>8.4f}")
            print(f"  {'Known accept rate':<28} {results['known_accept_rate']:>8.4f}")
            print(f"  {'Unknown detection rate':<28} {results['open_unk_det']:>8.4f}")
            print(f"  {'AUROC':<28} {results['auroc']:>8.4f}")
            print(f"  {'F1 macro':<28} {results['f1_macro']:>8.4f}")
            print(sep)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trajectory to a .npz file."""
        arrays = {}
        for dev, entries in self._history.items():
            for i, e in enumerate(entries):
                prefix = f'dev{dev}_day{i}'
                arrays[f'{prefix}_day_id'] = np.array([e['day']])
                arrays[f'{prefix}_mean']   = e['mean']
                arrays[f'{prefix}_cov']    = e['cov']
                arrays[f'{prefix}_n']      = np.array([e['n']])
                if 'cos_proto' in e:
                    arrays[f'{prefix}_cos_proto'] = e['cos_proto']
                    arrays[f'{prefix}_cos_weight'] = e['cos_weight']
        arrays['_day_ids']    = np.array(self._day_ids)
        arrays['_device_ids'] = np.array(sorted(self._history.keys()))
        arrays['_n_cosine_prototypes'] = np.array([self.n_cosine_prototypes])
        arrays['_cosine_days_keep'] = np.array([self.cosine_days_keep])
        arrays['_cosine_day_decay'] = np.array([self.cosine_day_decay])
        np.savez(path, **arrays)
        print(f"[trajectory] saved → {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> 'TemporalTrajectory':
        """Load a previously saved trajectory."""
        traj = cls(**kwargs)
        data = np.load(path, allow_pickle=False)
        traj._day_ids = list(data['_day_ids'])
        device_ids    = list(data['_device_ids'])

        for dev in device_ids:
            traj._history[int(dev)] = []
            i = 0
            while True:
                prefix = f'dev{dev}_day{i}'
                if f'{prefix}_mean' not in data:
                    break
                cov = data[f'{prefix}_cov'].astype(np.float32)
                entry = {
                    'day':     int(data[f'{prefix}_day_id'][0]),
                    'mean':    data[f'{prefix}_mean'].astype(np.float32),
                    'cov':     cov,
                    'cov_inv': _safe_inv(cov.astype(np.float64)).astype(np.float32),
                    'n':       int(data[f'{prefix}_n'][0]),
                }
                if f'{prefix}_cos_proto' in data:
                    entry['cos_proto'] = data[f'{prefix}_cos_proto'].astype(np.float32)
                    entry['cos_weight'] = data[f'{prefix}_cos_weight'].astype(np.float32)
                traj._history[int(dev)].append(entry)
                i += 1

        print(f"[trajectory] loaded from {path}  "
              f"devices={len(traj._history)}  days={traj._day_ids}")
        return traj

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def known_device_ids(self) -> list[int]:
        return sorted(self._history.keys())

    @property
    def n_days_observed(self) -> int:
        return len(self._day_ids)

    def latest_mean(self, device_id: int) -> np.ndarray:
        return self._history[device_id][-1]['mean']

    def extrapolate_mean(self, device_id: int, target_day: int) -> np.ndarray:
        """
        Linearly extrapolate device mean to target_day using the last two
        observed trajectory points.  Falls back to the latest mean when
        fewer than two points exist or the day delta is zero.
        """
        history = self._history[device_id]
        if len(history) < 2:
            return history[-1]['mean'].astype(np.float32)
        last = history[-1]
        prev = history[-2]
        day_delta = float(last['day'] - prev['day'])
        if day_delta <= 0:
            return last['mean'].astype(np.float32)
        target_delta = float(target_day - last['day'])
        drift = (last['mean'].astype(np.float64) - prev['mean'].astype(np.float64)) / day_delta
        mean_ext = last['mean'].astype(np.float64) + drift * target_delta
        norm = np.linalg.norm(mean_ext) + 1e-8
        return (mean_ext / norm).astype(np.float32)

    def drift_vector(self, device_id: int, target_day: int) -> np.ndarray:
        """
        Raw-space drift vector for one device, projecting the last observed
        per-day movement forward to target_day.

        Unlike ``extrapolate_mean`` (which extrapolates and re-normalises the
        centroid), this returns the *additive shift* to apply to any embedding
        of this device so it lands where the device cloud is expected to be on
        target_day. Used to drift-correct a full gallery before KNN matching,
        which preserves intra-class variance that prototype averaging discards.

        Returns a zero vector when fewer than two trajectory points exist.
        """
        history = self._history[device_id]
        ref = history[-1]['mean']
        if len(history) < 2:
            return np.zeros_like(ref, dtype=np.float32)
        last = history[-1]
        prev = history[-2]
        day_delta = float(last['day'] - prev['day'])
        if day_delta <= 0:
            return np.zeros_like(ref, dtype=np.float32)
        target_delta = float(target_day - last['day'])
        drift = (last['mean'].astype(np.float64)
                 - prev['mean'].astype(np.float64)) / day_delta
        return (drift * target_delta).astype(np.float32)

    def classify_extrapolated(
        self,
        emb: np.ndarray,
        target_day: int,
        threshold: float,
        known_ids: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cosine open-set classifier using 2-point linearly extrapolated means.

        Projects each device's observed drift direction forward to target_day,
        then classifies by nearest L2-normalised extrapolated centroid.
        Useful when the test day is not in the trajectory — compensates for
        predictable temporal drift without any labelled test-day data.

        Parameters
        ----------
        emb        : (N, D)  L2-normalised test-day embeddings
        target_day : int     day ID to extrapolate to
        threshold  : float   cosine-distance rejection threshold
        known_ids  : list    device IDs to classify (None = all)

        Returns
        -------
        preds     : (N,) int32  predicted device IDs, -1 = unknown
        min_dists : (N,) float32 cosine distance to nearest extrapolated centroid
        """
        emb = np.asarray(emb, dtype=np.float32)
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        if known_ids is None:
            known_ids = sorted(self._history.keys())

        centroids = np.stack([
            self.extrapolate_mean(dev, target_day) for dev in known_ids
        ])
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

        sim    = emb_n @ centroids.T
        best_j = np.argmax(sim, axis=1)
        dist   = 1.0 - sim[np.arange(len(emb)), best_j]
        preds  = np.array([known_ids[j] for j in best_j], dtype=np.int32)
        preds[dist > threshold] = -1
        return preds, dist.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"TemporalTrajectory("
            f"devices={len(self._history)}, "
            f"days={self._day_ids})"
        )


# ---------------------------------------------------------------------------
# Threshold sweep helper
# ---------------------------------------------------------------------------

def trajectory_threshold_sweep(
    traj:        TemporalTrajectory,
    emb_known:   np.ndarray,
    y_known:     np.ndarray,
    emb_unknown: np.ndarray,
    known_ids:   Optional[list[int]] = None,
    n_thresholds: int = 50,
) -> dict:
    """
    Sweep Mahalanobis thresholds to find the best operating point.

    Returns the threshold that maximises  0.5*(closed_acc + unknown_det),
    plus the full sweep arrays for plotting.

    Parameters
    ----------
    traj        : fitted TemporalTrajectory
    emb_known   : (N, D)  known-device test embeddings
    y_known     : (N,)    true labels
    emb_unknown : (M, D)  unknown-device test embeddings
    known_ids   : list of known device IDs (None = all)
    n_thresholds: number of threshold points to sweep

    Returns
    -------
    dict with keys:
        best_threshold, best_score,
        thresholds, closed_accs, unk_det_rates, auroc,
        tpr_at_fpr5 (TPR at FPR <= 5%)
    """
    emb_known   = np.asarray(emb_known,   dtype=np.float32)
    emb_unknown = np.asarray(emb_unknown, dtype=np.float32)
    y_known     = np.asarray(y_known,     dtype=np.int32)

    if known_ids is None:
        known_ids = traj.known_device_ids

    # Pre-compute distances once
    _, dists_known = traj.classify_open_set(
        emb_known, threshold=np.inf, known_ids=known_ids
    )
    _, dists_unk = traj.classify_open_set(
        emb_unknown, threshold=np.inf, known_ids=known_ids
    )

    # Closed-set labels (no rejection) — reuse
    preds_closed, _ = traj.classify_open_set(
        emb_known, threshold=np.inf, known_ids=known_ids
    )
    closed_acc_base = float(np.mean(preds_closed == y_known))

    all_dists  = np.concatenate([dists_known, dists_unk])
    thresholds = np.linspace(all_dists.min(), all_dists.max(), n_thresholds)

    closed_accs   = []
    unk_det_rates = []
    scores        = []

    # Minimum fraction of known samples that must be accepted.
    # Without this constraint, the sweep is gamed by very tight thresholds
    # that reject everything (unk_det=1.0, ca=1.0 on the empty accepted set)
    # and score ~0.5, beating any real operating point.
    MIN_ACCEPT_RATE = 0.50

    for thr in thresholds:
        # Open-set: reject known if distance > threshold
        accept_known = dists_known <= thr
        accept_rate  = float(accept_known.mean())

        if accept_rate < MIN_ACCEPT_RATE:
            # Penalise thresholds that reject too aggressively
            closed_accs.append(0.0)
            unk_det_rates.append(float(np.mean(dists_unk > thr)))
            scores.append(0.0)
            continue

        ca = float(np.mean(
            preds_closed[accept_known] == y_known[accept_known]
        )) if accept_known.sum() > 0 else 0.0

        ud = float(np.mean(dists_unk > thr))
        closed_accs.append(ca)
        unk_det_rates.append(ud)
        scores.append(0.5 * (ca + ud))

    best_idx   = int(np.argmax(scores))
    best_thr   = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    # AUROC
    auroc = float('nan')
    tpr_at_fpr5 = float('nan')
    if _HAS_SKLEARN:
        binary = np.concatenate([
            np.zeros(len(dists_known), dtype=np.int32),
            np.ones( len(dists_unk),   dtype=np.int32),
        ])
        all_scores = np.concatenate([dists_known, dists_unk])
        try:
            auroc = float(roc_auc_score(binary, all_scores))
            fpr, tpr, _ = roc_curve(binary, all_scores)
            # TPR at FPR <= 5%
            tpr_at_fpr5 = float(tpr[fpr <= 0.05][-1]) if (fpr <= 0.05).any() else 0.0
        except ValueError:
            pass

    print(
        f"\n[trajectory] Threshold sweep complete.\n"
        f"  best_threshold={best_thr:.3f}  score={best_score:.4f}  "
        f"closed_acc={closed_accs[best_idx]:.4f}  "
        f"unk_det={unk_det_rates[best_idx]:.4f}  "
        f"AUROC={auroc:.4f}"
    )

    return {
        'best_threshold': best_thr,
        'best_score':     best_score,
        'closed_acc_at_best': closed_accs[best_idx],
        'unk_det_at_best':    unk_det_rates[best_idx],
        'thresholds':         thresholds,
        'closed_accs':        np.array(closed_accs),
        'unk_det_rates':      np.array(unk_det_rates),
        'auroc':              auroc,
        'tpr_at_fpr5':        tpr_at_fpr5,
    }


def trajectory_history_threshold_sweep(
    traj:        TemporalTrajectory,
    emb_known:   np.ndarray,
    y_known:     np.ndarray,
    emb_unknown: np.ndarray,
    known_ids:   Optional[list[int]] = None,
    days_keep:   Optional[int] = None,
    day_decay:   float = 1.0,
    n_thresholds: int = 50,
) -> dict:
    """Sweep thresholds for the multi-day history Mahalanobis classifier."""
    emb_known   = np.asarray(emb_known,   dtype=np.float32)
    emb_unknown = np.asarray(emb_unknown, dtype=np.float32)
    y_known     = np.asarray(y_known,     dtype=np.int32)

    if known_ids is None:
        known_ids = traj.known_device_ids

    preds_closed, dists_known = traj.classify_open_set_history(
        emb_known, threshold=np.inf, known_ids=known_ids,
        days_keep=days_keep, day_decay=day_decay,
    )
    _, dists_unk = traj.classify_open_set_history(
        emb_unknown, threshold=np.inf, known_ids=known_ids,
        days_keep=days_keep, day_decay=day_decay,
    )

    all_dists = np.concatenate([dists_known, dists_unk])
    thresholds = np.linspace(all_dists.min(), all_dists.max(), n_thresholds)
    closed_accs, unk_det_rates, scores = [], [], []
    min_accept_rate = 0.50

    for thr in thresholds:
        accept_known = dists_known <= thr
        if float(accept_known.mean()) < min_accept_rate:
            closed_accs.append(0.0)
            unk_det_rates.append(float(np.mean(dists_unk > thr)))
            scores.append(0.0)
            continue
        ca = float(np.mean(preds_closed[accept_known] == y_known[accept_known]))
        ud = float(np.mean(dists_unk > thr))
        closed_accs.append(ca)
        unk_det_rates.append(ud)
        scores.append(0.5 * (ca + ud))

    best_idx = int(np.argmax(scores))
    best_thr = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    auroc = float('nan')
    tpr_at_fpr5 = float('nan')
    if _HAS_SKLEARN:
        binary = np.concatenate([
            np.zeros(len(dists_known), dtype=np.int32),
            np.ones(len(dists_unk), dtype=np.int32),
        ])
        all_scores = np.concatenate([dists_known, dists_unk])
        try:
            auroc = float(roc_auc_score(binary, all_scores))
            fpr, tpr, _ = roc_curve(binary, all_scores)
            tpr_at_fpr5 = float(tpr[fpr <= 0.05][-1]) if (fpr <= 0.05).any() else 0.0
        except ValueError:
            pass

    print(
        f"\n[trajectory-history] Threshold sweep complete.\n"
        f"  days_keep={days_keep}  decay={day_decay:.2f}  "
        f"best_threshold={best_thr:.3f}  score={best_score:.4f}  "
        f"closed_acc={closed_accs[best_idx]:.4f}  "
        f"unk_det={unk_det_rates[best_idx]:.4f}  AUROC={auroc:.4f}"
    )

    return {
        'best_threshold': best_thr,
        'best_score': best_score,
        'closed_acc_at_best': closed_accs[best_idx],
        'unk_det_at_best': unk_det_rates[best_idx],
        'thresholds': thresholds,
        'closed_accs': np.array(closed_accs),
        'unk_det_rates': np.array(unk_det_rates),
        'auroc': auroc,
        'tpr_at_fpr5': tpr_at_fpr5,
        'days_keep': days_keep,
        'day_decay': day_decay,
    }


def trajectory_cosine_threshold_sweep(
    traj:        TemporalTrajectory,
    emb_known:   np.ndarray,
    y_known:     np.ndarray,
    emb_unknown: Optional[np.ndarray] = None,
    known_ids:   Optional[list[int]] = None,
    days_keep:   Optional[int] = None,
    day_decay:   float = 1.0,
    n_thresholds: int = 100,
) -> dict:
    """Sweep thresholds for the multi-day cosine prototype trajectory.

    When emb_unknown is None or empty, calibrates using known-device distances
    only (no unknown data seen), setting threshold at _KNOWN_ONLY_ACCEPT_RATE
    quantile and scoring by closed_acc * accept_rate.
    """
    emb_known = np.asarray(emb_known, dtype=np.float32)
    y_known = np.asarray(y_known, dtype=np.int32)
    use_unknown_calib = _has_calib_unknown(emb_unknown)
    if use_unknown_calib:
        emb_unknown = np.asarray(emb_unknown, dtype=np.float32)

    if known_ids is None:
        known_ids = traj.known_device_ids

    preds_closed, dists_known = traj.classify_cosine_prototypes(
        emb_known, threshold=np.inf, known_ids=known_ids,
        days_keep=days_keep, day_decay=day_decay,
    )
    if use_unknown_calib:
        _, dists_unk = traj.classify_cosine_prototypes(
            emb_unknown, threshold=np.inf, known_ids=known_ids,
            days_keep=days_keep, day_decay=day_decay,
        )
    else:
        dists_unk = np.empty((0,), dtype=np.float32)

    if use_unknown_calib:
        all_dists = np.concatenate([dists_known, dists_unk])
        thresholds = np.linspace(all_dists.min(), all_dists.max(), n_thresholds)
    else:
        thresholds = np.array([_known_only_distance_threshold(dists_known)])

    closed_accs, unk_det_rates, scores = [], [], []
    min_accept_rate = 0.35

    for thr in thresholds:
        accept_known = dists_known <= thr
        if float(accept_known.mean()) < min_accept_rate:
            closed_accs.append(0.0)
            unk_det_rates.append(float(np.mean(dists_unk > thr)) if use_unknown_calib else float('nan'))
            scores.append(0.0)
            continue
        ca = float(np.mean(preds_closed[accept_known] == y_known[accept_known]))
        ud = float(np.mean(dists_unk > thr)) if use_unknown_calib else float('nan')
        closed_accs.append(ca)
        unk_det_rates.append(ud)
        scores.append(0.5 * (ca + ud) if use_unknown_calib else ca * float(accept_known.mean()))

    best_idx = int(np.argmax(scores))
    best_thr = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    auroc = float('nan')
    tpr_at_fpr5 = float('nan')
    if _HAS_SKLEARN and use_unknown_calib:
        binary = np.concatenate([
            np.zeros(len(dists_known), dtype=np.int32),
            np.ones(len(dists_unk), dtype=np.int32),
        ])
        all_scores = np.concatenate([dists_known, dists_unk])
        try:
            auroc = float(roc_auc_score(binary, all_scores))
            fpr, tpr, _ = roc_curve(binary, all_scores)
            tpr_at_fpr5 = float(tpr[fpr <= 0.05][-1]) if (fpr <= 0.05).any() else 0.0
        except ValueError:
            pass

    print(
        f"\n[trajectory-cosine] Threshold sweep complete.\n"
        f"  days_keep={days_keep}  decay={day_decay:.2f}  "
        f"best_threshold={best_thr:.3f}  score={best_score:.4f}  "
        f"closed_acc={closed_accs[best_idx]:.4f}  "
        f"unk_det={unk_det_rates[best_idx]:.4f}  AUROC={auroc:.4f}"
    )

    return {
        'best_threshold': best_thr,
        'best_score': best_score,
        'closed_acc_at_best': closed_accs[best_idx],
        'unk_det_at_best': unk_det_rates[best_idx],
        'thresholds': thresholds,
        'closed_accs': np.array(closed_accs),
        'unk_det_rates': np.array(unk_det_rates),
        'auroc': auroc,
        'tpr_at_fpr5': tpr_at_fpr5,
        'days_keep': days_keep,
        'day_decay': day_decay,
    }


# ---------------------------------------------------------------------------
# Integration helper — extract embeddings from a Keras model
# ---------------------------------------------------------------------------

def extract_embeddings_from_model(
    model,
    X:          np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Extract L2-normalised embeddings from a trained DF model.

    Tries 'emb_l2norm' first, falls back to 'embedding', then to the
    penultimate layer.

    Parameters
    ----------
    model      : tf.keras.Model
    X          : (N, slice_len, 2) preprocessed IQ data
    batch_size : int

    Returns
    -------
    (N, emb_dim) float32
    """
    import tensorflow as tf

    layer_names = [l.name for l in model.layers]
    if 'emb_l2norm' in layer_names:
        emb_layer = model.get_layer('emb_l2norm')
    elif 'embedding' in layer_names:
        emb_layer = model.get_layer('embedding')
    else:
        emb_layer = model.layers[-2]

    emb_model = tf.keras.Model(inputs=model.input, outputs=emb_layer.output)
    X_arr = np.asarray(X, dtype=np.float32)
    if len(X_arr) == 0:
        emb_dim = emb_model.output_shape[-1]
        return np.empty((0, emb_dim), dtype=np.float32)
    return emb_model.predict(X_arr, batch_size=batch_size, verbose=0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    D, N, K = 64, 300, 8

    # Simulate known-device embeddings over 3 days with slight drift
    device_ids = list(range(K))
    centres    = rng.standard_normal((K, D)).astype(np.float32)
    centres   /= np.linalg.norm(centres, axis=1, keepdims=True)

    traj = TemporalTrajectory(ewma_alpha=0.8)

    for day in range(1, 4):
        drift = rng.standard_normal((K, D)).astype(np.float32) * 0.05 * day
        emb   = np.vstack([
            centres[d] + drift[d] + rng.standard_normal((N//K, D)).astype(np.float32) * 0.1
            for d in device_ids
        ])
        emb  /= np.linalg.norm(emb, axis=1, keepdims=True)
        labs  = np.repeat(np.arange(K), N//K).astype(np.int32)
        traj.update(day, emb, labs)

    print(f"\nTrajectory: {traj}")

    # Day 4 test — known devices
    emb_test = np.vstack([
        centres[d] + rng.standard_normal((N//K, D)).astype(np.float32) * 0.1
        for d in device_ids
    ])
    emb_test /= np.linalg.norm(emb_test, axis=1, keepdims=True)
    y_test    = np.repeat(np.arange(K), N//K).astype(np.int32)

    # Day 4 test — unknown devices (2 extra)
    emb_unk  = rng.standard_normal((60, D)).astype(np.float32)
    emb_unk /= np.linalg.norm(emb_unk, axis=1, keepdims=True)

    # Threshold sweep
    sweep = trajectory_threshold_sweep(traj, emb_test, y_test, emb_unk)

    # Final evaluation at best threshold
    results = traj.evaluate(
        emb_test, y_test, device_ids, emb_unk,
        threshold=sweep['best_threshold'],
    )

    assert results['closed_acc'] > 0.5, "Closed-set accuracy too low"
    assert results['auroc'] > 0.5,      "AUROC too low"

    # Save / load round-trip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        tmp = f.name
    traj.save(tmp)
    traj2 = TemporalTrajectory.load(tmp)
    assert traj2.known_device_ids == traj.known_device_ids
    os.remove(tmp)

    print("\nAll self-tests passed.")
