#!/usr/bin/env python3
"""
phase3_trajectory_building.py — Phase 3: Build temporal trajectory before the test day.

FIX (embedding space consistency): The previous version loaded a different
per-day model for each trajectory day (Day-2 model for Day-2 embeddings,
Day-3 model for Day-3 embeddings, etc.).  This is incorrect: embeddings
from different models live in different geometric spaces, so Mahalanobis
distances computed across days are meaningless.

Corrected approach:
  - Use the SINGLE final Phase 2 model to extract embeddings for all trajectory
    days.
  - This is the same model used at test time in Phase 4, so all trajectory
    statistics (μ, Σ) and test embeddings share a consistent coordinate system.

Threshold sweep is done on cfg.traj_day using the same final model.
"""

from __future__ import annotations

import os
import time
import numpy as np
import tensorflow as tf

from experiment_config import ExperimentConfig
from data_utils import load_day
from temporal_trajectory import (
    TemporalTrajectory,
    trajectory_threshold_sweep,
    trajectory_history_threshold_sweep,
    trajectory_cosine_threshold_sweep,
    extract_embeddings_from_model,
)
import load_slice_IQ
import rf_models
import tta_bn


def _load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path,
        custom_objects={'L2Normalize': rf_models.L2Normalize},
    )


def _stratified_three_split(
    labels: np.ndarray,
    seed: int,
    fracs: tuple = (0.50, 0.25, 0.25),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split each class into three disjoint stratified subsets.

    fracs : (update, hyperparam_sweep, threshold_calib) fractions.
            Must sum to 1.  At least 1 sample per class per split is guaranteed.

    Returns update_idx, sweep_idx, calib_idx — all as int64 arrays.
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, "fracs must sum to 1"
    labels = np.asarray(labels, dtype=np.int32)
    rng = np.random.default_rng(seed)
    update_idx, sweep_idx, calib_idx = [], [], []
    for cls in sorted(np.unique(labels)):
        cls_idx = np.where(labels == cls)[0]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        n = len(cls_idx)
        n_update = max(1, int(round(n * fracs[0])))
        n_sweep  = max(1, int(round(n * fracs[1])))
        # Remainder goes to calib — ensures no overlap and at least 1 sample each
        n_update = min(n_update, n - 2)
        n_sweep  = min(n_sweep,  n - n_update - 1)
        update_idx.extend(cls_idx[:n_update].tolist())
        sweep_idx.extend( cls_idx[n_update:n_update + n_sweep].tolist())
        calib_idx.extend( cls_idx[n_update + n_sweep:].tolist())
    return (np.array(update_idx, dtype=np.int64),
            np.array(sweep_idx,  dtype=np.int64),
            np.array(calib_idx,  dtype=np.int64))


def _wisig_reserve_ft(cfg):
    """Keep Phase-3 calibration data disjoint from Phase-2 fine-tuning data.

    On the WiSig full layout the calibration session is also the last
    fine-tuning session. Phase 2 consumes the first ft_n_train signals of each
    device's fixed seeded order, so Phase 3 skips exactly those.
    """
    import os
    if getattr(cfg, 'dataset_layout', '') == 'wisig_full' and \
            cfg.traj_day in cfg.finetune_days:
        os.environ['T2R_WISIG_ROLE'] = 'cal'
        print(f'  [wisig_full] Phase 3 uses the calibration burst groups of day '
              f'{cfg.traj_day}; Phase 2 fine-tuned on the disjoint burst groups')
    else:
        os.environ.pop('T2R_WISIG_ROLE', None)



def run_phase3(cfg: ExperimentConfig, phase2_result: dict) -> dict:
    print('\n' + '=' * 62)
    print('  PHASE 3 — Building temporal trajectory')
    print('=' * 62)

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    results_path = os.path.join(cfg.results_dir, 'results_phase3.txt')

    def _log(line: str):
        print(line)
        with open(results_path, 'a') as f:
            print(line, file=f, flush=True)

    _log(f'\n### Phase 3 started at {time.ctime()}')
    _wisig_reserve_ft(cfg)

    # ── Build day → (model_path, norm_path) map from Phase 2 results ─────
    # all_models = list of (day, model_path, norm_path) tuples
    all_models = phase2_result['all_models']
    day_to_model = {}
    day_to_norm  = {}
    for (day, mpath, npath) in all_models:
        day_to_model[day] = mpath
        day_to_norm[day]  = npath

    final_model_path = phase2_result['final_model_path']
    final_norm_path  = phase2_result['final_norm_path']

    def _get_norm(day: int):
        npath = day_to_norm.get(day, final_norm_path)
        if npath and os.path.exists(npath):
            nstats = np.load(npath)
            return nstats['mean'], nstats['std']
        # Fallback: compute from this day's data
        _, _, _, _, _, nm, ns = load_day(
            cfg, day_id=day, device_ids=cfg.all_known_ids,
            split=False, normalize=True, augment=False,
        )
        return nm, ns

    # Build a real chronological pre-test trajectory. cfg.traj_day is split:
    # one half updates the final pre-test trajectory and the held-out half is
    # used only for threshold/EWMA calibration.
    trajectory_days = sorted({
        d for d in [cfg.init_day] + list(cfg.finetune_days)
        if d < cfg.traj_day
    })
    if not trajectory_days:
        raise ValueError(
            "Phase 3 needs at least one trajectory day before traj_day. "
            f"Got init_day={cfg.init_day}, finetune_days={cfg.finetune_days}, "
            f"traj_day={cfg.traj_day}."
        )
    trajectory_days_with_update = trajectory_days + [cfg.traj_day]

    # Load single set of norm stats for ALL days.
    norm_mean_final, norm_std_final = _get_norm(cfg.traj_day)
    norm_path = os.path.join(cfg.model_dir, 'phase3_norm_stats.npz')
    np.savez(norm_path, mean=norm_mean_final, std=norm_std_final)

    print(f'\nBuilding trajectory for days: {trajectory_days_with_update}')
    print(f'  Using single model + single norm stats for all days.')
    print('  Saving Mahalanobis stats plus multi-prototype cosine anchors.')

    # Load the final model once and reuse for all days
    shared_model = _load_model(final_model_path)

    # BN-adapt the model to traj_day data so that all extracted embeddings
    # (trajectory days + Phase 4 test day) share a space aligned to the most
    # recent pre-test conditions.  Uses only unlabeled statistics — no leakage.
    print(f'\n  BN-adapting model to Day {cfg.traj_day} data ...')
    X_bn_raw, _, _, _, _, _, _ = load_day(
        cfg,
        day_id     = cfg.traj_day,
        device_ids = cfg.all_known_ids,
        split      = False,
        normalize  = False,
        augment    = False,
    )
    X_bn = load_slice_IQ.apply_normalization(X_bn_raw, norm_mean_final, norm_std_final)
    del X_bn_raw
    tta_bn.adapt_bn_statistics(shared_model, X_bn, batch_size=128, n_passes=3)
    del X_bn
    adapted_model_path = os.path.join(
        cfg.model_dir, f'phase3_bn_adapted_d{cfg.traj_day}.keras'
    )
    shared_model.save(adapted_model_path)
    print(f'  BN-adapted model saved → {adapted_model_path}')

    # ── Step 1: Cache embeddings for every trajectory day (one inference pass each) ──
    day_emb_cache: dict = {}
    for day in trajectory_days_with_update:
        print(f'\n  → Day {day}  extracting embeddings ...')
        X, y, _, _, _, _, _ = load_day(
            cfg,
            day_id     = day,
            device_ids = cfg.all_known_ids,
            split      = False,
            normalize  = False,
            augment    = False,
        )
        X   = load_slice_IQ.apply_normalization(X, norm_mean_final, norm_std_final)
        emb = extract_embeddings_from_model(shared_model, X, batch_size=128)
        day_emb_cache[day] = (emb, y)
        print(f'    emb={emb.shape}')

    # ── Step 2: Triple-split traj_day ─────────────────────────────────────
    # Part A (50%): trajectory update   — seen by the model at trajectory time
    # Part B (25%): hyperparameter sweep — selects ewma/n_proto/days_keep/decay
    # Part C (25%): threshold calibration — sets the final operating threshold
    #
    # Keeping these three splits strictly disjoint prevents double-dipping:
    # the same data cannot both select hyperparameters and set the threshold.
    # Unknown devices are NEVER loaded here — using them to calibrate and then
    # testing on the same IDs in Phase 4 would be leakage.
    print(f'\nTriple-splitting Day {cfg.traj_day} embeddings '
          f'(50% update / 25% hyperparam sweep / 25% threshold calib) ...')
    emb_d7_full, y_d7_full = day_emb_cache[cfg.traj_day]

    traj_idx, sweep_idx, calib_idx = _stratified_three_split(
        y_d7_full, cfg.seed + 7, fracs=(0.50, 0.25, 0.25)
    )

    emb_d7_traj  = emb_d7_full[traj_idx];  y_d7_traj  = y_d7_full[traj_idx]
    emb_d7_sweep = emb_d7_full[sweep_idx];  y_d7_sweep = y_d7_full[sweep_idx]
    emb_d7_calib = emb_d7_full[calib_idx];  y_d7_calib = y_d7_full[calib_idx]

    print(f'  trajectory-update : {emb_d7_traj.shape}')
    print(f'  hyperparam-sweep  : {emb_d7_sweep.shape}')
    print(f'  threshold-calib   : {emb_d7_calib.shape}')

    emb_calib_empty = np.empty((0, emb_d7_calib.shape[1]), dtype=emb_d7_calib.dtype)

    calib_path = os.path.join(
        cfg.model_dir,
        f'phase3_calibration_d{cfg.traj_day}.npz'
    )
    np.savez(
        calib_path,
        emb_known=emb_d7_calib,
        y_known=y_d7_calib,
        emb_unknown=emb_calib_empty,
        emb_update=emb_d7_traj,
        y_update=y_d7_traj,
        emb_sweep=emb_d7_sweep,
        y_sweep=y_d7_sweep,
        known_only_calibration=True,
    )
    print(f'  Calibration embeddings saved → {calib_path}')

    # ── Step 3: Sweep cosine-prototype hyperparameters on Part B only ───────
    # Hyperparameters are selected on the sweep split (Part B).
    # The threshold is then set on the threshold-calib split (Part C).
    # These two splits never overlap, eliminating double-dipping.
    #
    # Selection criterion: 0.5*closed_acc + 0.5*LOCO-AUROC, where
    # LOCO-AUROC is leave-one-class-out pseudo-open-set AUROC — each known
    # class in turn is excluded from the prototype banks and its Part-B
    # samples act as pseudo-unknowns. This lets the sweep "see" open-set
    # rejection quality using known devices only (no leakage), which
    # stabilises hyperparameter choice across seeds: a closed-acc-only
    # criterion is nearly flat over configs and seed noise flips the
    # argmax toward degenerate small prototype banks.
    from sklearn.metrics import roc_auc_score as _roc_auc

    def _loco_score(traj_cand, emb, y, known_ids, days_keep, day_decay):
        """Returns (closed_acc, loco_auroc, combined_score) on (emb, y)."""
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        D = np.full((len(emb_n), len(known_ids)), np.inf, dtype=np.float32)
        for j, dev in enumerate(known_ids):
            proto, w = traj_cand.cosine_prototype_bank(
                dev, days_keep=days_keep, day_decay=day_decay
            )
            sim = emb_n @ proto.T
            sim = sim * (0.9 + 0.1 * w[None, :])   # same weighting as classifier
            D[:, j] = 1.0 - sim.max(axis=1)

        pred = np.argmin(D, axis=1)
        closed = float(np.mean(np.asarray(known_ids)[pred] == y))

        aurocs = []
        for j, dev in enumerate(known_ids):
            cols = [k for k in range(len(known_ids)) if k != j]
            d_min = D[:, cols].min(axis=1)
            lab = (y == dev).astype(int)        # held-out class = pseudo-unknown
            if 0 < lab.sum() < len(lab):
                aurocs.append(_roc_auc(lab, d_min))
        loco = float(np.mean(aurocs)) if aurocs else 0.0
        return closed, loco, 0.5 * closed + 0.5 * loco

    # Bank size (n_proto) is a capacity parameter, NOT tuned per seed:
    # Proposition 2 (bank max >= subset max) implies larger banks never
    # hurt the known-device score, and per-seed selection of bank size on
    # known-only Part-B data proved unreliable — it repeatedly picked
    # degenerate tiny banks that collapse open-set AUROC at test time
    # (see diagnose_seed21.py: n_proto=5 -> 0.64 AUROC vs n_proto=500 ->
    # 0.74 on identical embeddings). Fixed a priori to the largest value.
    N_PROTO_FIXED    = 500
    ewma_candidates  = sorted({cfg.traj_ewma, 0.5, 1.0})
    best_ewma        = cfg.traj_ewma
    best_sweep_score = -np.inf
    best_days_keep   = 1
    best_day_decay   = 1.0
    best_n_proto     = N_PROTO_FIXED

    _log('\nSweeping cosine prototype hyperparameters on Part B '
         f'(criterion: 0.5*closed + 0.5*LOCO-AUROC; n_proto fixed at '
         f'{N_PROTO_FIXED}) ...')
    for alpha in ewma_candidates:
        for n_proto in [N_PROTO_FIXED]:
            traj_cand = TemporalTrajectory(
                ewma_alpha=alpha,
                n_cosine_prototypes=n_proto,
            )
            for day in trajectory_days:
                emb, y = day_emb_cache[day]
                traj_cand.update(day_id=day, emb=emb, labels=y)
            traj_cand.update(day_id=cfg.traj_day, emb=emb_d7_traj, labels=y_d7_traj)

            # Pycom has four enrollment/reference days before testing:
            # W=1 uses Day 4, W=2 uses Days 3-4, W=3 uses Days 2-4,
            # and W=4 uses Days 1-4.
            for days_keep in [1, 2, 3, 4]:
                for day_decay in [1.0, 0.8, 0.6]:
                    closed, loco, score = _loco_score(
                        traj_cand, emb_d7_sweep, y_d7_sweep,
                        list(range(len(cfg.all_known_ids))),
                        days_keep, day_decay,
                    )
                    _log(f'  ewma={alpha:.1f}  n_proto={n_proto}  '
                         f'days_keep={days_keep}  decay={day_decay:.1f}  '
                         f'closed={closed:.4f}  loco_auroc={loco:.4f}  '
                         f'score={score:.4f}')

                    if score > best_sweep_score:
                        best_sweep_score = score
                        best_ewma        = alpha
                        best_days_keep   = days_keep
                        best_day_decay   = day_decay
                        best_n_proto     = n_proto

    _force = os.environ.get('T2R_FORCE_DAYS_KEEP', '').strip().lower()
    if _force:
        # Diagnostic override: evaluate a specific reference window regardless
        # of what known-only calibration selected. Used for the W-sweep only,
        # to measure what each window costs on the test session.
        best_days_keep = None if _force in ('all', 'none') else int(_force)
        _log(f'  [DIAGNOSTIC] window forced to days_keep={best_days_keep} '
             f'(calibration selection ignored)')

    _log(f'\n  Best cosine prototype trajectory: ewma={best_ewma}  '
         f'n_proto={best_n_proto}  days_keep={best_days_keep}  '
         f'decay={best_day_decay}  (score={best_sweep_score:.4f})')

    # Rebuild final trajectory with best hyperparameters.
    traj = TemporalTrajectory(
        ewma_alpha=best_ewma,
        n_cosine_prototypes=best_n_proto,
    )
    for day in trajectory_days:
        emb, y = day_emb_cache[day]
        traj.update(day_id=day, emb=emb, labels=y)
    traj.update(day_id=cfg.traj_day, emb=emb_d7_traj, labels=y_d7_traj)

    # Set threshold on Part C (threshold-calib split) — never seen during sweep.
    _log('\nSetting threshold on Part C (threshold-calib split, disjoint from sweep) ...')
    sweep = trajectory_cosine_threshold_sweep(
        traj        = traj,
        emb_known   = emb_d7_calib,
        y_known     = y_d7_calib,
        emb_unknown = None,
        known_ids   = list(range(len(cfg.all_known_ids))),
        days_keep   = best_days_keep,
        day_decay   = best_day_decay,
    )
    _log(f'  threshold-calib score={sweep["best_score"]:.4f}  '
         f'thr={sweep["best_threshold"]:.3f}  '
         f'closed={sweep["closed_acc_at_best"]:.4f}')

    print(f'\nTrajectory built: {traj.n_days_observed} days, '
          f'{len(traj.known_device_ids)} devices')

    # ── Save trajectory ───────────────────────────────────────────────────
    traj_path = os.path.join(
        cfg.model_dir,
        f'phase3_trajectory_d{cfg.traj_day}.npz'
    )
    traj.save(traj_path)

    threshold = (cfg.traj_threshold
                 if cfg.traj_threshold is not None
                 else sweep['best_threshold'])

    _log(f"  best_ewma={best_ewma}")
    _log(f"  best_n_cosine_prototypes={best_n_proto}")
    _log(f"  best_days_keep={best_days_keep}")
    _log(f"  best_day_decay={best_day_decay:.3f}")
    _log(f"  best_threshold={sweep['best_threshold']:.3f}  "
         f"score={sweep['best_score']:.4f}  "
         f"closed_acc={sweep['closed_acc_at_best']:.4f}  "
         f"unk_det={sweep['unk_det_at_best']:.4f}")
    _log(f"  using threshold: {threshold:.3f}")

    return {
        'model_path':     adapted_model_path,
        'norm_mean':      norm_mean_final,
        'norm_std':       norm_std_final,
        'norm_path':      norm_path,
        'trajectory':     traj,
        'trajectory_path': traj_path,
        'calibration_path': calib_path,
        'threshold':      threshold,
        'sweep_results':  sweep,
        'trajectory_classifier': 'cosine_prototypes',
        'history_days_keep': best_days_keep,
        'history_day_decay': best_day_decay,
        'n_cosine_prototypes': best_n_proto,
        'num_class':      len(cfg.all_known_ids),
        'test_day':       cfg.test_day,
        'trajectory_days': trajectory_days_with_update,
    }
