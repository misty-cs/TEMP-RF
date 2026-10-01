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


def _group_three_split(labels, groups, seed, fracs=(0.50, 0.25, 0.25)):
    """Per class, assign whole groups (WiSig bursts / Pycom capture files) to
    Part A, then B, then C, in a seeded random order: A takes groups until it
    holds fracs[0] of the class's slices, B until fracs[0]+fracs[1], C the rest.
    At least one group is always left for each later part. No group is shared.
    If B or C would get less than half of its share (a few very large groups),
    groups are balanced greedily instead: largest first, each to the part
    furthest below its share; every part keeps >= 1 group."""
    labels = np.asarray(labels, dtype=np.int32); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    A, B, C = [], [], []
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        ug = rng.permutation(np.unique(groups[idx]))
        if len(ug) < 3:
            raise ValueError(f'class {cls}: {len(ug)} groups, need >= 3 for A/B/C')
        members = [idx[groups[idx] == gv] for gv in ug]
        n = len(idx); cum = 0; k = 0
        a_end = b_end = None
        for k, m in enumerate(members):
            cum += len(m)
            if a_end is None and (cum >= fracs[0] * n or len(members) - (k + 1) == 2):
                a_end = k + 1
            elif a_end is not None and b_end is None and (cum >= (fracs[0] + fracs[1]) * n
                                                         or len(members) - (k + 1) == 1):
                b_end = k + 1
                break
        nB = sum(len(m) for m in members[a_end:b_end]); nC = sum(len(m) for m in members[b_end:])
        if nB < 0.5 * fracs[1] * n or nC < 0.5 * fracs[2] * n:
            # a few very large groups (WiSig: one burst can hold >50% of a
            # device's slices): greedy balance -- largest group first, each to
            # the part furthest below its share; every part keeps >= 1 group
            order = sorted(range(len(members)), key=lambda j: -len(members[j]))   # stable: seeded order breaks ties
            parts, have = ([], [], []), [0, 0, 0]
            for r, j in enumerate(order):
                empty = [p for p in range(3) if not parts[p]]
                if len(order) - r <= len(empty):
                    p = empty[0]
                else:
                    p = int(np.argmax([fracs[q] * n - have[q] for q in range(3)]))
                parts[p].append(j); have[p] += len(members[j])
            A += [i for j in parts[0] for i in members[j]]
            B += [i for j in parts[1] for i in members[j]]
            C += [i for j in parts[2] for i in members[j]]
            continue
        A += [i for m in members[:a_end] for i in m]
        B += [i for m in members[a_end:b_end] for i in m]
        C += [i for m in members[b_end:] for i in m]
    return (np.array(A, dtype=np.int64), np.array(B, dtype=np.int64), np.array(C, dtype=np.int64))


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

    # BN-adapt the model to the reference session so that all extracted
    # embeddings (trajectory days + test day) share a space aligned to the most
    # recent pre-test conditions. Unlabelled statistics of Part A ONLY: the
    # A/B/C group split depends only on labels and groups, so it is computed
    # first, and Parts B (selection) and C (thresholds) never influence the model.
    from data_utils import load_day_with_groups
    print(f'\n  BN-adapting model to Day {cfg.traj_day} Part A ...')
    X_bn_raw, _yb, _gb = load_day_with_groups(cfg, day_id=cfg.traj_day, device_ids=cfg.all_known_ids)
    _gb = _gb[:, 0] if np.ndim(_gb) == 2 else _gb
    _iA_bn, _, _ = _group_three_split(np.asarray(_yb).astype(int), _gb, cfg.seed + 7, fracs=(0.50, 0.25, 0.25))
    X_bn = load_slice_IQ.apply_normalization(X_bn_raw[_iA_bn], norm_mean_final, norm_std_final)
    _log(f'  BN adaptation on Part A only: {len(_iA_bn)} of {len(X_bn_raw)} reference slices')
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
    traj_groups = None
    for day in trajectory_days_with_update:
        print(f'\n  → Day {day}  extracting embeddings ...')
        if day == cfg.traj_day:
            # reference session with its groups (bursts / capture files), for a
            # group-disjoint A/B/C split
            from data_utils import load_day_with_groups
            X, y, traj_groups = load_day_with_groups(cfg, day_id=day, device_ids=cfg.all_known_ids)
            traj_groups = traj_groups[:, 0] if np.ndim(traj_groups) == 2 else traj_groups
        else:
            # Earlier session: prototypes only from groups the model was NOT
            # trained on (Phase 1 val/test groups; Phase 2 non-training groups).
            from data_utils import load_day_with_groups
            X, y, gd = load_day_with_groups(cfg, day_id=day, device_ids=cfg.all_known_ids)
            gd = gd[:, 0] if np.ndim(gd) == 2 else gd
            tg = os.path.join(cfg.model_dir, f'phase1_train_groups_d{day}.npz' if day == cfg.init_day
                              else f'phase2_train_groups_d{day}.npz')
            if os.path.exists(tg):
                z = np.load(tg)
                trained = set(zip(z['labels'].astype(int).tolist(), z['groups'].astype(int).tolist()))
                keep = np.array([(int(a), int(b)) not in trained for a, b in zip(y, gd)])
                _log(f'  day {day}: prototypes from {keep.sum()} of {len(keep)} slices '
                     f'(groups used for training excluded)')
                if keep.sum() == 0 or len(np.unique(y[keep])) < len(np.unique(y)):
                    raise RuntimeError(f'day {day}: no held-out slices left for some device')
                np.savez(os.path.join(cfg.model_dir, f'phase3_bank_groups_d{day}.npz'),
                         labels=np.asarray(y)[keep].astype(int), groups=gd[keep].astype(int),
                         idx=np.flatnonzero(keep))
                X, y = X[keep], y[keep]
            else:
                raise RuntimeError(f'day {day}: {os.path.basename(tg)} missing; cannot exclude '
                                   'training groups from the prototype banks')
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

    traj_idx, sweep_idx, calib_idx = _group_three_split(
        y_d7_full, traj_groups, cfg.seed + 7, fracs=(0.50, 0.25, 0.25)
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
        idx_update=traj_idx, idx_sweep=sweep_idx, idx_known=calib_idx,
        g_update=traj_groups[traj_idx], g_sweep=traj_groups[sweep_idx], g_known=traj_groups[calib_idx],
        idx_bn=_iA_bn,
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

    # Selection on Part B only (known devices; leave-one-device-out AUROC):
    #   1) EWMA alpha: only the single-centroid rule uses the EWMA mean, so alpha
    #      is chosen with that rule's own score (P = 1 banks).
    #   2) Bank size P, window and decay: chosen with the multi-prototype score.
    #      P = 'all' keeps every reference embedding as a prototype.
    # (Earlier code fixed P = 500 from a test-day diagnostic; that choice is
    # replaced by this selection.)
    def _centroid_loco(traj_cand, emb, y, known_ids):
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        M = np.stack([traj_cand._history[d][-1]['mean'] for d in known_ids]).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
        D = 1.0 - emb_n @ M.T
        closed = float(np.mean(np.asarray(known_ids)[D.argmin(1)] == y))
        au = []
        for j, dev in enumerate(known_ids):
            cols = [k for k in range(len(known_ids)) if k != j]
            lab = (y == dev).astype(int)
            if 0 < lab.sum() < len(lab):
                au.append(_roc_auc(lab, D[:, cols].min(1)))
        loco = float(np.mean(au)) if au else 0.0
        return closed, loco, 0.5 * closed + 0.5 * loco

    def _build(alpha, n_proto):
        tr = TemporalTrajectory(ewma_alpha=alpha, n_cosine_prototypes=n_proto)
        for day in trajectory_days:
            emb, y = day_emb_cache[day]
            tr.update(day_id=day, emb=emb, labels=y)
        tr.update(day_id=cfg.traj_day, emb=emb_d7_traj, labels=y_d7_traj)
        return tr

    known0 = list(range(len(cfg.all_known_ids)))
    sel_table = {'alpha': [], 'bank': []}       # saved for sensitivity analysis
    ewma_candidates = sorted({cfg.traj_ewma, 0.5, 1.0})
    best_ewma, best_c = cfg.traj_ewma, -np.inf
    _log('\nSelecting EWMA alpha on Part B with the single-centroid score ...')
    for alpha in ewma_candidates:
        c_closed, c_loco, c_score = _centroid_loco(_build(alpha, 1), emb_d7_sweep, y_d7_sweep, known0)
        _log(f'  ewma={alpha:.1f}  closed={c_closed:.4f}  loco_auroc={c_loco:.4f}  score={c_score:.4f}')
        sel_table['alpha'].append({'alpha': alpha, 'closed': c_closed, 'loco': c_loco})
        if c_score > best_c:
            best_c, best_ewma = c_score, alpha

    P_ALL = 10 ** 9
    p_candidates = [1, 5, 20, 50, 100, P_ALL]           # ascending: ties keep the smaller P
    best_sweep_score = -np.inf
    best_days_keep, best_day_decay, best_n_proto = 1, 1.0, 1
    _log('\nSelecting bank size P, window and decay on Part B '
         '(criterion: 0.5*closed + 0.5*LOCO-AUROC) ...')
    for n_proto in p_candidates:
        traj_cand = _build(best_ewma, n_proto)
        for days_keep in range(1, len(trajectory_days) + 2):
            for day_decay in [1.0, 0.8, 0.6]:
                closed, loco, score = _loco_score(
                    traj_cand, emb_d7_sweep, y_d7_sweep, known0, days_keep, day_decay,
                )
                _log(f'  n_proto={"all" if n_proto == P_ALL else n_proto}  days_keep={days_keep}  '
                     f'decay={day_decay:.1f}  closed={closed:.4f}  loco_auroc={loco:.4f}  score={score:.4f}')
                sel_table['bank'].append({'P': 'all' if n_proto == P_ALL else n_proto, 'W': days_keep,
                                          'decay': day_decay, 'closed': closed, 'loco': loco})
                if score > best_sweep_score:
                    best_sweep_score = score
                    best_days_keep, best_day_decay, best_n_proto = days_keep, day_decay, n_proto
    # Shared tie rule (selection.py): within 0.001 of the best keep the prior
    # default, else the simplest candidate.
    import selection as _S
    for r_ in sel_table['alpha']:
        r_['score'] = _S.sel_score(r_['closed'], r_['loco'])
    for r_ in sel_table['bank']:
        r_['score'] = _S.sel_score(r_['closed'], r_['loco'])
    _a = _S.choose(sel_table['alpha'], lambda r_: r_['alpha'] == cfg.traj_ewma,
                   lambda r_: abs(1.0 - r_['alpha']))
    _Pv = lambda r_: P_ALL if r_['P'] == 'all' else r_['P']
    _b = _S.choose(sel_table['bank'], lambda r_: (r_['P'], r_['W'], r_['decay']) == ('all', 1, 1.0),
                   lambda r_: (r_['W'], 1.0 - r_['decay'], _Pv(r_)))
    best_ewma = _a['alpha']
    best_days_keep, best_day_decay, best_n_proto = _b['W'], _b['decay'], _Pv(_b)
    best_sweep_score = _b['score']
    sel_table['chosen'] = {'alpha': best_ewma, 'P': _b['P'], 'W': best_days_keep, 'decay': best_day_decay,
                           'rule': f'max SEL, ties within {_S.TIE_TOL} keep the prior default else the simplest'}
    import json as _json
    _json.dump(sel_table, open(os.path.join(cfg.model_dir, 'phase3_selection.json'), 'w'), indent=1)

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

    # Rebuild final trajectory with the selected hyperparameters.
    traj = _build(best_ewma, best_n_proto)
    # Same sessions with one plain mean per device and session (P = 1, no EWMA):
    # the baseline "TEMP-RFF, P = 1" separates multi-prototype from multi-session.
    _build(1.0, 1).save(os.path.join(cfg.model_dir, f'phase3_trajectory_p1_d{cfg.traj_day}.npz'))
    # Every reference embedding as a prototype (the untuned default of the method).
    _build(1.0, P_ALL).save(os.path.join(cfg.model_dir, f'phase3_trajectory_pall_d{cfg.traj_day}.npz'))

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
