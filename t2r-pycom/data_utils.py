#!/usr/bin/env python3
"""
data_utils.py  —  Data loading for the multi-day temporal trajectory experiment.

Uses the same raw-IQ path as the main experiment pipeline:
  1. load_slice_IQ.LoadDataOpts  with root_dir = cfg.day_path(day)
  2. load_slice_IQ.load_data()   scans device subfolders, returns raw IQ
  3. signal_preprocessing.preprocess_batch()  CFO + RMS norm + optional aug
  4. load_slice_IQ.compute_normalization_stats()  global z-score on train
  5. load_slice_IQ.apply_normalization()

Supported folder structures:
  <data_root>/day1/equ/device_0/*.bin
  <data_root>/day1/equ/device_1/*.bin
  ...
  <data_root>/day8/equ/device_19/*.bin

  <data_root>/Day_1/device_1/*.dat
  <data_root>/Day_1/device_2/*.dat
  ...
  <data_root>/Day_5/device_50/*.dat
"""

from __future__ import annotations

import os
import pickle

import os
import numpy as np

import load_slice_IQ
import signal_preprocessing as sp


def _preprocess(X, augment, cfg):
    return sp.preprocess_batch(
        X,
        do_remove_cfo      = cfg.use_preprocessing,
        do_per_slice_norm  = cfg.use_preprocessing,
        augment            = augment and cfg.use_augmentation,
        phase_rot_range    = cfg.aug_phase_rot,
        amp_jitter_db      = cfg.aug_amp_db,
        noise_snr_db       = cfg.aug_snr_db,
        augment_apply_prob = cfg.aug_prob,
        multipath_taps     = getattr(cfg, 'aug_multipath_taps', 0),
        multipath_mag      = getattr(cfg, 'aug_multipath_mag', 0.0),
    )


def _filter_by_device(X, y, device_ids):
    mask = np.isin(y, device_ids)
    if not mask.any():
        raise RuntimeError(
            f"No samples found for devices {device_ids}. "
            "Check device subfolder names match expected indices."
        )
    return X[mask], y[mask]


def _remap_labels(y, device_ids):
    id_map = {dev: idx for idx, dev in enumerate(sorted(device_ids))}
    return np.array([id_map[int(v)] for v in y], dtype=np.int32)


def _make_opts(cfg, root, start_idx=0, device_ids=None):
    return load_slice_IQ.LoadDataOpts(
        root_dir  = root,
        file_key  = cfg.file_key,
        location  = cfg.location,
        num_slice = cfg.num_slice,
        slice_len = cfg.slice_len,
        start_idx = start_idx,
        stride    = cfg.stride,
        mul_trans = cfg.mul_trans,
        window    = cfg.window,
        data_type = cfg.data_type,
        device_ids= device_ids,
    )



# ── WiSig support ───────────────────────────────────────────────────────────
_WISIG_CACHE = {}

_WISIG_FULL_CACHE = {}

WISIG_FULL_DATES = {1: '2021_03_01', 2: '2021_03_08', 3: '2021_03_15', 4: '2021_03_23'}

def _wisig_path(cfg):
    if os.path.isdir(cfg.data_root):
        return os.path.join(cfg.data_root, 'ManySig.pkl')
    return cfg.data_root



def _load_wisig_manysig(cfg):
    path = _wisig_path(cfg)
    if path not in _WISIG_CACHE:
        print(f"[data_utils] loading WiSig ManySig pickle: {path}")
        with open(path, 'rb') as f:
            _WISIG_CACHE[path] = pickle.load(f)
    return _WISIG_CACHE[path]



def _load_wisig_day_raw(cfg, day_id, device_ids):
    if not 1 <= day_id <= 4:
        raise ValueError(f"WiSig ManySig day_id must be 1..4, got {day_id}")

    ds = _load_wisig_manysig(cfg)
    rx = int(cfg.wisig_rx_index)
    eq = int(cfg.wisig_equalized)
    if not 0 <= rx < len(ds['rx_list']):
        raise ValueError(f"wisig_rx_index={rx} out of range 0..{len(ds['rx_list']) - 1}")
    if eq not in (0, 1):
        raise ValueError(f"wisig_equalized must be 0 or 1, got {eq}")

    n_take = min(int(cfg.num_slice), int(ds['max_sig']))
    rng = np.random.default_rng(int(cfg.seed) * 100_000 + day_id * 100 + rx * 2 + eq)

    xs, ys = [], []
    for dev in sorted(device_ids):
        arr = ds['data'][int(dev)][rx][day_id - 1][eq]
        if arr.shape[1] < cfg.slice_len:
            raise ValueError(
                f"WiSig slice length {arr.shape[1]} is shorter than cfg.slice_len={cfg.slice_len}"
            )
        arr = arr[:, :cfg.slice_len, :].astype(np.float32, copy=False)
        if n_take < arr.shape[0]:
            idx = rng.choice(arr.shape[0], size=n_take, replace=False)
            idx.sort()
            arr = arr[idx]
        xs.append(arr)
        ys.append(np.full(arr.shape[0], int(dev), dtype=np.int32))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(
        f"  WiSig ManySig: day={day_id} date={ds['capture_date_list'][day_id - 1]} "
        f"rx={rx}:{ds['rx_list'][rx]} equalized={eq} X={X.shape}"
    )
    return X, y



def _burst_groups(X, thr=0.9):
    """Group near-identical signals (repeat transmissions of one burst).

    Every WiSig signal is the same WiFi preamble, so raw waveforms are almost
    collinear. After projecting out the common template, signals recorded back
    to back still match at cosine > 0.9 while signals from another date fall to
    ~0.7. Splitting by these groups instead of by signal keeps near-twins on the
    same side of a train/test split.
    """
    A = X.reshape(len(X), -1).astype(np.float64)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    t = A.mean(0); t /= max(np.linalg.norm(t), 1e-12)
    R = A - (A @ t)[:, None] * t[None, :]
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    S = R @ R.T
    parent = list(range(len(A)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(S, 1) > thr)):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj: parent[ri] = rj
    roots = {}
    out = np.empty(len(A), dtype=np.int64)
    for i in range(len(A)):
        r = find(i)
        out[i] = roots.setdefault(r, len(roots))
    return out



def _wisig_full_file(cfg, day_id):
    return os.path.join(
        cfg.data_root,
        f'dataset_{WISIG_FULL_DATES[day_id]}_{cfg.wisig_full_receiver}.pkl')



def _load_wisig_full(cfg, day_id):
    path = _wisig_full_file(cfg, day_id)
    if path not in _WISIG_FULL_CACHE:
        print(f'[data_utils] loading WiSig full pickle: {os.path.basename(path)}')
        with open(path, 'rb') as f:
            _WISIG_FULL_CACHE[path] = pickle.load(f)
    return _WISIG_FULL_CACHE[path]



def wisig_full_device_order(cfg):
    """Transmitters common to all four dates, ranked by their worst-date count.

    The first n_known entries are the enrolled devices and the next n_unknown
    are held out, so the split is deterministic and reproducible from the data
    alone. device_shift rotates the list, which rotates which transmitters are
    unknown.
    """
    per_day = {d: _load_wisig_full(cfg, d) for d in WISIG_FULL_DATES}
    common = set(per_day[1]['node_list'])
    for d in (2, 3, 4):
        common &= set(per_day[d]['node_list'])
    # Rank by the number of distinct bursts, not raw signal count. Some
    # transmitters send the same packet hundreds of times, so a high signal
    # count can mean very few independent observations; ranking by bursts
    # keeps the enrolled set at a comparable effective sample size.
    # Require a minimum raw signal count as well, so the fine-tuning and test
    # budgets can be met without unbalancing classes.
    min_sig = int(os.environ.get('T2R_WISIG_MIN_SIG', '780'))
    common = {tx for tx in common
              if min(len(per_day[d]['data'][per_day[d]['node_list'].index(tx)])
                     for d in WISIG_FULL_DATES) >= min_sig}
    cache_key = ('burst_order', cfg.wisig_full_receiver, int(cfg.slice_len), min_sig)
    if cache_key not in _WISIG_FULL_CACHE:
        mins = {}
        for tx in common:
            counts = []
            for d in WISIG_FULL_DATES:
                arr = np.asarray(per_day[d]['data'][per_day[d]['node_list'].index(tx)],
                                 dtype=np.float32)[:, :cfg.slice_len, :]
                if len(arr) < 200:
                    counts = []; break
                counts.append(len(np.unique(_burst_groups(arr[:800]))))
            if counts:
                mins[tx] = min(counts)
        _WISIG_FULL_CACHE[cache_key] = mins
    mins = _WISIG_FULL_CACHE[cache_key]
    common = set(mins)
    order = sorted(common, key=lambda t: (-mins[t], t))
    total = cfg.n_known + cfg.n_val_dev + cfg.n_unknown
    if len(order) < total:
        raise ValueError(f'WiSig full: only {len(order)} transmitters common to all '
                         f'dates, need {total}.')
    shift = getattr(cfg, 'device_shift', 0) % len(order)
    return order[shift:] + order[:shift]



def wisig_full_day_with_groups(cfg, day_id, device_ids):
    """Signals plus burst-group ids, with an optional Phase-2/Phase-3 role split.

    T2R_WISIG_ROLE=ft|cal partitions each device's burst groups deterministically
    (60% fine-tuning, 40% calibration) so that the session shared by Phase 2 and
    Phase 3 never puts a burst on both sides.
    """
    ds = _load_wisig_full(cfg, day_id)
    order = wisig_full_device_order(cfg)
    role = os.environ.get('T2R_WISIG_ROLE', '')
    n_take = int(cfg.num_slice)
    xs, ys, gs = [], [], []
    for dev in sorted(device_ids):
        tx = order[int(dev)]
        arr = np.asarray(ds['data'][ds['node_list'].index(tx)], dtype=np.float32)
        arr = arr[:, :cfg.slice_len, :]
        grp = _burst_groups(arr)
        uniq = np.unique(grp)
        rng = np.random.default_rng(int(cfg.seed) * 7919 + int(dev) * 31 + day_id)
        rng.shuffle(uniq)
        if role in ('ft', 'cal'):
            cut = max(1, int(0.6 * len(uniq)))
            keep = set(uniq[:cut].tolist()) if role == 'ft' else set(uniq[cut:].tolist())
            sel = np.flatnonzero(np.isin(grp, list(keep)))
        else:
            sel = np.arange(len(arr))
        rng2 = np.random.default_rng(int(cfg.seed) * 104729 + int(dev) + day_id)
        sel = sel[rng2.permutation(len(sel))][:n_take]
        xs.append(arr[sel]); ys.append(np.full(len(sel), int(dev), dtype=np.int32))
        gs.append(grp[sel].astype(np.int64) * 10000 + int(dev))
    X = np.concatenate(xs); y = np.concatenate(ys); g = np.concatenate(gs)
    print(f'  WiSig full: day={day_id} role={role or "all"} X={X.shape} '
          f'bursts={len(np.unique(g))}')
    return X, y, g



def load_day(cfg, day_id, device_ids, split=False, normalize=True,
             augment=False, start_idx=0):
    """
    Load one day, filter to device_ids, preprocess + normalise.

    Pipeline:
      raw load → preprocess_batch → compute_normalization_stats → apply_normalization

    Returns
    -------
    X_tr, y_tr, X_te, y_te, n_devices, norm_mean, norm_std
    Labels are remapped to 0-based within device_ids.
    X_te / y_te are empty when split=False.
    norm_mean / norm_std are None when normalize=False.
    """
    root = cfg.day_path(day_id)
    print(f"\n[data_utils] day={day_id}  devices={device_ids}")
    _layout = getattr(cfg, 'dataset_layout', 'folder_iq')
    if _layout in ('wisig_full', 'wisig_manysig'):
        if _layout == 'wisig_full':
            X_all_raw, y_all, g_all = wisig_full_day_with_groups(cfg, day_id, device_ids)
        else:
            X_all_raw, y_all = _load_wisig_day_raw(cfg, day_id, device_ids); g_all = None
        X_te_all_raw = np.empty((0, cfg.slice_len, 2), dtype=np.float32)
        y_te_all = np.empty((0,), dtype=np.int32)
        if split:
            rng = np.random.default_rng(int(cfg.seed) * 10_000 + day_id)
            tr_idx, te_idx = [], []
            for dev in sorted(device_ids):
                idx = np.where(y_all == dev)[0]
                if g_all is not None:          # hold out whole bursts
                    groups = np.unique(g_all[idx]); rng.shuffle(groups)
                    te_g = set(groups[:max(1, int(round(0.2 * len(groups))))].tolist())
                    mask = np.isin(g_all[idx], list(te_g))
                    te_idx.extend(idx[mask].tolist()); tr_idx.extend(idx[~mask].tolist())
                else:
                    idx = idx[rng.permutation(len(idx))]
                    n_tr = int(round(0.8 * len(idx)))
                    tr_idx.extend(idx[:n_tr].tolist()); te_idx.extend(idx[n_tr:].tolist())
            tr_idx = np.array(tr_idx, dtype=np.int64); te_idx = np.array(te_idx, dtype=np.int64)
            X_te_all_raw, y_te_all = X_all_raw[te_idx], y_all[te_idx]
            X_all_raw, y_all = X_all_raw[tr_idx], y_all[tr_idx]
    else:
        print(f"  root: {root}")

        opts = _make_opts(cfg, root, start_idx, device_ids=device_ids)

        X_all_raw, y_all, X_te_all_raw, y_te_all, _ = load_slice_IQ.load_data(
            opts, split=split, normalize=False
        )

    X_tr_raw, y_tr_raw = _filter_by_device(X_all_raw, y_all, device_ids)

    if split and len(X_te_all_raw) > 0:
        X_te_raw, y_te_raw = _filter_by_device(X_te_all_raw, y_te_all, device_ids)
    else:
        X_te_raw = np.empty((0, cfg.slice_len, 2), dtype=np.float32)
        y_te_raw = np.empty((0,), dtype=np.int32)

    X_tr_pp = _preprocess(X_tr_raw, augment=augment, cfg=cfg)
    X_te_pp = _preprocess(X_te_raw, augment=False,   cfg=cfg) \
              if len(X_te_raw) > 0 else X_te_raw

    norm_mean = norm_std = None
    if normalize:
        norm_mean, norm_std = load_slice_IQ.compute_normalization_stats(X_tr_pp)
        X_tr = load_slice_IQ.apply_normalization(X_tr_pp, norm_mean, norm_std)
        X_te = load_slice_IQ.apply_normalization(X_te_pp, norm_mean, norm_std) \
               if len(X_te_pp) > 0 else X_te_pp
    else:
        X_tr, X_te = X_tr_pp, X_te_pp

    y_tr = _remap_labels(y_tr_raw, device_ids)
    y_te = _remap_labels(y_te_raw, device_ids) if len(y_te_raw) > 0 \
           else np.empty((0,), dtype=np.int32)

    n_devices = len(device_ids)
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  n_devices={n_devices}")
    return X_tr, y_tr, X_te, y_te, n_devices, norm_mean, norm_std


def load_day_raw(cfg, day_id, device_ids, start_idx=0):
    """
    Load raw IQ without preprocessing or normalisation.
    Used by phase2/phase3 where finetune.CNN handles its own preprocessing.
    """
    _layout = getattr(cfg, 'dataset_layout', 'folder_iq')
    if _layout == 'wisig_full':
        return wisig_full_day_with_groups(cfg, day_id, device_ids)[:2]
    if _layout == 'wisig_manysig':
        return _load_wisig_day_raw(cfg, day_id, device_ids)
    root = cfg.day_path(day_id)
    print(f"\n[data_utils] raw load  day={day_id}  root={root}")

    opts = _make_opts(cfg, root, start_idx, device_ids=device_ids)
    X_all, y_all, _, _, _ = load_slice_IQ.load_data(
        opts, split=False, normalize=False
    )

    X, y_raw = _filter_by_device(X_all, y_all, device_ids)
    y        = _remap_labels(y_raw, device_ids)
    print(f"  X={X.shape}  labels={sorted(set(y.tolist()))}")
    return X, y


def load_day_raw_with_groups(cfg, day_id, device_ids, start_idx=0):
    """
    Load raw IQ for one day, keeping each slice's recording provenance.

    Same data as load_day_raw, but also returns which capture file each
    slice came from and its position in that file. Needed to build
    leak-free support/query splits: a random split puts slices from the
    same instant of the same capture on both sides, which lets a model
    match on channel/AGC state rather than on the device.

    Returns
    -------
    X       : (N, slice_len, 2) raw IQ
    y       : (N,) int  labels remapped to 0-based within device_ids
    groups  : (N, 2) int  [file_index, position_within_file]
    """
    if getattr(cfg, 'dataset_layout', '') == 'wisig_full':
        # a burst plays the role of a capture file: [burst_id, 0]
        X, y, g = wisig_full_day_with_groups(cfg, day_id, device_ids)
        return X, y, np.stack([g, np.zeros_like(g)], axis=1)
    import os
    root = cfg.day_path(day_id)
    print(f"\n[data_utils] raw load + groups  day={day_id}  root={root}")

    if getattr(cfg, 'auto_unzip', True):
        load_slice_IQ.unzip_dataset(root, delete_archives=False)

    dev_names = sorted(
        (d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))),
        key=load_slice_IQ.natural_key,
    )
    wanted = {int(v) for v in device_ids}
    opts   = _make_opts(cfg, root, start_idx, device_ids=device_ids)

    X_list, y_list, g_list = [], [], []
    for label, dev_name in enumerate(dev_names):
        if label not in wanted:
            continue
        dev_dir  = os.path.join(root, dev_name)
        glob_pat = os.path.join(dev_dir, cfg.location, cfg.file_key) \
                   if cfg.location else os.path.join(dev_dir, cfg.file_key)

        Xd, gd = load_slice_IQ.build_device_dataset(
            glob_pattern     = glob_pat,
            n_slices_per_dev = opts.num_slice,
            slice_len        = opts.slice_len,
            start_idx        = opts.start_idx,
            stride           = opts.stride,
            mul_trans        = opts.mul_trans,
            data_type        = opts.data_type,
            window           = opts.window,
            return_groups    = True,
        )
        X_list.append(Xd)
        y_list.append(np.full(len(Xd), label, dtype=np.int32))
        g_list.append(gd)

    X = np.concatenate(X_list, axis=0)
    y = _remap_labels(np.concatenate(y_list, axis=0), device_ids)
    g = np.concatenate(g_list, axis=0)
    print(f"  X={X.shape}  files/dev={len(np.unique(g[:, 0]))}  "
          f"labels={sorted(set(y.tolist()))}")
    return X, y, g


def load_day_with_groups(cfg, day_id, device_ids, start_idx=0):
    """
    Preprocessed (but not z-scored) IQ for one day, with capture provenance.

    Mirrors load_day(split=False, normalize=False, augment=False) so callers
    can apply their own normalisation stats, and additionally returns the
    per-slice [file_index, position] groups needed to build capture-aware
    splits and burst-level decisions.

    Returns
    -------
    X, y, groups
    """
    X_raw, y, groups = load_day_raw_with_groups(cfg, day_id, device_ids,
                                                start_idx=start_idx)
    X = _preprocess(X_raw, augment=False, cfg=cfg)
    return X, y, groups


def apply_norm(X, mean, std):
    """Apply pre-fitted z-score stats."""
    return load_slice_IQ.apply_normalization(X, mean, std)


def split_known_unknown(X, y, known_ids, unknown_ids):
    """
    Partition into known and unknown subsets.
    Labels are NOT remapped — returned as-is.
    """
    mask_k = np.isin(y, known_ids)
    mask_u = np.isin(y, unknown_ids)
    return X[mask_k], y[mask_k], X[mask_u], y[mask_u]
