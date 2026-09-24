#!/usr/bin/env python3


import os
import glob
import gzip
import random
import shutil
import struct
import tarfile
import time
import zipfile
import argparse

import numpy as np
from scipy import signal

import re


# ------------------------------------------------------------------
# Natural sorting helper (FIX)
# ------------------------------------------------------------------
def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]

# ---------------------------------------------------------------------------
# Dataset extraction  (ZIP / TAR.GZ / GZ / BZ2 / XZ)
# ---------------------------------------------------------------------------

def _detect_format(path: str) -> str:
    """Return format string by reading magic bytes, ignoring file extension."""
    try:
        with open(path, 'rb') as f:
            h = f.read(16)
    except OSError:
        return 'unreadable'
    if h[:4] == b'PK\x03\x04':
        return 'zip'
    if h[:2] == b'\x1f\x8b':
        return 'gz'       # gzip or .tar.gz
    if h[:3] == b'BZh':
        return 'bz2'
    if h[:6] == b'\xfd7zXZ\x00':
        return 'xz'
    if h[:5] == b'ustar':
        return 'tar'
    try:
        with open(path, 'rb') as f:
            f.seek(257)
            if f.read(5) == b'ustar':
                return 'tar'
    except OSError:
        pass
    return 'unknown'


def _is_already_bin(path: str) -> bool:
    """Heuristic: file looks like interleaved float32 IQ data."""
    try:
        size = os.path.getsize(path)
        if size == 0 or size % 8 != 0:
            return False
        with open(path, 'rb') as f:
            raw = f.read(64)
        floats = struct.unpack('<16f', raw)
        return all(-1e6 < v < 1e6 for v in floats)
    except Exception:
        return False


def unzip_dataset(root_dir: str, delete_archives: bool = False,
                  dry_run: bool = False) -> int:

    archive_exts = {'.zip', '.gz', '.tgz', '.tar', '.bz2', '.xz', '.tbz2'}
    archives = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in archive_exts):
                archives.append(os.path.join(dirpath, fname))
    archives.sort()

    if not archives:
        print(f"[unzip_dataset] No archives found under {root_dir}")
        return 0

    print(f"[unzip_dataset] Found {len(archives)} archive(s) under {root_dir}")
    extracted = skipped = failed = 0

    for apath in archives:
        dest      = os.path.dirname(apath)
        rel       = os.path.relpath(apath, root_dir)
        fmt       = _detect_format(apath)
        size_mb   = os.path.getsize(apath) / (1024 * 1024)

        print(f"  {'[DRY]' if dry_run else '[ARC]'} {rel}  "
              f"({size_mb:.1f} MB, {fmt})")

        if dry_run:
            print(f"         → would extract to {os.path.relpath(dest, root_dir)}/")
            continue

        # ── check already extracted ──────────────────────────────────────
        already = False
        try:
            if fmt == 'zip':
                with zipfile.ZipFile(apath) as zf:
                    members = zf.namelist()
                already = all(os.path.exists(os.path.join(dest, m))
                              for m in members)
            elif fmt in ('gz', 'bz2', 'xz', 'tar'):
                if tarfile.is_tarfile(apath):
                    with tarfile.open(apath) as tf:
                        members = tf.getnames()
                    already = all(os.path.exists(os.path.join(dest, m))
                                  for m in members)
                else:
                    # plain single-file gzip/bz2/xz
                    out_name = os.path.basename(apath)
                    for ext in ('.gz', '.bz2', '.xz'):
                        if out_name.lower().endswith(ext):
                            out_name = out_name[:-len(ext)]
                            break
                    already = os.path.exists(os.path.join(dest, out_name))
        except Exception:
            already = False

        if already:
            print(f"         → already extracted, skipping")
            skipped += 1
            if delete_archives:
                os.remove(apath)
            continue

        # ── extract ──────────────────────────────────────────────────────
        try:
            if fmt == 'zip':
                with zipfile.ZipFile(apath) as zf:
                    zf.extractall(dest)
                    n = len(zf.namelist())
                print(f"         → extracted {n} file(s)")

            elif fmt in ('gz', 'bz2', 'xz', 'tar'):
                if tarfile.is_tarfile(apath):
                    with tarfile.open(apath) as tf:
                        tf.extractall(dest)
                        n = len(tf.getnames())
                    print(f"         → extracted {n} file(s) (tar)")
                else:
                    # single-file compressed
                    out_name = os.path.basename(apath)
                    for ext in ('.gz', '.bz2', '.xz'):
                        if out_name.lower().endswith(ext):
                            out_name = out_name[:-len(ext)]
                            break
                    out_path = os.path.join(dest, out_name)
                    open_fn  = (gzip.open   if fmt == 'gz'  else
                                bz2_open    if fmt == 'bz2' else
                                lzma_open)
                    # import lazily so missing modules only fail if needed
                    if fmt == 'bz2':
                        import bz2
                        open_fn = bz2.open
                    elif fmt == 'xz':
                        import lzma
                        open_fn = lzma.open
                    with open_fn(apath, 'rb') as src, \
                         open(out_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"         → extracted to {os.path.basename(out_path)}")

            else:
                print(f"         → unsupported format '{fmt}', skipping")
                failed += 1
                continue

            extracted += 1
            if delete_archives:
                os.remove(apath)
                print(f"         → deleted archive")

        except Exception as e:
            print(f"         → FAILED: {e}")
            failed += 1

    if not dry_run:
        print(f"\n[unzip_dataset] Done — "
              f"extracted={extracted}  skipped={skipped}  failed={failed}\n")
    return extracted


# ---------------------------------------------------------------------------
# Low-level binary reader
# ---------------------------------------------------------------------------

def read_complex_bin(filename, start_idx=0, max_samples=None,
                     max_retries=12, retry_delay=5.0):

    raw = None
    for attempt in range(1, max_retries + 1):
        try:
            if max_samples is None:
                raw = np.fromfile(filename, dtype='<f4')
            else:
                raw = np.fromfile(
                    filename,
                    dtype='<f4',
                    count=int(max_samples) * 2,
                    offset=int(start_idx) * 8,
                )
            break
        except (FileNotFoundError, OSError) as e:
            if attempt == max_retries:
                raise
            print(f"  WARNING: transient read error on "
                  f"{os.path.basename(filename)} (attempt {attempt}/"
                  f"{max_retries}): {e} — retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    if raw.size == 0:
        raise ValueError(f"File is empty: {filename}")
    if raw.size % 2 != 0:
        raise ValueError(
            f"{filename}: odd number of float32 values — "
            "cannot interpret as interleaved IQ."
        )

    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    # Replace any NaN/Inf values with zero so a single bad sample
    # does not contaminate every slice drawn from this file.
    bad_mask = ~np.isfinite(iq.real) | ~np.isfinite(iq.imag)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        print(f"  WARNING: {n_bad} NaN/Inf sample(s) in "
              f"{os.path.basename(filename)} — zeroed out.")
        iq[bad_mask] = 0.0

    if np.abs(iq).std() < 1e-9:
        print(f"  WARNING: near-zero variance in {os.path.basename(filename)}"
              " — check file format/content.")

    if max_samples is None:
        iq = iq[start_idx:]
    if iq.size == 0:
        raise ValueError(
            f"start_idx={start_idx} is past end of {filename} "
            f"({iq.size + start_idx} total samples)."
        )
    return iq


# ---------------------------------------------------------------------------
# Per-recording pre-conditioning
# ---------------------------------------------------------------------------

def precondition_recording(iq):

    iq = iq - iq.mean()
    rms = np.sqrt((np.abs(iq) ** 2).mean()) + 1e-8
    return (iq / rms).astype(np.complex64)


# ---------------------------------------------------------------------------
# Complex → 2-channel real representation
# ---------------------------------------------------------------------------

def complex_to_iq(iq):

    return np.stack([iq.real.astype(np.float32),
                     iq.imag.astype(np.float32)], axis=-1)


def complex_to_magphase(iq):
    """Legacy mag+phase — kept for ablation studies only."""
    mag   = np.abs(iq).astype(np.float32)
    phase = np.angle(iq).astype(np.float32)
    return np.stack([mag, phase], axis=-1)


# ---------------------------------------------------------------------------
# IQ drift augmentation  (cross-day robustness)
# ---------------------------------------------------------------------------

def augment_iq(x, freq_shift_max=0.02, phase_jitter_max=0.1,
               amp_jitter_max=0.05, rng=None):
    """
    Simulate cross-day hardware drift on a single IQ slice.

    What each perturbation models
    ─────────────────────────────
    freq_shift  : oscillator frequency drift between sessions.
                  Applied as a complex rotation that ramps across samples.
    phase_jitter: random phase offset from connector / cable changes.
                  Applied as a fixed complex rotation to the whole slice.
    amp_jitter  : path-loss / gain variation.
                  Applied as a scalar amplitude scale.

    Parameters
    ──────────
    x               : np.ndarray  shape (slice_len, 2)  — I/Q channels
    freq_shift_max  : float  max ±fractional frequency offset   (default 0.02)
    phase_jitter_max: float  max ±phase rotation in radians     (default 0.1)
    amp_jitter_max  : float  max ±fractional amplitude change   (default 0.05)
    rng             : np.random.Generator | None  — pass for reproducibility

    Returns
    ───────
    np.ndarray  shape (slice_len, 2)  float32
    """
    if rng is None:
        rng = np.random.default_rng()

    N  = x.shape[0]
    cx = x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64)

    # 1. Frequency offset — ramps across the slice (oscillator drift)
    delta = rng.uniform(-freq_shift_max, freq_shift_max)
    t     = np.arange(N, dtype=np.float64)
    cx    = cx * np.exp(1j * 2 * np.pi * delta * t)

    # 2. Phase rotation — fixed offset across whole slice (connector change)
    phi = rng.uniform(-phase_jitter_max, phase_jitter_max)
    cx  = cx * np.exp(1j * phi)

    # 3. Amplitude scale — overall gain variation (path loss)
    scale = 1.0 + rng.uniform(-amp_jitter_max, amp_jitter_max)
    cx   *= scale

    return np.stack([cx.real, cx.imag], axis=-1).astype(np.float32)


def augment_batch(X, aug_prob=0.5, freq_shift_max=0.02,
                  phase_jitter_max=0.1, amp_jitter_max=0.05, seed=None):
    """
    Apply augment_iq to a random subset of slices in a batch.

    Parameters
    ──────────
    X            : np.ndarray  shape (N, slice_len, 2)
    aug_prob     : float  probability of augmenting each slice  (default 0.5)
    seed         : int | None

    Returns
    ───────
    np.ndarray  shape (N, slice_len, 2)  float32  — augmented copy
    """
    rng    = np.random.default_rng(seed)
    X_out  = X.copy()
    mask   = rng.random(len(X)) < aug_prob
    for i in np.where(mask)[0]:
        X_out[i] = augment_iq(
            X[i],
            freq_shift_max   = freq_shift_max,
            phase_jitter_max = phase_jitter_max,
            amp_jitter_max   = amp_jitter_max,
            rng              = rng,
        )
    return X_out


# ---------------------------------------------------------------------------
# Normalisation helpers (fit on training split, apply to both)
# ---------------------------------------------------------------------------

def compute_normalization_stats(X):

    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return mean, std


def apply_normalization(X, mean, std):
    """Apply pre-computed mean/std. Returns float32."""
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram (STFT) path
# ---------------------------------------------------------------------------

def stft_spectrogram(iq, seg_len, fs=2_000_000):
    _, _, Zxx = signal.stft(iq, nperseg=seg_len, fs=fs,
                            return_onesided=False)
    return (1000.0 * np.abs(Zxx) ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def _slice_recording(iq_2ch, slice_len, stride, n_requested):

    total     = iq_2ch.shape[0]
    max_start = total - slice_len

    if max_start <= 0:
        print(f"  WARNING: recording too short ({total} samples) "
              f"for slice_len={slice_len}. Skipping.")
        return np.empty((0, slice_len, 2), dtype=np.float32), 0

    if stride == 'r':
        pool   = list(range(max_start + 1))
        n      = min(n_requested, len(pool))
        starts = sorted(random.sample(pool, n))
    else:
        overlap_pct = max(0, (slice_len - stride) / slice_len * 100)
        if overlap_pct > 0:
            print(f"  INFO: stride={stride}, slice_len={slice_len} → "
                  f"{overlap_pct:.0f}% overlap. "
                  f"Use stride>={slice_len} for zero overlap.")
        starts = [i * stride for i in range(n_requested)
                  if i * stride <= max_start]

    if len(starts) < n_requested:
        print(f"  WARNING: requested {n_requested} slices, "
              f"only {len(starts)} fit "
              f"(total={total}, slice_len={slice_len}, stride={stride}).")

    slices = np.stack(
        [iq_2ch[s: s + slice_len] for s in starts], axis=0
    ).astype(np.float32)

    return slices, len(starts)


def read_spread_slices(filename, n_requested, slice_len, max_retries=12,
                       retry_delay=5.0):
    """Read n_requested slices spread evenly across the WHOLE recording.

    The default numeric-stride path walks a contiguous block from the start
    of the file, which for these captures covers well under 1% of the
    recording — every slice then shares one channel/AGC realisation and the
    train/test split leaks. Spreading the start offsets samples the full
    capture instead, and memory-maps so only the slices themselves are read
    rather than the entire 400 MB file.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Seek-and-read each slice rather than memory-mapping the whole
            # file: a 400 MB mapping per recording adds page-cache pressure
            # that can push Phase 4 (which already holds the test day plus
            # its embeddings) into an OOM kill on a 16 GB machine.
            total = os.path.getsize(filename) // 8
            max_start = total - slice_len
            if max_start <= 0:
                print(f"  WARNING: recording too short ({total} samples) "
                      f"for slice_len={slice_len}. Skipping.")
                return np.empty((0, slice_len), dtype=np.complex64), 0

            starts = np.unique(
                np.linspace(0, max_start, n_requested).astype(np.int64)
            )
            out = np.empty((len(starts), slice_len), dtype=np.complex64)
            with open(filename, 'rb') as fh:
                for i, s in enumerate(starts):
                    fh.seek(int(s) * 8)
                    chunk = np.frombuffer(
                        fh.read(slice_len * 8), dtype='<f4', count=slice_len * 2
                    )
                    out[i] = chunk[0::2] + 1j * chunk[1::2]

            bad = ~np.isfinite(out.real) | ~np.isfinite(out.imag)
            if bad.any():
                print(f"  WARNING: {int(bad.sum())} NaN/Inf sample(s) in "
                      f"{os.path.basename(filename)} — zeroed.")
                out[bad] = 0

            if len(starts) < n_requested:
                print(f"  WARNING: requested {n_requested} slices, "
                      f"only {len(starts)} distinct offsets fit "
                      f"(total={total}, slice_len={slice_len}).")
            return out, len(starts)

        except (FileNotFoundError, OSError) as e:
            if attempt == max_retries:
                raise
            print(f"  WARNING: transient read error on "
                  f"{os.path.basename(filename)} (attempt {attempt}/"
                  f"{max_retries}): {e} — retrying in {retry_delay}s...")
            time.sleep(retry_delay)


# ---------------------------------------------------------------------------
# Per-device dataset builder
# ---------------------------------------------------------------------------

def build_device_dataset(glob_pattern, n_slices_per_dev, slice_len,
                         start_idx, stride, mul_trans, data_type, window,
                         precondition=True, return_groups=False):

    filelist = sorted(glob.glob(glob_pattern))
    if not filelist:
        # A transient drive hiccup can make glob briefly see an empty dir
        # even though the files are really there — retry before giving up.
        for attempt in range(1, 6):
            print(f"  WARNING: no files matched {glob_pattern} "
                  f"(attempt {attempt}/5) — retrying in 5s...")
            time.sleep(5.0)
            filelist = sorted(glob.glob(glob_pattern))
            if filelist:
                break
    if not filelist:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    num_files  = len(filelist)
    all_slices = []
    all_groups = []   # (file_idx, time_rank) per slice — used for leak-free splits

    if mul_trans:
        base, rem = divmod(n_slices_per_dev, num_files)
        counts = [base + (1 if i < rem else 0) for i in range(num_files)]
    else:
        counts = [n_slices_per_dev] + [0] * (num_files - 1)

    for i, fpath in enumerate(filelist):
        n_req = counts[i]
        if n_req == 0:
            continue

        print(f"  {os.path.basename(fpath)}  requesting {n_req} slices")

        if data_type == 'IQ' and stride == 'spread':
            # Offsets spread across the whole recording, read via memmap.
            iq_slices, got = read_spread_slices(fpath, n_req, slice_len)
            if got > 0:
                if precondition:
                    # Same normalisation as precondition_recording, but the
                    # statistics come from samples spanning the full capture.
                    iq_slices = iq_slices - iq_slices.mean()
                    rms = np.sqrt((np.abs(iq_slices) ** 2).mean()) + 1e-8
                    iq_slices = (iq_slices / rms).astype(np.complex64)
                all_slices.append(complex_to_iq(iq_slices))
                all_groups.append(
                    np.stack([np.full(got, i, dtype=np.int32),
                              np.arange(got, dtype=np.int32)], axis=1)
                )
            if not mul_trans:
                break
            continue

        try:
            max_samples = None
            if data_type == 'IQ' and stride != 'r':
                max_samples = (max(0, n_req - 1) * int(stride)) + slice_len
            iq_raw = read_complex_bin(
                fpath,
                start_idx=start_idx,
                max_samples=max_samples,
            )
        except ValueError as e:
            print(f"  ERROR: {e} — skipping.")
            continue

        if precondition:
            iq_raw = precondition_recording(iq_raw)

        if data_type == 'IQ':
            iq_2ch = complex_to_iq(iq_raw)
            slices, got = _slice_recording(iq_2ch, slice_len, stride, n_req)
            if got > 0:
                all_slices.append(slices)
                all_groups.append(
                    np.stack([np.full(got, i, dtype=np.int32),
                              np.arange(got, dtype=np.int32)], axis=1)
                )
            if not mul_trans:
                break

        elif data_type == 'spectrogram':
            spec   = stft_spectrogram(iq_raw, window)
            chunks = [
                spec[:, j: j + slice_len]
                for j in range(0, spec.shape[1] - slice_len, slice_len)
            ]
            chunks = chunks[:n_req]
            if chunks:
                all_slices.append(
                    np.stack(chunks, axis=0).astype(np.float32)
                )
        else:
            raise ValueError(f"Unknown data_type: {data_type!r}")

    if not all_slices:
        raise RuntimeError(f"No valid slices from: {glob_pattern}")

    X = np.concatenate(all_slices, axis=0)
    if return_groups:
        groups = (np.concatenate(all_groups, axis=0) if all_groups
                  else np.zeros((len(X), 2), dtype=np.int32))
        return X, groups
    return X

def natural_key(name):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', name)]
def load_data(args, split=True, normalize=True):

    print(f"\nLoading data from: {args.root_dir}")

    # Auto-extract any archives present before scanning for .bin files
    if getattr(args, 'auto_unzip', True):
        unzip_dataset(args.root_dir, delete_archives=False)

    # FIX: natural numeric sorting
    # Before: Device_1, Device_10, Device_11, ..., Device_2
    # After : Device_1, Device_2, Device_3, ..., Device_20
    dev_names = sorted(
        (
            d for d in os.listdir(args.root_dir)
            if os.path.isdir(os.path.join(args.root_dir, d))
        ),
        key=natural_key
    )

    n_devices = len(dev_names)
    print(f"Devices found: {n_devices}  {dev_names}")

    split_ratio = 0.8 if split else 1.0
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    selected_devices = getattr(args, 'device_ids', None)
    if selected_devices is not None:
        selected_devices = {int(v) for v in selected_devices}
        print(f"Loading selected device indices only: {sorted(selected_devices)}")

    for label, dev_name in enumerate(dev_names):
        if selected_devices is not None and label not in selected_devices:
            continue

        dev_dir = os.path.join(args.root_dir, dev_name)

        if args.location:
            glob_pat = os.path.join(dev_dir, args.location, args.file_key)
        else:
            glob_pat = os.path.join(dev_dir, args.file_key)

        X, groups = build_device_dataset(
            glob_pattern=glob_pat,
            n_slices_per_dev=args.num_slice,
            slice_len=args.slice_len,
            start_idx=args.start_idx,
            stride=args.stride,
            mul_trans=args.mul_trans,
            data_type=args.data_type,
            window=args.window,
            return_groups=True,
        )

        if split and args.stride == 'spread':
            # Leak-free split: within each transmission file, the LAST
            # (1-split_ratio) of slices by recording time become the test
            # set. A random shuffle here would interleave train and test
            # slices from the same instant of the same capture, which lets
            # the model match on channel/AGC state instead of on the device.
            order = np.lexsort((groups[:, 1], groups[:, 0]))
            X, groups = X[order], groups[order]
            keep = np.zeros(len(X), dtype=bool)
            for fidx in np.unique(groups[:, 0]):
                m = np.flatnonzero(groups[:, 0] == fidx)
                keep[m[:int(len(m) * split_ratio)]] = True
            X = np.concatenate([X[keep], X[~keep]], axis=0)
            n_train = int(keep.sum())
        else:
            # Shuffle per device before splitting
            perm = np.random.permutation(X.shape[0])
            X = X[perm]
            n_train = int(X.shape[0] * split_ratio)

        y = np.full(X.shape[0], label, dtype=np.int32)

        if normalize:
            mean, std = compute_normalization_stats(X[:n_train])
            X = apply_normalization(X, mean, std)

        x_train_list.append(X[:n_train])
        y_train_list.append(y[:n_train])

        if split:
            x_test_list.append(X[n_train:])
            y_test_list.append(y[n_train:])

        zero_count = (X[:, :, 0].sum(axis=1) == 0).sum()
        print(
            f"  Device {label} ({dev_name}): "
            f"{X.shape[0]} slices  "
            f"zero-I slices: {zero_count}"
            + (" ← WARNING" if zero_count > 0 else "")
        )

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    if split:
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
    else:
        x_test = np.empty((0, args.slice_len, 2), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.int32)

    print(f"\nTrain: {x_train.shape}   Test: {x_test.shape}")
    print(f"Label distribution (train): {np.bincount(y_train)}")

    nan_count = np.isnan(x_train).sum()
    if nan_count > 0:
        print(
            f"  WARNING: {nan_count} NaN value(s) in final array — "
            f"replacing with 0. Check source files for corruption."
        )
        x_train = np.nan_to_num(x_train, nan=0.0)

    return x_train, y_train, x_test, y_test, n_devices
# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------

class LoadDataOpts:

    def __init__(
        self,
        root_dir,
        file_key   = '*.bin',
        location   = 'equ',   # subfolder inside each device folder
        num_slice  = 3_000,
        start_idx  = 0,
        slice_len  = 288,
        stride     = 288,
        mul_trans  = True,
        window     = 64,
        data_type  = 'IQ',
        auto_unzip = True,
        device_ids = None,
    ):
        self.root_dir   = root_dir
        self.file_key   = file_key
        self.location   = location
        self.num_slice  = num_slice
        self.start_idx  = start_idx
        self.slice_len  = slice_len
        self.stride     = stride
        self.mul_trans  = mul_trans
        self.window     = window
        self.data_type  = data_type
        self.auto_unzip = auto_unzip
        self.device_ids = device_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(description='RF complex IQ data loader')
    p.add_argument('-d', '--root_dir',   required=True)
    p.add_argument('-k', '--file_key',   default='*.bin')
    p.add_argument('-n', '--num_slice',  type=int, default=10_000)
    p.add_argument('-i', '--start_idx',  type=int, default=0)
    p.add_argument('-l', '--slice_len',  type=int, default=288)
    p.add_argument('-s', '--stride',     type=int, default=288)
    p.add_argument('--no_mul_trans',     action='store_true')
    p.add_argument('--data_type',        default='IQ',
                   choices=['IQ', 'spectrogram'])
    return p.parse_args(argv)


if __name__ == '__main__':
    import sys
    a = _parse_args(sys.argv[1:])
    opts = LoadDataOpts(
        root_dir  = a.root_dir,
        file_key  = a.file_key,
        num_slice = a.num_slice,
        start_idx = a.start_idx,
        slice_len = a.slice_len,
        stride    = a.stride,
        mul_trans = not a.no_mul_trans,
        data_type = a.data_type,
    )
    x_tr, y_tr, x_te, y_te, n = load_data(opts, split=True)
    print(f"\nLoaded {n} devices.")
    print(f"Train: {x_tr.shape}   Test: {x_te.shape}")
    print(f"I range: [{x_tr[:,:,0].min():.3f}, {x_tr[:,:,0].max():.3f}]")
    print(f"Q range: [{x_tr[:,:,1].min():.3f}, {x_tr[:,:,1].max():.3f}]")
