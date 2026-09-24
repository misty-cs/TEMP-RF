#!/usr/bin/env python3
"""
signal_preprocessing.py  —  Channel-mitigating preprocessing for RF fingerprinting.

Three complementary operations that strip session-specific channel artefacts
while preserving device-intrinsic hardware impairments:

  1. remove_cfo(iq)
       Estimates and removes bulk carrier-frequency offset via consecutive-
       sample autocorrelation.  Removes day-to-day frequency drift while
       leaving device-level phase-noise texture intact.

  2. normalize_per_slice(iq)
       L2 (RMS) normalisation per slice.  Removes path-loss / AGC amplitude
       differences between sessions — the single biggest day-to-day variation.

  3. augment_channel(iq, ...)
       Training-time augmentation that simulates channel variation:
       random phase rotation, amplitude jitter, additive noise.

Public API
----------
preprocess_batch(X, remove_cfo, per_slice_norm, augment, ...)
remove_cfo(iq)
normalize_per_slice(iq)
augment_channel(iq, ...)
global_zscore_after_preprocess(X_train, X_test, X_cross=None)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. CFO removal
# ---------------------------------------------------------------------------

def remove_cfo(iq: np.ndarray) -> np.ndarray:
    """
    Estimate and remove bulk carrier-frequency offset from a single IQ slice.

    The CFO is estimated as the mean phase increment between consecutive
    samples via the angle of the autocorrelation at lag 1:

        ω_cfo = angle( E[ x[n] · conj(x[n-1]) ] )

    Parameters
    ----------
    iq : (slice_len, 2) float32 [I, Q]

    Returns
    -------
    (slice_len, 2) float32, CFO-corrected
    """
    c = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    r = np.mean(c[1:] * np.conj(c[:-1]))
    if np.abs(r) < 1e-12:
        return iq
    cfo   = np.angle(r)
    t     = np.arange(len(c), dtype=np.float64)
    c_out = c * np.exp(-1j * cfo * t)
    return np.stack(
        [c_out.real.astype(np.float32), c_out.imag.astype(np.float32)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# 1b. Spectral equalization (channel whitening)
# ---------------------------------------------------------------------------

def equalize_spectrum(iq: np.ndarray, smooth_bins: int = 9) -> np.ndarray:
    """
    Whiten the smoothed magnitude spectrum of a single IQ slice.

    Divides each FFT bin by a moving-average magnitude envelope, flattening
    the multipath/receiver frequency response while keeping phase and fine
    spectral structure (IQ imbalance images, spurs, filter ripple) that carry
    the device fingerprint.

    Parameters
    ----------
    iq          : (slice_len, 2) float32 [I, Q]
    smooth_bins : envelope moving-average width in FFT bins (odd, default 9)

    Returns
    -------
    (slice_len, 2) float32, spectrally equalized
    """
    c = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    F = np.fft.fft(c)
    mag = np.abs(F)
    if mag.max() < 1e-12:
        return iq
    k = max(3, int(smooth_bins) | 1)
    kernel = np.ones(k) / k
    # circular smoothing so band edges are treated like any other bin
    env = np.convolve(np.concatenate([mag[-k:], mag, mag[:k]]), kernel,
                      mode='same')[k:-k]
    F_eq = F / (env + 1e-8 * mag.max())
    c_out = np.fft.ifft(F_eq)
    return np.stack(
        [c_out.real.astype(np.float32), c_out.imag.astype(np.float32)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# 2. Per-slice L2 (RMS) normalisation
# ---------------------------------------------------------------------------

def normalize_per_slice(iq: np.ndarray) -> np.ndarray:
    """
    Normalise a single IQ slice to unit RMS power.

    Parameters
    ----------
    iq : (slice_len, 2) float32

    Returns
    -------
    (slice_len, 2) float32, unit-RMS
    """
    rms = float(np.sqrt(np.mean(iq.astype(np.float64) ** 2))) + 1e-8
    return (iq / rms).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Training-time channel augmentation
# ---------------------------------------------------------------------------

def augment_channel(
    iq: np.ndarray,
    phase_rot_range: float = np.pi,
    amp_jitter_db:   float = 3.0,
    noise_snr_db:    float = 25.0,
    apply_prob:      float = 0.8,
    multipath_taps:  int   = 0,
    multipath_mag:   float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Randomly simulate channel variation for one IQ slice (training only).

    Three independent operations each applied with probability `apply_prob`:
      a) Random phase rotation  — simulates phase offset between sessions.
      b) Amplitude jitter       — simulates path-loss / AGC variation.
      c) Additive Gaussian noise — simulates thermal noise floor changes.
      d) Random multipath        — convolves with a short random FIR, which
         simulates the frequency-selective fading that changes between
         recording days. Disabled unless multipath_taps > 0. The other three
         operations are all flat in frequency, so without this the model
         never sees a frequency-selective channel during training.

    Parameters
    ----------
    iq             : (slice_len, 2)
    phase_rot_range: max ± phase rotation in radians (default π)
    amp_jitter_db  : max ± amplitude change in dB (default 3 dB)
    noise_snr_db   : SNR of injected noise in dB (default 25 dB)
    apply_prob     : per-operation application probability (default 0.8)
    rng            : np.random.Generator (optional; created if None)

    Returns
    -------
    (slice_len, 2) float32
    """
    if rng is None:
        rng = np.random.default_rng()

    c = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)

    if rng.random() < apply_prob:
        c *= np.exp(1j * rng.uniform(-phase_rot_range, phase_rot_range))

    if rng.random() < apply_prob:
        c *= 10 ** (rng.uniform(-amp_jitter_db, amp_jitter_db) / 20.0)

    if rng.random() < apply_prob:
        pwr   = float(np.mean(np.abs(c) ** 2)) + 1e-12
        noise = rng.standard_normal(len(c)) + 1j * rng.standard_normal(len(c))
        noise *= np.sqrt(pwr / (10 ** (noise_snr_db / 10.0)) / 2.0)
        c += noise

    if multipath_taps > 0 and multipath_mag > 0.0 and rng.random() < apply_prob:
        # Tap 0 is the direct path, fixed at unit gain; the extra taps are
        # random complex echoes with magnitudes up to multipath_mag. Keeping
        # the direct path dominant stops the augmentation from inventing a
        # channel that buries the fingerprint.
        taps = np.zeros(multipath_taps + 1, dtype=np.complex128)
        taps[0] = 1.0
        mags = rng.uniform(0.0, multipath_mag, size=multipath_taps)
        phis = rng.uniform(-np.pi, np.pi, size=multipath_taps)
        taps[1:] = mags * np.exp(1j * phis)
        c = np.convolve(c, taps)[:len(c)]

    return np.stack(
        [c.real.astype(np.float32), c.imag.astype(np.float32)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# 4. Batch pipeline
# ---------------------------------------------------------------------------

def preprocess_batch(
    X: np.ndarray,
    do_remove_cfo:      bool  = True,
    do_per_slice_norm:  bool  = True,
    do_equalize:        bool  = False,
    equalize_smooth_bins: int = 9,
    augment:            bool  = False,
    phase_rot_range:    float = np.pi,
    amp_jitter_db:      float = 3.0,
    noise_snr_db:       float = 25.0,
    augment_apply_prob: float = 0.8,
    multipath_taps:     int   = 0,
    multipath_mag:      float = 0.0,
    seed: int | None          = None,
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a batch of IQ slices.

    Parameters
    ----------
    X                 : (N, slice_len, 2)
    do_remove_cfo     : apply CFO correction per slice (default True)
    do_per_slice_norm : apply per-slice RMS normalisation (default True)
    augment           : apply channel augmentation (training only)
    phase_rot_range   : augmentation max phase rotation in radians
    amp_jitter_db     : augmentation max amplitude jitter in dB
    noise_snr_db      : augmentation additive noise SNR in dB
    augment_apply_prob: augmentation per-op probability
    multipath_taps    : number of random echo taps (0 disables multipath)
    multipath_mag     : max echo magnitude relative to the direct path
    seed              : RNG seed for reproducibility

    Returns
    -------
    (N, slice_len, 2) float32
    """
    X   = np.asarray(X, dtype=np.float32)
    out = np.empty_like(X)
    rng = np.random.default_rng(seed) if augment else None

    for i in range(len(X)):
        s = X[i]
        if do_remove_cfo:
            s = remove_cfo(s)
        if do_equalize:
            s = equalize_spectrum(s, smooth_bins=equalize_smooth_bins)
        if do_per_slice_norm:
            s = normalize_per_slice(s)
        if augment:
            s = augment_channel(
                s,
                phase_rot_range = phase_rot_range,
                amp_jitter_db   = amp_jitter_db,
                noise_snr_db    = noise_snr_db,
                apply_prob      = augment_apply_prob,
                multipath_taps  = multipath_taps,
                multipath_mag   = multipath_mag,
                rng             = rng,
            )
        out[i] = s

    return out


# ---------------------------------------------------------------------------
# 5. Global z-score after per-slice preprocessing
# ---------------------------------------------------------------------------

def global_zscore_after_preprocess(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    X_cross: np.ndarray | None = None,
):
    """
    Fit global z-score on X_train (already preprocessed) and apply to
    X_test and optionally X_cross.  Mirrors the load_slice_IQ
    compute_normalization_stats / apply_normalization API.

    Returns
    -------
    X_train_norm, X_test_norm[, X_cross_norm], mean, std
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std  = (X_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

    X_train_n = ((X_train - mean) / std).astype(np.float32)
    X_test_n  = ((X_test  - mean) / std).astype(np.float32)

    if X_cross is not None:
        X_cross_n = ((X_cross - mean) / std).astype(np.float32)
        return X_train_n, X_test_n, X_cross_n, mean, std

    return X_train_n, X_test_n, mean, std


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    N, L = 200, 288

    # CFO removal
    t   = np.arange(L)
    cfo = 0.02
    sig = np.exp(1j * cfo * t) * (1.0 + 0.01 * rng.standard_normal(L))
    iq_raw  = np.stack([sig.real, sig.imag], axis=-1).astype(np.float32)
    iq_corr = remove_cfo(iq_raw)
    c_corr  = iq_corr[:, 0] + 1j * iq_corr[:, 1]
    r_corr  = np.angle(np.mean(c_corr[1:] * np.conj(c_corr[:-1])))
    print(f"CFO residual after correction: {r_corr:.6f} rad/sample")
    assert abs(r_corr) < 1e-6, "CFO removal failed!"

    # Per-slice normalisation
    iq_norm = normalize_per_slice(iq_raw)
    rms     = float(np.sqrt(np.mean(iq_norm ** 2)))
    print(f"RMS after per-slice norm: {rms:.6f}  (expected ~1.0)")
    assert abs(rms - 1.0) < 1e-4, "RMS normalisation failed!"

    # Batch
    X    = rng.standard_normal((N, L, 2)).astype(np.float32)
    X_pp = preprocess_batch(X, augment=True, seed=42)
    assert X_pp.shape == (N, L, 2)
    print(f"Batch preprocessed: {X_pp.shape}  dtype={X_pp.dtype}")

    # Global z-score wrapper
    X_tr   = preprocess_batch(X[:160], augment=False)
    X_te   = preprocess_batch(X[160:], augment=False)
    X_tr_n, X_te_n, mean, std = global_zscore_after_preprocess(X_tr, X_te)
    print(f"Z-score mean shape: {mean.shape}  std min: {std.min():.4f}")

    print("\nAll self-tests passed.")
