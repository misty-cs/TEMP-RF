"""Oregon State LoRa (Diff_Days_Indoor), step 1: packets in one 20 s transmission (read-only)."""
import sys, numpy as np
f = sys.argv[1]
x = np.fromfile(f, dtype='<f4'); x = (x[0::2] + 1j * x[1::2]).astype(np.complex64)
fs, N = 1e6, 1024                     # SF7, 125 kHz, 1 MS/s -> 1024 samples per chirp
print(f'{len(x)} samples = {len(x)/fs:.1f} s, median |x| {np.median(np.abs(x)):.4f}, max {np.abs(x).max():.3f}')
p = np.convolve(np.abs(x) ** 2, np.ones(256) / 256, 'same'); pdb = 10 * np.log10(p + 1e-20)
floor = np.percentile(pdb, 10); on = pdb > floor + 10
r = np.diff(np.r_[0, on.astype(np.int8), 0]); st = np.flatnonzero(r == 1); en = np.flatnonzero(r == -1)
L = en - st; keep = L > 5 * N; st, en, L = st[keep], en[keep], L[keep]
print(f'noise floor {floor:.1f} dB, bursts > 5 chirps: {len(L)}; length (samples) p10/p50/p90: {np.percentile(L,[10,50,90]).astype(int) if len(L) else None}')
if len(L) > 1:
    print(f'  gap between packet starts (ms): median {np.median(np.diff(st))/1e3:.1f}')
# dechirp the first packet with an ideal SF7 up-chirp: preamble chirps give a constant tone
t = np.arange(N) / fs; k = 125e3 / (N / fs)
up = np.exp(1j * np.pi * k * t ** 2 - 1j * np.pi * 125e3 * t)
if len(st):
    s0 = st[0]; seg = x[s0: s0 + 12 * N]
    peaks = []
    for i in range(12):
        d = seg[i * N:(i + 1) * N] * np.conj(up)
        F = np.abs(np.fft.fft(d)); peaks.append((int(F.argmax()), round(float(F.max() / F.mean()), 1)))
    print('  first 12 chirp windows, dechirped FFT peak (bin, peak/mean):', peaks)
