"""Time domain or symbol domain? kurtosis + OFDM lag signatures on the continuous stream."""
import sys, numpy as np
F = sys.argv[1]
x = np.fromfile(F, dtype=np.complex64, count=3_000_000).astype(np.complex128)
x = x / np.median(np.abs(x))
r = x.real; k = np.mean((r - r.mean()) ** 4) / np.var(r) ** 2
print('kurtosis of real part: %.2f (Gaussian 3.0; QPSK 1.0; 16QAM 1.32; 64QAM 1.38)' % k)
def ac(lag, n=2_000_000):
    a, b = x[:n], x[lag:n + lag]
    return abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
print('normalized autocorrelation by lag:', {L: round(ac(L), 3) for L in (1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 160)})
# local (windowed) CP correlation: for OFDM at 20 MS/s, x[n] ~ x[n+64] inside the 16-sample CP of each 80-sample symbol
w = x[:400000]; c = w[:-64] * np.conj(w[64:])
m = np.abs(np.convolve(c, np.ones(16) / 16, 'valid')) / (np.convolve(np.abs(w[:-64]) ** 2, np.ones(16) / 16, 'valid') + 1e-12)
print('windowed lag-64 correlation: p50 %.3f p90 %.3f p99 %.3f' % tuple(np.percentile(m, [50, 90, 99])))
