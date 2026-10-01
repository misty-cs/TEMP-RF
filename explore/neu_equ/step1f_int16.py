import sys, numpy as np
g = np.fromfile(sys.argv[1], dtype='<i2', count=2 * 3_000_000).astype(np.float64)
x = g[0::2] + 1j * g[1::2]
print('int16 view: |x| percentiles', np.percentile(np.abs(x), [1, 25, 50, 75, 99, 99.99]).round(1))
print('distinct int16 values in first 1e6:', len(np.unique(g[:1_000_000])))
def ac(lag, n=2_000_000):
    a, b = x[:n], x[lag:n + lag]; return abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
print('autocorr by lag:', {L: round(ac(L), 3) for L in (1, 2, 4, 16, 32, 48, 64, 80, 96)})
p = np.convolve(np.abs(x) ** 2, np.ones(20) / 20, 'same'); pdb = 10 * np.log10(p + 1e-9)
print('power dB percentiles', np.percentile(pdb, [1, 10, 50, 90, 99]).round(1))
