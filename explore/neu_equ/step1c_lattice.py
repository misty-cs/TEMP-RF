"""Hide noise-enhanced (faded-subcarrier) values, normalize per subcarrier, look for a QAM lattice."""
import sys, numpy as np
F = sys.argv[1]
B = np.fromfile(F, dtype=np.complex64, count=48 * 60000).reshape(-1, 48).astype(np.complex128)
a = np.abs(B); med = np.median(a, 0, keepdims=True)
Z = B / med                                   # per-subcarrier scale
keep = np.abs(Z) < 2.0                        # drop noise-enhanced outliers
phi = np.angle(np.sum(np.where(keep, Z, 0) ** 4, 1, keepdims=True)) / 4
Zr = (Z * np.exp(-1j * phi))[keep]
print('kept %.1f%% of values; kurtosis of real part %.2f' % (100 * keep.mean(),
      np.mean((Zr.real - Zr.real.mean()) ** 4) / np.var(Zr.real) ** 2))
H, _, _ = np.histogram2d(Zr.real, Zr.imag, bins=61, range=[[-1.8, 1.8], [-1.8, 1.8]])
chars = ' .:-=+*#%@'; m = H.max()
for j in range(60, -1, -2): print('  ' + ''.join(chars[min(9, int(9 * H[i, j] / m))] for i in range(61)))
# 1-D histogram of the real part: peaks = constellation levels
h, e = np.histogram(Zr.real, bins=72, range=(-1.8, 1.8))
print('real-part histogram:'); print('  ' + ''.join(chars[min(9, int(9 * v / h.max()))] for v in h))
