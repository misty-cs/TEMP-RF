"""NEU equalized data, step 1: what are the 48-value blocks? (read-only, small chunks)"""
import sys, glob, numpy as np
F = sys.argv[1]; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
x = np.fromfile(F, dtype=np.complex64, count=48 * NB); B = x.reshape(-1, 48).astype(np.complex128)
print('file', F.split('/')[-3:], 'blocks', len(B))
# per-subcarrier statistics
a = np.abs(B); print('per-subcarrier median |x|:', np.round(np.median(a, 0), 2).tolist())
# H1: per-block common phase (4th power) + robust gain removed
g = np.median(a, 1, keepdims=True); Z = B / g
phi = np.angle(np.sum(Z ** 4, 1, keepdims=True)) / 4; Zr = Z * np.exp(-1j * phi)
z = Zr.ravel(); z = z[np.abs(z) < 3]
H, _, _ = np.histogram2d(z.real, z.imag, bins=41, range=[[-2.5, 2.5], [-2.5, 2.5]])
chars = ' .:-=+*#%@'; m = H.max()
print('derotated constellation (imag +2.5 top):')
for j in range(40, -1, -1): print('  ' + ''.join(chars[min(9, int(9 * H[i, j] / m))] for i in range(41)))
# H1b: is the phase between consecutive blocks continuous (i.e. blocks are consecutive OFDM symbols)?
d = np.angle(np.exp(1j * (phi[1:, 0] - phi[:-1, 0]) * 4)) / 4
print('block-to-block common-phase change: median |d| %.3f rad (uniform would be ~0.39)' % np.median(np.abs(d)))
