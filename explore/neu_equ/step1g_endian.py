import sys, numpy as np
x = np.fromfile(sys.argv[1], dtype='>c8', count=48 * 60000).astype(np.complex128)
print('finite:', np.isfinite(x).all(), 'median |x| %.3g, max %.3g' % (np.median(np.abs(x)), np.abs(x).max()))
z = x / np.median(np.abs(x)); z = z[np.abs(z) < 3]
H, _, _ = np.histogram2d(z.real, z.imag, bins=41, range=[[-2, 2], [-2, 2]])
chars = ' .:-=+*#%@'; m = H.max()
for j in range(40, -1, -2): print('  ' + ''.join(chars[min(9, int(9 * H[i, j] / m))] for i in range(41)))
print('kurtosis real %.2f' % (np.mean((z.real - z.real.mean()) ** 4) / np.var(z.real) ** 2))
