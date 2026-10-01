import sys, numpy as np
for dt in ('<c16', '>c16'):
    x = np.fromfile(sys.argv[1], dtype=dt, count=48 * 60000)
    ok = np.isfinite(x).all() and np.nanmax(np.abs(x)) < 1e6
    print(dt, 'finite+sane', ok, 'median |x| %.3g' % np.nanmedian(np.abs(x)) if np.isfinite(x).any() else '')
    if not ok: continue
    z = x / np.median(np.abs(x)); z = z[np.abs(z) < 3]
    k = np.mean((z.real - z.real.mean()) ** 4) / np.var(z.real) ** 2
    H, _, _ = np.histogram2d(z.real, z.imag, bins=41, range=[[-2, 2], [-2, 2]])
    ch = ' .:-=+*#%@'; m = H.max()
    for j in range(40, -1, -2): print('  ' + ''.join(ch[min(9, int(9 * H[i, j] / m))] for i in range(41)))
    print('  kurtosis real %.2f (BPSK -> 1.0)' % k)
