import sys, numpy as np
f = np.fromfile(sys.argv[1], dtype='<f4', count=96 * 60000).astype(np.float64)
def score(z):
    z = z[np.isfinite(z)]; z = z / np.median(np.abs(z)); z = z[np.abs(z) < 2.5]
    r = z.real; k = np.mean((r - r.mean()) ** 4) / np.var(r) ** 2
    pts = {'QPSK': np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2) * 1.0,
           '16QAM': np.array([a+1j*b for a in (-3,-1,1,3) for b in (-3,-1,1,3)]) / np.sqrt(10)}
    return k
for n in (48, 52, 64):
    per = 2 * n; m = len(f) // per * per; F = f[:m].reshape(-1, per)
    planar = F[:, :n] + 1j * F[:, n:]
    inter = F[:, 0::2] + 1j * F[:, 1::2]
    print(f'block {n}: kurtosis planar {score(planar.ravel()):.2f} | interleaved {score(inter.ravel()):.2f}')
# also float16/int16 interpretations
for dt in ('<i2', '<f2'):
    try:
        g = np.fromfile(sys.argv[1], dtype=dt, count=96 * 60000).astype(np.float64)
        print(dt, 'interleaved kurtosis %.2f' % score((g[0::2] + 1j * g[1::2])))
    except Exception as e:
        print(dt, e)
