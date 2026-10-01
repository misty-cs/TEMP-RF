import sys, numpy as np
B = np.fromfile(sys.argv[1], dtype='<c8', count=48 * 200000).reshape(-1, 48).astype(np.complex128)
phi = np.angle(np.sum(B ** 2, 1, keepdims=True)) / 2          # BPSK phase per block
Zr = B * np.exp(-1j * phi)
R = (Zr.real ** 2).sum(1) / ((np.abs(Zr) ** 2).sum(1) + 1e-30)
h, e = np.histogram(R, bins=10, range=(0.5, 1.0))
print('per-block energy fraction on the real axis (BPSK ~1.0, noise ~0.5-0.6):')
for c, lo in zip(h, e[:-1]): print(f'  {lo:.2f}-{lo+0.05:.2f} {"#" * int(50 * c / h.max())} {c}')
good = R > 0.9
print('blocks with R>0.9: %.1f%%' % (100 * good.mean()))
if good.any():
    z = Zr[good].ravel(); z = z / np.median(np.abs(z))
    print('in those blocks: real-part histogram (-2..2):')
    hh, _ = np.histogram(z.real, bins=40, range=(-2, 2)); ch = ' .:-=+*#%@'
    print('  ' + ''.join(ch[min(9, int(9 * v / hh.max()))] for v in hh))
