import sys, numpy as np
B = np.fromfile(sys.argv[1], dtype=np.complex64, count=48 * 60000).reshape(-1, 48).astype(np.complex128)
a = np.abs(B); z = a < 1e-6
print('exact zeros: %.2f%% of values; blocks all-zero: %.2f%%' % (100 * z.mean(), 100 * z.all(1).mean()))
small = a < 0.05 * np.median(a)
print('near-zero (<5%% of median): %.1f%%, per subcarrier max %.1f%%' % (100 * small.mean(), 100 * small.mean(0).max()))
Z = B / np.median(a, 0, keepdims=True)
c2 = [abs(np.vdot(Z[:-2, k], Z[2:, k])) / np.sqrt(np.vdot(Z[:-2, k], Z[:-2, k]).real * np.vdot(Z[2:, k], Z[2:, k]).real) for k in range(48)]
c1 = [abs(np.vdot(Z[:-1, k], Z[1:, k])) / np.sqrt(np.vdot(Z[:-1, k], Z[:-1, k]).real * np.vdot(Z[1:, k], Z[1:, k]).real) for k in range(48)]
print('lag-1-block corr per subcarrier: max %.2f median %.3f' % (max(c1), np.median(c1)))
print('lag-2-block corr per subcarrier: max %.2f median %.3f; top subcarriers %s' % (max(c2), np.median(c2), np.argsort(c2)[::-1][:6].tolist()))
