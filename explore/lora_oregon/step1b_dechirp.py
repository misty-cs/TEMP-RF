import sys, numpy as np
f = sys.argv[1]
x = np.fromfile(f, dtype='<f4', count=2 * 2_000_000); x = x[0::2] + 1j * x[1::2]   # first 2 s
fs, N, BW = 1e6, 1024, 125e3
t = np.arange(N) / fs; k = BW / (N / fs)
up = np.exp(1j * np.pi * k * t ** 2 - 1j * np.pi * BW * t)
nw = len(x) // N
bins, ratio = [], []
for i in range(nw):
    F = np.abs(np.fft.fft(x[i * N:(i + 1) * N] * np.conj(up)))
    bins.append(int(F.argmax())); ratio.append(F.max() / F.mean())
bins, ratio = np.array(bins), np.array(ratio)
print('peak/mean over windows: p10 %.1f p50 %.1f p90 %.1f (a chirp gives a strong peak, noise ~3-5)' % tuple(np.percentile(ratio, [10, 50, 90])))
print('first 60 window bins:', bins[:60].tolist())
same = np.r_[False, bins[1:] == bins[:-1]]
runs = np.diff(np.r_[0, same.astype(int), 0]); st = np.flatnonzero(runs == 1); L = np.flatnonzero(runs == -1) - st
print('runs of equal bins (>=4 windows, candidate preambles):', int((L >= 4).sum()), 'lengths', L[L >= 4][:15].tolist())
# spectrum centre (CFO / band position)
S = np.abs(np.fft.fftshift(np.fft.fft(x[:262144]))) ** 2; fr = np.fft.fftshift(np.fft.fftfreq(262144, 1 / fs))
occ = fr[S > 0.1 * S.max()]; print('occupied band approx %.1f .. %.1f kHz' % (occ.min() / 1e3, occ.max() / 1e3))
