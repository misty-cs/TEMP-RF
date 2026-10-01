"""Oregon State LoRa, Diff_Days_Indoor, step 2: does a raw device fingerprint carry across days?
(EXPLORATORY, no network). Input: head2s/d{day}_dev{dev}_IQ_{1,2}.dat = first 2 s of two
transmissions per device and day (devices 1-10, days 1-5).

Packets: dechirp 1024-sample windows (SF7, 125 kHz, 1 MS/s); a run of >= 6 equal
FFT-peak bins marks a preamble; its start is refined by NON-COHERENT (per-chirp) correlation with an ideal
8-up-chirp preamble (label-free, same template for every device). Slice = the 8
preamble chirps (8192 samples), RMS-normalized.
Features:
  raw       aligned complex preamble (phase-invariant cosine)
  spectrum  log|FFT| of the preamble (8192 -> 512 bands)
  tone      dechirped preamble: magnitude spectrum around its peak (+-64 bins, peak-aligned)
  cfo       fine frequency of the dechirped tone (scalar; timing and CFO are mixed in a chirp)
Evaluation: nearest centroid / 5-NN, 10 devices (chance 0.10):
  A same day, IQ_1 -> IQ_2 (other transmission)
  B day1 -> day2     C day1 -> day5     D days1-2 -> day5
"""
import glob, os, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
fs, N, BW = 1e6, 1024, 125e3
t = np.arange(N) / fs
up = np.exp(1j * np.pi * (BW / (N / fs)) * t ** 2 - 1j * np.pi * BW * t)
ref = np.tile(up, 8)


def preambles(x):
    nw = len(x) // N
    b = np.array([int(np.abs(np.fft.fft(x[i * N:(i + 1) * N] * np.conj(up))).argmax()) for i in range(nw)])
    same = np.r_[False, np.abs(((b[1:] - b[:-1]) + N // 2) % N - N // 2) <= 3]
    r = np.diff(np.r_[0, same.astype(int), 0]); st = np.flatnonzero(r == 1) - 1; L = np.flatnonzero(r == -1) - st - 1
    out = []
    for s, l in zip(st, L):
        if l < 5:
            continue
        lo, hi = max(0, (s - 1) * N), min(len(x) - 9 * N, (s + 1) * N)
        if hi <= lo:
            continue
        seg = x[lo: hi + 8 * N]
        # non-coherent: per-chirp correlation magnitudes summed over the 8 chirps
        # (a large CFO rotates the phase between chirps; coherent correlation cancels)
        c1 = np.abs(np.correlate(seg, up, 'valid')) / (np.linalg.norm(up) + 1e-12)
        nlag = hi - lo + 1
        c = sum(c1[k * N: k * N + nlag] for k in range(8))
        a = lo + int(c.argmax())
        P = x[a: a + 8 * N]
        q = c.max() / (8 * np.sqrt(np.mean(np.abs(P) ** 2) * N) + 1e-12)
        if q > 0.3:
            out.append(P / (np.sqrt(np.mean(np.abs(P) ** 2)) + 1e-12))
    return np.array(out)


def feats(P):
    raw = P / np.linalg.norm(P, axis=1, keepdims=True)
    S = np.log(np.abs(np.fft.fftshift(np.fft.fft(P, axis=1), axes=1)) + 1e-9).reshape(len(P), 512, 16).mean(2)
    D = np.fft.fft(P * np.conj(ref)[None, :], axis=1); M = np.abs(D)
    pk = M.argmax(1)
    tone = np.stack([np.roll(M[i], -pk[i] + 64)[:129] for i in range(len(P))]); tone /= tone.max(1, keepdims=True)
    # fine frequency: parabolic interpolation of the peak, bins of fs/8192 Hz
    y0, y1, y2 = [M[np.arange(len(P)), (pk + d) % M.shape[1]] for d in (-1, 0, 1)]
    frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    f = ((pk + frac + M.shape[1] / 2) % M.shape[1] - M.shape[1] / 2) * fs / M.shape[1]
    return {'raw': raw, 'spectrum': S, 'tone': tone, 'cfo': f[:, None] / 1e3}


def run(tr, te, kind):
    X = np.concatenate([F[kind] for F, _ in tr]); y = np.concatenate([[l] * len(F[kind]) for F, l in tr])
    Xt = np.concatenate([F[kind] for F, _ in te]); yt = np.concatenate([[l] * len(F[kind]) for F, l in te])
    labs = np.unique(y)
    if kind == 'raw':
        A = X * np.exp(-1j * np.angle(X @ np.conj(X[0])))[:, None]
        C = np.stack([A[y == c].mean(0) for c in labs]); C /= np.linalg.norm(C, axis=1, keepdims=True)
        cen = labs[np.abs(Xt @ np.conj(C).T).argmax(1)]; S = np.abs(Xt @ np.conj(X).T)
    else:
        mu, sd = X.mean(0), X.std(0) + 1e-9; Z, Zt = (X - mu) / sd, (Xt - mu) / sd
        C = np.stack([Z[y == c].mean(0) for c in labs])
        cen = labs[((Zt[:, None, :] - C[None]) ** 2).sum(2).argmin(1)]; S = -((Zt[:, None, :] - Z[None]) ** 2).sum(2)
    nn = y[np.argsort(-S, 1)[:, :5]]
    knn = np.array([np.bincount(r).argmax() for r in nn])
    return float(np.mean(cen == yt)), float(np.mean(knn == yt)), len(yt)


if __name__ == '__main__':
    data = {}
    for p in sorted(glob.glob(f'{HERE}/head2s/*.dat')):
        m = re.search(r'd(\d)_dev(\d+)_IQ_(\d)', p); day, dev, f = map(int, m.groups())
        z = np.fromfile(p, dtype='<f4'); x = (z[0::2] + 1j * z[1::2]).astype(np.complex128)
        P = preambles(x)
        if len(P):
            data[(day, dev, f)] = feats(P)
    cnt = {k: len(v['raw']) for k, v in data.items()}
    print('files', len(data), '| preambles per 2-s piece: min %d median %d max %d' % (min(cnt.values()), np.median(list(cnt.values())), max(cnt.values())))
    cfo = {dev: [np.median(data[(d, dev, 1)]['cfo']) for d in range(1, 6) if (d, dev, 1) in data] for dev in range(1, 11)}
    print('dechirped tone frequency per device (kHz), days 1..5:')
    for dev, v in cfo.items():
        print(f'  device {dev:2d}: ' + ' '.join(f'{u:7.2f}' for u in v))
    sel = lambda days, files: [(data[(d, dev, f)], dev) for d in days for dev in range(1, 11) for f in files if (d, dev, f) in data]
    cases = {'A same day, IQ_1 -> IQ_2': (sel([1], [1]), sel([1], [2])),
             'B day1 -> day2': (sel([1], [1, 2]), sel([2], [1, 2])),
             'C day1 -> day5': (sel([1], [1, 2]), sel([5], [1, 2])),
             'D days1-2 -> day5': (sel([1, 2], [1, 2]), sel([5], [1, 2]))}
    print('accuracy, 10 devices (chance 0.10): centroid / 5-NN')
    for kind in ('raw', 'spectrum', 'tone', 'cfo'):
        print(f'  {kind:9s} ' + ' | '.join(f'{c}: ' + '%.2f / %.2f' % run(tr, te, kind)[:2] for c, (tr, te) in cases.items()), flush=True)
