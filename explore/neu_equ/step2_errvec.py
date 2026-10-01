"""NEU Setup 1 (equalized BPSK), step 2: error-vector features, no network (EXPLORATORY).

Per block (48 data subcarriers): robust gain, BPSK phase (squaring), hard decision
s = sign(Re), error e = z - s. Per window of W blocks: features
  - mean(e * s) per subcarrier (complex -> 96 reals): residual rotation / IQ-imbalance-like effects
  - mean |e|^2 per subcarrier (48)
  - mean |z| per subcarrier (48)                (baseline 'amplitude profile')
Nearest centroid (standardized, cosine) and 5-NN, 20 devices (chance 0.05):
  A  same day (day2): files 1-2 -> file 3
  B  day2 (files 1-3) -> day3 (files 1-3)
Reads NB blocks from the start of each file only.
"""
import glob, os, sys, numpy as np
ROOT = '/Volumes/TOSHIBA EXT/neu2'
NB, W, NF = int(os.environ.get('NB', 50000)), int(os.environ.get('W', 100)), 3


def feats(path):
    B = np.fromfile(path, dtype='<c8', count=48 * NB).reshape(-1, 48).astype(np.complex128)
    B = B / (np.median(np.abs(B), 1, keepdims=True) + 1e-12)
    B = B * np.exp(-1j * np.angle(np.sum(B ** 2, 1, keepdims=True)) / 2)
    s = np.sign(B.real); s[s == 0] = 1
    e = B - s
    n = len(B) // W
    es = (e * s)[:n * W].reshape(n, W, 48).mean(1)
    ev = (np.abs(e) ** 2)[:n * W].reshape(n, W, 48).mean(1)
    am = np.abs(B)[:n * W].reshape(n, W, 48).mean(1)
    return {'errvec': np.hstack([es.real, es.imag, np.log(ev + 1e-9)]), 'amplitude': np.log(am + 1e-9)}


def load(day):
    out = {}
    for dev in sorted(glob.glob(f'{ROOT}/day{day}/equ/Device_*'), key=lambda p: int(p.split('_')[-1])):
        files = sorted(glob.glob(f'{dev}/*.bin'))[:NF]
        out[int(dev.split('_')[-1])] = [feats(f) for f in files]
        print(f'  day{day} {os.path.basename(dev)}: {len(files)} files', flush=True)
    return out


def evaluate(tr, te, kind):
    labs = sorted(tr)
    Xtr = np.vstack([np.vstack([f[kind] for f in tr[d]]) for d in labs])
    ytr = np.concatenate([[i] * sum(len(f[kind]) for f in tr[d]) for i, d in enumerate(labs)])
    Xte = np.vstack([np.vstack([f[kind] for f in te[d]]) for d in labs])
    yte = np.concatenate([[i] * sum(len(f[kind]) for f in te[d]) for i, d in enumerate(labs)])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    l2 = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    C = l2(np.stack([Xtr[ytr == c].mean(0) for c in range(len(labs))]))
    cen = float(np.mean((l2(Xte) @ C.T).argmax(1) == yte))
    S = l2(Xte) @ l2(Xtr).T
    nn = ytr[np.argsort(-S, 1)[:, :5]]
    knn = float(np.mean([np.bincount(r, minlength=len(labs)).argmax() == t for r, t in zip(nn, yte)]))
    return cen, knn, len(yte)


if __name__ == '__main__':
    print(f'NB={NB} blocks per file, window W={W}')
    d2, d3 = load(2), load(3)
    A_tr = {d: v[:2] for d, v in d2.items()}; A_te = {d: v[2:3] for d, v in d2.items()}
    for kind in ('amplitude', 'errvec'):
        a = evaluate(A_tr, A_te, kind); b = evaluate(d2, d3, kind)
        print(f'{kind:10s} | A same day new file: centroid {a[0]:.3f} 5-NN {a[1]:.3f} (n={a[2]}) '
              f'| B day2 -> day3: centroid {b[0]:.3f} 5-NN {b[1]:.3f} (n={b[2]})', flush=True)
