"""Exploratory Pycom input diagnostics, step 2 (Days 1-3 only, no network).

1. labels: the same device folder has the same label on every day
2. packet structure: power envelope over 4096 samples (where does the packet end?)
3. alignment / content: per-file 256-sample templates; |corr| same device
   (other file, same day / other day) vs other devices
4. CFO per packet (squaring removes the DBPSK sign): stability within file,
   across files, across days, between devices
5. representations, raw nearest-centroid and 5-NN, equal budget (150 packets
   per file), capture-disjoint evaluation:
     V1 Day 1 files 1-3 -> files 4-5 (new files, same day)
     V2 Day 1 -> Day 2          V3 Day 2 -> Day 3 (new day)
Chance = 1/20.
"""
import glob, os, sys, numpy as np
sys.path.insert(0, '/Users/misty/Downloads/icc-rerun/t2r')
import load_slice_IQ as L
QMASK = '--qmask' in sys.argv
from collections import defaultdict
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pycom_diag')
FS = 25e6
rng = np.random.default_rng(0)
F = []
for p in sorted(glob.glob(f'{DIR}/*.npz')):
    z = np.load(p)
    P = z['P']
    if QMASK:
        P = P[L.packet_quality_mask(P[:, :256])]       # the pipeline's filter (first 256 samples)
    if len(P) < 10:
        print(f"   skipped (<10 good packets): {p.split('/')[-1]} ({len(P)})"); continue
    if len(P) > 150:
        P = P[np.sort(rng.choice(len(P), 150, replace=False))]
    F.append(dict(P=P, lab=int(z['label']), fo=str(z['folder']), day=int(z['day']), file=str(z['fname'])))
print(f'{len(F)} files, {sum(len(f["P"]) for f in F)} packets')

# 1. labels
m = defaultdict(set)
for f in F:
    m[f['lab']].add(f['fo'])
print('1. label -> folders:', {k: sorted(v) for k, v in sorted(m.items())}, 'consistent:', all(len(v) == 1 for v in m.values()))

# 2. structure
env = np.mean([np.mean(np.abs(f['P']) ** 2, 0) for f in F], 0)
blk = env.reshape(16, 256).mean(1); blk /= blk[:4].mean()
print('2. mean power per 256-sample block (0..4095), rel. to first 1024:', np.round(blk, 2))
drop = np.mean([np.mean(np.abs(f['P'][:, -256:]) ** 2) / np.mean(np.abs(f['P'][:, :256]) ** 2) for f in F])
print(f'   last/first block power ratio (mean over files): {drop:.2f}')

# 3. templates
def tmpl(P, off=0, L=256):
    S = P[:, off:off + L]; S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    A = S * np.exp(-1j * np.angle(S @ np.conj(S[0])))[:, None]
    t = A.mean(0); q = np.abs(S @ np.conj(t / (np.linalg.norm(t) + 1e-12)))
    return t / (np.linalg.norm(t) + 1e-12), float(np.mean(q))
def xc(a, b, lag=25):
    return max(abs(np.vdot(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)])) for k in range(-lag, lag + 1))
T = [tmpl(f['P']) for f in F]
print(f'3. within-file packet-template |corr|: median {np.median([q for _, q in T]):.3f}, min {min(q for _, q in T):.3f}')
same_d, same_x, other = [], [], []
idx = rng.choice(len(F), min(len(F), 120), replace=False)
for i in idx:
    for j in idx:
        if i >= j: continue
        c = xc(T[i][0], T[j][0])
        if F[i]['lab'] == F[j]['lab']:
            (same_d if F[i]['day'] == F[j]['day'] else same_x).append(c)
        else:
            other.append(c)
q = lambda v: f'median {np.median(v):.3f} (p10 {np.percentile(v, 10):.3f}, p90 {np.percentile(v, 90):.3f}, n {len(v)})'
print('   template |corr| same device, same day :', q(same_d))
print('   template |corr| same device, other day:', q(same_x))
print('   template |corr| other devices         :', q(other))

# 4. CFO
def cfo(P, L=2048):
    z = P[:, :L] ** 2
    return np.angle(np.sum(z[:, 1:] * np.conj(z[:, :-1]), 1)) / (2 * 2 * np.pi) * FS
for f in F:
    f['cfo'] = cfo(f['P'])
wf = np.median([np.std(f['cfo']) for f in F])
byfile = defaultdict(list)
for f in F:
    byfile[(f['lab'], f['day'])].append(np.median(f['cfo']))
xf = np.median([np.std(v) for v in byfile.values()])
bydev = defaultdict(list)
for f in F:
    bydev[f['lab']].append(np.median(f['cfo']))
xd = np.median([np.std(v) for v in bydev.values()])
dev_med = np.array([np.median(v) for v in bydev.values()])
print(f'4. CFO (Hz): std within file {wf:.0f}; across files same day {xf:.0f}; across files+days {xd:.0f}; '
      f'spread of device medians {np.std(dev_med):.0f} (range {dev_med.min():.0f}..{dev_med.max():.0f})')

# 5. representations
def rep_raw(off, L):
    def g(P):
        Z = P[:, off:off + L]; return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return g, 'phase'
def rep_raw_nocfo(off, L):
    def g(P):
        Z = P[:, off:off + L]; c = cfo(P)
        Z = Z * np.exp(-2j * np.pi * c[:, None] * np.arange(off, off + L)[None, :] / FS)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return g, 'phase'
def rep_spec(L):
    def g(P):
        S = np.log(np.abs(np.fft.fftshift(np.fft.fft(P[:, :L], axis=1), axes=1)) + 1e-9)
        S = S - S.mean(1, keepdims=True); return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    return g, 'real'
def rep_cfo():
    return (lambda P: cfo(P)[:, None] / 1e3), 'scalar'
REPS = {'start256 (current)': rep_raw(0, 256), 'start256 CFO removed': rep_raw_nocfo(0, 256),
        'off512 len256': rep_raw(512, 256), 'off1024 len256': rep_raw(1024, 256), 'off2048 len256': rep_raw(2048, 256),
        'start1024': rep_raw(0, 1024), 'start4096': rep_raw(0, 4096),
        'log|FFT| 1024': rep_spec(1024), 'log|FFT| 4096': rep_spec(4096), 'CFO only': rep_cfo()}
def split(name):
    if name == 'V1 day1 files1-3 -> 4-5':
        tr = [f for f in F if f['day'] == 1 and f['file'] in ('tx_1_iq.dat', 'tx_2_iq.dat', 'tx_3_iq.dat')]
        te = [f for f in F if f['day'] == 1 and f['file'] in ('tx_4_iq.dat', 'tx_5_iq.dat')]
    elif name == 'V2 day1 -> day2':
        tr = [f for f in F if f['day'] == 1]; te = [f for f in F if f['day'] == 2]
    else:
        tr = [f for f in F if f['day'] == 2]; te = [f for f in F if f['day'] == 3]
    return tr, te
def run(g, kind, tr, te):
    Xtr = np.concatenate([g(f['P']) for f in tr]); ytr = np.concatenate([[f['lab']] * len(f['P']) for f in tr])
    Xte = np.concatenate([g(f['P']) for f in te]); yte = np.concatenate([[f['lab']] * len(f['P']) for f in te])
    K = ytr.max() + 1
    if kind == 'phase':
        A = Xtr * np.exp(-1j * np.angle(Xtr @ np.conj(Xtr[0])))[:, None]
        M = np.stack([A[ytr == c].mean(0) for c in range(K)]); M /= np.linalg.norm(M, axis=1, keepdims=True)
        cen = (np.abs(Xte @ np.conj(M).T)).argmax(1)
        S = np.abs(Xte @ np.conj(Xtr).T)
    elif kind == 'real':
        M = np.stack([Xtr[ytr == c].mean(0) for c in range(K)]); M /= np.linalg.norm(M, axis=1, keepdims=True)
        cen = (Xte @ M.T).argmax(1); S = Xte @ Xtr.T
    else:
        M = np.stack([np.median(Xtr[ytr == c], 0) for c in range(K)])
        cen = np.abs(Xte - M.T).argmin(1); S = -np.abs(Xte - Xtr.T)
    nn = ytr[np.argsort(-S, axis=1)[:, :5]]
    knn = np.array([np.bincount(r, minlength=K).argmax() for r in nn])
    return float(np.mean(cen == yte)), float(np.mean(knn == yte))
print('5. accuracy, 20 devices (chance 0.050): centroid / 5-NN')
SPL = ('V1 day1 files1-3 -> 4-5', 'V2 day1 -> day2', 'V3 day2 -> day3')
print(f'   {"representation":24s} ' + ' | '.join(f'{s:>24s}' for s in SPL))
for n, (g, kind) in REPS.items():
    cells = []
    for s in SPL:
        tr, te = split(s); c, k = run(g, kind, tr, te); cells.append(f'{c:.3f} / {k:.3f}'.rjust(24))
    print(f'   {n:24s} ' + ' | '.join(cells), flush=True)
