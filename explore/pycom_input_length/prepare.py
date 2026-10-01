"""Frozen data for the Pycom input-length comparison (EXPLORATORY).

Days 1-3 only (Days 4-5 are never read); the 16 known devices of split 0
(pool positions 0-15). The packet set is the PIPELINE's packet set:
  1. detection and fine alignment exactly as t2r/load_slice_IQ.
     _read_packet_slices_nocache with slice_len = 256 (256-sample template);
  2. only packet starts with 4096 samples available after the aligned start
     are kept (the only difference from the pipeline; the count dropped is
     printed);
  3. the pipeline's quality filter on the first 256 samples;
  4. the pipeline's selection (build_device_dataset): 780 per device and day,
     156 per file, shortfall moved to other files one by one, then evenly
     spaced packets (np.linspace) within each file.
4096 samples are stored from each aligned start; train.py cuts the first L.
The per-file preconditioning of the pipeline (mean removal and RMS over the
selected slices) is applied in train.py for each L, as the pipeline would.
check: the first 256 samples equal the pipeline's own packets (verified per file).

    python prepare.py            -> data_d1.npz, data_d2.npz, data_d3.npz
"""
import os, sys, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
t2r = os.path.abspath(os.path.join(here, '..', '..', 't2r'))
sys.path.insert(0, t2r)
import load_slice_IQ as L, datasets as D

ROOT = os.path.join(t2r, D.DATASETS['pycom']['data_root'])
POOL = [int(v) for v in D.DATASETS['pycom']['env']['T2R_DEVICE_POOL'].split(',')]
N_KNOWN, PER_DEV, SEG, SL = 16, 780, 4096, 256


def aligned_packets(filename, gap_db=15.0, min_gap=125, win=25, search=25, chunk=4_000_000):
    """Same steps as load_slice_IQ._read_packet_slices_nocache(filename, None, 256),
    returning 4096 samples from each aligned start. Returns (P4096, n_dropped_end)."""
    total = os.path.getsize(filename) // 8

    def rd(start, n):
        with open(filename, 'rb') as fh:
            fh.seek(int(start) * 8)
            return np.frombuffer(fh.read(int(n) * 8), dtype='<f4').astype(np.float64)
    dc = 0j
    for a in range(0, total, chunk):
        x = rd(a, min(total, a + chunk) - a); dc += (x[0::2] + 1j * x[1::2]).sum()
    dc /= total
    starts = []
    for a in range(0, total, chunk):
        b = min(total, a + chunk); x = rd(a, b - a)
        c = (x[0::2] + 1j * x[1::2]) - dc
        p = 10 * np.log10(np.convolve(np.abs(c) ** 2, np.ones(win) / win, 'same') + 1e-30)
        med = np.median(p); sil = p < med - gap_db
        runs = np.diff(np.r_[0, sil.astype(np.int8), 0]); st = np.flatnonzero(runs == 1); en = np.flatnonzero(runs == -1)
        starts.extend((a + en[(en - st) >= min_gap]).tolist())
    starts = np.asarray([e for e in starts if search <= e < total - SL - search])   # pipeline's rule
    if len(starts) == 0:
        return np.empty((0, SEG), np.complex64), 0

    def seg(e, off=0, n=SL):
        x = rd(e + off, n); return (x[0::2] + 1j * x[1::2]) - dc
    tsel = starts[np.linspace(0, len(starts) - 1, min(64, len(starts))).astype(int)]
    T = np.stack([seg(e) for e in tsel]); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    T = T * np.exp(-1j * np.angle(T @ np.conj(T[0])))[:, None]
    tmpl = T.mean(0); tmpl /= np.linalg.norm(tmpl) + 1e-12            # the pipeline's 256-sample template
    out, drop = [], 0
    for e in starts:                                                    # n_requested=None: every start
        w = seg(e, -search, SL + 2 * search)
        o = int(np.argmax(np.abs(np.correlate(w, tmpl, 'valid'))))
        a0 = e - search + o
        if a0 + SEG > total:
            drop += 1; continue
        out.append(seg(a0, 0, SEG))
    return np.asarray(out, dtype=np.complex64), drop


for day in (1, 2, 3):
    root = f'{ROOT}/Day_{day}'
    folders = sorted((x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))), key=L.natural_key)
    X, y, g, names = [], [], [], []
    for lab in range(N_KNOWN):
        fo = folders[POOL[lab]]
        files = sorted(f for f in os.listdir(os.path.join(root, fo)) if f.endswith('.dat'))
        good, drops = [], []
        for f in files:
            fp = os.path.join(root, fo, f)
            P, dr = aligned_packets(fp); drops.append(dr)
            ref, _ = L._read_packet_slices_nocache(fp, None, SL)            # the pipeline's own packets
            if not np.array_equal(P[:, :SL], ref[:len(P)]):
                raise SystemExit(f'CHECK FAILED: {fp}: first 256 samples differ from the pipeline packets')
            good.append(P[L.packet_quality_mask(P[:, :SL])])
        num = len(files); base, rem = divmod(PER_DEV, num)
        counts = [base + (1 if i < rem else 0) for i in range(num)]         # build_device_dataset
        avail = [len(p) for p in good]
        alloc = [min(c, a) for c, a in zip(counts, avail)]
        short = PER_DEV - sum(alloc)
        while short > 0 and any(al < av for al, av in zip(alloc, avail)):
            for j in range(num):
                if short > 0 and alloc[j] < avail[j]:
                    alloc[j] += 1; short -= 1
        for fi, (P, n) in enumerate(zip(good, alloc)):
            if n == 0:
                continue
            idx = np.unique(np.linspace(0, len(P) - 1, n).astype(int))
            X.append(np.stack([P[idx].real, P[idx].imag], -1).astype(np.float32)); y += [lab] * len(idx)
            g.append(np.stack([np.full(len(idx), fi), np.arange(len(idx))], 1)); names.append(f'{fo}/{files[fi]}:{len(idx)}/{len(P)}')
        print(f'day {day} {fo}: good {avail} -> used {alloc}; starts dropped (no 4096 samples) {drops}', flush=True)
    X, y, g = np.concatenate(X), np.asarray(y), np.concatenate(g)
    np.savez(os.path.join(here, f'data_d{day}.npz'), X=X, y=y, g=g, files=np.array(names))
    cnt = np.bincount(y, minlength=N_KNOWN)
    print(f'DAY {day}: {len(y)} packets, per device min {cnt.min()} max {cnt.max()}, shape {X.shape}', flush=True)
