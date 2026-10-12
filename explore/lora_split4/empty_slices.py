"""Find LoRa slices that are mostly zero (cause of the spikes after per-slice RMS normalization)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import datasets as D
from data_utils import load_day_raw_with_groups
for k, v in D.DATASETS['lora']['env'].items(): os.environ[k] = v
cfg = D.make_cfg('lora', 'res/lora/shift4/seed42', 42, 4)
pool = [int(v) for v in D.DATASETS['lora']['env']['T2R_DEVICE_POOL'].split(',')]
out = open(os.path.join(os.path.dirname(__file__), 'empty_slices.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
P('day id  n_slices  n_bad(rms<20% of median)  zero-samples share in bad slices  files of bad slices')
tot = 0
for day in (1, 2, 3, 4, 5):
    for d in pool:
        X, y, g = load_day_raw_with_groups(cfg, day_id=day, device_ids=[d])
        rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=(1, 2))); bad = rms < 0.2 * np.median(rms)
        if bad.any():
            z = np.mean(np.all(X[bad] == 0, axis=2)); tot += bad.sum()
            P(f'{day} {d:3d}  {len(X)}  {bad.sum():4d}  {z:.2f}  files {sorted(set(g[bad, 0].tolist()))}')
P('total bad slices', tot)
