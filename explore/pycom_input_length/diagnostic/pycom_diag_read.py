"""Exploratory Pycom input diagnostics, step 1: read packets (Days 1-3 only).

Read-only. Day 4 (reference) and Day 5 (test) are never opened. For every
device of the pool and every file: up to 200 packet-aligned 4096-sample
segments (same detector and alignment as the pipeline), stored for step 2.
"""
import os, sys, time, numpy as np
sys.path.insert(0, '/Users/misty/Downloads/icc-rerun/t2r')
import load_slice_IQ as L, datasets as D

ROOT = '/Users/misty/Downloads/icc-rerun/t2r/data/pycom_indoor'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pycom_diag')
os.makedirs(OUT, exist_ok=True)
pool = [int(v) for v in D.DATASETS['pycom']['env']['T2R_DEVICE_POOL'].split(',')]
DAYS = (1, 2, 3)
for day in DAYS:
    root = f'{ROOT}/Day_{day}'
    folders = sorted((x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))), key=L.natural_key)
    for lab, fi in enumerate(pool):
        fo = folders[fi]
        for f in sorted(os.listdir(os.path.join(root, fo))):
            if not f.endswith('.dat'):
                continue
            dst = os.path.join(OUT, f'd{day}_{fo}_{f[:-4]}.npz')
            if os.path.exists(dst):
                continue
            t = time.time()
            P, n = L._read_packet_slices_nocache(os.path.join(root, fo, f), 200, 4096)
            np.savez(dst, P=P, label=lab, folder=fo, day=day, fname=f)
            print(f'day {day} {fo:10s} {f} packets {n} ({time.time() - t:.0f}s)', flush=True)
print('READ DONE')
