"""Pycom input-length comparison, one length per call (EXPLORATORY).

    python train.py <L> [epochs]        L in {256, 1024, 4096}

Phase 1 exactly as in the pipeline (t2r/phase1_initial_training.run_phase1:
same model, CE + SupCon with the code defaults lambda 0.5, T 0.07, same
preprocessing, augmentation, init probes and capture split: per device one
validation file and one test file, the rest train, chosen by seed 42), with one
difference: the input is the first L samples of the SAME packets
(prepare.py: all packets, quality filter, 780 per device and day). Seed 42, the same maximum epochs (100 = the pipeline) and the pipeline's
validation-based checkpoint and early stopping for every L. The pipeline's
per-file preconditioning is applied to the first L samples.

Selection among lengths: Day-1 validation only (val_sel, val_acc).
Reported, not used for selection: Day 2 and Day 3
  softmax      the Phase-1 classifier on every slice of the day
  centroid     centroids of the Day-1 TRAINING slices' embeddings (no new enrolment)
  re-enrolled  centroids from the day's files 1-3, scored on files 4-5
Days 4 and 5 are never read.
"""
import json, os, sys, time, numpy as np
L_ = int(sys.argv[1]); EP = sys.argv[2] if len(sys.argv) > 2 else '100'
here = os.path.dirname(os.path.abspath(__file__))
t2r = os.path.abspath(os.path.join(here, '..', '..', 't2r'))
sys.path.insert(0, t2r)
os.environ['T2R_P1_EPOCHS'] = EP
out = os.path.join(here, 'runs', f'L{L_}')
os.makedirs(out, exist_ok=True)
import datasets as D
cfg = D.make_cfg('pycom', out, 42, 0)
cfg.slice_len = L_
import repro; repro.set_all(42)
import data_utils


def _cut(z):
    """First L samples + the pipeline's per-file preconditioning
    (load_slice_IQ.build_device_dataset, precondition=True): per device and
    file, remove the mean and divide by the RMS of that file's selected slices."""
    X = z['X'][:, :L_, :].astype(np.float64); y, g = z['y'], z['g']
    Z = X[..., 0] + 1j * X[..., 1]
    for dev in np.unique(y):
        for f in np.unique(g[y == dev, 0]):
            m = (y == dev) & (g[:, 0] == f)
            s = Z[m] - Z[m].mean(); Z[m] = s / (np.sqrt((np.abs(s) ** 2).mean()) + 1e-8)
    return np.stack([Z.real, Z.imag], -1).astype(np.float32), y, g


DATA = {d: _cut(np.load(os.path.join(here, f'data_d{d}.npz'))) for d in (1, 2, 3)}


def _day1(cfg_, day_id, device_ids, start_idx=0):          # frozen packets, first L samples
    assert day_id == 1, f'only Day 1 is used for training (asked for day {day_id})'
    return DATA[1]


data_utils.load_day_raw_with_groups = _day1
from phase1_initial_training import run_phase1
t0 = time.time()
r = run_phase1(cfg)
t_train = time.time() - t0

import tensorflow as tf, load_slice_IQ, rf_models
from temporal_trajectory import extract_embeddings_from_model
md = cfg.model_dir
m = tf.keras.models.load_model(os.path.join(md, 'phase1_df_d1.keras'), custom_objects={'L2Normalize': rf_models.L2Normalize})
n = np.load(os.path.join(md, 'phase1_norm_d1.npz'))
prep = lambda X: load_slice_IQ.apply_normalization(data_utils._preprocess(X, False, cfg), n['mean'], n['std'])
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
emb = lambda X: l2n(extract_embeddings_from_model(m, prep(X), 256))
K = 16
tg = np.load(os.path.join(md, 'phase1_train_groups_d1.npz'))
train_pairs = set(zip(tg['labels'].astype(int).tolist(), tg['groups'].astype(int).tolist()))
X1, y1, g1 = DATA[1]
is_tr = np.array([(int(a), int(b)) in train_pairs for a, b in zip(y1, g1[:, 0])])
E1 = emb(X1[is_tr]); C1 = l2n(np.stack([E1[y1[is_tr] == c].mean(0) for c in range(K)]))
res = dict(L=L_, epochs=int(EP), seed=42, train_seconds=round(t_train), n_train_day1=int(is_tr.sum()),
           day1_val_acc=r['val_acc'], day1_val_sel=r['val_sel'], day1_val_loco=r['val_loco_auroc'],
           day1_test_file_acc=r['test_acc'])
for d in (2, 3):
    X, y, g = DATA[d]; f = g[:, 0]
    E = emb(X)
    sm = float(np.mean(np.argmax(m.predict(prep(X), batch_size=256, verbose=0), 1) == y))
    cen = float(np.mean((E @ C1.T).argmax(1) == y))
    enr, qry = f <= 2, f >= 3
    Cd = l2n(np.stack([E[enr & (y == c)].mean(0) for c in range(K)]))
    ren = float(np.mean((E[qry] @ Cd.T).argmax(1) == y[qry]))
    res[f'day{d}'] = dict(softmax=sm, centroid_day1=cen, reenrolled_files123_to_45=ren, n=int(len(y)))
json.dump(res, open(os.path.join(out, 'result.json'), 'w'), indent=1)
print('RESULT', json.dumps(res), flush=True)
