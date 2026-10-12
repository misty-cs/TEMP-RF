"""LoRa split 4 (unknown devices 0-3): why is the AUROC of every learned rule ~0?
Read-only analysis of a finished final run (res/lora/shift4/seed42); writes report.txt.
Uses the run's saved Phase-3 model, normalization and Part-A embeddings."""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import tensorflow as tf, datasets as D, load_slice_IQ, rf_models
from data_utils import load_day_with_groups, load_day_raw
from temporal_trajectory import extract_embeddings_from_model
run = sys.argv[1] if len(sys.argv) > 1 else 'res/lora/shift4/seed42'
shift = int(run.split('shift')[1].split('/')[0]); seed = int(run.split('seed')[1])
cfg = D.make_cfg('lora', run, seed, shift)
for k, v in D.DATASETS['lora']['env'].items(): os.environ[k] = v
cfg = D.make_cfg('lora', run, seed, shift)
md = cfg.model_dir; known, unk = cfg.all_known_ids, cfg.unknown_ids
out = open(os.path.join(os.path.dirname(__file__), f'report_shift{shift}_seed{seed}.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
P('run', run, 'known', known, 'unknown', unk)
model = tf.keras.models.load_model(os.path.join(md, f'phase3_bn_adapted_d{cfg.traj_day}.keras'), custom_objects={'L2Normalize': rf_models.L2Normalize})
nrm = np.load(os.path.join(md, 'phase3_norm_stats.npz'))
cal = np.load(os.path.join(md, f'phase3_calibration_d{cfg.traj_day}.npz'))
eA, yA = cal['emb_update'], cal['y_update'].astype(int)
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
C = l2n(np.stack([eA[yA == c].mean(0) for c in range(len(known))]))
devs = known + unk
res = {}
for day in (cfg.traj_day, cfg.test_day):
    X, y, g = load_day_with_groups(cfg, day_id=day, device_ids=devs)
    y = np.array([sorted(devs)[int(v)] for v in y])
    e = extract_embeddings_from_model(model, load_slice_IQ.apply_normalization(X, nrm['mean'], nrm['std']), 256)
    sim = l2n(e) @ C.T
    raw_rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=(1, 2)))
    P(f'\n== day {day} ({"reference" if day == cfg.traj_day else "test"}): per device')
    P('dev  known  n    best-centroid-sim(mean)  nearest-known(top)  share-of-top  embnorm-before-L2? raw_rms')
    for d in sorted(devs):
        m = y == d; s = sim[m]; top = s.argmax(1); vals, cnt = np.unique(top, return_counts=True)
        tk = sorted(known)[vals[cnt.argmax()]]
        P(f'{d:3d}  {"K" if d in known else "U"}  {m.sum():4d}  {s.max(1).mean():.3f}  {tk:3d}  {cnt.max()/m.sum():.2f}  {raw_rms[m].mean():.4f}')
    res[day] = (e, y, sim)
e, y, sim = res[cfg.test_day]
from sklearn.metrics import roc_auc_score
isu = np.isin(y, unk)
P('\nstatic-centroid AUROC on test day (score = 1 - max sim):', round(roc_auc_score(isu, 1 - sim.max(1)), 3))
P('mean max-sim known', round(sim.max(1)[~isu].mean(), 3), 'unknown', round(sim.max(1)[isu].mean(), 3))
