"""Norm of the embedding before the final unit-length scaling, per device (split 4, seed 42, test day)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import tensorflow as tf, datasets as D, load_slice_IQ, rf_models
from data_utils import load_day_with_groups
for k, v in D.DATASETS['lora']['env'].items(): os.environ[k] = v
run = 'res/lora/shift4/seed42'; cfg = D.make_cfg('lora', run, 42, 4); md = cfg.model_dir
m = tf.keras.models.load_model(os.path.join(md, f'phase3_bn_adapted_d{cfg.traj_day}.keras'), custom_objects={'L2Normalize': rf_models.L2Normalize})
names = [l.name for l in m.layers]; print(names[-6:])
sub = tf.keras.Model(m.input, [m.get_layer('embedding').output, m.get_layer('fc1').output])
nrm = np.load(os.path.join(md, 'phase3_norm_stats.npz'))
devs = cfg.all_known_ids + cfg.unknown_ids
X, y, g = load_day_with_groups(cfg, day_id=cfg.test_day, device_ids=devs)
y = np.array([sorted(devs)[int(v)] for v in y]); Xn = load_slice_IQ.apply_normalization(X, nrm['mean'], nrm['std'])
out = open(os.path.join(os.path.dirname(__file__), 'prenorm.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
P('normalized input: |x| mean / max abs per device; pre-L2 embedding norm (median); fc1 active units (median)')
for d in sorted(devs):
    xs = Xn[y == d][:200]; z, h = sub.predict(xs, verbose=0)
    P(f'{d:3d} {"U" if d in cfg.unknown_ids else "K"}  |x| {np.abs(xs).mean():.3f}/{np.abs(xs).max():.1f}  ||z|| {np.median(np.linalg.norm(z, axis=1)):.4f}  fc1>0 {np.median((h > 0).sum(1)):.0f}/128')
