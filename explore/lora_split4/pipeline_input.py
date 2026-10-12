"""Compare raw LoRa slices with the slices the pipeline feeds to the network (split 4, seed 42)."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import datasets as D, load_slice_IQ
from data_utils import load_day_with_groups, load_day_raw_with_groups
for k, v in D.DATASETS['lora']['env'].items(): os.environ[k] = v
cfg = D.make_cfg('lora', 'res/lora/shift4/seed42', 42, 4); md = cfg.model_dir
nrm = np.load(os.path.join(md, 'phase3_norm_stats.npz'))
out = open(os.path.join(os.path.dirname(__file__), 'pipeline_input.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
P('norm stats mean', np.round(nrm['mean'].ravel()[:4], 4), 'std', np.round(nrm['std'].ravel()[:4], 4), 'shapes', nrm['mean'].shape, nrm['std'].shape)
for d in (0, 9, 4, 5):
    Xr, yr, gr = load_day_raw_with_groups(cfg, day_id=5, device_ids=[d])
    Xp, yp, gp = load_day_with_groups(cfg, day_id=5, device_ids=[d])
    Xn = load_slice_IQ.apply_normalization(Xp, nrm['mean'], nrm['std'])
    for name, X in (('raw', Xr), ('preproc', Xp), ('normalized', Xn)):
        a = np.abs(X); P(f'dev {d} {name:10s} shape {X.shape} dtype {X.dtype}  mean|x| {a.mean():.4f}  max {a.max():.3f}  '
                          f'per-slice max (median) {np.median(a.reshape(len(X), -1).max(1)):.3f}  argmax pos (median) {int(np.median(a.reshape(len(X), -1).argmax(1)))}')
