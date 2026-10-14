"""Read-only analysis for the paper figures (no run, setting or result is changed).

For every device split of a dataset (seed 42 run) this loads the saved Phase-3 model,
normalization and calibration embeddings, embeds the test day, and stores what the
figures need in analysis/cache/<ds>_shift<S>.npz:
  test-day embeddings and device ids, classifier logits, Part-A / Part-C embeddings.
For split 0 it also stores per-day embeddings of the known devices (trajectories) and
test-day embeddings under the Phase-1 model (t-SNE).

    python analysis/compute_cache.py wisig|pycom|lora
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
ds = sys.argv[1]
import datasets as D
for k, v in D.DATASETS[ds].get('env', {}).items():
    os.environ[k] = v                                   # dataset settings (device pool, CFO)
import tensorflow as tf, load_slice_IQ, rf_models
from data_utils import load_day_with_groups
from temporal_trajectory import extract_embeddings_from_model

OUT = os.path.join(ROOT, 'analysis', 'cache'); os.makedirs(OUT, exist_ok=True)
CO = {'L2Normalize': rf_models.L2Normalize}
SUB = 150                                               # slices per device and day for trajectories / t-SNE


def load(path):
    return tf.keras.models.load_model(path, custom_objects=CO)


def embed_day(cfg, model, nrm, day, devs, sub=None, rng=None):
    X, y, _ = load_day_with_groups(cfg, day_id=day, device_ids=devs)
    y = np.array([sorted(devs)[int(v)] for v in y])
    if sub:
        keep = np.concatenate([rng.choice(np.flatnonzero(y == d), min(sub, (y == d).sum()), replace=False) for d in sorted(set(y))])
        X, y = X[keep], y[keep]
    Xn = load_slice_IQ.apply_normalization(X, nrm['mean'], nrm['std'])
    return extract_embeddings_from_model(model, Xn, 256), y, Xn


for sh in D.default_shifts(ds):
    run = f'res/{ds}/shift{sh}/seed42'
    out = os.path.join(OUT, f'{ds}_shift{sh}.npz')
    if os.path.exists(out):
        print('exists', out); continue
    cfg = D.make_cfg(ds, run, 42, sh); md = cfg.model_dir
    known, unk = list(cfg.all_known_ids), list(cfg.unknown_ids); devs = known + unk
    model = load(os.path.join(md, f'phase3_bn_adapted_d{cfg.traj_day}.keras'))
    nrm = np.load(os.path.join(md, 'phase3_norm_stats.npz'))
    W, b = model.get_layer('classifier').get_weights()
    e, y, _ = embed_day(cfg, model, nrm, cfg.test_day, devs)
    cal = np.load(os.path.join(md, f'phase3_calibration_d{cfg.traj_day}.npz'))
    extra = {}
    if sh == D.default_shifts(ds)[0]:
        rng = np.random.default_rng(0)
        days = list(range(1, cfg.test_day + 1))
        T_e, T_y, T_d = [], [], []
        for d in days:
            ed, yd, _ = embed_day(cfg, model, nrm, d, known, SUB, rng)
            T_e.append(ed); T_y.append(yd); T_d.append(np.full(len(yd), d))
        extra.update(traj_e=np.concatenate(T_e), traj_y=np.concatenate(T_y), traj_day=np.concatenate(T_d))
        m1 = load(os.path.join(md, 'phase1_df_d1.keras')); n1 = np.load(os.path.join(md, 'phase1_norm_d1.npz'))
        e1, y1, _ = embed_day(cfg, m1, n1, cfg.test_day, devs, SUB, np.random.default_rng(1))
        e3, y3, _ = embed_day(cfg, model, nrm, cfg.test_day, devs, SUB, np.random.default_rng(1))
        extra.update(tsne_p1_e=e1, tsne_p1_y=y1, tsne_full_e=e3, tsne_full_y=y3)
    np.savez_compressed(out, e=e, y=y, logits=e @ W + b, known=np.array(known), unknown=np.array(unk),
                        eA=cal['emb_update'], yA=cal['y_update'], eC=cal['emb_known'], yC=cal['y_known'], **extra)
    print('saved', out, e.shape)
