"""Joint-training control: one Phase-1 model on the init session plus ft_n_train/device
of every fine-tuning session (sampled like Phase 2), no sequential fine-tuning.
Evaluated like the stage ablation in common_eval.py.
Run from t2r/ (main.py --joint does this):  python joint_eval.py <dataset> <out_dir> <shift> <seed> [neu_slicing]
"""
import os, sys, json, numpy as np
ds, out, shift, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
slicing = sys.argv[5] if len(sys.argv) > 5 else 'contiguous'
sys.path.insert(0, os.getcwd())
import datasets as D
os.environ['T2R_JOINT_DAYS'] = ','.join(map(str, D.DATASETS[ds]['finetune_days']))
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from temporal_trajectory import extract_embeddings_from_model
from phase1_initial_training import run_phase1
import load_slice_IQ, rf_models
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
cfg = D.make_cfg(ds, out, seed, shift, slicing)
import repro; repro.set_all(seed)
import manifest; manifest.write_manifest(cfg, out, 'train')
run_phase1(cfg)
md = cfg.model_dir; known, unk = cfg.all_known_ids, cfg.unknown_ids; devs = known + unk; K = len(known)
kmap = {d: i for i, d in enumerate(sorted(known))}
from data_utils import load_day_with_groups as _ldg
Xt, yt, _ = _ldg(cfg, day_id=cfg.test_day, device_ids=devs)      # same loader and order as common_eval
yt = np.array([sorted(devs)[int(v)] for v in yt]); is_k = np.isin(yt, known)
import hashlib
TEST_INPUT_SHA1 = hashlib.sha1(np.ascontiguousarray(Xt).tobytes() + yt.astype(np.int64).tobytes()).hexdigest()
yk = np.array([kmap[int(v)] for v in yt[is_k]])
# Part A of the reference session, with the same group split and seed as Phase 3
from data_utils import load_day_with_groups
from phase3_trajectory_building import _group_three_split
Xr, yr, gr = load_day_with_groups(cfg, day_id=cfg.traj_day, device_ids=known)
yr = np.asarray(yr).astype(int); gr = gr[:, 0] if np.ndim(gr) == 2 else gr
iA, _, _ = _group_three_split(yr, gr, cfg.seed + 7, fracs=(0.50, 0.25, 0.25))
Xr, yr = Xr[iA], yr[iA]
n = np.load(os.path.join(md, f'phase1_norm_d{cfg.init_day}.npz'))
nz = lambda X: load_slice_IQ.apply_normalization(X, n['mean'], n['std'])
m = tf.keras.models.load_model(os.path.join(md, f'phase1_df_d{cfg.init_day}.keras'), custom_objects={'L2Normalize': rf_models.L2Normalize})
sm = float(np.mean(np.argmax(m.predict(nz(Xt[is_k]), batch_size=256, verbose=0), 1) == yk))
a_k, a_u, a_r = (l2n(extract_embeddings_from_model(m, nz(X), 256)) for X in (Xt[is_k], Xt[~is_k], Xr))
C = l2n(np.stack([a_r[yr == k].mean(0) for k in range(K)]))
dk, du = 1 - (a_k @ C.T).max(1), 1 - (a_u @ C.T).max(1)
r = dict(dataset=ds, shift=shift, seed=seed, softmax=sm, centroid=float(np.mean((a_k @ C.T).argmax(1) == yk)),
         auroc=float(roc_auc_score(np.r_[np.zeros(len(dk)), np.ones(len(du))], np.r_[dk, du])),
         test_input_sha1=TEST_INPUT_SHA1)
json.dump(r, open(os.path.join(out, 'joint_eval.json'), 'w'), indent=1); print('JOINT', r, flush=True)
manifest.write_manifest(cfg, out, 'eval')
# Joint-control audit (the pipeline audit does not apply: no Phase 2/3). It is a
# real pass/fail record; aggregate.py also checks compatibility with the paired
# pipeline run (same code, versions, configuration, data, test input).
from provenance_checks import disjoint_by_class
checks = []
def check(name, ok, detail=''):
    checks.append({'check': name, 'pass': bool(ok), 'detail': detail})
p1 = os.path.join(md, f'phase1_train_groups_d{cfg.init_day}.npz')
check('phase1_groups', *(disjoint_by_class(p1, ['', 'val_', 'test_'], K) if os.path.exists(p1) else (False, 'file missing')))
check('joint_days', os.environ.get('T2R_JOINT_DAYS') == ','.join(map(str, cfg.finetune_days)),
      f"T2R_JOINT_DAYS={os.environ.get('T2R_JOINT_DAYS')}")
check('devices', not (set(known) & set(unk)) and len(known) == cfg.n_known and len(unk) == cfg.n_unknown,
      f'known {sorted(known)} / unknown {sorted(unk)}')
_cnt = {int(d_): int((yt == d_).sum()) for d_ in devs}
check('test_set', all(v == int(cfg.num_slice) for v in _cnt.values()), f'per device {sorted(set(_cnt.values()))}')
mt_, me_ = (json.load(open(os.path.join(out, f'manifest_{s_}.json'))) for s_ in ('train', 'eval'))
check('same_code', mt_['code_sha256'] == me_['code_sha256'], 'training and evaluation code identical')
check('same_settings', mt_['env'] == me_['env'] and mt_.get('versions') == me_.get('versions'),
      'identical T2R_* settings and versions')
check('determinism', mt_['env'].get('TF_DETERMINISTIC_OPS') == '1', '')
aud = {'kind': 'joint', 'all_pass': all(c['pass'] for c in checks), 'checks': checks,
       'artifacts': {k: manifest.file_sha256(os.path.join(out, k))
                     for k in ('joint_eval.json', 'manifest_train.json', 'manifest_eval.json')}}
json.dump(aud, open(os.path.join(out, 'audit.json'), 'w'), indent=1)
for c in checks:
    print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:14s} {c['detail']}")
print('AUDIT', 'PASS' if aud['all_pass'] else 'FAIL')
sys.exit(0 if aud['all_pass'] else 1)
