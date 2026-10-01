"""Automatic audit of one finished run (after common_eval.py).

    python audit_run.py <dataset> <run_dir> <device_shift> <seed> [neu_slicing]

Writes <run_dir>/audit.json with one entry per check and all_pass. main.py runs
it after every evaluation and marks a failing run with AUDIT_FAILED.
Checks
  sessions        init / fine-tune / reference / test sessions are different days
  devices         known and unknown device sets are disjoint and complete
  phase1_groups   Phase-1 train / val / test parts present, non-empty, cover every
                  known device, and share no group
  phase2_groups   Phase-2 train / val / held-out groups are disjoint (every session)
  phase3_parts    reference parts A / B / C: saved sample indices and groups reproduce
                  the split, are group-disjoint; BN adaptation used Part A only
  phase3_banks    saved bank (device, group) identities of every earlier session
                  share no group with that session's training groups; all devices
  test_set        recomputed test-input hash matches; every device present;
                  per-device counts add up
  protocol_invariance  every threshold protocol keeps the rule's closed/AUROC
  selection_sensitivity weight 0.5 + tie rule reproduces the Phase-3 choice
  eval_complete   every rule has zero-shot, all threshold protocols and selection data
  partA_check     common_eval's reload of Part A matched Phase 3 exactly
  same_code       training and evaluation used identical code (SHA-256)
  artifacts       result, manifests, Phase-3 model and calibration exist; their
                  SHA-256 is stored (main.py / aggregate.py require them unchanged)
  determinism     deterministic TensorFlow ops were enabled
  same_settings   training and evaluation used identical T2R_* settings
  same_config     training and evaluation configurations are identical
  same_versions   training and evaluation used the same software versions
Missing provenance files make the check FAIL (fail closed).
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D

ds, run_dir, shift, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
slicing = sys.argv[5] if len(sys.argv) > 5 else 'contiguous'
cfg = D.make_cfg(ds, run_dir, seed, shift, slicing)
md = cfg.model_dir
checks = []


def check(name, ok, detail=''):
    checks.append({'check': name, 'pass': bool(ok), 'detail': detail})


from provenance_checks import disjoint_by_class


# sessions and devices
sess = [cfg.init_day, *cfg.finetune_days, cfg.traj_day, cfg.test_day]
check('sessions', len(set(sess)) == len(sess), f'init {cfg.init_day}, fine-tune {cfg.finetune_days}, '
      f'reference {cfg.traj_day}, test {cfg.test_day}')
kn, un = set(cfg.all_known_ids), set(cfg.unknown_ids)
check('devices', not (kn & un) and len(kn) == cfg.n_known and len(un) == cfg.n_unknown,
      f'known {sorted(kn)} / unknown {sorted(un)}')
# Phase 1 / Phase 2 groups
p1 = os.path.join(md, f'phase1_train_groups_d{cfg.init_day}.npz')
check('phase1_groups', *(disjoint_by_class(p1, ['', 'val_', 'test_'], cfg.n_known) if os.path.exists(p1) else (False, 'file missing')))
for d in cfg.finetune_days:
    p2 = os.path.join(md, f'phase2_train_groups_d{d}.npz')
    ok, det = disjoint_by_class(p2, ['', 'val_', 'test_'], cfg.n_known) if os.path.exists(p2) else (False, 'file missing')
    check(f'phase2_groups_d{d}', ok, det)
# Phase 3 parts: reproduce and compare with the saved calibration file
from data_utils import load_day_with_groups
from phase3_trajectory_building import _group_three_split
cal = np.load(os.path.join(md, f'phase3_calibration_d{cfg.traj_day}.npz'))
_, yr, gr = load_day_with_groups(cfg, day_id=cfg.traj_day, device_ids=cfg.all_known_ids)
yr = np.asarray(yr).astype(int); gr = gr[:, 0] if np.ndim(gr) == 2 else gr
iA, iB, iC = _group_three_split(yr, gr, cfg.seed + 7)
ids_ok = all(k in cal for k in ('idx_update', 'idx_sweep', 'idx_known', 'g_update', 'g_sweep', 'g_known', 'idx_bn'))
same = ids_ok and all(np.array_equal(i, cal[k]) for i, k in ((iA, 'idx_update'), (iB, 'idx_sweep'), (iC, 'idx_known')))
same = same and (np.array_equal(yr[iA], cal['y_update']) and np.array_equal(yr[iB], cal['y_sweep'])
                 and np.array_equal(yr[iC], cal['y_known']))
if ids_ok:                                          # groups as SAVED by Phase 3
    gA, gB, gC = (set(zip(cal[y_].astype(int).tolist(), cal[g_].astype(int).tolist()))
                  for y_, g_ in (('y_update', 'g_update'), ('y_sweep', 'g_sweep'), ('y_known', 'g_known')))
else:
    gA = gB = gC = {None}
bn_ok = ids_ok and np.array_equal(cal['idx_bn'], cal['idx_update'])
fresh_ok = ids_ok and all(np.array_equal(gr[cal[i_]].astype(int), cal[g_].astype(int)) and
                          np.array_equal(yr[cal[i_]], cal[y_].astype(int))
                          for i_, g_, y_ in (('idx_update', 'g_update', 'y_update'), ('idx_sweep', 'g_sweep', 'y_sweep'),
                                             ('idx_known', 'g_known', 'y_known')))
cover_ok = ids_ok and all(sorted(set(cal[y_].astype(int).tolist())) == list(range(cfg.n_known))
                          for y_ in ('y_update', 'y_sweep', 'y_known'))
same = same and fresh_ok and cover_ok
check('phase3_parts', same and bn_ok and not (gA & gB) and not (gA & gC) and not (gB & gC),
      f'sizes {len(iA)}/{len(iB)}/{len(iC)}; saved identities present {ids_ok}; shared groups A&B {len(gA & gB)}, '
      f'A&C {len(gA & gC)}, B&C {len(gB & gC)}; reproduces Phase 3 (indices, fresh groups, labels, all devices): {same}; '
      f'BN adapted on Part A only: {bn_ok}')
earlier = sorted({cfg.init_day, *[d for d in cfg.finetune_days if d < cfg.traj_day]})
det, ok_b = [], True
for d in earlier:
    bf = os.path.join(md, f'phase3_bank_groups_d{d}.npz')
    tf_ = os.path.join(md, f'phase1_train_groups_d{d}.npz' if d == cfg.init_day else f'phase2_train_groups_d{d}.npz')
    if not (os.path.exists(bf) and os.path.exists(tf_)):
        ok_b = False; det.append(f'day {d}: provenance file missing'); continue
    b, t = np.load(bf), np.load(tf_)
    bank = set(zip(b['labels'].tolist(), b['groups'].tolist()))
    trained = set(zip(t['labels'].astype(int).tolist(), t['groups'].astype(int).tolist()))
    cover = sorted(set(b['labels'].tolist())) == list(range(cfg.n_known))
    ok_b &= not (bank & trained) and cover
    det.append(f'day {d}: {len(b["labels"])} bank slices, {len(bank & trained)} trained groups used, all devices {cover}')
check('phase3_banks', ok_b, '; '.join(det))
# evaluation output
ce = json.load(open(os.path.join(run_dir, 'common_eval.json')))
ts = ce['test_set']
from data_utils import load_day_with_groups as _ldg
import hashlib
devs = cfg.all_known_ids + cfg.unknown_ids
Xt, yt, _ = _ldg(cfg, day_id=cfg.test_day, device_ids=devs)
yt = np.array([sorted(devs)[int(v)] for v in yt])
sha = hashlib.sha1(np.ascontiguousarray(Xt).tobytes() + yt.astype(np.int64).tobytes()).hexdigest()
pdv = {int(d_): int((yt == d_).sum()) for d_ in devs}                     # recomputed, not read from the result
stored = {int(k): v for k, v in ts.get('per_device', {}).items()}
budget = int(cfg.num_slice)
check('test_set', ts.get('input_sha1') == sha and stored == pdv and all(v == budget for v in pdv.values())
      and ts['n_known'] == sum(pdv[d] for d in cfg.all_known_ids) and ts['n_unknown'] == sum(pdv[d] for d in cfg.unknown_ids)
      and sorted(ce['known']) == sorted(cfg.all_known_ids) and sorted(ce['unknown']) == sorted(cfg.unknown_ids),
      f"{ts['n_known']} known + {ts['n_unknown']} unknown slices; recomputed per device min {min(pdv.values())} "
      f"max {max(pdv.values())} (budget {budget}); stored counts match: {stored == pdv}; "
      f"input hash recomputed matches: {ts.get('input_sha1') == sha}")
del Xt
bad_inv = [n for n, v in ce['rules'].items() if 'tuned' in v and
           (abs(v['tuned']['closed'] - v['closed']) > 1e-9 or abs(v['tuned']['auroc'] - v['auroc']) > 1e-6)]
check('protocol_invariance', not bad_inv, f'tuned protocol changed the rule: {bad_inv}' if bad_inv
      else 'every threshold protocol uses the rule\'s own prediction and score')
check('selection_sensitivity', ce.get('selection_sensitivity_w05_matches_phase3') is True,
      'weight 0.5 with the tie rule reproduces the Phase-3 choice')
need = ('closed', 'auroc', 'fixed90', 'objective', 'tuned')
missing = [n for n, v in ce['rules'].items() if n != 'softmax' and not all(k in v for k in need)]
check('eval_complete', not missing and all(k in ce.get('selection', {}) for k in ('knn', 'openmax', 'probe', 'static_window', 'knn_window', 'temp_rff_p1')),
      f'rules missing parts: {missing}' if missing else f"{len(ce['rules'])} rules; selection tables "
      f"{sorted(ce.get('selection', {}))}")
check('partA_check', ce.get('check_partA_max_abs_diff', 1.0) == 0.0,
      f"max |diff| {ce.get('check_partA_max_abs_diff')}")
mt, me = (os.path.join(run_dir, f'manifest_{s}.json') for s in ('train', 'eval'))
if os.path.exists(mt) and os.path.exists(me):
    a_, b_ = json.load(open(mt)), json.load(open(me))
    diff = [f for f in a_['code_sha256'] if a_['code_sha256'][f] != b_['code_sha256'].get(f)]
    check('same_code', not diff, f'differing files: {diff}' if diff else 'identical code')
    check('determinism', a_['env'].get('TF_DETERMINISTIC_OPS') == '1', f"TF_DETERMINISTIC_OPS={a_['env'].get('TF_DETERMINISTIC_OPS')}")
    import manifest as _M
    dc = sorted(k for k in set(a_['config']) | set(b_['config']) if a_['config'].get(k) != b_['config'].get(k)
                and k != 'output_root')
    check('same_config', not dc, f'training (run_experiment CLI) and evaluation (datasets.make_cfg) configs differ: {dc}'
          if dc else 'training and evaluation used the same configuration')
    dv = a_.get('versions') != b_.get('versions')
    check('same_versions', not dv, f"train {a_.get('versions')} vs eval {b_.get('versions')}" if dv
          else 'training and evaluation used the same software versions')
    de = sorted(set(a_['env'].items()) ^ set(b_['env'].items()))
    check('same_settings', not de, f'differing settings: {de}' if de else 'training and evaluation used identical T2R_* settings')
else:
    for n_ in ('same_code', 'determinism', 'same_settings', 'same_config', 'same_versions'):
        check(n_, False, 'manifest missing')
import manifest
arts = manifest.run_artifacts(run_dir, cfg.traj_day)
miss_a = [k for k, p_ in arts.items() if not os.path.exists(p_)]
check('artifacts', not miss_a, f'missing: {miss_a}' if miss_a else 'result, manifests, Phase-3 model and calibration present; hashes recorded')
out = {'run': run_dir, 'all_pass': all(c['pass'] for c in checks), 'checks': checks,
       'artifacts': {k: manifest.file_sha256(p_) for k, p_ in arts.items() if os.path.exists(p_)},
       'study': {'code_sha256_all': json.load(open(mt))['code_sha256_all'] if os.path.exists(mt) else None}}
json.dump(out, open(os.path.join(run_dir, 'audit.json'), 'w'), indent=1)
for c in checks:
    print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:16s} {c['detail']}")
print('AUDIT', 'PASS' if out['all_pass'] else 'FAIL')
sys.exit(0 if out['all_pass'] else 1)
