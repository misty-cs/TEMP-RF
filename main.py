"""Single entry point for every dataset.

    python main.py --dataset wisig                      # print the plan only
    python main.py --dataset pycom --shifts 0 --seeds 42 --go
    python main.py --dataset neu_A --neu-slicing chunk --go
    python main.py --dataset wisig --joint --go         # joint-training control

For each (device shift, seed): the full pipeline (Phase 1-4) once, then
common_eval.py and audit_run.py. A run is skipped only if its audit passed and it
was made with the current code and settings; a run with other code/settings is
reported STALE and left untouched. Nothing runs without --go.
Results: res/<dataset>[_<slicing>]/shift<S>/seed<s>/ (and res_joint/...).
"""
import argparse, os, subprocess, sys, time
import datasets as D
import manifest, repro

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=sorted(D.DATASETS))
p.add_argument('--shifts', type=int, nargs='+', default=None,
               help='default: every device unknown once (datasets.default_shifts)')
p.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 13, 21, 100])
p.add_argument('--neu-slicing', default='contiguous', choices=sorted(D.NEU_SLICING_ENV))
p.add_argument('--cfo', choices=['default', 'keep', 'remove'], default='default',
               help='override CFO handling (default: as set in datasets.py)')
p.add_argument('--joint', action='store_true', help='run the joint-training control instead')
p.add_argument('--allow-untuned', action='store_true', help='run a split without tuned settings (code defaults; pilot/debug only)')
p.add_argument('--pilot', action='store_true', help='pilot: results in res_pilot/ and res_joint_pilot/ (never mixed with res/), implies --allow-untuned')
p.add_argument('--root', default=None, help='results folder (default res; e.g. res_draft); joint runs go to res_joint<suffix>')
p.add_argument('--go', action='store_true', help='actually run (default: print the plan)')
a = p.parse_args()
if a.shifts is None:
    a.shifts = D.default_shifts(a.dataset)
if a.pilot:
    a.allow_untuned = True
RES = a.root or ('res_pilot' if a.pilot else 'res')
if not RES.startswith('res'):
    raise SystemExit('--root must start with "res" (res, res_draft, res_pilot_tuned, ...)')
RES_J = 'res_joint' + RES[3:]

PY = os.environ.get('PYTHON', sys.executable)
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
d = D.get(a.dataset, a.neu_slicing)
tag, d['env'] = D.variant(a.dataset, a.cfo, a.neu_slicing)
env = dict(os.environ, **d['env'])
data_probe = d['data_root']


def done(path, marker):
    return os.path.exists(os.path.join(path, marker))


import json as _json


DATA_ID = None


def data_id():
    global DATA_ID
    if DATA_ID is None:
        DATA_ID = manifest.data_identity(a.dataset, a.neu_slicing)
    return DATA_ID


def split_env(sh):
    """Settings tuned on this split's known devices (tune_phase1.py / tune_phase2.py).
    Returns (env, used) or (None, reason). Both selections are re-validated: their
    identity (code incl. imports, T2R_* settings, software versions, data
    inventory, seed, and for Phase 2 the Phase-1 checkpoint hash) must match
    what the current code and data give."""
    used = {}
    s1p, s2p = f'res_tune/{tag}/selected_phase1.json', f'res_tune/{tag}/selected_phase2.json'
    s1 = _json.load(open(s1p)).get(str(sh)) if os.path.exists(s1p) else None
    s2 = _json.load(open(s2p)).get(str(sh)) if os.path.exists(s2p) else None
    if s1 is None or s2 is None:
        return None, f'no tuned selection for split {sh} ({s1p}: {s1 is not None}, {s2p}: {s2 is not None})'
    try:
        did = data_id()
    except Exception as e:
        return None, f'data identity not available: {e}'
    if manifest.phase1_identity(a.dataset, env, s1['lambda'], s1['T'], s1['epochs'], s1['seeds'][0], sh, did) != s1['identity']:
        return None, f'split {sh}: Phase-1 selection was made with other code/settings/data; rerun tune_phase1.py'
    ck = os.path.join(s1['phase1_dir'], 'modelDir', f"phase1_df_d{d['init_day']}.keras")
    if not os.path.exists(ck) or manifest.file_sha256(ck) != s2.get('phase1_checkpoint_sha256'):
        return None, f'split {sh}: Phase-1 checkpoint {ck} missing or not the one Phase-2 tuning used'
    if s2.get('phase1_identity') != s1['identity'] or manifest.phase2_identity(
            a.dataset, env, s1, s2['replay'], s2['lr2'], s2['seed'], sh, s2['phase1_checkpoint_sha256'], did) != s2.get('identity'):
        return None, f'split {sh}: Phase-2 selection was made with other code/settings/data/Phase-1 model; rerun tune_phase2.py'
    e2 = dict(env, T2R_SUPCON_LAMBDA=str(s1['lambda']), T2R_SUPCON_T=str(s1['T']), T2R_P1_EPOCHS=str(s1['epochs']),
              T2R_FT_REPLAY=str(s2['replay']), T2R_FT_LR2=str(s2['lr2']))
    for var in ('T2R_SUPCON_LAMBDA', 'T2R_SUPCON_T', 'T2R_P1_EPOCHS', 'T2R_FT_REPLAY', 'T2R_FT_LR2'):
        used[var] = e2[var]
    return e2, used


def step(cmd, log, run_env=None):
    print('  $', ' '.join(cmd), '>', log, flush=True)
    if not a.go:
        return 0
    while not os.path.exists(data_probe):
        print(f'  {time.ctime()} data missing ({data_probe}), waiting', flush=True); time.sleep(120)
    with open(log, 'w') as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=run_env or env)


print(f'dataset={a.dataset} slicing={a.neu_slicing if d["layout"] == "neu" else "-"} '
      f'shifts={a.shifts} seeds={a.seeds} env={d["env"]} {"RUN" if a.go else "PLAN ONLY"}')
for sh in a.shifts if not a.joint else [a.shifts[0]]:
    renv, used = split_env(sh)
    if renv is None:
        if a.go and not a.allow_untuned:
            print(f'SKIP split {sh}: {used}', flush=True); continue
        print(f'  WARNING {used}; {"using code defaults (--allow-untuned)" if a.allow_untuned else "plan only"}', flush=True)
        renv, used = dict(env), {}
    else:
        print(f'split {sh}: tuned settings {used}', flush=True)
    for seed in a.seeds:
        senv = dict(renv, **repro.env(seed))          # same reproducibility env for every subprocess
        if a.joint:
            senv['T2R_JOINT_DAYS'] = ','.join(map(str, d['finetune_days']))
            out = f'{RES_J}/{tag}/shift{sh}/seed{seed}'
            st, why = manifest.run_status(out, senv, d['traj_day'], data_id() if a.go else None,
                                          manifest.expected_config(a.dataset, senv, seed, sh, a.neu_slicing))
            if st == 'complete':
                print(f'skip {out} (complete: audit passed, same code, settings, config, versions and data)'); continue
            if st == 'stale':
                print(f'STALE {out}: {why}. Not reused, not overwritten; move it away to rerun.', flush=True); continue
            if st == 'partial':
                k_ = 1
                while os.path.exists(f'{out}.failed{k_}'): k_ += 1
                if a.go: os.rename(out, f'{out}.failed{k_}')
                print(f'  earlier incomplete attempt kept as {out}.failed{k_}; running again', flush=True)
            os.makedirs(out, exist_ok=True) if a.go else None
            rc_j = step([PY, '-u', 'joint_eval.py', a.dataset, out, str(sh), str(seed), a.neu_slicing], f'{out}/joint.log', senv)
            if a.go and rc_j:
                open(f'{out}/AUDIT_FAILED', 'w').write('see joint.log'); print(f'  JOINT FAILED/AUDIT FAILED {out}', flush=True)
            continue
        out = f'{RES}/{tag}/shift{sh}/seed{seed}'
        st, why = manifest.run_status(out, senv, d['traj_day'], data_id() if a.go else None,
                                      manifest.expected_config(a.dataset, senv, seed, sh, a.neu_slicing))
        if st == 'complete':
            print(f'skip {out} (complete: audit passed, artifacts unchanged, same code, settings, config, versions and data)'); continue
        if st == 'stale':
            print(f'STALE {out}: {why}. Not reused, not overwritten; move it away to rerun.', flush=True)
            continue
        if st == 'partial' and not os.path.exists(f'{out}/PHASE3_DONE'):
            # training did not finish (e.g. drive disconnect): keep the attempt as evidence
            k_ = 1
            while os.path.exists(f'{out}.failed{k_}'): k_ += 1
            if a.go: os.rename(out, f'{out}.failed{k_}')
            print(f'  earlier incomplete attempt kept as {out}.failed{k_}; running again', flush=True)
            st = 'new'
        print(f'[{time.ctime()}] {out} ({st}{": " + why if why else ""})', flush=True)
        if a.go:
            os.makedirs(out, exist_ok=True)
        ok = a.go and os.path.exists(f'{out}/PHASE3_DONE')     # pipeline (Phases 1-3) finished, same code/settings
        if not ok:
            rc = step([PY, '-u', 'run_experiment.py', *D.cli_args(a.dataset), '--output_root', out,
                       '--seed', str(seed), '--device_shift', str(sh)], f'{out}/run.log', senv)
            if rc:
                print(f'  pipeline failed rc={rc}, see {out}/run.log'); continue
        rc_e = step([PY, '-u', 'common_eval.py', a.dataset, out, str(sh), str(seed), a.neu_slicing], f'{out}/common_eval.log', senv)
        if rc_e:
            print(f'  common_eval failed rc={rc_e}, see {out}/common_eval.log', flush=True)
            if a.go:
                open(f'{out}/AUDIT_FAILED', 'w').write('common_eval failed')
            continue
        rc_a = step([PY, '-u', 'audit_run.py', a.dataset, out, str(sh), str(seed), a.neu_slicing], f'{out}/audit.log', senv)
        if a.go:
            if rc_a:
                open(f'{out}/AUDIT_FAILED', 'w').write('see audit.log')
                print(f'  AUDIT FAILED for {out}; see {out}/audit.log', flush=True)
            elif os.path.exists(f'{out}/AUDIT_FAILED'):
                os.remove(f'{out}/AUDIT_FAILED')
print('done' if a.go else 'plan printed; add --go to run')
