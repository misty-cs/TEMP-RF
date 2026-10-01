"""Select the Phase-2 replay weight and stage-2 learning rate on VALIDATION data
only, separately for every device split.

    python tune_phase2.py --dataset wisig            # print the plan only
    python tune_phase2.py --dataset pycom --go
    python tune_phase2.py --dataset wisig --cfo remove --go   # own folder res_tune/wisig_nocfo

Run after tune_phase1.py: for each split, the Phase-1 model of that split's
selected (lambda, T) is reused (res_tune/<tag>/selected_phase1.json), and
Phase 2 is repeated for every setting. Only the split's known devices and the
fine-tuning sessions are used. Selection metric (shared, selection.py): mean over
fine-tuning sessions of SEL = 0.5 x nearest-centroid accuracy + 0.5 x
leave-one-device-out AUROC on the Phase-2 validation set (group-disjoint: other
bursts / another capture file). The reference and test sessions and unknown devices are never used.
Grid: replay weight in {0, 0.1, 0.2, 0.5} x stage-2 lr in {1e-5, 5e-5, 1e-4}
(12 settings). Shared tie rule: within 0.001 of the best, keep the prior default
(0.2, 5e-5), else the simplest (smallest replay weight, then lr).
Identity: every result stores a hash of the Phase-2 code (with its imports), all
T2R_* settings, software versions, the data inventory, the seed and the Phase-1
selection identity + checkpoint SHA-256 (manifest.phase2_identity). The Phase-1
choice is re-validated first; main.py re-validates both before a run. A cached result
with another identity is STALE (not used, not overwritten). A split is selected
only when all 12 settings have a current result.
Results: res_tune/<tag>/shift<S>/phase2/tune.json; choice per split in
res_tune/<tag>/selected_phase2.json (read by main.py).
"""
import argparse, glob, json, os, re, shutil, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=['wisig', 'pycom', 'lora'])
p.add_argument('--shifts', type=int, nargs='+', default=None, help='default: every split of the dataset')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--cfo', choices=['default', 'keep', 'remove'], default='default')
p.add_argument('--go', action='store_true')
a = p.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here); sys.path.insert(0, here)
import datasets as D, manifest
PY = os.environ.get('PYTHON', sys.executable)
shifts = a.shifts if a.shifts is not None else D.default_shifts(a.dataset)
tag, venv = D.variant(a.dataset, a.cfo)
grid = [(r, lr) for r in (0.0, 0.1, 0.2, 0.5) for lr in (1e-5, 5e-5, 1e-4)]
sel1_path = f'res_tune/{tag}/selected_phase1.json'
sel1 = json.load(open(sel1_path)) if os.path.exists(sel1_path) else {}
sel_path = f'res_tune/{tag}/selected_phase2.json'
selected = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
driver = r'''
import os, sys
sys.path.insert(0, os.getcwd())
import datasets as D
cfg = D.make_cfg(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
import repro; repro.set_all(cfg.seed)
import manifest; manifest.write_manifest(cfg, sys.argv[2], 'tune_phase2')
import run_experiment as R
from phase2_sequential_finetuning import run_phase2
run_phase2(cfg, R._resume_phase1(cfg))
'''
init_day = D.DATASETS[a.dataset]['init_day']
import repro
BASE = dict(os.environ, **venv)
DATA_ID = manifest.data_identity(a.dataset)
print(f'dataset={a.dataset} tag={tag} shifts={shifts} settings={len(grid)} {"RUN" if a.go else "PLAN ONLY"}')
for sh in shifts:
    s1 = sel1.get(str(sh))
    if s1 is None or 'identity' not in s1:
        print(f'split {sh}: no Phase-1 choice (run tune_phase1.py first)'); continue
    if manifest.phase1_identity(a.dataset, BASE, s1['lambda'], s1['T'], s1['epochs'], s1['seeds'][0], sh, DATA_ID) != s1['identity']:
        print(f'split {sh}: Phase-1 choice was made with other code/settings/data (rerun tune_phase1.py)'); continue
    src = os.path.join(s1['phase1_dir'], 'modelDir')
    ck = os.path.join(src, f'phase1_df_d{init_day}.keras')
    if not os.path.exists(ck):
        print(f'split {sh}: Phase-1 checkpoint {ck} missing'); continue
    ck_sha = manifest.file_sha256(ck)
    env1 = {'T2R_SUPCON_LAMBDA': str(s1['lambda']), 'T2R_SUPCON_T': str(s1['T']), 'T2R_P1_EPOCHS': str(s1['epochs'])}
    root = f'res_tune/{tag}/shift{sh}/phase2'
    res = json.load(open(f'{root}/tune.json')) if os.path.exists(f'{root}/tune.json') else {}
    current = {}
    for r, lr in grid:
        key = f'replay{r}_lr{lr:g}'
        out = f'{root}/{key}'
        env = dict(BASE, **repro.env(a.seed), **env1, T2R_FT_REPLAY=str(r), T2R_FT_LR2=str(lr))
        idn = manifest.phase2_identity(a.dataset, BASE, s1, r, lr, a.seed, sh, ck_sha, DATA_ID)
        if key in res:
            if res[key].get('identity') == idn:
                current[key] = res[key]; print('skip (current)', sh, key, res[key]['val_mean']); continue
            print(f'STALE {sh} {key}: made with other code/settings/Phase-1 model; not used, not overwritten'); continue
        if a.go and os.path.exists(out) and os.listdir(out):
            # an earlier attempt that left no result (e.g. drive disconnect): keep
            # it as evidence under a new name and run the setting again
            k_ = 1
            while os.path.exists(f'{out}.failed{k_}'): k_ += 1
            os.rename(out, f'{out}.failed{k_}'); print(f'  earlier failed attempt kept as {out}.failed{k_}')
        print('  ', f'split {sh}', key, '->', out)
        if not a.go:
            continue
        os.makedirs(f'{out}/modelDir', exist_ok=True)
        for f in glob.glob(os.path.join(src, 'phase1_*')):
            dst = os.path.join(out, 'modelDir', os.path.basename(f))
            (shutil.copytree if os.path.isdir(f) else shutil.copy2)(f, dst, **({'dirs_exist_ok': True} if os.path.isdir(f) else {}))
        with open(f'{out}/run.log', 'w') as fh:
            rc = subprocess.call([PY, '-u', '-c', driver, a.dataset, out, str(a.seed), str(sh)],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        txt = open(f'{out}/results_phase2.txt').read() if os.path.exists(f'{out}/results_phase2.txt') else ''
        vals = [float(v) for v in re.findall(r'val_sel=([0-9.]+)', txt)]
        n_ft = len(D.DATASETS[a.dataset]['finetune_days'])
        if rc == 0 and len(vals) == n_ft:
            res[key] = {'replay': r, 'lr2': lr, 'val_per_session': vals, 'val_mean': sum(vals) / len(vals), 'identity': idn}
            current[key] = res[key]
            os.makedirs(root, exist_ok=True); json.dump(res, open(f'{root}/tune.json', 'w'), indent=1)
            print('     ', res[key], flush=True)
        else:
            print(f'      FAILED rc={rc}, {len(vals)} of {n_ft} session scores, see {out}/run.log', flush=True)
    miss = [f'replay{r}_lr{lr:g}' for r, lr in grid if f'replay{r}_lr{lr:g}' not in current]
    if miss:
        print(f'split {sh}: grid incomplete ({len(miss)} of {len(grid)} missing/stale/failed); NOT selected')
        continue
    import selection as S
    best = S.choose([dict(v, score=v['val_mean']) for v in current.values()],
                    lambda v: (v['replay'], v['lr2']) == (0.2, 5e-5), lambda v: (v['replay'], v['lr2']))
    print(f"split {sh}: SELECTED replay={best['replay']} lr2={best['lr2']:g} (val {best['val_mean']:.4f})")
    if not a.go:
        continue
    selected[str(sh)] = {'replay': best['replay'], 'lr2': best['lr2'], 'val_mean': best['val_mean'],
                         'candidates': [{k_: v[k_] for k_ in ('replay', 'lr2', 'val_mean')} for v in current.values()],
                         'seed': a.seed, 'phase1_identity': s1['identity'], 'phase1_checkpoint_sha256': ck_sha,
                         'identity': best['identity']}
    os.makedirs(f'res_tune/{tag}', exist_ok=True); json.dump(selected, open(sel_path, 'w'), indent=1)
