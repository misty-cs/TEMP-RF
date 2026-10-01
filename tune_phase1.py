"""Select the SupCon weight (lambda) and temperature (T) on VALIDATION data only,
separately for every device split.

    python tune_phase1.py --dataset wisig            # print the plan only
    python tune_phase1.py --dataset pycom --go
    python tune_phase1.py --dataset wisig --shifts 0 --go
    python tune_phase1.py --dataset wisig --cfo remove --go   # own folder res_tune/wisig_nocfo

Phase 1 only, init session only. For each split, only that split's known
devices are used (its unknown devices are never loaded), so a device that is
unknown in a split never influences that split's hyperparameters.
Selection metric (shared, selection.py): SEL = 0.5 x nearest-centroid accuracy
+ 0.5 x leave-one-device-out AUROC on the Phase-1 validation split
    WiSig: held-out bursts          Pycom: a held-out capture file
The test session is never touched; the Phase-1 'test' split is a diagnostic only.
Grid: lambda in {0, 0.25, 0.5, 1}, T in {0.05, 0.07, 0.1}; lambda = 0 ignores T,
so 10 settings. Shared tie rule: within 0.001 of the best, keep the prior
default (0.5, 0.07), else the simplest (smallest lambda, then T).
Identity: every result stores a hash of the Phase-1 code (with its imports), all
T2R_* settings, software versions, the data inventory, seed, epochs and split
(manifest.phase1_identity; main.py recomputes it before using a selection). A cached result with another
identity is STALE: it is not used and not overwritten. A split is selected
only when every setting x seed of the grid has a current result.
Results: res_tune/<tag>/shift<S>/tune.json; choice per split in
res_tune/<tag>/selected_phase1.json (read by main.py).
"""
import argparse, json, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True, choices=['wisig', 'pycom', 'lora'])
p.add_argument('--seeds', type=int, nargs='+', default=[42])
p.add_argument('--shifts', type=int, nargs='+', default=None, help='default: every split of the dataset')
p.add_argument('--epochs', type=int, default=100, help='Phase-1 epochs per setting (same as the final runs)')
p.add_argument('--cfo', choices=['default', 'keep', 'remove'], default='default')
p.add_argument('--go', action='store_true')
a = p.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here); sys.path.insert(0, here)
import datasets as D, manifest
PY = os.environ.get('PYTHON', sys.executable)
shifts = a.shifts if a.shifts is not None else D.default_shifts(a.dataset)
tag, venv = D.variant(a.dataset, a.cfo)
grid = [(0.0, 0.07)] + [(l, t) for l in (0.25, 0.5, 1.0) for t in (0.05, 0.07, 0.1)]
driver = r'''
import os, sys, json
sys.path.insert(0, os.getcwd())
import datasets as D
cfg = D.make_cfg(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
import repro; repro.set_all(cfg.seed)
import manifest; manifest.write_manifest(cfg, sys.argv[2], 'tune_phase1')
from phase1_initial_training import run_phase1
r = run_phase1(cfg)
json.dump({k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
          open(os.path.join(sys.argv[2], 'phase1_acc.json'), 'w'))
'''


import repro
BASE = dict(os.environ, **venv)
DATA_ID = manifest.data_identity(a.dataset)


def env_of(lam, T, seed):
    return dict(BASE, **repro.env(seed), T2R_SUPCON_LAMBDA=str(lam), T2R_SUPCON_T=str(T), T2R_P1_EPOCHS=str(a.epochs))


def ident(lam, T, seed, sh):
    return manifest.phase1_identity(a.dataset, BASE, lam, T, a.epochs, seed, sh, DATA_ID)


sel_path = f'res_tune/{tag}/selected_phase1.json'
selected = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
print(f'dataset={a.dataset} tag={tag} shifts={shifts} seeds={a.seeds} epochs={a.epochs} settings={len(grid)} '
      f'{"RUN" if a.go else "PLAN ONLY"}')
for sh in shifts:
    root = f'res_tune/{tag}/shift{sh}'
    results = json.load(open(f'{root}/tune.json')) if os.path.exists(f'{root}/tune.json') else {}
    current = {}
    for lam, T in grid:
        for seed in a.seeds:
            key = f'lam{lam}_T{T}_seed{seed}'
            out = f'{root}/{key}'; idn = ident(lam, T, seed, sh)
            if key in results:
                if results[key].get('identity') == idn:
                    current[key] = results[key]; print('skip (current)', sh, key); continue
                print(f'STALE {sh} {key}: made with other code/settings; not used, not overwritten '
                      f'(move {out} and its tune.json entry away to rerun)'); continue
            if a.go and os.path.exists(out) and os.listdir(out):
                # an earlier attempt that left no result (e.g. drive disconnect): keep
                # it as evidence under a new name and run the setting again
                k_ = 1
                while os.path.exists(f'{out}.failed{k_}'): k_ += 1
                os.rename(out, f'{out}.failed{k_}'); print(f'  earlier failed attempt kept as {out}.failed{k_}')
            print('  ', f'split {sh}', key, '->', out)
            if not a.go:
                continue
            os.makedirs(out, exist_ok=True)
            with open(f'{out}/run.log', 'w') as f:
                rc = subprocess.call([PY, '-u', '-c', driver, a.dataset, out, str(seed), str(sh)],
                                     stdout=f, stderr=subprocess.STDOUT, env=env_of(lam, T, seed))
            if rc == 0 and os.path.exists(f'{out}/phase1_acc.json'):
                r_ = json.load(open(f'{out}/phase1_acc.json'))
                if 'val_sel' not in r_:
                    print(f'      FAILED: no val_sel in {out}/phase1_acc.json', flush=True); continue
                results[key] = dict(r_, identity=idn); current[key] = results[key]
                os.makedirs(root, exist_ok=True)
                json.dump(results, open(f'{root}/tune.json', 'w'), indent=1)
                print('     ', r_, flush=True)
            else:
                print(f'      FAILED rc={rc}, see {out}/run.log', flush=True)
    need = [f'lam{l}_T{t}_seed{s}' for l, t in grid for s in a.seeds]
    miss = [k for k in need if k not in current]
    if miss:
        print(f'split {sh}: grid incomplete ({len(miss)} of {len(need)} missing/stale/failed); NOT selected')
        continue
    import selection as S
    cands = [{'lambda': l, 'T': t, 'score': sum(current[f'lam{l}_T{t}_seed{s}']['val_sel'] for s in a.seeds) / len(a.seeds)}
             for l, t in grid]
    b = S.choose(cands, lambda c: (c['lambda'], c['T']) == (0.5, 0.07), lambda c: (c['lambda'], c['T']))
    print(f'split {sh}: SEL per setting:', {f"({c['lambda']},{c['T']})": round(c['score'], 4) for c in cands})
    print(f'split {sh}: SELECTED lambda={b["lambda"]} T={b["T"]} (SEL {b["score"]:.4f})')
    if not a.go:
        continue
    selected[str(sh)] = {'lambda': b['lambda'], 'T': b['T'], 'val_sel': b['score'], 'candidates': cands,
                         'epochs': a.epochs, 'seeds': [a.seeds[0]],
                         'identity': ident(b['lambda'], b['T'], a.seeds[0], sh),
                         'phase1_dir': f'{root}/lam{b["lambda"]}_T{b["T"]}_seed{a.seeds[0]}',
                         'selection_seeds': a.seeds}
    os.makedirs(f'res_tune/{tag}', exist_ok=True)
    json.dump(selected, open(sel_path, 'w'), indent=1)
