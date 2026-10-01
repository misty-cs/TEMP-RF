"""Run manifest: records exactly what produced a result, so every number can be
traced back later (code, settings, data, devices, software versions).

write_manifest(cfg, path, stage) is called by run_experiment.py (training) and
common_eval.py (evaluation); it writes <run_dir>/manifest_<stage>.json.
"""
import dataclasses, glob, hashlib, json, os, platform, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _versions():
    out = {'python': sys.version.split()[0], 'platform': platform.platform()}
    for m in ('numpy', 'scipy', 'sklearn', 'tensorflow', 'keras'):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


_HASH_CACHE = os.path.join(HERE, '.data_hash_cache.json')


def _cached_hash(path, sampled):
    """Content hash of a data file, cached on (real path, size, mtime_ns).
    sampled=False: SHA-256 of the whole file. sampled=True: SHA-256 of three
    64 KiB blocks (start, middle, end) + size (large raw recordings)."""
    st = os.stat(path); rp = os.path.realpath(path)
    k = json.dumps([rp, st.st_size, st.st_mtime_ns, sampled])
    try:
        cache = json.load(open(_HASH_CACHE))
    except Exception:
        cache = {}
    if k not in cache:
        if not sampled:
            cache[k] = _sha256(path)
        else:
            h = hashlib.sha256(str(st.st_size).encode())
            with open(path, 'rb') as f:
                for off in (0, max(0, st.st_size // 2 - 32768), max(0, st.st_size - 65536)):
                    f.seek(off); h.update(f.read(65536))
            cache[k] = h.hexdigest()
        tmp = _HASH_CACHE + f'.tmp{os.getpid()}'
        json.dump(cache, open(tmp, 'w')); os.replace(tmp, _HASH_CACHE)
    return cache[k]


def _data_inventory(cfg):
    """Data identity of every session the configuration reads.
    WiSig: file name, size, mtime and full SHA-256 of each session pickle.
    Folder datasets: per session, every file of the device pool (T2R_DEVICE_POOL,
    else all device folders) with size, mtime and a sampled content hash (start,
    middle, end blocks; the raw Pycom recordings are 466 GB). Packet slices also
    depend on the detector code, recorded as packet_code. A missing or empty
    session / device folder is recorded as 'missing: ...'."""
    inv = {}
    days = sorted({cfg.init_day, cfg.traj_day, cfg.test_day, *cfg.finetune_days})
    for d in days:
        if cfg.dataset_layout == 'wisig_full':
            from data_utils import _wisig_full_file
            f = _wisig_full_file(cfg, d)
            if not os.path.exists(f):
                inv[str(d)] = f'missing: {f}'; continue
            st = os.stat(f)
            inv[str(d)] = [{'file': os.path.basename(f), 'size': st.st_size, 'mtime': int(st.st_mtime),
                            'sha256': _cached_hash(f, sampled=False)}]
        else:
            import load_slice_IQ as L
            root = cfg.day_path(d)
            if not os.path.isdir(root):
                inv[str(d)] = f'missing: {root}'; continue
            folders = sorted((x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))), key=L.natural_key)
            pool = os.environ.get('T2R_DEVICE_POOL')
            if pool:
                idx = [int(v) for v in pool.split(',')]
                byidx = {L.folder_index(n, i): n for i, n in enumerate(folders)}
                miss = [i for i in idx if i not in byidx]
                if miss:
                    inv[str(d)] = f'missing: device folders for ids {miss}'; continue
                folders = [byidx[i] for i in idx]
            rows, empty = [], []
            for fo in folders:
                files = sorted(glob.glob(os.path.join(root, fo, cfg.location or '', cfg.file_key)))
                if not files:
                    empty.append(fo)
                for f in files:
                    st = os.stat(f)
                    rows.append(f'{os.path.relpath(f, root)}:{st.st_size}:{int(st.st_mtime)}:{_cached_hash(f, sampled=True)}')
            if empty or not rows:
                inv[str(d)] = f'missing: no files for {empty or "any device"}'; continue
            inv[str(d)] = {'n_files': len(rows), 'devices': folders,
                           'sha256_of_listing': hashlib.sha256('\n'.join(rows).encode()).hexdigest()}
            if os.environ.get('T2R_PACKET') == '1':
                inv[str(d)]['packet_code'] = L.packet_code_id()
    return inv


def code_sha256():
    return {os.path.relpath(f, HERE): _sha256(f) for f in sorted(glob.glob(os.path.join(HERE, '*.py')))}


REPRO_KEYS = ('TF_DETERMINISTIC_OPS', 'PYTHONHASHSEED')


def t2r_env(env):
    return {k: v for k, v in sorted(env.items()) if k.startswith('T2R_') or k in REPRO_KEYS}


def _local_imports(fn):
    import ast
    tree = ast.parse(open(os.path.join(HERE, fn)).read()); mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split('.')[0])
    return {m + '.py' for m in mods if os.path.exists(os.path.join(HERE, m + '.py'))}


def local_deps(entries):
    """Every local module an entry file can import (transitively, static)."""
    seen, todo = set(), list(entries)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f); todo += list(_local_imports(f) - seen)
    return tuple(sorted(seen))


# code that determines a Phase-1 / Phase-2 tuning result (transitive imports;
# manifest.py only records and is excluded)
PHASE1_CODE = tuple(f for f in local_deps(('phase1_initial_training.py', 'datasets.py', 'repro.py')) if f != 'manifest.py')
PHASE2_CODE = tuple(f for f in local_deps(PHASE1_CODE + ('phase2_sequential_finetuning.py', 'run_experiment.py'))
                    if f != 'manifest.py')


def data_identity(dataset, neu_slicing='contiguous'):
    """Hash of the data inventory (file names, sizes, modification times) of all
    sessions of a dataset. Fails if the data cannot be listed."""
    import tempfile, datasets as D
    saved = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as t:
            cfg = D.make_cfg(dataset, t, 0, 0, neu_slicing)
            inv = _data_inventory(cfg)
    finally:
        os.environ.clear(); os.environ.update(saved)
    if any(isinstance(v, str) for v in inv.values()):
        raise RuntimeError(f'data inventory incomplete for {dataset}: {inv}')
    return hashlib.sha256(json.dumps(inv, sort_keys=True).encode()).hexdigest()


def identity(files, env, **extra):
    """Hash of what determines a tuning result: the code files it runs, all
    T2R_* settings, software versions and extra values (seed, epochs, data
    identity, source checkpoint, ...). The per-process reproducibility
    variables are left out; the seed is an explicit extra value."""
    code = code_sha256()
    ver = {k: v for k, v in _versions().items() if k != 'platform'}
    blob = json.dumps({'code': {f: code.get(f) for f in files},
                       'env': {k: v for k, v in t2r_env(env).items() if k not in REPRO_KEYS},
                       'versions': ver, 'extra': extra}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def phase1_identity(dataset, env, lam, T, epochs, seed, shift, data_id):
    import datasets as D
    e = dict(env, T2R_SUPCON_LAMBDA=str(lam), T2R_SUPCON_T=str(T), T2R_P1_EPOCHS=str(epochs))
    return identity(PHASE1_CODE, D.effective_env(dataset, e), seeds=[seed], epochs=epochs, shift=shift, data=data_id)


def phase2_identity(dataset, env, s1, replay, lr2, seed, shift, ck_sha, data_id):
    import datasets as D
    e = dict(env, T2R_SUPCON_LAMBDA=str(s1['lambda']), T2R_SUPCON_T=str(s1['T']), T2R_P1_EPOCHS=str(s1['epochs']),
             T2R_FT_REPLAY=str(replay), T2R_FT_LR2=str(lr2))
    return identity(PHASE2_CODE, D.effective_env(dataset, e), seed=seed, shift=shift,
                    phase1_identity=s1['identity'], phase1_checkpoint_sha256=ck_sha, data=data_id)


def file_sha256(path):
    return _sha256(path)


# artifacts whose hashes the audit records; completion and aggregation require
# them unchanged
def run_artifacts(run_dir, traj_day):
    return {'common_eval.json': os.path.join(run_dir, 'common_eval.json'),
            'manifest_train.json': os.path.join(run_dir, 'manifest_train.json'),
            'manifest_eval.json': os.path.join(run_dir, 'manifest_eval.json'),
            'results_phase3.txt': os.path.join(run_dir, 'results_phase3.txt'),
            'phase3_model': os.path.join(run_dir, 'modelDir', f'phase3_bn_adapted_d{traj_day}.keras'),
            'phase3_calibration': os.path.join(run_dir, 'modelDir', f'phase3_calibration_d{traj_day}.npz')}


def artifacts_match(run_dir, recorded, traj_day):
    """(ok, reason): every recorded artifact exists and has the recorded hash."""
    if recorded and 'joint_eval.json' in recorded:          # joint-training control
        arts = {k: os.path.join(run_dir, k) for k in ('joint_eval.json', 'manifest_train.json', 'manifest_eval.json')}
    else:
        arts = run_artifacts(run_dir, traj_day)
    if not recorded or set(recorded) != set(arts):
        return False, 'audit has no artifact hashes'
    for k, pth in arts.items():
        if not os.path.exists(pth):
            return False, f'{k} missing'
        if _sha256(pth) != recorded[k]:
            return False, f'{k} changed after the audit'
    return True, ''


RUN_SPECIFIC = ('output_root', 'seed', 'device_shift')


def norm_config(cfgd, drop=RUN_SPECIFIC):
    """Configuration as the manifest stores it, without run-specific fields."""
    return {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
            for k, v in cfgd.items() if k not in drop}


def expected_config(dataset, env, seed, shift, neu_slicing='contiguous'):
    """The configuration a run launched with `env` will use (make_cfg; identical
    to run_experiment.py's CLI configuration, checked by the audit)."""
    import tempfile, datasets as D
    saved = dict(os.environ)
    try:
        os.environ.update({k: v for k, v in env.items() if k.startswith('T2R_')})
        with tempfile.TemporaryDirectory() as t:
            c = D.make_cfg(dataset, t, seed, shift, neu_slicing)
            out = norm_config(dataclasses.asdict(c))
    finally:
        os.environ.clear(); os.environ.update(saved)
    return out


def current_versions():
    return {k: v for k, v in _versions().items() if k != 'platform'}


def run_status(run_dir, env, traj_day, data_id=None, config=None):
    """'complete' only if the audit passed, every audited artifact is present and
    unchanged, and the run was made with the current code, settings (and data,
    if data_id is given). 'stale' if it exists with other code/settings/data or
    without a training manifest; 'partial' if the pipeline started with the
    current code/settings but evaluation/audit is not complete; 'new' if the
    folder is empty or absent."""
    if not os.path.isdir(run_dir) or not os.listdir(run_dir):
        return 'new', ''
    mt = os.path.join(run_dir, 'manifest_train.json')
    if not os.path.exists(mt):
        return 'stale', 'files exist but no manifest_train.json (unknown provenance)'
    m = json.load(open(mt)); cur_code = code_sha256()
    old_env = {k: v for k, v in m['env'].items() if k not in REPRO_KEYS}
    new_env = {k: v for k, v in t2r_env(env).items() if k not in REPRO_KEYS}
    d_env = sorted(set(old_env.items()) ^ set(new_env.items()))
    d_code = sorted(f for f in set(m['code_sha256']) | set(cur_code) if m['code_sha256'].get(f) != cur_code.get(f))
    d_rep = [k for k in REPRO_KEYS if m['env'].get(k) != env.get(k)]
    d_data = []
    if data_id is not None:
        inv = m.get('data')
        if not isinstance(inv, dict) or hashlib.sha256(json.dumps(inv, sort_keys=True).encode()).hexdigest() != data_id:
            d_data = ['data inventory differs']
    cur_v = current_versions(); d_ver = []
    for st_ in ('train', 'eval'):
        mp = os.path.join(run_dir, f'manifest_{st_}.json')
        if os.path.exists(mp):
            v_ = {k: v for k, v in json.load(open(mp)).get('versions', {}).items() if k != 'platform'}
            if v_ != cur_v:
                d_ver.append(f'{st_} versions {v_} != current {cur_v}')
    d_cfg = []
    if config is not None:
        old_c = norm_config(m.get('config', {}))
        d_cfg = sorted(k for k in set(old_c) | set(config) if old_c.get(k) != config.get(k))
    if d_env or d_code or d_rep or d_data or d_ver or d_cfg:
        return 'stale', (f'settings differ {d_env}; reproducibility env differs {d_rep}; code differs {d_code}; '
                         f'{d_data}; {d_ver}; config differs {d_cfg}')
    au = os.path.join(run_dir, 'audit.json')
    if not os.path.exists(au) or os.path.exists(os.path.join(run_dir, 'AUDIT_FAILED')):
        return 'partial', ''
    A = json.load(open(au))
    ok, why = artifacts_match(run_dir, A.get('artifacts'), traj_day)
    if A.get('all_pass') and ok:
        return 'complete', ''
    return 'partial', why or 'audit did not pass'


def write_manifest(cfg, run_dir, stage, extra=None):
    cfgd = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(vars(cfg))
    code = code_sha256()
    m = {
        'stage': stage,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': sys.argv,
        'config': {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in cfgd.items()},
        'derived': {'known_ids': cfg.all_known_ids, 'unknown_ids': cfg.unknown_ids, 'stride': cfg.stride},
        'env': t2r_env(os.environ),
        'code_sha256': code,
        'code_sha256_all': hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
        'versions': _versions(),
    }
    try:
        m['data'] = _data_inventory(cfg)
    except Exception as e:
        m['data'] = f'inventory failed: {e}'
    if extra:
        m.update(extra)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f'manifest_{stage}.json'), 'w') as f:
        json.dump(m, f, indent=1, default=str)
    return m
