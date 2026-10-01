"""One table for every dataset / protocol the pipeline can run.

Each entry gives the run_experiment.py arguments and the environment it needs.
main.py, common_eval.py and joint_eval.py all build their configuration from
here, so a dataset is defined in exactly one place.

Sessions:  init = Phase 1, finetune = Phase 2, traj = Phase 3 reference and
calibration, test = Phase 4 open-set test (with never-enrolled devices).
"""

DATASETS = {
    # WiSig ManySig (equalized), receiver node24-16, 16 transmitters.
    # Packet-level release: every input is the first 256 samples of a packet.
    'wisig': dict(
        layout='wisig_full', data_root='data/wisig', extra=['--wisig_full_receiver', 'node24-16'],
        n_known=12, n_unknown=4, init_day=1, finetune_days=[2], traj_day=3, test_day=4,
        split_mode='random', env={'T2R_KEEP_CFO': '1'},   # keep transmitter CFO (as released)
    ),
    # Pycom indoor WiFi (raw I/Q), 20 of 50 devices, capture-disjoint splits.
    # Continuous recordings: slices spread over the whole capture (default).
    'pycom': dict(
        layout='pycom_indoor', data_root='data/pycom_indoor', extra=[],
        n_known=16, n_unknown=4, init_day=1, finetune_days=[2, 3], traj_day=4, test_day=5,
        split_mode='capture', env={'T2R_PACKET': '1', 'T2R_KEEP_CFO': '1',   # packet-aligned slices, CFO kept
                                   # Pool = the first 20 devices (folder order) that pass pool_check.py on
                                   # the VERIFIED copy (PROTOCOL.md 14): no duplicate files (none exist);
                                   # >= 780 good packets and >= 3 good files on every pre-test day; test day
                                   # available. Fail: device_5, 14, 20, 28, 29. Folders 1-4, 6-13, 15-19,
                                   # 21-23. 0-based folder indices:
                                   'T2R_DEVICE_POOL': '0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,20,21,22'},
    ),
    # Oregon State LoRa (Elmaghbub & Hamdaoui, IEEE Access 2021), Diff_Days_Indoor:
    # 25 Pycom LoRa devices (SF7, 125 kHz), USRP B210 at 1 MS/s, raw float32 IQ,
    # 10 x 20 s transmissions per device and day, five days. Slice = one preamble
    # chirp (1024 samples). Device9 is missing on Day 2 at the source, so the
    # pool is the first 20 devices present on all five days (folders 1-8, 10-21).
    'lora': dict(
        layout='lora_oregon', data_root='data/lora_indoor', extra=[],
        n_known=16, n_unknown=4, init_day=1, finetune_days=[2, 3], traj_day=4, test_day=5,
        split_mode='capture', slice_len=1024,
        env={'T2R_KEEP_CFO': '1',
             'T2R_DEVICE_POOL': '0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20'},
    ),
    # NEU (equalized release). Day 1 is incomplete (single-capture devices), so
    # Phase 1 uses day 2. day3/day5, day4/day6, day7/day8 hold the same
    # recordings; each run excludes the twin of its test day. Directory
    # numbers are not capture order (T2R_ALLOW_ANY_ORDER=1).
    'neu_A': dict(
        layout='neu', data_root='data/neu2', extra=[],
        n_known=16, n_unknown=4, init_day=2, finetune_days=[4, 7], traj_day=9, test_day=5,
        split_mode='capture', env={'T2R_ALLOW_ANY_ORDER': '1'},
    ),
    'neu_B': dict(
        layout='neu', data_root='data/neu2', extra=[],
        n_known=16, n_unknown=4, init_day=2, finetune_days=[3, 6], traj_day=9, test_day=8,
        split_mode='capture', env={'T2R_ALLOW_ANY_ORDER': '1'},
    ),
    'neu_C': dict(
        layout='neu', data_root='data/neu2', extra=[],
        n_known=16, n_unknown=4, init_day=2, finetune_days=[3, 6], traj_day=8, test_day=9,
        split_mode='capture', env={'T2R_ALLOW_ANY_ORDER': '1'},
    ),
}

# NEU slicing of the continuous recordings. contiguous = first ~0.7 MB of each
# capture (original); spread = evenly spaced single slices over the whole
# capture; chunk = 10 evenly spaced blocks of back-to-back slices.
NEU_SLICING_ENV = {'contiguous': {}, 'spread': {'T2R_SPREAD': '1'}, 'chunk': {'T2R_CHUNK': '1'}}

# Shared by every dataset.
COMMON = dict(num_slice=780, slice_len=256, ft_n_train=150)
COMMON_ENV = {'T2R_FT_NTEST': '60', 'T2R_MINIMAL': '1', 'T2R_SKIP_PHASE4': '1'}


def get(name, neu_slicing='contiguous'):
    d = dict(DATASETS[name]); env = dict(COMMON_ENV); env.update(d['env'])
    if d['layout'] == 'neu':
        env.update(NEU_SLICING_ENV[neu_slicing])
    d['env'] = env
    return d


def make_cfg(name, output_root, seed, device_shift, neu_slicing='contiguous'):
    """ExperimentConfig for this dataset. Sets the dataset env first, because
    ExperimentConfig reads the slicing switches from the environment."""
    import os
    d = get(name, neu_slicing)
    for k, v in d['env'].items():      # values set by the caller (main.py) win
        os.environ.setdefault(k, v)
    from experiment_config import ExperimentConfig
    return ExperimentConfig(
        data_root=d['data_root'], dataset_layout=d['layout'], output_root=output_root,
        wisig_full_receiver='node24-16', ft_split_mode=d['split_mode'],
        n_known=d['n_known'], n_unknown=d['n_unknown'], init_day=d['init_day'],
        finetune_days=list(d['finetune_days']), traj_day=d['traj_day'], test_day=d['test_day'],
        seed=seed, device_shift=device_shift, ft_always_reinit=False,
        **dict(COMMON, slice_len=d.get('slice_len', COMMON['slice_len'])))


def default_shifts(name):
    """One split per group of n_unknown devices, so that every device is
    unknown exactly once: WiSig (16 devices) -> 0,4,8,12; Pycom and NEU
    (20 devices) -> 0,4,8,12,16."""
    d = DATASETS[name]
    return list(range(0, d['n_known'] + d['n_unknown'], d['n_unknown']))


def cli_args(name):
    """Arguments for run_experiment.py (without output_root / seed / shift)."""
    d = DATASETS[name]
    a = ['--dataset_layout', d['layout'], '--data_root', d['data_root'], *d['extra'],
         '--n_known', str(d['n_known']), '--n_unknown', str(d['n_unknown']),
         '--ft_split_mode', d['split_mode'],
         '--init_day', str(d['init_day']), '--finetune_days', ','.join(map(str, d['finetune_days'])),
         '--traj_day', str(d['traj_day']), '--test_day', str(d['test_day']),
         '--num_slice', str(COMMON['num_slice']), '--slice_len', str(d.get('slice_len', COMMON['slice_len'])),
         '--ft_n_train', str(COMMON['ft_n_train']), '--no_ft_always_reinit', '--skip_plots']
    return a


def variant(name, cfo='default', neu_slicing='contiguous'):
    """(tag, env) of one dataset variant. The tag names the results and the
    tuning folders (res/<tag>, res_tune/<tag>), so a CFO variant never shares
    tuned settings or results with the default."""
    d = get(name, neu_slicing)
    tag = name + (f'_{neu_slicing}' if d['layout'] == 'neu' else '')
    env = dict(d['env'])
    if cfo == 'keep':
        env['T2R_KEEP_CFO'] = '1'; tag += '_keepcfo'
    elif cfo == 'remove':
        env['T2R_KEEP_CFO'] = '0'; tag += '_nocfo'
    return tag, env


def effective_env(name, env, neu_slicing='contiguous'):
    """T2R_* settings a subprocess started with `env` will use (make_cfg adds
    the dataset values with setdefault, so values in `env` win)."""
    out = dict(get(name, neu_slicing)['env'])
    out.update({k: v for k, v in env.items() if k.startswith('T2R_')})
    return out
