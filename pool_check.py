"""Check a device pool against the data-quality rule (label-free).

    python pool_check.py                          # Pycom, the pool in datasets.py
    python pool_check.py --dataset lora           # LoRa, the pool in datasets.py
    python pool_check.py --dataset pycom 25 26    # candidate folder numbers

Rule, applied identically to every device of a dataset:
  - no duplicate recordings with another device: the first 64 KiB of every file
    compared with every other device's files, all days (data integrity,
    label-free). The Pycom source has no duplicates (PROTOCOL.md section 14).
  - quality on the PRE-TEST days only (Days 1-4):
      >= 780 good slices per day
        Pycom: packets matching their own file's preamble template (|corr| >= 0.8)
        LoRa:  preamble chirps with non-coherent alignment quality >= 0.5
      >= 3 capture files with >= 50 good slices per day (Phase 1 needs separate
      train / val / test files; Phases 2-3 need >= 3 groups);
  - availability on the test day (Day 5): all capture files present
    (Pycom 5, LoRa 10 per device). Test-day counts are printed, not used.
LoRa: the data root is the de-duplicated tree (data_prep/lora_dedup.py,
PROTOCOL.md 16.1); "all capture files" means 10 minus the files excluded there
as duplicated recordings (data_prep/lora_duplicates.json).
"""
import argparse, glob, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D, load_slice_IQ as L

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='pycom', choices=['pycom', 'lora'])
p.add_argument('--root', default=None, help='data root (default: data_root in datasets.py)')
p.add_argument('devices', nargs='*', type=int, help='folder numbers (default: the pool in datasets.py)')
a = p.parse_args()
CFG = {'pycom': dict(day='Day_{d}', dev='device_{k}', files='*.dat', n_files=5,
                     count=lambda f: L.read_packet_slices(f, None, 256)[1]),
       'lora':  dict(day='Day{d}', dev='Device{k}', files='IQ_*.dat', n_files=10,
                     count=lambda f: L.read_lora_slices(f, None, 1024)[1])}[a.dataset]
TEST_DAY = D.DATASETS[a.dataset]['test_day']
root = a.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), D.DATASETS[a.dataset]['data_root'])
EXCLUDED = {}                               # (day, device) -> files excluded as duplicated recordings
if a.dataset == 'lora':
    import json
    for g in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_prep', 'lora_duplicates.json')))['duplicates']:
        for f in g['files']:
            _day, _dev = f.split('/')[:2]
            k = (int(_day[3:]), int(_dev[6:])); EXCLUDED[k] = EXCLUDED.get(k, 0) + 1


def _h(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read(65536)).hexdigest()


_all = {}                                   # hash -> set of devices, every device folder and day
for _d in range(1, 6):
    for _dev in glob.glob(os.path.join(root, CFG['day'].format(d=_d), '*')):
        if not os.path.isdir(_dev):
            continue
        for _f in glob.glob(os.path.join(_dev, CFG['files'])):
            _all.setdefault(_h(_f), set()).add(L.folder_index(os.path.basename(_dev), -1) + 1)
DUPLICATE_DEVICES = {d for s_ in _all.values() if len(s_) > 1 for d in s_}
devs = a.devices or [int(v) + 1 for v in D.get(a.dataset)['env']['T2R_DEVICE_POOL'].split(',')]
all_ok = True
for dev in devs:
    rows, ok = [], dev not in DUPLICATE_DEVICES
    for day in range(1, 6):
        files = sorted(glob.glob(os.path.join(root, CFG['day'].format(d=day), CFG['dev'].format(k=dev), CFG['files'])))
        n = [CFG['count'](f) for f in files]
        good_files = sum(1 for x in n if x >= 50)
        day_ok = len(files) == CFG['n_files'] - EXCLUDED.get((day, dev), 0) and (day == TEST_DAY or (sum(n) >= 780 and good_files >= 3))
        ok &= day_ok
        rows.append(f'D{day}:{sum(n)}/{good_files}f' + ('' if day_ok else '!'))
    all_ok &= ok
    print(f"{CFG['dev'].format(k=dev):<10s} {'PASS' if ok else 'FAIL'}  " + '  '.join(rows), flush=True)
print('devices with duplicate files:', sorted(DUPLICATE_DEVICES) or 'none')
print('POOL', 'PASS' if all_ok else 'FAIL')
sys.exit(0 if all_ok else 1)
