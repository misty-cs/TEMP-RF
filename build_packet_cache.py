"""Build the label-free packet cache for the Pycom device pool and report, per
file, the packet count and alignment quality. Reads the drive once; later runs
read the cache.   python build_packet_cache.py  ->  cache_packets/report.json
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D, load_slice_IQ as L
d = D.DATASETS['pycom']; pool = [int(v) for v in D.get('pycom')['env']['T2R_DEVICE_POOL'].split(',')]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), d['data_root'])
report = {}
for day in range(1, 6):
    for idx in pool:
        files = sorted(glob.glob(f'{root}/Day_{day}/device_{idx + 1}/*.dat'))
        rows = []
        for f in files:
            P, n = L.read_packet_slices(f, None, 256)
            rows.append({'file': os.path.basename(f), 'packets': int(n), 'quality': L.packet_alignment_quality(P)})
        report[f'Day_{day}/device_{idx + 1}'] = rows
        tot = sum(r['packets'] for r in rows)
        flag = '' if len(files) == 5 and tot >= 780 and min(r['packets'] for r in rows) >= 156 else '   <-- check'
        print(f'Day_{day} device_{idx + 1}: files {len(files)}, packets {[r["packets"] for r in rows]} (total {tot}), '
              f'quality {[round(r["quality"], 3) for r in rows]}{flag}', flush=True)
    json.dump(report, open(os.path.join(L.PACKET_CACHE_DIR, 'report.json'), 'w'), indent=1)
print('DONE')
