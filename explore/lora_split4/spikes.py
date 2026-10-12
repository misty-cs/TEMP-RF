"""Where are the large samples inside LoRa slices? (raw slices from the pipeline reader)"""
import os, sys, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import datasets as D, load_slice_IQ as L
root = D.DATASETS['lora']['data_root']
out = open(os.path.join(os.path.dirname(__file__), 'spikes.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
P('id day  share of slices with peak/RMS > 4 | median peak/RMS | position of peak (sample idx, median, IQR) | |x| first 16 vs rest')
for i in [0, 1, 2, 3, 9, 4, 5, 10, 13, 20]:
    for day in (1, 5):
        S = np.concatenate([L.read_lora_slices(f, None, 1024)[0][:60] for f in sorted(glob.glob(f'{root}/Day{day}/Device{i+1}/IQ_*.dat'))[:4]])
        a = np.abs(S); rms = np.sqrt((a ** 2).mean(1)); pr = a.max(1) / rms; pos = a.argmax(1)
        P(f'{i:3d} {day}  {np.mean(pr > 4):.2f}  {np.median(pr):5.2f}  {int(np.median(pos)):5d} ({int(np.percentile(pos,25))}-{int(np.percentile(pos,75))})  {a[:, :16].mean():.4f} vs {a[:, 16:].mean():.4f}')
