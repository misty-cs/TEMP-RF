"""Raw LoRa slices per device and day: are some devices' slices degenerate?"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import datasets as D, load_slice_IQ as L, glob
root = D.DATASETS['lora']['data_root']
out = open(os.path.join(os.path.dirname(__file__), 'raw_check.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
P('device day files  slices(q>=0.5)  mean|IQ|  std-across-slices  mean-pairwise-corr(10)  quality(mean)')
for dev in [1,2,3,4,5,10,11]:
    for day in [1,2,3,4,5]:
        fs = sorted(glob.glob(f'{root}/Day{day}/Device{dev}/IQ_*.dat'))
        S, Q = [], []
        for f in fs:
            s, n = L.read_lora_slices(f, None, 1024)
            if n: S.append(s[:50])
        if not S: P(dev, day, len(fs), 0); continue
        S = np.concatenate(S); c = S[..., 0] + 1j * S[..., 1] if S.ndim == 3 else S
        c = c[:200]; cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
        cor = np.abs(cn[:10] @ cn[:10].conj().T); cor = cor[~np.eye(len(cor), dtype=bool)].mean()
        P(dev, day, len(fs), len(S), round(float(np.abs(c).mean()), 5), round(float(np.abs(c).std(0).mean()), 5), round(float(cor), 3))
