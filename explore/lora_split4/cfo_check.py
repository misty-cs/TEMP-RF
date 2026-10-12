"""Dechirped-peak frequency (CFO proxy) of the LoRa preamble slices per device and day."""
import os, sys, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import datasets as D, load_slice_IQ as L
root = D.DATASETS['lora']['data_root']; N = 1024; fs = 1e6; bw = 125e3
t = np.arange(N) / fs; k = bw / (N / fs)
up = np.exp(1j * np.pi * (k * t ** 2 - bw * t)); down = np.conj(up)
out = open(os.path.join(os.path.dirname(__file__), 'cfo_check.txt'), 'w')
P = lambda *a: (print(*a), print(*a, file=out))
pool = [int(v) for v in D.DATASETS['lora']['env']['T2R_DEVICE_POOL'].split(',')]
P('id(folder-1)  day: peak frequency of the dechirped slice in kHz (median over slices; spread = IQR)')
for i in pool:
    row = []
    for day in (1, 4, 5):
        f = []
        for fn in sorted(glob.glob(f'{root}/Day{day}/Device{i+1}/IQ_*.dat'))[:4]:
            s, n = L.read_lora_slices(fn, None, N)
            if not n: continue
            c = s[:40]
            spec = np.abs(np.fft.fft(c * down, axis=1)); pk = spec.argmax(1)
            fr = np.fft.fftfreq(N, 1 / fs)[pk] / 1e3; f += list(fr)
        f = np.array(f); row.append(f'D{day} {np.median(f):7.2f} ({np.subtract(*np.percentile(f, [75, 25])):.2f})')
    P(f'{i:3d}  ' + '   '.join(row))
