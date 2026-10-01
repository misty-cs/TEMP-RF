"""Download the 43 wrong/missing Pycom files into their correct place (safe order).
Old wrong copies are moved to TOSHIBA EXT/pycom_wrong_copies/ (not deleted)."""
import os, json, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify as V                      # reuses check(), B, L (runs nothing on import? guarded below)
bad = [k for k, v in json.load(open('verify_report.json')).items() if v != 'ok']
BK = '/Volumes/TOSHIBA EXT/pycom_wrong_copies'
done = 0
for rel in bad:
    dst = f'{V.L}/{rel}'; part = dst + '.part'
    if os.path.exists(dst) and V.check(rel)[1] == 'ok':
        print('already ok', rel, flush=True); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(5):
        rc = subprocess.call(['curl', '-sk', '-C', '-', '--retry', '5', '-o', part, f'{V.B}/{rel}'])
        if os.path.exists(part) and os.path.getsize(part) == 400000000:
            break
    if not (os.path.exists(part) and os.path.getsize(part) == 400000000):
        print('FAILED download', rel, flush=True); continue
    # verify the downloaded file with the same block check (compare .part against server)
    os.replace(part, part + '.chk')
    tmp_rel_ok = True
    with open(part + '.chk', 'rb') as f:
        for off in (0, 200000000, 400000000 - V.BLK):
            f.seek(off)
            if f.read(V.BLK) != V.req(f'{V.B}/{rel}', (off, off + V.BLK - 1)):
                tmp_rel_ok = False
    if not tmp_rel_ok:
        print('FAILED verify', rel, flush=True); os.replace(part + '.chk', part); continue
    if os.path.exists(dst):
        b = f'{BK}/{rel}'; os.makedirs(os.path.dirname(b), exist_ok=True); shutil.move(dst, b)
    os.replace(part + '.chk', dst)
    done += 1; print(f'fixed {done}/{len(bad)}: {rel}', flush=True)
print('FIX DONE', done, 'of', len(bad))
