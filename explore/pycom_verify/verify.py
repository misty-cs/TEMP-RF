"""Compare the local Pycom WiFi Indoor copy with the Oregon State server (read-only).
Per file: exact size (HTTP HEAD) + three 64 KiB blocks (start, middle, end) byte-equal."""
import os, ssl, urllib.request, json, concurrent.futures as cf
B = 'https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/WiFi-Dataset/Indoor'
L = '/Volumes/TOSHIBA EXT/pycom/Indoor'
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE   # server chain incomplete
BLK = 65536

def req(url, rng=None, head=False):
    for attempt in range(5):
        try:
            r = urllib.request.Request(url, method='HEAD' if head else 'GET')
            if rng: r.add_header('Range', 'bytes=%d-%d' % rng)
            with urllib.request.urlopen(r, context=ctx, timeout=60) as h:
                return (int(h.headers['Content-Length']) if head else h.read())
        except Exception as e:
            err = e
    raise err

def check(rel):
    lp = f'{L}/{rel}'; url = f'{B}/{rel}'
    try:
        size = req(url, head=True)
    except Exception as e:
        return rel, 'server error: %s' % e
    if not os.path.exists(lp):
        return rel, 'MISSING locally (server %d bytes)' % size
    ls = os.path.getsize(lp)
    if ls != size:
        return rel, 'SIZE differs: local %d server %d' % (ls, size)
    with open(lp, 'rb') as f:
        for off in (0, size // 2, size - BLK):
            f.seek(off); a = f.read(BLK)
            b = req(url, (off, off + BLK - 1))
            if a != b:
                return rel, 'CONTENT differs at offset %d' % off
    return rel, 'ok'

if __name__ == '__main__':
    rels = [f'Day_{d}/device_{i}/tx_{t}_iq.dat' for d in range(1, 6) for i in range(1, 51) for t in range(1, 6)]
    res = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for n, (rel, st) in enumerate(ex.map(check, rels), 1):
            res[rel] = st
            if st != 'ok': print(rel, st, flush=True)
            if n % 100 == 0: print(f'.. {n}/{len(rels)} checked', flush=True)
    bad = {k: v for k, v in res.items() if v != 'ok'}
    json.dump(res, open('verify_report.json', 'w'), indent=1)
    print(f'DONE: {len(rels)} files, ok {len(rels) - len(bad)}, problems {len(bad)}')
