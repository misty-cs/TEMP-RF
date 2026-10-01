"""Where does each local file really come from? Match first 64 KiB of every local file
against the first 64 KiB of every server file (read-only). Also: duplicates on the server."""
import os, ssl, urllib.request, json, hashlib, collections, concurrent.futures as cf
B = 'https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/WiFi-Dataset/Indoor'
L = '/Volumes/TOSHIBA EXT/pycom/Indoor'
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
BLK = 65536
rels = [f'Day_{d}/device_{i}/tx_{t}_iq.dat' for d in range(1, 6) for i in range(1, 51) for t in range(1, 6)]

def srv(rel):
    for a in range(5):
        try:
            r = urllib.request.Request(f'{B}/{rel}'); r.add_header('Range', 'bytes=0-%d' % (BLK - 1))
            with urllib.request.urlopen(r, context=ctx, timeout=60) as h:
                return rel, hashlib.sha1(h.read()).hexdigest()
        except Exception as e:
            err = e
    return rel, 'ERR ' + str(err)

def loc(rel):
    p = f'{L}/{rel}'
    if not os.path.exists(p): return rel, None
    with open(p, 'rb') as f: return rel, hashlib.sha1(f.read(BLK)).hexdigest()

with cf.ThreadPoolExecutor(6) as ex:
    S = dict(ex.map(srv, rels))
Lh = dict(map(loc, rels))
by_hash = collections.defaultdict(list)
for r, h in S.items(): by_hash[h].append(r)
print('SERVER duplicates (same first 64 KiB in different server files):')
dups = [v for v in by_hash.values() if len(v) > 1]
for v in dups: print('  ', ' = '.join(v))
if not dups: print('   none')
print('LOCAL files that are not where they should be:')
wrong = {}
for r in rels:
    if Lh[r] is None:
        wrong[r] = 'missing'; print(f'   {r}: MISSING'); continue
    if Lh[r] == S[r]: continue
    src = by_hash.get(Lh[r], [])
    wrong[r] = src or 'unknown'
    print(f'   {r}: actually = {", ".join(src) if src else "NOT any server file (corrupt/other)"}')
json.dump({'server_duplicates': dups, 'local_wrong': wrong}, open('locate_report.json', 'w'), indent=1)
print('DONE: local files wrong/missing:', len(wrong))
