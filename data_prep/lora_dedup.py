"""LoRa Indoor (Oregon State): remove duplicated recordings (PROTOCOL.md 16.1).

The published Diff_Days_Indoor_Setup contains recordings that are stored,
byte for byte, under two device IDs (same day, same file number, neighbouring
devices; the server copies are identical too, so this is a source error, not
a download error). Two independent 20 s over-the-air captures cannot be equal,
and the metadata does not say which device sent the signal, so BOTH copies of
every duplicated recording are excluded. Label-free: only file contents are
compared, every device is treated the same.

    python data_prep/lora_dedup.py SRC DST    # find duplicates, build DST
    python data_prep/lora_dedup.py --check DST

SRC  the downloaded tree (Day1..Day5/DeviceK/IQ_i.dat + .sigmf-meta), unchanged
DST  a tree of symbolic links to SRC without the duplicated recordings
Writes data_prep/lora_duplicates.json (every duplicated group with its SHA-256).
--check exits 0 only if DST matches SRC minus the listed files.
"""
import glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(HERE, 'lora_duplicates.json')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rel_files(root):
    return sorted(os.path.relpath(f, root) for f in glob.glob(os.path.join(root, 'Day*', 'Device*', 'IQ_*.dat')))


def check(dst):
    L = json.load(open(LIST)); src = L['source']
    excl = {f for g in L['duplicates'] for f in g['files']}
    want = [f for f in rel_files(src) if f not in excl]
    have = rel_files(dst)
    ok = len(want) == L['n_source_files'] - L['n_excluded'] and have == want and all(os.path.realpath(os.path.join(dst, f)) == os.path.realpath(os.path.join(src, f))
                              and os.path.getsize(os.path.join(dst, f)) == L['file_size'] for f in have)
    print(f'{dst}: {len(have)} files, expected {len(want)} ({len(excl)} excluded):', 'OK' if ok else 'MISMATCH')
    return ok


def build(src, dst):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    files = rel_files(src)
    sizes = {os.path.getsize(os.path.join(src, f)) for f in files}
    assert len(sizes) == 1, f'files of different size in {src}: {sizes}'
    first = {}                                          # cheap pre-filter: first 64 KiB
    for f in files:
        with open(os.path.join(src, f), 'rb') as fh:
            first.setdefault(hashlib.sha1(fh.read(65536)).hexdigest(), []).append(f)
    full = {}
    for group in first.values():
        if len(group) > 1:
            for f in group:
                full.setdefault(sha256(os.path.join(src, f)), []).append(f)
    dups = [{'sha256': h, 'files': sorted(g)} for h, g in sorted(full.items()) if len(g) > 1]
    excl = {f for g in dups for f in g['files']}
    for f in files:
        if f in excl:
            continue
        for ext in ('.dat', '.sigmf-meta'):
            s = os.path.join(src, f[:-4] + ext); d = os.path.join(dst, f[:-4] + ext)
            if os.path.exists(s) and not os.path.lexists(d):
                os.makedirs(os.path.dirname(d), exist_ok=True); os.symlink(s, d)
    json.dump({'source': src, 'file_size': sizes.pop(), 'n_source_files': len(files),
               'n_excluded': len(excl), 'duplicates': dups}, open(LIST, 'w'), indent=1)
    for g in dups:
        print('duplicate:', ' = '.join(g['files']))
    print(f'{len(dups)} duplicated recordings, {len(excl)} files excluded, {len(files) - len(excl)} kept')
    return check(dst)


if __name__ == '__main__':
    if sys.argv[1] == '--check':
        sys.exit(0 if check(os.path.abspath(sys.argv[2])) else 1)
    sys.exit(0 if build(sys.argv[1], sys.argv[2]) else 1)
