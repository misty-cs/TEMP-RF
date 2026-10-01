"""Checks shared by audit_run.py (pipeline runs) and joint_eval.py (joint control)."""
import numpy as np


def disjoint_by_class(npz, prefix_pairs, n_dev):
    """Named parts are present, non-empty, cover every known device (labels
    0..n_dev-1), and share no (device, group) pair."""
    z = np.load(npz); bad = []
    miss = [p or 'train' for p in prefix_pairs if f'{p}labels' not in z]
    if miss:
        return False, f'parts missing: {miss}'
    parts = {p: set(zip(z[f'{p}labels'].astype(int).tolist(), z[f'{p}groups'].astype(int).tolist()))
             for p in prefix_pairs}
    for p in prefix_pairs:
        devs_ = sorted({d_ for d_, _ in parts[p]})
        if devs_ != list(range(n_dev)):
            bad.append(f'{p or "train"}: devices {devs_} (expected 0..{n_dev - 1})')
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = len(parts[names[i]] & parts[names[j]])
            if n:
                bad.append(f'{names[i] or "train"}&{names[j]}: {n} shared')
    return not bad, '; '.join(bad) or ('parts ' + ', '.join(f'{p or "train"} {len(parts[p])} groups' for p in names)
                                       + ' non-empty, every device, disjoint')
