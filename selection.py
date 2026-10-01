"""One selection score and one tie rule for every tuned choice in the pipeline.

Score (all selections): SEL = 0.5 x closed accuracy + 0.5 x leave-one-device-out
AUROC, computed on the selection data of that stage (Phase-1 val, Phase-2 val,
reference Part B). Leave-one-device-out: each known device in turn is held out;
its slices are scored against the other devices only.

Tie rule (all selections): every candidate whose score is within TIE_TOL of the
best counts as tied; among tied candidates the prior default is kept if it is
tied, otherwise the simplest candidate (smallest `simplicity` value).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

TIE_TOL = 1e-3


def sel_score(closed, loco):
    return 0.5 * float(closed) + 0.5 * float(loco)


def centroid_sel(emb_ref, y_ref, emb_sel, y_sel):
    """closed accuracy, LOCO AUROC and SEL of nearest-centroid classification of
    emb_sel against class centroids of emb_ref (cosine), labels 0..K-1."""
    l2 = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    classes = np.unique(y_ref)
    C = l2(np.stack([l2(emb_ref)[y_ref == c].mean(0) for c in classes]))
    D = 1 - l2(emb_sel) @ C.T
    closed = float(np.mean(classes[D.argmin(1)] == y_sel))
    au = []
    for j, c in enumerate(classes):
        lab = (y_sel == c).astype(int)
        if 0 < lab.sum() < len(lab):
            au.append(roc_auc_score(lab, np.delete(D, j, axis=1).min(1)))
    loco = float(np.mean(au)) if au else 0.0
    return closed, loco, sel_score(closed, loco)


def choose(cands, is_default, simplicity, key='score', tol=TIE_TOL):
    """Pick from a list of dicts by `key` with the shared tie rule."""
    best = max(c[key] for c in cands)
    tied = [c for c in cands if c[key] >= best - tol]
    for c in tied:
        if is_default(c):
            return c
    return min(tied, key=simplicity)
