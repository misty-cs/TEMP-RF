"""Common open-set evaluation for all decision rules on one trained run.

Run from t2r/ after run_experiment.py finished (main.py does this):
    python common_eval.py <dataset> <run_dir> <device_shift> <seed> [neu_slicing]
Writes <run_dir>/common_eval.json.

Every rule uses the SAME Phase-3 model and the SAME reference-session parts
saved by Phase 3 (disjoint by construction):
    Part A (50%)  prototypes / centroids / k-NN support / OpenMax fit
    Part B (25%)  rule-internal selection (k-NN k); TEMP-RFF window was
                  selected on it in Phase 3
    Part C (25%)  thresholds. Split once, rule-independently, into
                  pseudo-unknown = hardest 25% by static-centroid distance
                  calib-known    = remaining 75%
Two threshold protocols, applied identically to every rule:
    fixed90   threshold = 90th percentile of the rule's calib-known scores
    objective max 0.5*(acc on accepted + pseudo-unknown rejection),
              subject to >= 50% calib-known acceptance
Unknown-device test data is used only for the final metrics.
"""
import copy, json, os, re, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import datasets as D
from data_utils import load_day, split_known_unknown
from temporal_trajectory import TemporalTrajectory, extract_embeddings_from_model
import load_slice_IQ, rf_models, openmax as OM

ds, run_dir, shift, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
slicing = sys.argv[5] if len(sys.argv) > 5 else 'contiguous'
KS = [1, 5, 10, 20, 50, 100, 200, 500]
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
cfg = D.make_cfg(ds, run_dir, seed, shift, slicing)
import repro; repro.set_all(seed)
md = cfg.model_dir
known, unk = cfg.all_known_ids, cfg.unknown_ids
K = len(known)
kmap = {d: i for i, d in enumerate(sorted(known))}

# ── shared model, reference parts, trajectory ───────────────────────────────
model = tf.keras.models.load_model(os.path.join(md, f'phase3_bn_adapted_d{cfg.traj_day}.keras'),
                                   custom_objects={'L2Normalize': rf_models.L2Normalize})
W_clf, b_clf = model.get_layer('classifier').get_weights()
logits = lambda e: e @ W_clf + b_clf
nrm = np.load(os.path.join(md, 'phase3_norm_stats.npz'))
norm = lambda X: load_slice_IQ.apply_normalization(X, nrm['mean'], nrm['std'])
cal = np.load(os.path.join(md, f'phase3_calibration_d{cfg.traj_day}.npz'))
eA, yA = cal['emb_update'], cal['y_update'].astype(int)
eB, yB = cal['emb_sweep'], cal['y_sweep'].astype(int)
eC, yC = cal['emb_known'], cal['y_known'].astype(int)
traj = TemporalTrajectory.load(os.path.join(md, f'phase3_trajectory_d{cfg.traj_day}.npz'))
p3 = open(os.path.join(run_dir, 'results_phase3.txt')).read()
sel_dk = int(re.findall(r'best_days_keep=(\d+)', p3)[-1])
sel_dec = float(re.findall(r'best_day_decay=([0-9.]+)', p3)[-1])
W_MAX = len(traj._history[next(iter(traj._history))])

# ── test session ────────────────────────────────────────────────────────────
devs = known + unk
from data_utils import load_day_with_groups
Xt, yt, gt = load_day_with_groups(cfg, day_id=cfg.test_day, device_ids=devs)
yt = np.array([sorted(devs)[int(v)] for v in yt])
import hashlib
TEST_INPUT_SHA1 = hashlib.sha1(np.ascontiguousarray(Xt).tobytes() + yt.astype(np.int64).tobytes()).hexdigest()
TEST_COUNTS = {int(d): int((yt == d).sum()) for d in sorted(devs)}
is_k = np.isin(yt, known)
Xk, Xu = norm(Xt[is_k]), norm(Xt[~is_k])
yk = np.array([kmap[int(v)] for v in yt[is_k]])
gk = None if gt is None else (gt[is_k, 0] if np.ndim(gt) == 2 else gt[is_k])
ek = extract_embeddings_from_model(model, Xk, 256)
eu = extract_embeddings_from_model(model, Xu, 256)

# ── rule-independent calibration split of Part C ────────────────────────────
Cstat = l2n(np.stack([eA[yA == c].mean(0) for c in range(K)]))
dC_stat = 1 - (l2n(eC) @ Cstat.T).max(1)
hard = np.zeros(len(eC), bool)
for c in range(K):                                   # hardest 25% per device
    i = np.flatnonzero(yC == c)
    hard[i[np.argsort(dC_stat[i])[-max(1, len(i) // 4):]]] = True
eCk, yCk, eCu = eC[~hard], yC[~hard], eC[hard]


# ── rules ────────────────────────────────────────────────────────────────
# Every rule returns (pred, score): score = how unknown a slice looks (reject if
# score > threshold). The SAME (pred, score) is used by every threshold protocol.
# Each rule also has a leave-one-device-out function loco(c) -> score function of
# the same rule with device c removed: refitted (probe, k-NN), recalibrated
# (OpenMax), renormalised (MSP: softmax over the other K-1 logits) or with the
# device's references removed (centroid / prototype rules). A slice of device c
# scored by loco(c) is a pseudo-unknown of the tuned protocol. The shared network
# has still seen device c; this is the same for every rule.
# Tunable rules are selected on Part B with the Phase-3 score
#   SEL = 0.5 x closed accuracy + 0.5 x leave-one-device-out AUROC,
# and every candidate with its score is stored in res['selection'].
SELECTION = {}


import selection as _S


def _sel_score(closed, loco):                  # shared score (selection.py)
    return _S.sel_score(closed, loco)


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


# softmax head: MSP rejection (maximum softmax probability, Hendrycks & Gimpel 2017)
def r_msp(e):
    p_ = _softmax(logits(e)); return p_.argmax(1), 1 - p_.max(1)


def _cached(make):
    memo = {}
    def get(c):
        if c not in memo:
            memo[c] = make(c)
        return memo[c]
    return get


def _D_loco(Dfn):
    """loco for rules that are nearest-reference on a class-distance matrix D:
    removing device c removes its column (its references)."""
    return lambda c: (lambda e: np.delete(Dfn(e), c, axis=1).min(1))


loco_msp = lambda c: (lambda e: 1 - _softmax(np.delete(logits(e), c, axis=1)).max(1))


# static centroid: one Part-A mean per device, cosine distance (no parameters)
def r_static(e):
    s_ = l2n(e) @ Cstat.T
    return s_.argmax(1), 1 - s_.max(1)


def D_static(e): return 1 - l2n(e) @ Cstat.T


# linear probe: multinomial logistic regression on Part-A embeddings, MSP rejection
from sklearn.linear_model import LogisticRegression


def _probe_fit(X, y, C):
    return LogisticRegression(C=C, max_iter=3000).fit(l2n(X), y)


def make_probe(C=None):
    def rule_from(C_):
        m = _probe_fit(eA, yA, C_)
        def r(e):
            p_ = m.predict_proba(l2n(e)); return m.classes_[p_.argmax(1)], 1 - p_.max(1)
        def refit(c):
            mc = _probe_fit(eA[yA != c], yA[yA != c], C_)
            return lambda e: 1 - mc.predict_proba(l2n(e)).max(1)
        return r, _cached(refit)
    if C is not None:
        r, D = rule_from(C); return r, D, C
    cands = []
    for C_ in (0.01, 0.1, 1.0, 10.0):
        m = _probe_fit(eA, yA, C_)
        closed = float(np.mean(m.predict(l2n(eB)) == yB)); au = []
        for c in range(K):                                  # refit without device c
            mc = _probe_fit(eA[yA != c], yA[yA != c], C_)
            au.append(roc_auc_score((yB == c).astype(int), 1 - mc.predict_proba(l2n(eB)).max(1)))
        cands.append(dict(C=C_, closed=closed, loco=float(np.mean(au)), score=_sel_score(closed, float(np.mean(au)))))
    SELECTION['probe'] = cands
    best = _S.choose(cands, lambda d: d['C'] == 1.0, lambda d: d['C'])
    r, D = rule_from(best['C']); return r, D, best['C']


# k-NN: vote of the k nearest Part-A embeddings (cosine), unknown score = mean
# distance to them; search space k x (uniform | distance-weighted vote)
from sklearn.neighbors import NearestNeighbors
_An = l2n(eA)
_nn = NearestNeighbors(n_neighbors=20, metric='cosine').fit(_An)


def _knn_fn(k, weighted, model=_nn, labels=yA):
    def f(e):
        d, ix = model.kneighbors(l2n(e), n_neighbors=k)
        lab = labels[ix]; w = 1.0 / (d + 1e-6) if weighted else np.ones_like(d)
        votes = np.zeros((len(e), K)); np.add.at(votes, (np.repeat(np.arange(len(e)), k), lab.ravel()), w.ravel())
        return votes.argmax(1), d.mean(1)
    return f


def _knn_loco(k, weighted, Xn, y):
    """k-NN refitted on the support without device c (native k-NN score)."""
    def refit(c):
        m = y != c
        nn_c = NearestNeighbors(n_neighbors=k, metric='cosine').fit(Xn[m])
        f = _knn_fn(k, weighted, nn_c, y[m])
        return lambda e: f(e)[1]
    return _cached(refit)


def make_knn(k=None, weighted=None):
    if k is not None:
        return _knn_fn(k, weighted), _knn_loco(k, weighted, _An, yA), (k, weighted)
    cands = []
    for k_ in (1, 5, 10, 20):
        for wt in (False, True):
            closed = float(np.mean(_knn_fn(k_, wt)(eB)[0] == yB)); au = []
            for c in range(K):
                m = yA != c
                nn_c = NearestNeighbors(n_neighbors=k_, metric='cosine').fit(_An[m])
                au.append(roc_auc_score((yB == c).astype(int), _knn_fn(k_, wt, nn_c, yA[m])(eB)[1]))
            cands.append(dict(k=k_, weighted=wt, closed=closed, loco=float(np.mean(au)),
                              score=_sel_score(closed, float(np.mean(au)))))
    SELECTION['knn'] = cands
    b_ = _S.choose(cands, lambda d: (d['k'], d['weighted']) == (5, False), lambda d: (d['weighted'], d['k']))
    return _knn_fn(b_['k'], b_['weighted']), _knn_loco(b_['k'], b_['weighted'], _An, yA), (b_['k'], b_['weighted'])


# OpenMax (Weibull fit on Part-A logits, rank-weighted revision); search space
# distance {euclidean, eucos} x tail {10, 20, 50} x alpha {1, 2, 5, min(10, K), K}
# (min(10, K) is the prior default, so the tie rule can keep it)
def _om_loco(LA, tail, dist, al):
    """OpenMax recalibrated without device c: its logit column and its Part-A
    slices removed, alpha capped at K-1."""
    def refit(c):
        cols = [j for j in range(K) if j != c]; ma = yA != c
        remap = {o: n_ for n_, o in enumerate(cols)}
        cal_c = OM.calibrate(LA[ma][:, cols], np.array([remap[v] for v in yA[ma]]), K - 1, tail, dist)
        return lambda e: OM.score(logits(e)[:, cols], cal_c, min(al, K - 1))[1]
    return _cached(refit)


def _om_rule(LA, tail, dist, al):
    cal = OM.calibrate(LA, yA, K, tail, dist)
    def r(e):
        pr, u = OM.score(logits(e), cal, al); return pr[:, :K].argmax(1), u
    return r, _om_loco(LA, tail, dist, al)


def make_openmax(dist=None, tail=None, al=None):
    LA, LB = logits(eA), logits(eB)
    if dist is not None:
        r, D = _om_rule(LA, tail, dist, al); return r, D, (dist, tail, al)
    cands = []
    for dist_ in ('euclidean', 'eucos'):
        for tail_ in (10, 20, 50):
            for al_ in sorted({1, 2, 5, min(10, K), K}):
                cal = OM.calibrate(LA, yA, K, tail_, dist_)
                pB, _ = OM.score(LB, cal, al_)
                closed = float(np.mean(pB[:, :K].argmax(1) == yB)); au = []
                lo_ = _om_loco(LA, tail_, dist_, al_)
                for c in range(K):
                    au.append(roc_auc_score((yB == c).astype(int), lo_(c)(eB)))
                cands.append(dict(dist=dist_, tail=tail_, alpha=al_, closed=closed, loco=float(np.mean(au)),
                                  score=_sel_score(closed, float(np.mean(au)))))
    SELECTION['openmax'] = cands
    b_ = _S.choose(cands, lambda d: (d['dist'], d['tail'], d['alpha']) == ('euclidean', 20, min(10, K)),
                   lambda d: (d['dist'] != 'euclidean', d['alpha'], d['tail']))
    r, D = _om_rule(LA, b_['tail'], b_['dist'], b_['alpha'])
    return r, D, (b_['dist'], b_['tail'], b_['alpha'])


# multi-prototype rules (TEMP-RFF family); W/P/decay/alpha selected in Phase 3
def mp_distance(dk, dec, tr=None, weighted=True):
    """Class-distance matrix of the multi-prototype rule (n x K)."""
    tr = tr or traj
    banks = [tr.cosine_prototype_bank(d, days_keep=dk, day_decay=dec) for d in range(K)]
    def D(e):
        en = l2n(e); out = np.empty((len(e), K), np.float32)
        for j, (P, w) in enumerate(banks):
            s_ = en @ P.T
            out[:, j] = 1 - ((s_ * (0.9 + 0.1 * w[None, :])) if weighted else s_).max(1)
        return out
    return D


def rule_from_D(Dfn):
    def f(e):
        D = Dfn(e); return D.argmin(1), D.min(1)
    return f


def make_mp(dk, dec, tr=None, weighted=True):
    return rule_from_D(mp_distance(dk, dec, tr, weighted))


traj_p1 = TemporalTrajectory.load(os.path.join(md, f'phase3_trajectory_p1_d{cfg.traj_day}.npz'))
traj_pall = TemporalTrajectory.load(os.path.join(md, f'phase3_trajectory_pall_d{cfg.traj_day}.npz'))

# Equal-data baselines: the same reference embeddings as TEMP-RFF (all
# embeddings of the window sessions: earlier sessions from non-training groups
# + Part A, i.e. the P = all banks). Each rule selects its own window W on B.
def _window_support(W):
    X_, y_ = [], []
    for d in range(K):
        P_, _w = traj_pall.cosine_prototype_bank(d, days_keep=W, day_decay=1.0)
        X_.append(P_); y_.append(np.full(len(P_), d))
    return np.concatenate(X_), np.concatenate(y_)


def _loco_D(Dfn, e, y):
    """Leave-one-device-out AUROC from a class-distance matrix (distance to the
    held-out device removed), as in Phase 3."""
    D = Dfn(e); au = []
    for c in range(K):
        cols = [j for j in range(K) if j != c]
        au.append(roc_auc_score((y == c).astype(int), D[:, cols].min(1)))
    return float((D.argmin(1) == y).mean()), float(np.mean(au))


def make_window_rules():
    cands_s, cands_k = [], []
    for W in range(1, W_MAX + 1):
        Xs, ys = _window_support(W)
        Cw = l2n(np.stack([Xs[ys == c].mean(0) for c in range(K)]))
        D_s = (lambda Cw_: (lambda e: 1 - l2n(e) @ Cw_.T))(Cw)
        cl, lo = _loco_D(D_s, eB, yB)
        cands_s.append(dict(W=W, closed=cl, loco=lo, score=_sel_score(cl, lo)))
        nnw = NearestNeighbors(n_neighbors=20, metric='cosine').fit(Xs)
        for k_ in (1, 5, 10, 20):
            for wt in (False, True):
                closed = float(np.mean(_knn_fn(k_, wt, nnw, ys)(eB)[0] == yB)); au = []
                for c in range(K):
                    m = ys != c
                    nn_c = NearestNeighbors(n_neighbors=k_, metric='cosine').fit(Xs[m])
                    au.append(roc_auc_score((yB == c).astype(int), _knn_fn(k_, wt, nn_c, ys[m])(eB)[1]))
                cands_k.append(dict(W=W, k=k_, weighted=wt, closed=closed, loco=float(np.mean(au)),
                                    score=_sel_score(closed, float(np.mean(au)))))
    SELECTION['static_window'] = cands_s; SELECTION['knn_window'] = cands_k
    bs = _S.choose(cands_s, lambda d: d['W'] == 1, lambda d: d['W'])
    bk = _S.choose(cands_k, lambda d: (d['W'], d['k'], d['weighted']) == (1, 5, False),
                   lambda d: (d['W'], d['weighted'], d['k']))
    Xs, ys = _window_support(bs['W']); Cw = l2n(np.stack([Xs[ys == c].mean(0) for c in range(K)]))
    D_s = lambda e: 1 - l2n(e) @ Cw.T
    Xk, yk_ = _window_support(bk['W']); Xk_n = l2n(Xk)
    nnk = NearestNeighbors(n_neighbors=20, metric='cosine').fit(Xk_n)
    r_k = _knn_fn(bk['k'], bk['weighted'], nnk, yk_)
    return (rule_from_D(D_s), _D_loco(D_s), bs), (r_k, _knn_loco(bk['k'], bk['weighted'], Xk_n, yk_), bk)


r_probe, D_probe, PROBE_C = make_probe()
r_knn, D_knn, KNN_SEL = make_knn()
r_openmax, D_openmax, OM_SEL = make_openmax()
KNN_K = KNN_SEL[0]
# untuned defaults (same values for every run; shows the effect of tuning)
DEF = dict(knn=(5, False), openmax=('euclidean', 20, min(10, K)), probe=1.0, temp_rff=('all', 1, 1.0))
r_probe_def, D_probe_def, _ = make_probe(DEF['probe'])
r_knn_def, D_knn_def, _ = make_knn(*DEF['knn'])
r_om_def, D_om_def, _ = make_openmax(*DEF['openmax'])

_SW, _KW = make_window_rules()


# temp_rff_p1 (one plain mean per device and session) selects its OWN window W
# and session decay on Part B, with the same score and tie rule (default W 1,
# decay 1; simplicity: smaller W, then less decay). temp_rff_nowt keeps the
# full method's settings on purpose: it is a fixed-setting ablation of the
# prototype weight only.
def select_p1():
    cands = []
    for W in range(1, W_MAX + 1):
        for dec in (1.0, 0.8, 0.6):
            cl, lo = _loco_D(mp_distance(W, dec, traj_p1), eB, yB)
            cands.append(dict(W=W, decay=dec, closed=cl, loco=lo, score=_sel_score(cl, lo)))
    SELECTION['temp_rff_p1'] = cands
    return _S.choose(cands, lambda d: (d['W'], d['decay']) == (1, 1.0), lambda d: (d['W'], 1.0 - d['decay']))


P1_SEL = select_p1()
def _mp(dk, dec, tr=None, weighted=True):          # (rule, loco) of a prototype rule
    D_ = mp_distance(dk, dec, tr, weighted); return rule_from_D(D_), _D_loco(D_)


RULE_D = {                              # rule -> (rule fn, loco fn)
    'msp': (r_msp, loco_msp),
    'probe': (r_probe, D_probe),
    'openmax': (r_openmax, D_openmax),
    'static': (r_static, _D_loco(D_static)),
    'knn': (r_knn, D_knn),
    'temp_rff_p1': _mp(P1_SEL['W'], P1_SEL['decay'], traj_p1),
    'static_window': (_SW[0], _SW[1]),
    'knn_window': (_KW[0], _KW[1]),
    'temp_rff': _mp(sel_dk, sel_dec),
    'temp_rff_nowt': _mp(sel_dk, sel_dec, weighted=False),
    'probe_default': (r_probe_def, D_probe_def),
    'knn_default': (r_knn_def, D_knn_def),
    'openmax_default': (r_om_def, D_om_def),
    'temp_rff_default': _mp(1, 1.0, traj_pall),
}
for w_ in range(1, W_MAX + 1):
    RULE_D[f'temp_rff_W{w_}'] = _mp(w_, 1.0)


# ── thresholds and metrics ──────────────────────────────────────────────────
# fixed90 uses ALL of Part C (accepts 90% of it). The objective protocol uses the
# rule-independent split of C (calib-known 75% / pseudo-unknown 25%, hardest by
# static-centroid distance; a stress subset defined once for every rule).
def thresholds(rule, Ck=None, Cu=None):
    Ck = eCk if Ck is None else Ck; Cu = eCu if Cu is None else Cu
    pk, sk = rule(Ck); _, su = rule(Cu)
    out = {'fixed90': float(np.quantile(np.r_[sk, su], 0.90))}
    best = (-1.0, float(np.quantile(sk, 0.5)))
    for t in np.linspace(min(sk.min(), su.min()), max(sk.max(), su.max()), 200):
        a = sk <= t
        if a.mean() < 0.5:
            continue
        sc = 0.5 * (np.mean(pk[a] == yCk[a]) + np.mean(su > t))
        if sc > best[0]:
            best = (sc, float(t))
    out['objective'] = best[1]
    return out


def metrics(rule, thr, e_k=None, y_k=None, e_u=None):
    e_k = ek if e_k is None else e_k; y_k = yk if y_k is None else y_k
    e_u = eu if e_u is None else e_u
    pk, sk = rule(e_k); _, su = rule(e_u)
    auroc = float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su]))
    out = {'closed': float(np.mean(pk == y_k)), 'auroc': auroc}
    for name, t in thr.items():
        a = sk <= t
        out[name] = dict(thr=t, accept=float(a.mean()),
                         open=float(np.mean(pk[a] == y_k[a])) if a.any() else float('nan'),
                         unk_det=float(np.mean(su > t)),
                         known_ok=float(np.mean(a & (pk == y_k))))
    return out


import hashlib
res = dict(dataset=ds, shift=shift, seed=seed, known=known, unknown=unk,
           sel_days_keep=sel_dk, sel_decay=sel_dec, knn_k=KNN_K, knn_weighted=KNN_SEL[1],
           openmax_dist=OM_SEL[0], openmax_tail=OM_SEL[1], openmax_alpha=OM_SEL[2], probe_C=PROBE_C,
           window_rules=dict(static_window=_SW[2], knn_window=_KW[2], temp_rff_p1=P1_SEL),
           defaults=DEF, rules={},
           test_set=dict(n_known=int(len(yk)), n_unknown=int(len(eu)),
                         sha1=hashlib.sha1(np.ascontiguousarray(ek).tobytes() + np.ascontiguousarray(eu).tobytes()
                                           + yk.astype(np.int32).tobytes()).hexdigest(),
                         input_sha1=TEST_INPUT_SHA1, per_device=TEST_COUNTS))
res['rules']['softmax'] = {'closed': float(np.mean(logits(ek).argmax(1) == yk))}
for name, (rule, _D) in RULE_D.items():
    res['rules'][name] = metrics(rule, thresholds(rule))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['rules'][name].items()
                 if k in ('closed', 'auroc')}, flush=True)

# ── raw-input baseline: nearest centroid on the loader output, no network ────
# Reference session reloaded exactly as Phase 3 loads it and split with the same
# function and seed, so Part A / Part C are the same slices as above.
from phase3_trajectory_building import _group_three_split
from data_utils import load_day_with_groups as _ldg
Xr_raw, yr_raw, gr_raw = _ldg(cfg, day_id=cfg.traj_day, device_ids=known)
yr_raw = np.asarray(yr_raw).astype(int)
gr_raw = gr_raw[:, 0] if np.ndim(gr_raw) == 2 else gr_raw
iA, iB, iC = _group_three_split(yr_raw, gr_raw, cfg.seed + 7, fracs=(0.50, 0.25, 0.25))
chk = extract_embeddings_from_model(model, norm(Xr_raw[iA[:256]]), 256)
dmax = float(np.abs(chk - eA[:256]).max())
res['check_partA_max_abs_diff'] = dmax
if not (np.array_equal(yr_raw[iA], yA) and np.array_equal(yr_raw[iC], yC) and dmax < 1e-3):
    raise SystemExit(f'CHECK FAILED: Part A reload does not match Phase 3 (label match '
                     f'{np.array_equal(yr_raw[iA], yA)}, max emb diff {dmax:.2e})')
cplx = lambda X: X[..., 0] + 1j * X[..., 1]
RA = cplx(Xr_raw[iA]); RA /= np.linalg.norm(RA, axis=1, keepdims=True)
ref = RA[0]
RA = RA * np.exp(-1j * np.angle(RA @ np.conj(ref)))[:, None]      # common phase removed
Mraw = np.stack([RA[yA == c].mean(0) for c in range(K)])
Mraw /= np.linalg.norm(Mraw, axis=1, keepdims=True)


def r_raw(X):
    Z = cplx(X); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    S = np.abs(Z @ np.conj(Mraw).T)                                  # phase-invariant cosine
    return S.argmax(1), 1 - S.max(1)


RC = Xr_raw[iC]; RCk, RCu = RC[~hard], RC[hard]                      # same pseudo-unknown mask
Xt_k, Xt_u = Xt[is_k], Xt[~is_k]
res['rules']['raw_centroid'] = metrics(r_raw, thresholds(r_raw, RCk, RCu), Xt_k, yk, Xt_u)
Mraw_pp = Mraw.copy()                          # centroids of the pipeline-input raw rule
print('raw_centroid', {k: round(v, 3) for k, v in res['rules']['raw_centroid'].items() if k in ('closed', 'auroc')},
      '| partA check max diff', f'{dmax:.1e}', flush=True)

# Same rule on the input WITHOUT slice preprocessing (no CFO removal, no per-slice
# normalisation): WiSig as released keeps the transmitter CFO on purpose.
from data_utils import load_day_raw
Xr_nopp, yr_nopp = load_day_raw(cfg, day_id=cfg.traj_day, device_ids=known)
Xt_nopp, yt_nopp = load_day_raw(cfg, day_id=cfg.test_day, device_ids=devs)
yt_nopp = np.array([sorted(devs)[int(v)] for v in yt_nopp])
if not (np.array_equal(np.asarray(yr_nopp).astype(int), yr_raw) and np.array_equal(yt_nopp, yt)):
    raise SystemExit('CHECK FAILED: unprocessed reload order differs from the processed load')
RA = cplx(Xr_nopp[iA]); RA /= np.linalg.norm(RA, axis=1, keepdims=True)
RA = RA * np.exp(-1j * np.angle(RA @ np.conj(RA[0])))[:, None]
Mraw = np.stack([RA[yA == c].mean(0) for c in range(K)]); Mraw /= np.linalg.norm(Mraw, axis=1, keepdims=True)
RC = Xr_nopp[iC]; tk = np.isin(yt_nopp, known)
res['rules']['raw_centroid_cfo'] = metrics(r_raw, thresholds(r_raw, RC[~hard], RC[hard]), Xt_nopp[tk], yk, Xt_nopp[~tk])
print('raw_centroid_cfo', {k: round(v, 3) for k, v in res['rules']['raw_centroid_cfo'].items() if k in ('closed', 'auroc')}, flush=True)

# ── tuned calibration (all rules, one procedure) ────────────────────────────
# Every rule keeps its own (pred, score). Pseudo-unknowns: leave-one-device-out
# -- a slice of device c scored by the same rule with device c removed (loco(c),
# see the rules section). Candidate thresholds are computed on Part C
# (known-accept quantiles 80/90/95 %, Youden's J = accept rate + pseudo-unknown
# rejection - 1); the method is chosen on Part B by open-set balanced accuracy
# 0.5 x (accepted and correct) + 0.5 x (pseudo-unknown rejected), ties -> q90.
# Part C sets the threshold, Part B chooses the method; no part does both.
# Test data only scores. Closed accuracy / AUROC are therefore identical to the
# threshold-free values of the rule (checked below).
def _chunked(f, X, bs=2048):
    outs = [f(X[i:i + bs]) for i in range(0, len(X), bs)]
    if isinstance(outs[0], tuple):
        return tuple(np.concatenate(o) for o in zip(*outs))
    return np.concatenate(outs)


def _loco_scores(loco, X, y):
    u = np.empty(len(y), np.float64)
    for c in np.unique(y):
        i = np.flatnonzero(y == c); u[i] = _chunked(loco(int(c)), X[i])
    return u


def tuned_calibration(rule, loco, B_in, yB_, C_in, yC_, T_k, yT, T_u):
    pB, sB = _chunked(rule, B_in); uB = _loco_scores(loco, B_in, yB_)
    _, sC = _chunked(rule, C_in); uC = _loco_scores(loco, C_in, yC_)
    cands = {f'q{q}': float(np.quantile(sC, q / 100)) for q in (80, 90, 95)}
    ts = np.unique(np.quantile(np.r_[sC, uC], np.linspace(0, 1, 400)))
    J = np.array([np.mean(sC <= t) + np.mean(uC > t) - 1 for t in ts])
    cands['youden'] = float(ts[int(J.argmax())])
    scoreB = {m: float(0.5 * np.mean((sB <= t) & (pB == yB_)) + 0.5 * np.mean(uB > t)) for m, t in cands.items()}
    best = max(scoreB, key=lambda m: (round(scoreB[m], 6), m == 'q90'))
    pk, sk = _chunked(rule, T_k); _, su = _chunked(rule, T_u)
    t = cands[best]; a = sk <= t
    return dict(method=best, thr=t, candidates=cands, scoreB=scoreB,
                closed=float(np.mean(pk == yT)),
                auroc=float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su])),
                accept=float(a.mean()), open=float(np.mean(pk[a] == yT[a])) if a.any() else float('nan'),
                unk_det=float(np.mean(su > t)), known_ok=float(np.mean(a & (pk == yT))))


def D_raw_factory(M):
    def D(X):
        Z = cplx(X); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        return 1 - np.abs(Z @ np.conj(M).T)
    return D


for name, (_r, _lo) in RULE_D.items():
    res['rules'][name]['tuned'] = tuned_calibration(_r, _lo, eB, yB, eC, yC, ek, yk, eu)
_Dr = D_raw_factory(Mraw_pp)
res['rules']['raw_centroid']['tuned'] = tuned_calibration(
    rule_from_D(_Dr), _D_loco(_Dr), Xr_raw[iB], yr_raw[iB], Xr_raw[iC], yr_raw[iC], Xt_k, yk, Xt_u)
_Dr = D_raw_factory(Mraw)                       # centroids of the CFO-kept raw rule (last built)
res['rules']['raw_centroid_cfo']['tuned'] = tuned_calibration(
    rule_from_D(_Dr), _D_loco(_Dr), Xr_nopp[iB], yr_raw[iB], Xr_nopp[iC], yr_raw[iC], Xt_nopp[tk], yk, Xt_nopp[~tk])
for name in list(RULE_D) + ['raw_centroid', 'raw_centroid_cfo']:
    t_, r_ = res['rules'][name]['tuned'], res['rules'][name]
    if abs(t_['closed'] - r_['closed']) > 1e-9 or abs(t_['auroc'] - r_['auroc']) > 1e-6:
        raise SystemExit(f'CHECK FAILED: {name} tuned protocol changed the rule '
                         f'(closed {t_["closed"]} vs {r_["closed"]}, auroc {t_["auroc"]} vs {r_["auroc"]})')
    print(f"tuned {name:15s} method={t_['method']:6s} accept={t_['accept']:.2f} unk_det={t_['unk_det']:.2f} "
          f"known_ok={t_['known_ok']:.2f}", flush=True)

# ── selection-weight sensitivity (Phase 3 table) ────────────────────────────
_selp = os.path.join(md, 'phase3_selection.json')
if os.path.exists(_selp):
    _sel = json.load(open(_selp)); sens = {}
    # same tie rule, defaults and simplicity order as Phase 3; only the weight changes
    _Pv = lambda r: 10 ** 9 if r['P'] == 'all' else r['P']
    for wgt in (0.3, 0.5, 0.7):
        rb = [dict(r, score=wgt * r['closed'] + (1 - wgt) * r['loco']) for r in _sel['bank']]
        ra = [dict(r, score=wgt * r['closed'] + (1 - wgt) * r['loco']) for r in _sel['alpha']]
        b = _S.choose(rb, lambda r: (r['P'], r['W'], r['decay']) == ('all', 1, 1.0),
                      lambda r: (r['W'], 1.0 - r['decay'], _Pv(r)))
        a_ = _S.choose(ra, lambda r: r['alpha'] == cfg.traj_ewma, lambda r: abs(1.0 - r['alpha']))
        sens[str(wgt)] = {'P': b['P'], 'W': b['W'], 'decay': b['decay'], 'alpha': a_['alpha']}
    res['selection_sensitivity'] = sens
    ch = _sel.get('chosen', {})
    res['selection_sensitivity_w05_matches_phase3'] = bool(ch) and all(
        sens['0.5'][k] == ch[k] for k in ('P', 'W', 'decay', 'alpha'))
    print('selection by weight on closed accuracy:', sens, flush=True)

# ── few-shot: the same target-session support added to each rule's references ─
# Support and query come from different groups (bursts / capture files) of the
# target session; 3 draws. TEMP-RFF adds the support as a new session; the
# static centroid and k-NN add it to their Part-A reference set; the
# equal-history controls static_window / knn_window add it to their own window
# reference set (the same earlier sessions as TEMP-RFF + Part A), so a
# difference to TEMP-RFF is not only more history. Thresholds are calibrated on
# the reference session exactly as in the zero-shot case.
def _fs_rules(sup):
    es, ys = ek[sup], yk[sup]
    tr = copy.deepcopy(traj); tr.update(day_id=cfg.test_day, emb=es, labels=ys)
    D_t = mp_distance(sel_dk + 1, sel_dec, tr)
    Ra, Ry = np.concatenate([eA, es]), np.concatenate([yA, ys])
    Cs = l2n(np.stack([Ra[Ry == c].mean(0) for c in range(K)]))
    D_s = lambda e: 1 - l2n(e) @ Cs.T
    Rn = l2n(Ra); nnf = NearestNeighbors(n_neighbors=KNN_SEL[0], metric='cosine').fit(Rn)
    kf = _knn_fn(KNN_SEL[0], KNN_SEL[1], nnf, Ry)
    # equal-history controls: the window rules' own reference embeddings (the
    # same earlier sessions as TEMP-RFF + Part A) plus the same support
    Xw, yw = _window_support(_SW[2]['W']); Xw, yw = np.concatenate([Xw, es]), np.concatenate([yw, ys])
    Cw = l2n(np.stack([Xw[yw == c].mean(0) for c in range(K)])); D_sw = lambda e: 1 - l2n(e) @ Cw.T
    Xq, yq = _window_support(_KW[2]['W']); Xq, yq = l2n(np.concatenate([Xq, es])), np.concatenate([yq, ys])
    nnq = NearestNeighbors(n_neighbors=20, metric='cosine').fit(Xq)
    kq = _knn_fn(_KW[2]['k'], _KW[2]['weighted'], nnq, yq)
    return {'temp_rff': (rule_from_D(D_t), _D_loco(D_t)), 'static': (rule_from_D(D_s), _D_loco(D_s)),
            'knn': (kf, _knn_loco(KNN_SEL[0], KNN_SEL[1], Rn, Ry)),
            'static_window': (rule_from_D(D_sw), _D_loco(D_sw)),
            'knn_window': (kq, _knn_loco(_KW[2]['k'], _KW[2]['weighted'], Xq, yq))}


fs = {n: {} for n in ('temp_rff', 'static', 'knn', 'static_window', 'knn_window')}
for k in KS:
    runs = {n: [] for n in fs}; k_eff = []
    for r in range(3):
        rng = np.random.default_rng(seed + 1000 * r)
        sup, qry = [], []
        for c in range(K):
            idx = np.flatnonzero(yk == c)
            if gk is not None:                       # group-disjoint support vs query
                g = gk[idx]; ug = rng.permutation(np.unique(g))
                ns = max(1, min(len(ug) - 1, int(round(0.5 * len(ug)))))
                m = np.isin(g, ug[:ns]); cand = rng.permutation(idx[m])
                sup += cand[:k].tolist(); qry += idx[~m].tolist()
            else:
                p_ = rng.permutation(idx); sup += p_[:k].tolist(); qry += p_[k:].tolist()
        sup, qry = np.array(sup), np.array(qry)
        k_eff.append(len(sup) / K)
        for n, (rule, lo) in _fs_rules(sup).items():
            mm = metrics(rule, thresholds(rule), ek[qry], yk[qry])
            mm['tuned'] = tuned_calibration(rule, lo, eB, yB, eC, yC, ek[qry], yk[qry], eu)
            runs[n].append(mm)
    for n in fs:
        rr = runs[n]
        fs[n][k] = {m: float(np.mean([x[m] for x in rr])) for m in ('closed', 'auroc')}
        fs[n][k]['k_eff'] = float(np.mean(k_eff))
        for p_ in ('fixed90', 'objective', 'tuned'):
            fs[n][k][p_] = {m: float(np.nanmean([x[p_][m] for x in rr])) for m in ('accept', 'open', 'unk_det', 'known_ok')}
    print('fewshot', k, {n: round(fs[n][k]['closed'], 3) for n in fs}, flush=True)
res['fewshot'] = fs

# ── pipeline-stage ablation (softmax and Part-A centroid) ────────────────────
# Part A of the reference session (same group split as every other rule)
Xr, yr = Xr_raw[iA], yr_raw[iA]
last = cfg.finetune_days[-1]
stages = {}
for tag, mf, nf in [('phase1', f'phase1_df_d{cfg.init_day}.keras', f'phase1_norm_d{cfg.init_day}.npz'),
                    ('phase2', f'phase2_ft_d{last}.keras', f'phase2_norm_d{last}.npz'),
                    ('phase3', f'phase3_bn_adapted_d{cfg.traj_day}.keras', 'phase3_norm_stats.npz')]:
    n2 = np.load(os.path.join(md, nf)); nz = lambda X: load_slice_IQ.apply_normalization(X, n2['mean'], n2['std'])
    m2 = tf.keras.models.load_model(os.path.join(md, mf), custom_objects={'L2Normalize': rf_models.L2Normalize})
    sm = float(np.mean(np.argmax(m2.predict(nz(Xt[is_k]), batch_size=256, verbose=0), 1) == yk))
    a_k, a_u, a_r = (l2n(extract_embeddings_from_model(m2, nz(X), 256)) for X in (Xt[is_k], Xt[~is_k], Xr))
    Cc = l2n(np.stack([a_r[yr == c].mean(0) for c in range(K)]))
    dk_, du_ = 1 - (a_k @ Cc.T).max(1), 1 - (a_u @ Cc.T).max(1)
    stages[tag] = dict(softmax=sm, centroid=float(np.mean((a_k @ Cc.T).argmax(1) == yk)),
                       auroc=float(roc_auc_score(np.r_[np.zeros(len(dk_)), np.ones(len(du_))], np.r_[dk_, du_])))
    print('stage', tag, {k: round(v, 3) for k, v in stages[tag].items()}, flush=True)
res['stages'] = stages

res['selection'] = SELECTION
import manifest
manifest.write_manifest(cfg, run_dir, 'eval', extra={'test_set': res['test_set']})
json.dump(res, open(os.path.join(run_dir, 'common_eval.json'), 'w'), indent=1)
print('WROTE', os.path.join(run_dir, 'common_eval.json'))
