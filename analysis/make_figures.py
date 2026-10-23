"""Paper figures for Experiments 1-7 (read-only; PROTOCOL.md 16.1).

Sources: res/<ds>/shift*/seed*/common_eval.json (all runs, 5 seeds) and
analysis/cache/<ds>_shift*.npz (compute_cache.py, seed-42 run of every split).
Figures that use the cache describe the decision rules with their default settings
(MP-NN: all part-A embeddings, W=1; k-NN: k=5); this is stated in the captions.

    python analysis/make_figures.py            # writes analysis/figs/*.pdf
"""
import os, sys, glob, json, collections, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
# FIG_OUT / FIG_EXCLUDE: optional variant output (e.g. ICC figures without the linear probe:
# FIG_OUT=analysis/figs_icc FIG_EXCLUDE=probe); defaults give the full figure set.
FIG = os.path.join(ROOT, os.environ.get('FIG_OUT', os.path.join('analysis', 'figs'))); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.linewidth': 0.3, 'grid.alpha': 0.5,
                     'legend.frameon': False, 'legend.fontsize': 6.5})
DS = ['wisig', 'pycom', 'lora']; DSN = {'wisig': 'WiSig (equalized WiFi)', 'pycom': 'Pycom (raw WiFi)', 'lora': 'LoRa'}
RULES = [r for r in ['msp', 'probe', 'openmax', 'static', 'knn', 'temp_rff'] if r not in os.environ.get('FIG_EXCLUDE', '').split(',')]
RN = {'msp': 'MSP', 'probe': 'Probe', 'openmax': 'OpenMax', 'static': 'Static', 'knn': 'k-NN', 'temp_rff': 'MP-NN',
      'raw_centroid': 'Raw centroid'}
# fixed categorical order (never cycled); bank-based rules in the blue/green pair
COL = {'msp': '#8c8c8c', 'probe': '#b07aa1', 'openmax': '#e15759', 'static': '#f28e2b', 'knn': '#4e79a7',
       'temp_rff': '#59a14f', 'raw_centroid': '#76b7b2'}
W1, W2 = 3.45, 7.16                                    # IEEE column / page width (inches)


def runs(ds):
    out = []
    for f in sorted(glob.glob(f'res/{ds}/shift*/seed*/common_eval.json')):
        if '.failed' in f: continue
        A = json.load(open(f.replace('common_eval.json', 'audit.json')))
        if A.get('all_pass'): out.append(json.load(open(f)))
    return out


R = {ds: runs(ds) for ds in DS}
save = lambda fig, n: (fig.tight_layout(rect=(0, 0, 1, 0.9 if fig.legends else 1)), fig.savefig(os.path.join(FIG, n), bbox_inches='tight'), plt.close(fig), print('wrote', n))


# ---------- E1b: AUROC per device split ----------
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.1), sharey=True)
for ax, ds in zip(axs, DS):
    by = collections.defaultdict(list)
    for E in R[ds]: by[E['shift']].append(E)
    sh = sorted(by); x = np.arange(len(sh)); w = 0.8 / (len(RULES) + 1)
    for i, r in enumerate(RULES + ['raw_centroid']):
        v = [np.mean([E['rules'][r]['auroc'] for E in by[s]]) for s in sh]
        ax.bar(x + (i - len(RULES) / 2) * w, v, w * 0.9, color=COL[r], label=RN[r])
    ax.axhline(0.5, ls=':', lw=0.7, color='k'); ax.set_xticks(x); ax.set_xticklabels([str(s) for s in sh])
    ax.set_xlabel('device split'); ax.set_title(DSN[ds]); ax.set_ylim(0, 1)
axs[0].set_ylabel('AUROC (mean over 5 seeds)')
fig.legend(*axs[0].get_legend_handles_labels(), ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=6)
fig.subplots_adjust(top=0.78)
save(fig, 'E1b_split_auroc.pdf')

# ---------- E2: contribution of each phase ----------
fig, axs = plt.subplots(1, 2, figsize=(W2, 1.9))
st = ['phase1', 'phase2', 'phase3']; lab = ['Phase 1', 'Phase 2', 'Phase 3 (BN)', 'Full (MP-NN)']
for ax, met, yl in zip(axs, ['centroid', 'auroc'], ['closed-set accuracy', 'AUROC']):
    x = np.arange(4); w = 0.26
    for i, ds in enumerate(DS):
        v = [np.mean([E['stages'][s][met] for E in R[ds]]) for s in st]
        v.append(np.mean([E['rules']['temp_rff']['closed' if met == 'centroid' else 'auroc'] for E in R[ds]]))
        e = [np.std([E['stages'][s][met] for E in R[ds]], ddof=1) for s in st] + \
            [np.std([E['rules']['temp_rff']['closed' if met == 'centroid' else 'auroc'] for E in R[ds]], ddof=1)]
        ax.bar(x + (i - 1) * w, v, w * 0.9, yerr=e, capsize=1.5, error_kw={'lw': 0.5},
               color=['#4e79a7', '#f28e2b', '#59a14f'][i], label=DSN[ds])
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel(yl); ax.set_ylim(0, 1)
axs[1].axhline(0.5, ls=':', lw=0.7, color='k'); fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02))
save(fig, 'E2_phases.pdf')

# ---------- E3b: window W ----------
fig, ax = plt.subplots(figsize=(W1, 1.7))
for i, ds in enumerate(DS):
    Ws = sorted({int(k[len('temp_rff_W'):]) for E in R[ds] for k in E['rules'] if k.startswith('temp_rff_W')})
    v = [np.mean([E['rules'][f'temp_rff_W{w}']['auroc'] for E in R[ds]]) for w in Ws]
    ax.plot(Ws, v, 'o-', ms=3, color=['#4e79a7', '#f28e2b', '#59a14f'][i], label=DSN[ds])
ax.set_xlabel('window W (enrollment days whose prototypes are used)'); ax.set_ylabel('AUROC'); ax.legend()
save(fig, 'E3b_window.pdf')

# ---------- E4c: score shift (fixed90 accept on the test day) ----------
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.7), sharey=True)
for ax, ds in zip(axs, DS):
    v = [np.mean([E['rules'][r]['fixed90']['accept'] for E in R[ds]]) for r in RULES]
    s = [np.std([E['rules'][r]['fixed90']['accept'] for E in R[ds]], ddof=1) for r in RULES]
    ax.bar([RN[r] for r in RULES], v, yerr=s, capsize=1.5, error_kw={'lw': 0.5}, color=[COL[r] for r in RULES])
    ax.axhline(0.9, ls='--', lw=0.7, color='k'); ax.set_title(DSN[ds]); ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45)
axs[0].set_ylabel('known accept rate on test day\n(target 0.90 on reference day)')
save(fig, 'E4c_threshold_shift.pdf')

# ---------- E5: few-shot re-enrollment ----------
fig, axs = plt.subplots(2, 3, figsize=(W2, 3.0), sharex=True)
for j, ds in enumerate(DS):
    for i, met in enumerate(['closed', 'auroc']):
        ax = axs[i, j]
        for r in ['temp_rff', 'knn', 'static']:
            ks = sorted({int(k) for E in R[ds] for k in E['fewshot'][r]})
            v = [np.mean([E['fewshot'][r][str(k)][met] for E in R[ds] if str(k) in E['fewshot'][r]]) for k in ks]
            v0 = np.mean([E['rules'][r][met] for E in R[ds]])
            ax.plot([0.4] + ks, [v0] + v, 'o-', ms=2.5, color=COL[r], label=RN[r])
        ax.set_xscale('log'); ax.set_ylabel('closed-set accuracy' if met == 'closed' else 'AUROC')
        if i == 0: ax.set_title(DSN[ds])
        ax.set_xticks([0.4, 1, 10, 100]); ax.set_xticklabels(['0', '1', '10', '100']); ax.minorticks_off()
        if i == 1: ax.set_xlabel('labeled test-day samples per device')
axs[0, 0].legend()
save(fig, 'E5_fewshot.pdf')

# ---------- E6b: operating points (tuned thresholds) ----------
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
for ax, ds in zip(axs, DS):
    for r in RULES + ['raw_centroid']:
        a = np.mean([E['rules'][r]['tuned']['accept'] for E in R[ds]]); u = np.mean([E['rules'][r]['tuned']['unk_det'] for E in R[ds]])
        ax.scatter(a, u, s=22, color=COL[r], label=RN[r], zorder=3, edgecolor='white', linewidth=0.6)
    ax.scatter(1, 1, marker='+', s=40, color='k'); ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_title(DSN[ds]); ax.set_xlabel('known accept rate')
axs[0].set_ylabel('unknown-device detection'); axs[0].legend(fontsize=5.5, loc='lower left')
save(fig, 'E6b_operating_points.pdf')


# ---------- figures from the inference cache ----------
l2n = lambda e: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def scores(c):
    """(pred, rejection score) of MP-NN (all part-A embeddings), static centroid, k-NN (k=5), MSP."""
    known = sorted(c['known'].tolist()); e = l2n(c['e']); eA = l2n(c['eA']); yA = c['yA']
    out = {}
    S = e @ eA.T
    best = np.full((len(e), len(known)), -1.0); topk = np.zeros((len(e), len(known)))
    for i in range(len(known)):
        s = S[:, yA == i]; best[:, i] = s.max(1); topk[:, i] = np.sort(s, 1)[:, -5:].mean(1)
    out['temp_rff'] = (best.argmax(1), 1 - best.max(1))
    out['knn'] = (topk.argmax(1), 1 - topk.max(1))
    C = l2n(np.stack([eA[yA == i].mean(0) for i in range(len(known))])); sc = e @ C.T
    out['static'] = (sc.argmax(1), 1 - sc.max(1))
    z = c['logits'] - c['logits'].max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    out['msp'] = (p.argmax(1), 1 - p.max(1))
    return known, out


def cache(ds):
    return {int(f.split('shift')[1].split('.')[0]): dict(np.load(f)) for f in sorted(glob.glob(f'analysis/cache/{ds}_shift*.npz'))}


C = {ds: cache(ds) for ds in DS}
have = [ds for ds in DS if C[ds]]
print('cache available for', have)

# E1a: ROC curves (split 0, seed 42)
fig, axs = plt.subplots(1, len(have), figsize=(W2 * len(have) / 3, 2.1), squeeze=False)
for ax, ds in zip(axs[0], have):
    c = C[ds][min(C[ds])]; known, sc = scores(c); isu = np.isin(c['y'], c['unknown'])
    for r in ['msp', 'static', 'knn', 'temp_rff']:
        fpr, tpr, _ = roc_curve(isu, sc[r][1]); ax.plot(fpr, tpr, color=COL[r], lw=1.2,
                                                        label=f'{RN[r]} ({roc_auc_score(isu, sc[r][1]):.2f})')
    ax.plot([0, 1], [0, 1], ':', color='k', lw=0.7); ax.set_title(f'{DSN[ds]}, split {min(C[ds])}, seed 42', fontsize=7)
    ax.set_xlabel('known rejected (FPR)'); ax.set_ylabel('unknown rejected (TPR)'); ax.legend(loc='lower right')
save(fig, 'E1a_roc.pdf')

# E3a: bank size P (AUROC with P k-means prototypes of part A per device, W=1)
from sklearn.cluster import MiniBatchKMeans
fig, ax = plt.subplots(figsize=(W1, 1.8))
for i, ds in enumerate(have):
    Ps = [1, 5, 20, 50, 100, 'all']; v = []
    for P in Ps:
        a = []
        for sh, c in C[ds].items():
            e = l2n(c['e']); eA = l2n(c['eA']); yA = c['yA']; K = len(c['known']); best = np.full((len(e), K), -1.0)
            for k in range(K):
                X = eA[yA == k]
                B = X if P == 'all' or P >= len(X) else l2n(MiniBatchKMeans(P, random_state=0, n_init=3).fit(X).cluster_centers_)
                best[:, k] = (e @ B.T).max(1)
            a.append(roc_auc_score(np.isin(c['y'], c['unknown']), 1 - best.max(1)))
        v.append(np.mean(a))
    ax.plot(range(len(Ps)), v, 'o-', ms=3, color=['#4e79a7', '#f28e2b', '#59a14f'][DS.index(ds)], label=DSN[ds])
ax.set_xticks(range(6)); ax.set_xticklabels(['1', '5', '20', '50', '100', 'all']); ax.set_xlabel('prototypes per device P')
ax.set_ylabel('AUROC (seed 42, all splits)'); ax.legend()
save(fig, 'E3a_bank_size.pdf')

# E4a: device trajectories (PCA of daily means, split 0)
from sklearn.decomposition import PCA
fig, axs = plt.subplots(1, len(have), figsize=(W2 * len(have) / 3, 2.2), squeeze=False)
for ax, ds in zip(axs[0], have):
    c = C[ds][min(C[ds])]
    if 'traj_e' not in c: continue
    te, ty, td = l2n(c['traj_e']), c['traj_y'], c['traj_day']; days = sorted(set(td))
    M = {(d, k): te[(ty == k) & (td == d)].mean(0) for d in days for k in sorted(set(ty))}
    pca = PCA(2).fit(np.stack(list(M.values())))
    move = {k: np.linalg.norm(M[(days[-1], k)] - M[(days[0], k)]) for k in sorted(set(ty))}
    top = sorted(move, key=move.get)[-5:]
    for k in sorted(set(ty)):
        P_ = pca.transform(np.stack([M[(d, k)] for d in days]))
        ax.plot(P_[:, 0], P_[:, 1], '-', lw=1.2 if k in top else 0.6, color=None if k in top else '#c0c0c0',
                marker='o', ms=2, label=f'dev {k}' if k in top else None)
        ax.annotate('', P_[-1], P_[-2], arrowprops=dict(arrowstyle='->', lw=0.6, color='k' if k in top else '#c0c0c0'))
    ax.set_title(f'{DSN[ds]}\ndevice means, days {days[0]}-{days[-1]}', fontsize=7); ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2'); ax.legend(fontsize=5)
save(fig, 'E4a_trajectories.pdf')

# E4b: t-SNE of test-day embeddings, Phase-1 model vs. full TEMP-RFF (split 0)
from sklearn.manifold import TSNE
fig, axs = plt.subplots(len(have), 2, figsize=(W1, 1.7 * len(have)), squeeze=False)
for row, ds in zip(axs, have):
    c = C[ds][min(C[ds])]
    if 'tsne_p1_e' not in c: continue
    for ax, key, t in zip(row, ['tsne_p1', 'tsne_full'], ['Phase-1 model', 'full TEMP-RFF']):
        e, y = l2n(c[key + '_e']), c[key + '_y']; Z = TSNE(2, random_state=0, init='pca', perplexity=30).fit_transform(e)
        unk = np.isin(y, c['unknown'])
        ax.scatter(Z[~unk, 0], Z[~unk, 1], s=1, c=y[~unk], cmap='tab20', alpha=0.7)
        ax.scatter(Z[unk, 0], Z[unk, 1], s=1, c='k', alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{DSN[ds]}: {t}', fontsize=6); ax.grid(False)
save(fig, 'E4b_tsne.pdf')

# E6a: threshold sweep of MP-NN (split 0)
fig, axs = plt.subplots(1, len(have), figsize=(W2 * len(have) / 3, 1.9), squeeze=False)
for ax, ds in zip(axs[0], have):
    c = C[ds][min(C[ds])]; known, sc = scores(c); isu = np.isin(c['y'], c['unknown']); s = sc['temp_rff'][1]
    eC = l2n(c['eC']); eA = l2n(c['eA']); sC = 1 - (eC @ eA.T).max(1); thr90 = np.quantile(sC, 0.9)
    t = np.linspace(0, np.quantile(s, 0.999), 200)
    ax.plot(t, [(s[~isu] <= x).mean() for x in t], color=COL['temp_rff'], label='known accepted')
    ax.plot(t, [(s[isu] > x).mean() for x in t], '--', color=COL['openmax'], label='unknown detected')
    ax.axvline(thr90, ls=':', color='k', lw=0.7); ax.set_title(f'{DSN[ds]}, split {min(C[ds])}, seed 42', fontsize=7)
    ax.set_xlabel('rejection threshold (1 - q.p)'); ax.set_ylabel('rate'); ax.legend()
save(fig, 'E6a_threshold_sweep.pdf')

# E7: confusion matrix (known devices, all splits pooled) and per-device results
for ds in have:
    allk = sorted({int(v) for c in C[ds].values() for v in c['y']})
    cm = np.zeros((len(allk), len(allk))); dev_acc = collections.defaultdict(list); dev_auc = {}
    for sh, c in C[ds].items():
        known, sc = scores(c); pred, s = sc['temp_rff']; isu = np.isin(c['y'], c['unknown'])
        for yt, pk in zip(c['y'][~isu], pred[~isu]):
            cm[allk.index(int(yt)), allk.index(int(known[pk]))] += 1
            dev_acc[int(yt)].append(int(yt) == int(known[pk]))
        for u in c['unknown']:
            m = (c['y'] == u) | ~isu; dev_auc[int(u)] = roc_auc_score((c['y'][m] == u), s[m])
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={'width_ratios': [1, 1.4]})
    im = axs[0].imshow(cmn, cmap='Blues', vmin=0, vmax=1); axs[0].grid(False)
    axs[0].set_xticks(range(len(allk))); axs[0].set_xticklabels(allk, fontsize=4.5); axs[0].set_yticks(range(len(allk))); axs[0].set_yticklabels(allk, fontsize=4.5)
    axs[0].set_xlabel('predicted device'); axs[0].set_ylabel('true device'); axs[0].set_title(f'{DSN[ds]}: known devices (all splits)')
    x = np.arange(len(allk)); w = 0.4
    axs[1].bar(x - w / 2, [np.mean(dev_acc[d]) if dev_acc[d] else 0 for d in allk], w, color=COL['temp_rff'], label='accuracy when known')
    axs[1].bar(x + w / 2, [dev_auc.get(d, 0) for d in allk], w, color=COL['openmax'], label='AUROC when unknown')
    axs[1].axhline(0.5, ls=':', lw=0.7, color='k'); axs[1].set_xticks(x); axs[1].set_xticklabels(allk, fontsize=5)
    axs[1].set_xlabel('device'); axs[1].set_ylim(0, 1.05); axs[1].legend(fontsize=5.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, f'E7_devices_{ds}.pdf')
