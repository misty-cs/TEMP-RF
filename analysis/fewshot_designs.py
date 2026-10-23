"""Design variants of the few-shot figure (ICC Fig. 4), previews only.
Same data as make_figures.py E5 (official common_eval.json, audited runs).
Writes analysis/figs_preview/fewshot_D*.pdf/.png. 2026-10-23."""
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.argv = sys.argv[:1]
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import importlib.util
spec = importlib.util.spec_from_file_location('mf_data', os.path.join(here, 'make_figures.py'))
src = open(os.path.join(here, 'make_figures.py')).read().split('# ---------- E1b')[0]   # settings + data only
ns = {'__file__': os.path.join(here, 'make_figures.py')}; exec(compile(src, 'make_figures_head', 'exec'), ns)
R, DS, DSN, COL, W2 = ns['R'], ns['DS'], ns['DSN'], ns['COL'], ns['W2']
OUT = os.path.join(here, 'figs_preview')
RULES = [('temp_rff', 'MP-NN'), ('knn', 'k-NN'), ('static', 'Static centroid')]

def curve(ds, r, met):
    ks = sorted({int(k) for E in R[ds] for k in E['fewshot'][r]})
    per = [[E['fewshot'][r][str(k)][met] for E in R[ds] if str(k) in E['fewshot'][r]] for k in ks]
    v0 = [E['rules'][r][met] for E in R[ds]]
    return [0] + ks, [np.mean(v0)] + [np.mean(p) for p in per], [np.std(v0, ddof=1)] + [np.std(p, ddof=1) for p in per]

def save(fig, n):
    fig.savefig(os.path.join(OUT, n + '.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, n + '.png'), dpi=200, bbox_inches='tight'); plt.close(fig); print('wrote', n)

pos = lambda ks: [0.4 if k == 0 else k for k in ks]

# D1: AUROC only, one row, MP-NN emphasized with a +-1 sd band, direct labels, chance line
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9))
for ax, ds in zip(axs, DS):
    for r, lab in RULES:
        ks, m, s = curve(ds, r, 'auroc'); x = pos(ks)
        lw, al = (1.8, 1) if r == 'temp_rff' else (1.0, 0.75)
        ax.plot(x, m, '-o', ms=2.5, lw=lw, alpha=al, color=COL[r])
        if r == 'temp_rff': ax.fill_between(x, np.array(m) - s, np.array(m) + s, color=COL[r], alpha=0.12, lw=0)
        ax.annotate(lab, (x[-1], m[-1]), xytext=(3, 0), textcoords='offset points', fontsize=6, color=COL[r], va='center')
    ax.axhline(0.5, ls=':', lw=0.7, color='k'); ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([0.4, 1, 10, 100]); ax.set_xticklabels(['0', '1', '10', '100']); ax.set_xlim(0.3, 1500)
    ax.set_title(DSN[ds], fontsize=7); ax.set_xlabel('labeled test-day samples per device')
axs[0].set_ylabel('AUROC'); save(fig, 'fewshot_D1_auroc_row')

# D2: heatmap of AUROC (rules x n) per dataset with values
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.5))
for ax, ds in zip(axs, DS):
    M = []; 
    for r, lab in RULES:
        ks, m, s = curve(ds, r, 'auroc'); M.append(m)
    M = np.array(M); im = ax.imshow(M, cmap='Blues', vmin=0.4, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks], fontsize=5.5)
    ax.set_yticks(range(len(RULES))); ax.set_yticklabels([l for _, l in RULES], fontsize=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=4.8, color='white' if M[i, j] > 0.78 else 'k')
    ax.set_title(DSN[ds], fontsize=7); ax.set_xlabel('labeled samples per device', fontsize=6); ax.grid(False)
save(fig, 'fewshot_D2_heatmap')

# D3: dot (slope) chart at n = 0, 1, 5, 500, accuracy and AUROC, evenly spaced categories
sel = [0, 1, 5, 500]
fig, axs = plt.subplots(2, 3, figsize=(W2, 2.6), sharex=True)
for j, ds in enumerate(DS):
    for i, met in enumerate(['closed', 'auroc']):
        ax = axs[i, j]
        for r, lab in RULES:
            ks, m, s = curve(ds, r, met); d = dict(zip(ks, m))
            ax.plot(range(len(sel)), [d.get(k, np.nan) for k in sel], '-o', ms=3.5, lw=1, color=COL[r], label=lab, mec='white', mew=0.6)
        ax.set_xticks(range(len(sel))); ax.set_xticklabels([str(k) for k in sel])
        if i == 0: ax.set_title(DSN[ds], fontsize=7)
        if j == 0: ax.set_ylabel('accuracy' if met == 'closed' else 'AUROC')
        if i == 1: ax.set_xlabel('labeled samples per device')
fig.legend(*axs[0, 0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.04), fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, 'fewshot_D3_dots')

# D4: gain over n=0 (delta AUROC), MP-NN and k-NN, datasets as panels, zero line
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.8), sharey=True)
for ax, ds in zip(axs, DS):
    for r, lab in RULES[:2]:
        ks, m, s = curve(ds, r, 'auroc'); x = pos(ks)
        ax.plot(x[1:], np.array(m[1:]) - m[0], '-o', ms=2.5, color=COL[r], label=lab)
    ax.axhline(0, lw=0.7, color='k'); ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(['1', '10', '100']); ax.set_title(DSN[ds], fontsize=7)
    ax.set_xlabel('labeled samples per device')
axs[0].set_ylabel('AUROC gain over 0 labels'); axs[0].legend(fontsize=6); save(fig, 'fewshot_D4_gain')

# D5: current 2x3 layout, refined labels and line design
STY = {'temp_rff': dict(ls='-', marker='o', lw=1.6, ms=3.2), 'knn': dict(ls='--', marker='s', lw=1.1, ms=2.8),
       'static': dict(ls=':', marker='^', lw=1.1, ms=3.0)}
fig, axs = plt.subplots(2, 3, figsize=(W2, 2.9), sharex=True)
for j, ds in enumerate(DS):
    for i, met in enumerate(['closed', 'auroc']):
        ax = axs[i, j]
        for r, lab in RULES:
            ks, m, s = curve(ds, r, met)
            ax.plot(pos(ks), m, color=COL[r], label=lab, mec='white', mew=0.5, **STY[r])
        if met == 'auroc': ax.axhline(0.5, ls=(0, (1, 2)), lw=0.7, color='0.35')
        ax.set_xscale('log'); ax.minorticks_off(); ax.set_xlim(0.3, 750)
        ax.set_xticks([0.4, 1, 10, 100, 500]); ax.set_xticklabels(['0', '1', '10', '100', '500'])
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        from matplotlib.ticker import MultipleLocator
        lo, hi = ax.get_ylim(); stp = next(t for t in (0.01, 0.02, 0.05, 0.1, 0.2) if (hi - lo) / t <= 5.5)
        ax.yaxis.set_major_locator(MultipleLocator(stp)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=6.5, length=2.5, width=0.6, direction='out', pad=1.5)
        if i == 0: ax.set_title(DSN[ds], fontsize=7.5)
    axs[0, 0].set_ylabel('Accuracy'); axs[1, 0].set_ylabel('AUROC')
fig.supxlabel('Labeled test-day samples per device', fontsize=7.5, y=0.045)
fig.legend(*axs[0, 0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=7, handlelength=2.8)
fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=0.6); save(fig, 'fewshot_D5_refined')

# D6: line-design / color variants of D5 with more ticks
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
VARIANTS = {
 'V1_default':   {'temp_rff': ('#59a14f', '-', 'o', 1.6, 'full'), 'knn': ('#4e79a7', '--', 's', 1.1, 'full'), 'static': ('#f28e2b', ':', '^', 1.1, 'full')},
 'V2_okabe_ito': {'temp_rff': ('#0072B2', '-', 'o', 1.6, 'full'), 'knn': ('#D55E00', '--', 's', 1.1, 'full'), 'static': ('#009E73', ':', '^', 1.1, 'full')},
 'V3_grayscale': {'temp_rff': ('#000000', '-', 'o', 1.6, 'full'), 'knn': ('#6e6e6e', '--', 's', 1.1, 'none'), 'static': ('#a8a8a8', ':', '^', 1.1, 'none')},
 'V4_purple_teal': {'temp_rff': ('#6a3d9a', '-', 'o', 1.6, 'full'), 'knn': ('#1b9e77', '--', 'D', 1.1, 'none'), 'static': ('#b8860b', '-.', 'v', 1.1, 'full')},
 'V5_red_navy':  {'temp_rff': ('#c0392b', '-', 'o', 1.6, 'full'), 'knn': ('#1f3a5f', '--', 's', 1.1, 'none'), 'static': ('#7f8c8d', ':', '^', 1.1, 'full')},
}
for vname, sty in VARIANTS.items():
    fig, axs = plt.subplots(2, 3, figsize=(W2, 2.9), sharex=True)
    for j, ds in enumerate(DS):
        for i, met in enumerate(['closed', 'auroc']):
            ax = axs[i, j]
            for r, lab in RULES:
                c, ls, mk, lw, fill = sty[r]; ks, m, _ = curve(ds, r, met)
                ax.plot(pos(ks), m, color=c, ls=ls, marker=mk, lw=lw, ms=3.0, label=lab,
                        mfc=(c if fill == 'full' else 'white'), mec=(c if fill == 'none' else 'white'), mew=0.7)
            if met == 'auroc': ax.axhline(0.5, ls=(0, (1, 2)), lw=0.7, color='0.35')
            ax.set_xscale('log'); ax.minorticks_off(); ax.set_xlim(0.3, 750)
            ax.set_xticks([0.4, 1, 5, 10, 50, 100, 500]); ax.set_xticklabels(['0', '1', '5', '10', '50', '100', '500'])
            lo, hi = ax.get_ylim(); stp = next(t for t in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1) if (hi - lo) / t <= 7.5)
            ax.yaxis.set_major_locator(MultipleLocator(stp)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if stp < 0.01 or stp == 0.025 else '%.2f'))
            ax.tick_params(labelsize=6, length=2.5, width=0.6, direction='out', pad=1.5)
            if i == 0: ax.set_title(DSN[ds], fontsize=7.5)
        axs[0, 0].set_ylabel('Accuracy'); axs[1, 0].set_ylabel('AUROC')
    fig.supxlabel('Labeled test-day samples per device', fontsize=7.5, y=0.045)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=7, handlelength=2.8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=0.6); save(fig, 'fewshot_D6_' + vname)

# D7: V4 colors, evenly spaced x positions (one per measured n)
sty = VARIANTS['V4_purple_teal']
fig, axs = plt.subplots(2, 3, figsize=(W2, 2.9), sharex=True)
for j, ds in enumerate(DS):
    for i, met in enumerate(['closed', 'auroc']):
        ax = axs[i, j]
        for r, lab in RULES:
            c, ls, mk, lw, fill = sty[r]; ks, m, _ = curve(ds, r, met)
            ax.plot(range(len(ks)), m, color=c, ls=ls, marker=mk, lw=lw, ms=3.0, label=lab,
                    mfc=(c if fill == 'full' else 'white'), mec=(c if fill == 'none' else 'white'), mew=0.7)
        if met == 'auroc': ax.axhline(0.5, ls=(0, (1, 2)), lw=0.7, color='0.35')
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks]); ax.set_xlim(-0.4, len(ks) - 0.6)
        lo, hi = ax.get_ylim(); stp = next(t for t in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1) if (hi - lo) / t <= 7.5)
        ax.yaxis.set_major_locator(MultipleLocator(stp)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if stp < 0.01 or stp == 0.025 else '%.2f'))
        ax.tick_params(labelsize=6, length=2.5, width=0.6, direction='out', pad=1.5)
        if i == 0: ax.set_title(DSN[ds], fontsize=7.5)
    axs[0, 0].set_ylabel('Accuracy'); axs[1, 0].set_ylabel('AUROC')
fig.supxlabel('Labeled test-day samples per device', fontsize=7.5, y=0.045)
fig.legend(*axs[0, 0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=7, handlelength=2.8)
fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=0.6); save(fig, 'fewshot_D7_V4_even')
