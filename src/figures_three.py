"""Figures 2, 3 and 6 for the three frameworks (A, B: benzochalcogendiazole; C: chalcogendiazoloquinoxaline)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures import CA, CB, GRAY, INK, RED, ORDER

CC = '#2e8b57'
COL = {'A': CA, 'B': CB, 'C': CC}


def fig2(p, out):
    stat_outlier = p.z_monomer.abs() > 3.5
    fig, ax = plt.subplots(2, 3, figsize=(6.9, 4.6), sharex=True, sharey='row')
    for j, fam in enumerate(['A', 'B', 'C']):
        for i, (q, lab) in enumerate([('dE_HOMO', r'$\Delta E_{HOMO}$ (PCM − isolated), eV'),
                                      ('dE_LUMO', r'$\Delta E_{LUMO}$ (PCM − isolated), eV')]):
            a = ax[i, j]; g = p[p.system == fam]
            for x, s in g.groupby('XYX'):
                s = s.sort_values('n'); so = bool(stat_outlier[s.index].any()) and fam != 'C'
                mp = bool(s.mixed_provenance.any())
                dashed = so or mp
                a.plot(s.n, s[q], color=INK if dashed else GRAY, lw=1.1 if dashed else 0.6, ls='--' if dashed else '-', alpha=1 if dashed else .7)
                fl = s[s.flag]; a.scatter(fl.n, fl[q], marker='x', s=20, color=RED, lw=1.0, zorder=4)
                if dashed and i == 0:
                    dy = {'TeTeTe': -5, 'SeSeSe': 4}.get(x, 0)
                    a.annotate(f'{fam}-{x}', (s.n.iloc[-1], s[q].iloc[-1]), xytext=(3, dy), textcoords='offset points', fontsize=6, va='center')
            m = g[g.S3].groupby('n')[q].mean(); a.plot(m.index, m.values, color=COL[fam], lw=2.2, marker='o', ms=4, zorder=5)
            if q == 'dE_LUMO':
                a.set_ylim(-0.62, 0.75)
            a.axhline(0, color='#bbb', lw=0.5, zorder=0); a.set_xlim(0.7, 7.2); a.set_xticks(range(1, 7))
            if j == 0: a.set_ylabel(lab)
            if i == 0: a.set_title(f'System $\\mathit{{{fam}}}$', loc='left', fontsize=9)
            if i == 1: a.set_xlabel('Number of repeat units, $n$')
    h = [Line2D([], [], color=GRAY, lw=.8, label='individual systems'),
         Line2D([], [], color=CA, lw=2, marker='o', ms=4, label=r'$\mathit{A}$ mean (screened)'),
         Line2D([], [], color=CB, lw=2, marker='o', ms=4, label=r'$\mathit{B}$ mean (screened)'),
         Line2D([], [], color=CC, lw=2, marker='o', ms=4, label=r'$\mathit{C}$ mean'),
         Line2D([], [], color=INK, ls='--', lw=1.1, label='mixed-provenance series'),
         Line2D([], [], color=RED, marker='x', ls='', label='row flagged by audit')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f'{out}/Fig2.png', dpi=300); fig.savefig(f'{out}/Fig2.tiff', dpi=600); plt.close(fig)


def fig3(p, out):
    s3 = p[p.S3]
    cmap = LinearSegmentedColormap.from_list('div', [CA, '#f0efec', RED])
    fig, ax = plt.subplots(2, 3, figsize=(6.9, 4.9), gridspec_kw=dict(hspace=0.5, wspace=0.12))
    for i, (q, lab) in enumerate([('dE_HOMO', r'$\Delta E_{HOMO}$'), ('dE_LUMO', r'$\Delta E_{LUMO}$')]):
        for j, fam in enumerate(['A', 'B', 'C']):
            M = s3[s3.system == fam].pivot_table(index='donor', columns='acceptor', values=q).reindex(index=ORDER, columns=ORDER)
            a = ax[i, j]; im = a.imshow(M.values, cmap=cmap, norm=TwoSlopeNorm(0, -0.55, 0.55))
            for r in range(4):
                for c in range(4):
                    v = M.values[r, c]
                    a.text(c, r, '—' if np.isnan(v) else f'{v:+.2f}', ha='center', va='center', fontsize=6.3,
                           color='white' if (not np.isnan(v) and abs(v) > 0.35) else INK)
            a.set_xticks(range(4), ORDER); a.set_yticks(range(4), ORDER if j == 0 else [''] * 4)
            a.set_xlabel('Acceptor chalcogen (Y)', fontsize=7.5); a.set_ylabel('Donor chalcogen (X)' if j == 0 else '', fontsize=7.5)
            a.set_title(f'System $\\mathit{{{fam}}}$, mean {lab}', fontsize=8, loc='left')
            for sp in a.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.set_label('PCM − isolated (eV)')
    fig.savefig(f'{out}/Fig3.png', bbox_inches='tight', dpi=300); fig.savefig(f'{out}/Fig3.tiff', bbox_inches='tight', dpi=600); plt.close(fig)


def fig6(p, pred, out):
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.35))
    g = p[p.system == 'C']
    for k, (a, (t, tgt, lab, lo, hi)) in enumerate(zip(ax, [('HOMO', 'E_HOMO_pcm', r'$E_{HOMO}$', -5.9, -4.2),
                                                         ('LUMO', 'E_LUMO_pcm', r'$E_{LUMO}$', -4.7, -3.3),
                                                         ('Eg', 'E_g_pcm', r'$E_g$', -0.05, 2.1)])):
        a.plot([lo, hi], [lo, hi], color='#999', lw=0.7, ls='--', zorder=0)
        a.scatter(g[tgt], pred.loc[g.index, f'direct_S3_M3_{t}'], s=11, facecolor='none', edgecolor=INK, lw=0.6, label='A/B model only')
        a.scatter(g[tgt], pred.loc[g.index, f'pooled_S3_M3_{t}'], s=10, color=CC, lw=0.3, edgecolor='white', label='pooled model')
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
        a.set_xlabel(f'Reported PCM {lab} (eV)'); a.set_ylabel(f'Predicted PCM {lab} (eV)')
        a.set_title('(' + 'abc'[k] + ')', loc='left', fontsize=9)
    ax[0].legend(fontsize=6, frameon=False, loc='lower right', handletextpad=0.2, borderaxespad=0.2)
    plt.tight_layout(); fig.savefig(f'{out}/Fig6.png', dpi=300); fig.savefig(f'{out}/Fig6.tiff', dpi=600); plt.close(fig)


if __name__ == '__main__':
    import sys, os
    root = sys.argv[1]
    p = pd.read_csv(os.path.join(root, 'results/extension/combined_pairs.csv'))
    pred = pd.read_csv(os.path.join(root, 'results/extension/predictions.csv'))
    out = os.path.join(root, 'figures')
    fig2(p, out); fig3(p, out); fig6(p, pred, out)
    print('figures 2, 3, 6 written')
