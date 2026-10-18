"""Figures 2-6 of the manuscript."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555',
                     'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
CA, CB, GRAY, INK, RED = '#2a78d6', '#eb6834', '#9a9a96', '#222', '#d03b3b'
ORDER = ['O', 'S', 'Se', 'Te']


def fig2(p, out):
    stat_outlier = p.z_monomer.abs() > 3.5
    fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True)
    for j, (fam, col) in enumerate([('A', CA), ('B', CB)]):
        for i, (q, lab) in enumerate([('dE_HOMO', r'$\Delta E_{HOMO}$ (PCM − isolated), eV'),
                                      ('dE_LUMO', r'$\Delta E_{LUMO}$ (PCM − isolated), eV')]):
            a = ax[i, j]; g = p[p.system == fam]
            for x, s in g.groupby('XYX'):
                s = s.sort_values('n'); so = bool(stat_outlier[s.index].any())
                a.plot(s.n, s[q], color=INK if so else GRAY, lw=1.2 if so else 0.6, ls='--' if so else '-', alpha=1 if so else .7)
                fl = s[s.flag]; a.scatter(fl.n, fl[q], marker='x', s=22, color=RED, lw=1.1, zorder=4)
                if so and i == 0:
                    a.annotate(f'{fam}-{x}', (s.n.iloc[-1], s[q].iloc[-1]), xytext=(3, 0), textcoords='offset points', fontsize=6.5, va='center')
            m = g[g.S3].groupby('n')[q].mean(); a.plot(m.index, m.values, color=col, lw=2.2, marker='o', ms=4, zorder=5)
            a.axhline(0, color='#bbb', lw=0.5, zorder=0); a.set_xlim(0.7, 7.0); a.set_xticks(range(1, 7))
            if j == 0: a.set_ylabel(lab)
            if i == 0: a.set_title(f'System $\\mathit{{{fam}}}$', loc='left', fontsize=9)
            if i == 1: a.set_xlabel('Number of repeat units, $n$')
    h = [Line2D([], [], color=GRAY, lw=.8, label='individual systems'),
         Line2D([], [], color=CA, lw=2, marker='o', ms=4, label=r'$\mathit{A}$ mean (screened)'),
         Line2D([], [], color=CB, lw=2, marker='o', ms=4, label=r'$\mathit{B}$ mean (screened)'),
         Line2D([], [], color=INK, ls='--', lw=1.2, label='mixed-provenance series'),
         Line2D([], [], color=RED, marker='x', ls='', label='row flagged by audit')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f'{out}/Fig2.png'); fig.savefig(f'{out}/Fig2.tiff', dpi=600); plt.close(fig)


def fig3(p, out):
    s3 = p[p.S3]
    cmap = LinearSegmentedColormap.from_list('div', [CA, '#f0efec', RED])
    fig, ax = plt.subplots(2, 2, figsize=(5.6, 5.2), gridspec_kw=dict(hspace=0.45, wspace=0.15))
    for i, (q, lab) in enumerate([('dE_HOMO', r'$\Delta E_{HOMO}$'), ('dE_LUMO', r'$\Delta E_{LUMO}$')]):
        for j, fam in enumerate(['A', 'B']):
            M = s3[s3.system == fam].pivot_table(index='donor', columns='acceptor', values=q).reindex(index=ORDER, columns=ORDER)
            a = ax[i, j]; im = a.imshow(M.values, cmap=cmap, norm=TwoSlopeNorm(0, -0.55, 0.55))
            for r in range(4):
                for c in range(4):
                    v = M.values[r, c]
                    a.text(c, r, '—' if np.isnan(v) else f'{v:+.2f}', ha='center', va='center', fontsize=6.5,
                           color='white' if (not np.isnan(v) and abs(v) > 0.35) else INK)
            a.set_xticks(range(4), ORDER); a.set_yticks(range(4), ORDER)
            a.set_xlabel('Acceptor chalcogen (Y)'); a.set_ylabel('Donor chalcogen (X)' if j == 0 else '')
            a.set_title(f'System $\\mathit{{{fam}}}$, mean {lab}', fontsize=8.5, loc='left')
            for sp in a.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03); cb.set_label('PCM − isolated (eV)')
    fig.savefig(f'{out}/Fig3.png', bbox_inches='tight'); fig.savefig(f'{out}/Fig3.tiff', dpi=600, bbox_inches='tight'); plt.close(fig)


def fig4(p, preds, out):
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.35))
    for k, (a, (t, tgt, lab)) in enumerate(zip(ax, [('HOMO', 'E_HOMO_pcm', r'$E_{HOMO}$'), ('LUMO', 'E_LUMO_pcm', r'$E_{LUMO}$'), ('Eg', 'E_g_pcm', r'$E_g$')])):
        pr = preds[('S1', t, 'M3')]; y = p[tgt].values; ok = p.S3.values
        lo, hi = {'HOMO': (-6.0, -3.8), 'LUMO': (-3.8, -2.4), 'Eg': (0.7, 2.7)}[t]
        a.plot([lo, hi], [lo, hi], color='#999', lw=0.7, ls='--', zorder=0)
        for fam, col, mk in [('A', CA, 'o'), ('B', CB, 's')]:
            sel = ok & (p.system == fam).values
            a.scatter(y[sel], pr[sel], s=9, color=col, marker=mk, lw=0.3, edgecolor='white', label=f'System $\\mathit{{{fam}}}$')
        bad = ~ok & (y > lo - 0.5) & (y < hi + 0.5)
        a.scatter(y[bad], pr[bad], s=14, facecolor='none', edgecolor=INK, lw=0.7, marker='D', label='flagged / mixed provenance')
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
        a.set_xlabel(f'Reported PCM {lab} (eV)'); a.set_ylabel(f'Predicted PCM {lab} (eV)')
        a.set_title('(' + 'abc'[k] + ')', loc='left', fontsize=9)
    ax[0].legend(fontsize=6, frameon=False, loc='upper left', handletextpad=0.2)
    plt.tight_layout(); fig.savefig(f'{out}/Fig4.png'); fig.savefig(f'{out}/Fig4.tiff', dpi=600); plt.close(fig)


def optical_table(p, data_dir):
    o = pd.read_csv(f'{data_dir}/monomer_optical_thesis_table_4.4.csv')
    m1 = p[p.n == 1][['system', 'XYX', 'E_g_iso', 'dE_g', 'S3']]
    o = o.merge(m1, on=['system', 'XYX'])
    o['dE_opt'] = o.E_PCM_eV - o.E_iso_eV; o['f_ratio'] = o.f_PCM / o.f_iso
    o['band_change'] = (o.f_ratio > 2) | (o.f_ratio < 0.5) | (o.dE_opt.abs() > 0.25)
    o['consistent'] = ~o.band_change & o.S3
    return o


def fig6(o, out):
    c = o.consistent
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for fam, col, mk in [('A', CA, 'o'), ('B', CB, 's')]:
        g = o[o.system == fam]
        ax[0].scatter(g.E_g_iso, g.E_iso_eV, s=10 + 60 * g.f_iso, color=col, marker=mk, lw=0.4, edgecolor='white', label=f'System $\\mathit{{{fam}}}$')
    ax[0].plot([1.8, 2.7], [1.8, 2.7], ls='--', color='#999', lw=0.7)
    ax[0].set_xlabel(r'Isolated monomer $E_g$ (eV)'); ax[0].set_ylabel(r'Isolated monomer $E_{max}$ (eV)')
    ax[0].legend(fontsize=6.5, frameon=False, loc='lower left'); ax[0].set_title('(a) marker area = 10 + 60f', fontsize=7.5, loc='left')
    for fam, col, mk in [('A', CA, 'o'), ('B', CB, 's')]:
        g = o[c & (o.system == fam)]
        ax[1].scatter(g.dE_g, g.dE_opt, s=16, color=col, marker=mk, lw=0.4, edgecolor='white')
    b = np.polyfit(o.dE_g[c], o.dE_opt[c], 1); xx = np.linspace(-0.26, 0.14, 10)
    ax[1].plot(xx, np.polyval(b, xx), color='#777', lw=0.8)
    ax[1].axhline(0, color='#ccc', lw=.5); ax[1].axvline(0, color='#ccc', lw=.5)
    ax[1].set_xlabel(r'$\Delta E_g$ monomer (PCM − isolated, eV)'); ax[1].set_ylabel(r'$\Delta E_{max}$ (PCM − isolated, eV)')
    r = np.corrcoef(o.dE_g[c], o.dE_opt[c])[0, 1]
    ax[1].set_title(f'(b) screened pairs, n = {c.sum()}, r = {r:.2f}', fontsize=7.5, loc='left')
    plt.tight_layout(); fig.savefig(f'{out}/Fig7.png'); fig.savefig(f'{out}/Fig7.tiff', dpi=600); plt.close(fig)
    return dict(n_consistent=int(c.sum()), r=r, slope=b[0])


def fig_learning_curve(sm, out, setting='S3'):
    """Median and 10-90 % range of the test MAE vs number of training combinations."""
    s = sm[(sm.setting == setting) & (sm.k <= 14)]
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), sharex=True)
    styles = {'M1': (GRAY, 's'), 'M2': (CA, 'o'), 'M3': (CB, '^')}
    for k_, (a, (t, lab)) in enumerate(zip(ax, [('HOMO', r'$E_{HOMO}^{PCM}$'), ('LUMO', r'$E_{LUMO}^{PCM}$'), ('Eg', r'$E_{g}^{PCM}$')])):
        m0 = s[(s.target == t) & (s.model == 'M0')]['median'].median()
        a.axhline(m0, color=INK, ls=':', lw=0.8)
        if t != 'Eg': a.text(2, m0 * 1.08, 'M0 (no correction)', fontsize=6.3, ha='left', va='bottom')
        for m, (col, mk) in styles.items():
            g = s[(s.target == t) & (s.model == m) & (s.n >= 50)].sort_values('k')
            a.fill_between(g.k, g.p10, g.p90, color=col, alpha=0.18, lw=0)
            a.plot(g.k, g['median'], color=col, marker=mk, ms=3, lw=1.4, label=m)
        a.set_yscale('log'); a.set_ylim(0.008, 0.5)
        a.set_xlabel('Combinations with PCM data, $k$'); a.set_title('(' + 'abc'[k_] + ') ' + lab, loc='left', fontsize=8)
        a.set_xticks([2, 4, 6, 8, 10, 12, 14])
        a.axhline(0.05, color='#bbb', lw=0.5, ls='--')
    ax[0].set_ylabel('MAE on withheld combinations (eV)')
    ax[2].legend(fontsize=6.5, frameon=False, loc='lower left')
    plt.tight_layout(); fig.savefig(f'{out}/Fig5.png'); fig.savefig(f'{out}/Fig5.tiff', dpi=600); plt.close(fig)
