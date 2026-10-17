"""Source-data audit and isolated/PCM pairing.

Rules (fixed before any modelling):
  R1  |E_LUMO - E_HOMO - E_g| > 0.025 eV          (internal consistency)
  R2  PCM entry identical to the isolated entry     (copied row)
  R3  HOMO decreases or LUMO increases with n by
      more than 0.02 eV w.r.t. the last accepted entry (monotonicity)
A pair is flagged if either member violates a rule. Series whose isolated
values were adopted from earlier studies are marked by provenance, and a
robust z-score of the monomer HOMO shift is reported as an independent check.
"""
import re
import numpy as np
import pandas as pd


def load(data_dir):
    d = pd.read_csv(f'{data_dir}/orbital_energies_thesis_tables_4.2_4.3.csv')
    prov = pd.read_csv(f'{data_dir}/mixed_provenance_series.csv')
    return d, prov


def flag_entries(d):
    d = d.copy()
    d['R1'] = ((d.E_LUMO - d.E_HOMO) - d.E_g).abs() > 0.025
    iso = d[d.environment == 'isolated'].set_index(['system', 'XYX', 'n'])
    def same_as_iso(r):
        if r.environment != 'PCM' or (r.system, r.XYX, r.n) not in iso.index:
            return False
        i = iso.loc[(r.system, r.XYX, r.n)]
        return (i.E_HOMO, i.E_LUMO, i.E_g) == (r.E_HOMO, r.E_LUMO, r.E_g)
    d['R2'] = d.apply(same_as_iso, axis=1)
    d['R3'] = False
    for _, s in d.groupby(['environment', 'system', 'XYX']):
        ref = None
        for idx, r in s.sort_values('n').iterrows():
            if ref is not None and (r.E_HOMO < ref.E_HOMO - 0.02 or r.E_LUMO > ref.E_LUMO + 0.02):
                d.loc[idx, 'R3'] = True
                continue
            ref = r
    d['flag'] = d.R1 | d.R2 | d.R3
    return d


def make_pairs(d, prov):
    key = ['system', 'XYX', 'n']
    iso = d[d.environment == 'isolated'].set_index(key)
    pcm = d[d.environment == 'PCM'].set_index(key)
    cols = ['E_HOMO', 'E_LUMO', 'E_g', 'flag']
    p = iso[cols].join(pcm[cols], lsuffix='_iso', rsuffix='_pcm', how='inner').reset_index()
    p['flag'] = p.flag_iso | p.flag_pcm
    for q in ['E_HOMO', 'E_LUMO', 'E_g']:
        p['d' + q] = p[q + '_pcm'] - p[q + '_iso']
    tok = lambda x: re.findall('Te|Se|O|S', x)
    p['donor'] = [tok(x)[0] for x in p.XYX]
    p['acceptor'] = [tok(x)[1] for x in p.XYX]
    p['inv_n'] = 1 / p.n
    p['is_B'] = (p.system == 'B').astype(float)
    # provenance
    p['mixed_provenance'] = False
    for _, r in prov.iterrows():
        lo, hi = map(int, r.n_range.split('-'))
        m = (p.system == r.system) & (p.XYX == r.XYX) & p.n.between(lo, hi)
        p.loc[m, 'mixed_provenance'] = True
    # robust z of monomer HOMO shift within each system (independent check)
    p['z_monomer'] = np.nan
    mono = p[(p.n == 1) & ~p.flag]
    for fam, g in mono.groupby('system'):
        med = g.dE_HOMO.median(); mad = 1.4826 * np.median(np.abs(g.dE_HOMO - med))
        for x, v in zip(g.XYX, g.dE_HOMO):
            p.loc[(p.system == fam) & (p.XYX == x), 'z_monomer'] = (v - med) / mad
    p['S2'] = ~p.flag
    p['S3'] = ~p.flag & ~p.mixed_provenance
    return p
