"""Descriptive sensitivity of optical-shift correlation to screening thresholds.

Thresholds are heuristic checks of tabulated bands, not state assignments.
The selection uses the response itself; these correlations are exploratory.
"""
import numpy as np
import pandas as pd


def run(o, out):
    rows = []
    def record(label, mask, cutoff=np.nan, ratio=np.nan):
        a = o.loc[mask]
        rows.append(dict(selection=label, energy_cutoff_eV=cutoff,
                         oscillator_ratio_max=ratio, N=len(a),
                         r=np.corrcoef(a.dE_g, a.dE_opt)[0, 1],
                         median_optical_shift_eV=a.dE_opt.median()))
    record('all_pairs', np.ones(len(o), bool))
    record('orbital_S3_only', o.S3)
    record('band_screen_only', ~o.band_change, .25, 2)
    for cutoff in [.20, .25, .30, .40]:
        for ratio in [1.5, 2, 3]:
            ok = o.S3 & (o.dE_opt.abs() <= cutoff) & o.f_ratio.between(1 / ratio, ratio)
            record('orbital_S3_plus_band_screen', ok, cutoff, ratio)
    pd.DataFrame(rows).to_csv(f'{out}/optical_screen_sensitivity.csv', index=False)
