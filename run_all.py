"""Reproduce all numbers, tables and figures of the manuscript.

    python run_all.py

Outputs are written to results/ (CSV tables) and figures/ (PNG, plus Fig1 as TIFF/PDF/SVG).
The learning-curve step takes a few minutes.
"""
import os, sys, subprocess
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
# Avoid nested BLAS threading for these small regression problems.
threadpool_limits(limits=1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
import audit, models, figures, learning_curve

DATA, RES, FIG = (os.path.join(HERE, d) for d in ('data', 'results', 'figures'))
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# 1. audit and pairing
d, prov = audit.load(DATA)
d = audit.flag_entries(d)
p = audit.make_pairs(d, prov)
d[d.flag].to_csv(f'{RES}/table1_flagged_entries.csv', index=False)
print(f'entries: {len(d)}  pairs: {len(p)}  rule-flagged pairs: {p.flag.sum()}  '
      f'S2: {p.S2.sum()}  S3: {p.S3.sum()}')
print('statistical screen |z|>3.5:', sorted(p.loc[p.z_monomer.abs() > 3.5, ['system', 'XYX']].drop_duplicates().apply('-'.join, axis=1)))

# 2. mean shifts (Table 2)
s3 = p[p.S3]
t2 = s3.groupby(['n', 'system'])[['dE_HOMO', 'dE_LUMO', 'dE_g']].mean().unstack('system')
t2.loc['All'] = s3.groupby('system')[['dE_HOMO', 'dE_LUMO', 'dE_g']].mean().unstack()
t2.round(3).to_csv(f'{RES}/table2_mean_shifts.csv'); print(t2.round(2))

# 3. models (Tables 3 and 4)
p, perf, coefs, preds = models.run(p)
perf.to_csv(f'{RES}/table3_model_performance.csv', index=False)
coefs.to_csv(f'{RES}/table4_coefficients.csv', index=False)
main = perf[perf.cv == 'leave-one-XYX-out'].pivot_table(index=['setting', 'model'], columns='target', values='MAE')
print(main.round(3))
p.to_csv(f'{RES}/pairs_with_flags.csv', index=False)

# 4. figures
subprocess.run([sys.executable, os.path.join(HERE, 'src', 'fig1_structures.py'), FIG], check=True)
figures.fig2(p, FIG); figures.fig3(p, FIG); figures.fig4(p, preds, FIG)

# 5. learning curve: how many PCM calculations are needed (S3/OLS and S1/Huber, evaluated on screened pairs)
X_M3 = models.X_M2 + [c for c in p.columns if c.startswith(('donor_', 'acceptor_'))]
lc_s3 = learning_curve.run(p, X_M3, 'S3', 'ols', n_rep=200)
lc_s1 = learning_curve.run(p, X_M3, 'S1', 'huber', n_rep=60)
pd.concat([lc_s3.attrs['diagnostics'], lc_s1.attrs['diagnostics']]).to_csv(f'{RES}/learning_curve_rank_diagnostics.csv', index=False)
lc_s3.attrs = {}; lc_s1.attrs = {}
lc = pd.concat([lc_s3, lc_s1])
lc_sum = learning_curve.summarize(lc); lc_sum.to_csv(f'{RES}/learning_curve_summary.csv', index=False)
figures.fig_learning_curve(lc_sum, FIG)
print(lc_sum[(lc_sum.k.isin([4, 8])) & lc_sum.model.isin(['M2', 'M3'])].pivot_table(index=['setting', 'model', 'k'], columns='target', values='median').round(3))

# 6. monomer optical data
o = figures.optical_table(p, DATA); o.to_csv(f'{RES}/optical_monomer.csv', index=False)
print('optical:', figures.fig6(o, FIG))
import sensitivity
sensitivity.run(o, RES)
import report_checks
report_checks.run(p, o, coefs, preds, RES)

# 7. basis-set check of relative orbital energies (B3LYP/6-31G(d) vs LANL2DZ, Ref. 2019 Table 4)
b = pd.read_csv(f'{DATA}/basis_comparison_2019_table4.csv')
for a6, al, lab in [('H6', 'HL', 'HOMO'), ('L6', 'LL', 'LUMO'), ('E6', 'EL', 'Eg')]:
    print(f'basis check {lab}: r = {np.corrcoef(b[a6], b[al])[0, 1]:.3f}, mean offset = {(b[a6] - b[al]).mean():.2f} eV (n = {len(b)})')
print('done ->', RES, FIG)
