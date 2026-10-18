"""Build Supplementary Data S1 (Excel workbook) from the source data and the results of run_all.py/run_extension.py."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
import audit, models

D, R = os.path.join(HERE, 'data'), os.path.join(HERE, 'results')
d, prov = audit.load(D)
d = audit.flag_entries(d)
p = audit.make_pairs(d, prov)
p, perf, coefs, preds = models.run(p)

s1 = d.rename(columns={'environment': 'env', 'XYX': 'xyx', 'E_HOMO': 'H', 'E_LUMO': 'L', 'E_g': 'Eg'})[
    ['env', 'system', 'xyx', 'n', 'H', 'L', 'Eg', 'R1', 'R2', 'R3']]
s2 = p[['system', 'XYX', 'n', 'E_HOMO_iso', 'E_LUMO_iso', 'E_g_iso', 'E_HOMO_pcm', 'E_LUMO_pcm', 'E_g_pcm',
        'dE_HOMO', 'dE_LUMO', 'dE_g', 'flag', 'mixed_provenance']].copy()
for t in ['HOMO', 'LUMO', 'Eg']:
    s2[f'pred_PCM_{t}_M3_robust'] = pd.Series(preds[('S1', t, 'M3')], index=p.index).round(4)
s3 = pd.read_csv(os.path.join(D, 'monomer_optical_thesis_table_4.4.csv'))
s4 = pd.read_csv(os.path.join(D, 'emire_orbital_table_4.2.csv'))[['system', 'XYX', 'n', 'environment', 'E_HOMO', 'E_LUMO', 'E_g', 'source_table', 'source_print_page']]
cp = pd.read_csv(os.path.join(R, 'extension', 'combined_pairs.csv'))
pr = pd.read_csv(os.path.join(R, 'extension', 'predictions.csv'))
s5 = cp[['system', 'XYX', 'n', 'E_HOMO_iso', 'E_LUMO_iso', 'E_g_iso', 'E_HOMO_pcm', 'E_LUMO_pcm', 'E_g_pcm',
         'dE_HOMO', 'dE_LUMO', 'dE_g', 'flag', 'mixed_provenance', 'S3']].copy()
for t in ['HOMO', 'LUMO', 'Eg']:
    s5[f'pred_pooled_S1_M3_{t}'] = pr[f'pooled_S1_M3_{t}'].round(4)
    s5[f'pred_pooled_S3_M3_{t}'] = pr[f'pooled_S3_M3_{t}'].round(4)
    s5[f'pred_direct_AB_S3_M3_{t}'] = pr[f'direct_S3_M3_{t}'].round(4)
s6 = pd.read_csv(os.path.join(D, 'emire_optical_tables_4.5_4.6.csv'))[['system', 'XYX', 'n', 'environment', 'wavelength_nm', 'Emax', 'f', 'source_table', 'source_print_page']]
corr = pd.read_csv(os.path.join(D, 'source_corrections.csv'))
readme = pd.DataFrame([
    ('S1_entries', '383 entries of systems A and B (Özkılınç 2021, MSc thesis, Tables 4.2 and 4.3) with audit rules R1–R3'),
    ('S2_pairs_predictions', '191 A/B isolated/PCM pairs, shifts (PCM − isolated), audit flags and leave-one-combination-out predictions of the robust model M3'),
    ('S3_monomer_optical', 'A/B monomer TD-B3LYP/LANL2DZ bands (Özkılınç 2021, Table 4.4)'),
    ('S4_C_entries', '192 entries of system C (Uyanık 2021, MSc thesis, Table 4.2) with source page'),
    ('S5_all_pairs_predictions', 'All 287 pairs of A, B and C with flags and pooled/direct-transfer predictions of M3'),
    ('S6_C_optical', 'C monomer and hexamer TD-B3LYP/LANL2DZ bands (Uyanık 2021, Tables 4.5 and 4.6)'),
    ('S7_source_corrections', 'Documented correction of a sign typo in the source table'),
], columns=['Sheet', 'Content'])
out = os.path.join(HERE, 'supplementary', 'Supplementary_Data_S1.xlsx')
with pd.ExcelWriter(out) as w:
    for name, df in [('README', readme), ('S1_entries', s1), ('S2_pairs_predictions', s2), ('S3_monomer_optical', s3),
                     ('S4_C_entries', s4), ('S5_all_pairs_predictions', s5), ('S6_C_optical', s6), ('S7_source_corrections', corr)]:
        df.to_excel(w, sheet_name=name, index=False)
print('written', out)
