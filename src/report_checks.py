"""Supplementary numerical checks and descriptive manuscript statistics."""
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import models


def run(p, o, coefs, preds, out):
    stats = {}
    for name, tgt, base in models.TARGETS:
        err = np.abs(preds[('S1', name, 'M3')] - p[tgt].values)
        stats[name] = dict(mean_all=float(err.mean()), median_all=float(np.median(err)),
                           mean_screened=float(err[p.S3].mean()),
                           mean_flagged_or_mixed=float(err[~p.S3].mean()),
                           within_0_05=int((err <= .05).sum()))
    stats['screened_counts'] = p[p.S3].groupby('system').size().to_dict()
    stats['screened_shift_HOMO_iso_r'] = float(p.loc[p.S3, ['dE_HOMO', 'E_HOMO_iso']].corr().iloc[0, 1])
    stats['optical_greater_than_gap_1_eV'] = int(((o.E_iso_eV-o.E_g_iso)>1).sum())
    strong=(o.f_iso>=.3)&(o.f_PCM>=.3)
    stats['strong_optical_N'] = int(strong.sum())
    stats['strong_optical_absolute_r'] = float(np.corrcoef(o.loc[strong,'E_g_iso'],o.loc[strong,'E_iso_eV'])[0,1])
    stats['M2_VIF']={}
    X=p.loc[p.S3, models.X_M2]
    for c in X:
        r2=LinearRegression().fit(X.drop(columns=c), X[c]).score(X.drop(columns=c), X[c])
        stats['M2_VIF'][c]=float(1/(1-r2))
    a=coefs[(coefs.setting=='S3')&(coefs.target=='HOMO')&(coefs.model=='M2')].iloc[0]
    stats['M2_midpoint_form'] = dict(intercept=float(a.intercept),
        midpoint=float(a.E_HOMO_iso-1+a.E_LUMO_iso),
        gap=float((a.E_LUMO_iso-(a.E_HOMO_iso-1))/2),
        inv_n=float(a.inv_n), framework_B=float(a.is_B))
    stats['gap_difference_prediction_MAE']={}
    for setting in ['S1','S3']:
        mask=p.S3
        diff=preds[(setting,'LUMO','M3')]-preds[(setting,'HOMO','M3')]
        stats['gap_difference_prediction_MAE'][setting]=dict(
            from_difference=float(np.abs(diff[mask]-p.loc[mask,'E_g_pcm']).mean()),
            direct=float(np.abs(preds[(setting,'Eg','M3')][mask]-p.loc[mask,'E_g_pcm']).mean()))
    with open(f'{out}/additional_manuscript_checks.json','w') as f:
        json.dump(stats,f,indent=2)
    return stats
