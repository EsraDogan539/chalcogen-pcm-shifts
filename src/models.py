"""Isolated-to-PCM correction models and grouped cross-validation.

M0  E_PCM = E_iso
M1  E_PCM = E_iso + mean shift of the system (A or B) in the training data
M2  E_PCM = b0 + b1 E_HOMO_iso + b2 E_LUMO_iso + b3/n + b4 [B system]
M3  M2 + donor and acceptor chalcogen indicators (O as reference)

S1: all pairs, Huber loss; S2: rule-screened pairs, OLS; S3: S2 without
mixed-provenance series, OLS. Leave-one-XYX-out (both A and B withheld),
plus leave-one-donor-out / leave-one-acceptor-out for M1 and M2.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor

TARGETS = [('HOMO', 'E_HOMO_pcm', 'E_HOMO_iso'), ('LUMO', 'E_LUMO_pcm', 'E_LUMO_iso'), ('Eg', 'E_g_pcm', 'E_g_iso')]
X_M2 = ['E_HOMO_iso', 'E_LUMO_iso', 'inv_n', 'is_B']


def add_indicators(p):
    D = pd.get_dummies(p[['donor', 'acceptor']], drop_first=True).astype(float)
    return pd.concat([p, D], axis=1), X_M2 + list(D.columns)


def _regressor(fitter):
    return LinearRegression() if fitter == 'ols' else HuberRegressor(epsilon=1.35, max_iter=5000)


def _fit_predict(tr, te, tgt, base, model, fitter, X_M3):
    if model == 'M0':
        return te[base].values
    if model == 'M1':
        off = (tr[tgt] - tr[base]).groupby(tr.system).mean()
        return (te[base] + te.system.map(off)).values
    X = X_M2 if model == 'M2' else X_M3
    return _regressor(fitter).fit(tr[X], tr[tgt]).predict(te[X])


def cross_validate(p, mask, tgt, base, model, fitter, X_M3, group='XYX'):
    """Train on rows in `mask` outside the held-out group; predict every row of that group."""
    pred = np.full(len(p), np.nan)
    for g in p[group].unique():
        te = (p[group] == g).values
        tr = mask & ~te
        pred[te] = _fit_predict(p[tr], p[te], tgt, base, model, fitter, X_M3)
    return pred


def grouped_bootstrap(err, groups, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(groups); idx = {k: np.where(groups == k)[0] for k in u}
    means = [err[np.concatenate([idx[k] for k in rng.choice(u, len(u))])].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def run(p):
    p, X_M3 = add_indicators(p)
    settings = {'S1': (np.ones(len(p), bool), 'huber'), 'S2': (p.S2.values, 'ols'), 'S3': (p.S3.values, 'ols')}
    rows, preds, coefs = [], {}, []
    for s, (mask, fitter) in settings.items():
        for name, tgt, base in TARGETS:
            y = p[tgt].values
            for model in ['M0', 'M1', 'M2', 'M3']:
                pr = cross_validate(p, mask, tgt, base, model, fitter, X_M3)
                preds[(s, name, model)] = pr
                e = np.abs(pr[mask] - y[mask]); lo, hi = grouped_bootstrap(e, p.XYX.values[mask])
                rows.append(dict(setting=s, fitter=fitter, N=int(mask.sum()), target=name, model=model, cv='leave-one-XYX-out',
                                 MAE=e.mean(), CI_low=lo, CI_high=hi, RMSE=np.sqrt((e ** 2).mean()), max_error=e.max(),
                                 within_0_05=int((e <= 0.05).sum()), median_AE=np.median(e)))
            for model in ['M1', 'M2']:
                for grp in ['donor', 'acceptor']:
                    pr = cross_validate(p, mask, tgt, base, model, fitter, X_M3, grp)
                    rows.append(dict(setting=s, fitter=fitter, N=int(mask.sum()), target=name, model=model,
                                     cv=f'leave-one-{grp}-out', MAE=np.abs(pr[mask] - y[mask]).mean()))
            for model, X in [('M2', X_M2), ('M3', X_M3)]:
                m = _regressor(fitter).fit(p.loc[mask, X], p.loc[mask, tgt])
                coefs.append(dict(setting=s, target=name, model=model, intercept=m.intercept_, **dict(zip(X, m.coef_))))
    return p, pd.DataFrame(rows), pd.DataFrame(coefs), preds
