"""How many PCM calculations are needed to build the correction?

For k = 2..15 training combinations (each combination = A and B system, n = 1-6),
random subsets are drawn, the model is fitted on their pairs and the PCM energies of
all remaining combinations are predicted (errors evaluated on the screened
pairs, setting S3, of the withheld combinations). Repeated `n_rep` times per k.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor

X_M2 = ['E_HOMO_iso', 'E_LUMO_iso', 'inv_n', 'is_B']
TARGETS = [('HOMO', 'E_HOMO_pcm', 'E_HOMO_iso'), ('LUMO', 'E_LUMO_pcm', 'E_LUMO_iso'), ('Eg', 'E_g_pcm', 'E_g_iso')]


def _fit(fitter):
    return LinearRegression() if fitter == 'ols' else HuberRegressor(epsilon=1.35, max_iter=5000)


def run(p, X_M3, setting='S3', fitter='ols', n_rep=300, seed=1, eval_on='S3'):
    rng = np.random.default_rng(seed)
    mask = p[setting].values if setting != 'S1' else np.ones(len(p), bool)
    combos = np.array(sorted(p.XYX.unique()))
    rows, diagnostics = [], []
    for k in range(2, 16):
        for rep in range(n_rep):
            train_c = rng.choice(combos, k, replace=False)
            tr = mask & p.XYX.isin(train_c).values
            te = p[eval_on].values & ~p.XYX.isin(train_c).values
            n_pcm = int((p.XYX.isin(train_c)).sum())          # PCM-optimized structures in training
            for name, tgt, base in TARGETS:
                y = p.loc[te, tgt].values
                res = {'M0': np.abs(p.loc[te, base].values - y).mean()}
                off = (p.loc[tr, tgt] - p.loc[tr, base]).groupby(p.loc[tr, 'system']).mean()
                if set(off.index) >= {'A', 'B'}:
                    res['M1'] = np.abs(p.loc[te, base].values + p.loc[te, 'system'].map(off).values - y).mean()
                X2 = p.loc[tr, X_M2]
                rank2 = np.linalg.matrix_rank(np.column_stack([np.ones(len(X2)), X2.values]))
                if tr.sum() > len(X_M2) + 1 and rank2 == len(X_M2) + 1:
                    res['M2'] = np.abs(_fit(fitter).fit(p.loc[tr, X_M2], p.loc[tr, tgt]).predict(p.loc[te, X_M2]) - y).mean()
                # M3 only when every chalcogen indicator varies in the training set
                Xt = p.loc[tr, X_M3]
                rank3 = np.linalg.matrix_rank(np.column_stack([np.ones(len(Xt)), Xt.values]))
                accepted = bool((Xt.std() > 0).all() and tr.sum() > len(X_M3) + 1 and rank3 == len(X_M3) + 1)
                diagnostics.append(dict(setting=setting, k=k, rep=rep, target=name,
                                        train_combinations=';'.join(sorted(train_c)), n_train=int(tr.sum()),
                                        rank=rank3, required_rank=len(X_M3) + 1, accepted=accepted))
                if accepted:
                    res['M3'] = np.abs(_fit(fitter).fit(Xt, p.loc[tr, tgt]).predict(p.loc[te, X_M3]) - y).mean()
                for m, v in res.items():
                    rows.append(dict(setting=setting, k=k, rep=rep, n_pcm=n_pcm, target=name, model=m, MAE=v))
    result = pd.DataFrame(rows)
    result.attrs['diagnostics'] = pd.DataFrame(diagnostics)
    return result


def summarize(lc):
    g = lc.groupby(['setting', 'target', 'model', 'k']).MAE
    s = pd.DataFrame({'median': g.median(), 'p10': g.quantile(0.10), 'p90': g.quantile(0.90), 'n': g.size()}).reset_index()
    return s
