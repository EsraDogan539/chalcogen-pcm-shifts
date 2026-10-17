"""Reproduce the quinoxaline-family audit, transfer and pooled experiments.
Run after or independently of run_all.py. Raw inputs include source locators.
"""
from pathlib import Path
import sys,json,warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,HuberRegressor
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
threadpool_limits(limits=1)
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import audit,models
RES=ROOT/'results/extension';RES.mkdir(exist_ok=True)
FIG=ROOT/'figures'
old,prov=audit.load(ROOT/'data')
raw=pd.read_csv(ROOT/'data/emire_orbital_table_4.2.csv')
allraw=pd.concat([old,raw],ignore_index=True)
d=audit.flag_entries(allraw)
p=audit.make_pairs(d,prov)
p,x3old=models.add_indicators(p)
p['is_C']=(p.system=='C').astype(float)
x2=models.X_M2+['is_C'];x3=x3old+['is_C']
for col in ['E_HOMO_iso','E_LUMO_iso','inv_n']:p['C_x_'+col]=p.is_C*p[col]
x4=x3+['C_x_E_HOMO_iso','C_x_E_LUMO_iso','C_x_inv_n']
assert len(p)==287 and len(d)==575
p.to_csv(RES/'combined_pairs.csv',index=False)
d[d.flag].to_csv(RES/'flagged_entries.csv',index=False)
def reg(f):return LinearRegression() if f=='ols' else HuberRegressor(epsilon=1.35,max_iter=10000)
rank_rows=[]
def check(tr,x):
    z=np.column_stack([np.ones(len(tr)),tr[x]])
    rank=int(np.linalg.matrix_rank(z))
    assert rank==len(x)+1 and len(tr)>len(x)+1
    rank_rows.append(dict(N_train=len(tr),parameters_including_intercept=len(x)+1,rank=rank,features=';'.join(x),frameworks=';'.join(sorted(tr.system.unique())),training_combinations=';'.join(sorted(tr.XYX.unique()))))
def metric(y,pr,groups,**info):
    e=np.abs(pr-y);lo,hi=models.grouped_bootstrap(e,np.array(groups),n=5000)
    return dict(**info,N=len(y),MAE=float(e.mean()),median_AE=float(np.median(e)),RMSE=float(np.sqrt(np.mean(e**2))),bias=float(np.mean(pr-y)),CI_low=float(lo),CI_high=float(hi),max_error=float(e.max()),within_0_05=int((e<=.05).sum()))
rows=[];pred=p[['system','XYX','n','S3','flag','mixed_provenance']].copy()
for setting,mask,fitter in [('S1',np.ones(len(p),bool),'huber'),('S3',p.S3.values,'ols')]:
    # Direct transfer of the A/B model, without any C reference or C indicator.
    for mod,x in [('M2',models.X_M2),('M3',x3old)]:
        tr=p[mask&(p.system!='C')];te=p[p.system=='C'];check(tr,x)
        for target,tgt,base in models.TARGETS:
            pr=reg(fitter).fit(tr[x],tr[tgt]).predict(te[x]);pred.loc[te.index,f'direct_{setting}_{mod}_{target}']=pr
            for test in ['all','screened']:
                m=np.ones(len(te),bool) if test=='all' else te.S3.values
                rows.append(metric(te[tgt].values[m],pr[m],te.XYX.values[m],experiment='direct_A_B_to_C',setting=setting,model=mod,target=target,test_set=test,family='C'))
    # Pooled group CV: same XYX in every framework withheld together.
    for mod,x in [('M0',None),('M1',None),('M2',x2),('M3',x3),('M4',x4)]:
        for target,tgt,base in models.TARGETS:
            pr=np.empty(len(p))
            for g in p.XYX.unique():
                te=p[p.XYX==g];tr=p[mask&(p.XYX!=g)]
                if mod=='M0':v=te[base].values
                elif mod=='M1':v=(te[base]+te.system.map((tr[tgt]-tr[base]).groupby(tr.system).mean())).values
                else:check(tr,x);v=reg(fitter).fit(tr[x],tr[tgt]).predict(te[x])
                pr[te.index]=v
            pred[f'pooled_{setting}_{mod}_{target}']=pr
            for family in ['ALL','A','B','C']:
                for test in ['all','screened']:
                    m=np.ones(len(p),bool) if family=='ALL' else (p.system==family).values
                    if test=='screened':m=m&p.S3.values
                    rows.append(metric(p[tgt].values[m],pr[m],p.XYX.values[m],experiment='pooled_leave_one_XYX_out',setting=setting,model=mod,target=target,test_set=test,family=family))
perf=pd.DataFrame(rows);perf.to_csv(RES/'performance.csv',index=False);pred.to_csv(RES/'predictions.csv',index=False)
coefs=[]
for setting,mask,fitter in [('S1',np.ones(len(p),bool),'huber'),('S3',p.S3.values,'ols')]:
 for mod,x in [('M2',x2),('M3',x3),('M4',x4)]:
  tr=p[mask];check(tr,x)
  for target,tgt,base in models.TARGETS:
   m=reg(fitter).fit(tr[x],tr[tgt]);coefs.append(dict(setting=setting,model=mod,target=target,intercept=float(m.intercept_),**dict(zip(x,m.coef_))))
pd.DataFrame(coefs).to_csv(RES/'pooled_coefficients.csv',index=False)
# Add a small number of C references to fixed screened A/B training data.
rng=np.random.default_rng(17);draws=[]
for k in [2,4,8,12]:
    for rep in range(100):
        chosen=rng.choice(p.XYX.unique(),k,replace=False)
        tr=p[p.S3&((p.system!='C')|p.XYX.isin(chosen))]
        te=p[p.S3&(p.system=='C')&~p.XYX.isin(chosen)]
        check(tr,x2)
        for target,tgt,base in models.TARGETS:
            pr=reg('ols').fit(tr[x2],tr[tgt]).predict(te[x2])
            draws.append(dict(k=k,rep=rep,target=target,model='M2',N_train_C=int((tr.system=='C').sum()),N_test_C=len(te),rank=len(x2)+1,MAE=float(np.mean(np.abs(pr-te[tgt]))),chosen=';'.join(chosen)))
cal=pd.DataFrame(draws);cal.to_csv(RES/'C_reference_draws.csv',index=False)
cal.groupby(['k','target']).MAE.agg(median='median',p10=lambda x:x.quantile(.1),p90=lambda x:x.quantile(.9)).reset_index().to_csv(RES/'C_reference_summary.csv',index=False)
pd.DataFrame(rank_rows).to_csv(RES/'design_rank_checks.csv',index=False)
# Optical data retain bands as listed, applying the same heuristic, separately by n.
o=pd.read_csv(ROOT/'data/emire_optical_tables_4.5_4.6.csv')
iso=o[o.environment=='isolated'].set_index(['XYX','n']);pcm=o[o.environment=='PCM'].set_index(['XYX','n'])
op=iso[['Emax','f','wavelength_nm']].join(pcm[['Emax','f','wavelength_nm']],lsuffix='_iso',rsuffix='_pcm').reset_index()
op=op.merge(p[p.system=='C'][['XYX','n','S3','dE_g']],on=['XYX','n'])
op['dEmax']=op.Emax_pcm-op.Emax_iso;op['ratio_f']=op.f_pcm/op.f_iso
op['band_screen']=(op.dEmax.abs()<=.25)&op.ratio_f.between(.5,2)
op.to_csv(RES/'optical_pairs.csv',index=False)
optrows=[]
for n in [1,6]:
    sub=op[op.n==n]
    for mode,keep in [('all',np.ones(len(sub),bool)),('orbital',sub.S3.values),('band',sub.band_screen.values),('both',(sub.S3&sub.band_screen).values)]:
        g=sub[keep];optrows.append(dict(n=n,screen=mode,N=len(g),r=float(g.dEmax.corr(g.dE_g)),median_optical_shift=float(g.dEmax.median())))
pd.DataFrame(optrows).to_csv(RES/'optical_correlations.csv',index=False)
# Targeted optical threshold sensitivity, n=1/6 kept separate.
sens=[]
for n in [1,6]:
 for cut in [.2,.25,.3,.4]:
  for ratio in [1.5,2,3]:
   g=op[(op.n==n)&op.S3&(op.dEmax.abs()<=cut)&op.ratio_f.between(1/ratio,ratio)]
   sens.append(dict(n=n,energy_cutoff=cut,ratio_limit=ratio,N=len(g),r=float(g.dEmax.corr(g.dE_g))))
pd.DataFrame(sens).to_csv(RES/'optical_sensitivity.csv',index=False)
summary={'entries':len(d),'pairs':len(p),'C_entries':len(raw),'C_pairs':int((p.system=='C').sum()),'C_flagged_pairs':int(p.loc[p.system=='C','flag'].sum()),'C_screened_pairs':int(p.loc[p.system=='C','S3'].sum()),'combined_screened_pairs':int(p.S3.sum()),'flagged_C':d[(d.system=='C')&d.flag][['XYX','n','environment','E_HOMO','E_LUMO','E_g','R1','R2','R3']].to_dict('records'),'mean_shifts_C':p[p.S3&(p.system=='C')][['dE_HOMO','dE_LUMO','dE_g']].mean().to_dict(),'range_Eg_C':{'iso_min':float(p.loc[p.system=='C','E_g_iso'].min()),'iso_max':float(p.loc[p.system=='C','E_g_iso'].max())}}
(RES/'summary.json').write_text(json.dumps(summary,indent=2))
# Figures 2, 3 and 6 for all three frameworks (overwrite the A/B-only Fig. 2-3 written by run_all.py)
import figures_three
figures_three.fig2(p, FIG); figures_three.fig3(p, FIG); figures_three.fig6(p, pred, FIG)
# Submission TIFFs: RGB, LZW-compressed
from PIL import Image
for f in sorted(FIG.glob('Fig*.tiff')):
    im = Image.open(f); dpi = im.info.get('dpi', (600, 600))
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[3] if im.mode == 'RGBA' else None); im = bg
    im.save(f, compression='tiff_lzw', dpi=dpi)
print(json.dumps(summary,indent=2))
print(perf[(perf.setting=='S3')&(perf.model=='M3')&(perf.family.isin(['ALL','C']))&(perf.test_set=='screened')][['experiment','family','target','N','MAE','median_AE']].to_string(index=False))
