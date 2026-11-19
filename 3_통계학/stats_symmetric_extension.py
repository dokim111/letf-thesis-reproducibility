"""Reproduce symmetric downside/cubic comparisons without changing the models.

All four log models are refitted using mature training targets s+h <= t.
No additional significance tests, variable selection or forecast tuning is done.
The existing v0.2 semivariance MSEs form a numerical reproduction gate.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'symmetric_extension_results'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'krx_stock_daily_long_20000104_20261113.csv'
OLD = ROOT / 'graduation_extension_results/semivariance_log_results.csv'
START, LISTING, EVENT, EPS = 20150615, 20260527, 20260731, 1e-8
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
initial_hashes = {'data': sha(DATA), 'old_semivariance_results': sha(OLD)}
old = pd.read_csv(OLD)
df = pd.read_csv(DATA, dtype={'code': str}, encoding='utf-8-sig')
df['r'] = df.close / (df.close-df.change_from_base_price)-1
comparisons, correlations, forecasts, validation = [], [], [], []
pairs = [('M1','M2'), ('M1','MD'), ('MD','MDK'), ('M2','MDK'), ('M2','MD')]

def roll(x,w):
    return pd.Series(x).rolling(w).mean().to_numpy()

for stock, frame in df.groupby('stock'):
    frame=frame[frame.date>=START].reset_index(drop=True)
    r=frame.r.to_numpy(); dates=frame.date.to_numpy(); n=len(r)
    rv={w:roll(r*r,w) for w in (1,5,22)}
    d={w:roll(r*r*(r<0),w)/(rv[w]+EPS) for w in (1,5,22)}
    k={w:roll(r**3,w)/(rv[w]+EPS)**1.5 for w in (5,22)}
    X1=np.column_stack([np.ones(n)]+[np.log(rv[w]+EPS) for w in (1,5,22)]+[roll(r,22)])
    X2=np.column_stack([X1,k[5],k[22]])
    XD=np.column_stack([X1,d[1],d[5],d[22]])
    XDK=np.column_stack([XD,k[5],k[22]])
    designs={'M1':X1,'M2':X2,'MD':XD,'MDK':XDK}
    origin_by_h={h:np.arange(22+750+h,n-h) for h in (5,20)}
    pre_by_h={h:ids[dates[ids+h]<LISTING] for h,ids in origin_by_h.items()}
    corr_samples={
        'pre_common_forecast':np.intersect1d(pre_by_h[5],pre_by_h[20]),
        'pre_strict_h5':pre_by_h[5],
        'pre_strict_h20':pre_by_h[20],
        'pre_all_valid_features':np.flatnonzero((np.arange(n)>=21)&(dates<LISTING)),
    }
    for sample, ids in corr_samples.items():
        for w in (5,22):
            correlations.append(dict(stock=stock,sample=sample,w=w,n=len(ids),
                first_date=int(dates[ids[0]]),last_date=int(dates[ids[-1]]),
                pearson=float(np.corrcoef(d[w][ids],k[w][ids])[0,1])))
    daily=np.column_stack([beta*np.log1p(r)-np.log1p(beta*r) for beta in (2,-2)])
    event_index=int(np.flatnonzero(dates==EVENT)[0])
    for h, origins in origin_by_h.items():
        # Explicit forward sums independently check the old cumulative-sum target.
        target=np.full((n,2),np.nan)
        target[:n-h]=np.array([daily[s+1:s+h+1].sum(axis=0) for s in range(n-h)])
        assert np.nanmin(target)>0
        z=np.log(target+EPS)
        predictions={}
        for model,X in designs.items():
            pred=np.empty((len(origins),2))
            for j,t in enumerate(origins):
                training=slice(21,int(t-h+1))
                fit=np.linalg.lstsq(X[training],z[training],rcond=None)[0]
                pred[j]=X[t]@fit
            assert np.isfinite(pred).all()
            predictions[model]=pred
        event_affected=((origins<event_index)&(origins+h>=event_index))|((origins>=event_index)&(origins-21<=event_index))
        masks={
            'pre_strict':dates[origins+h]<LISTING,
            'post':dates[origins]>=LISTING,
            'post_no_event':(dates[origins]>=LISTING)&~event_affected,
        }
        for j,beta in enumerate((2,-2)):
            for sample,mask in masks.items():
                losses={model:float(np.mean((z[origins,j][mask]-pred[mask,j])**2)) for model,pred in predictions.items()}
                for reference,candidate in pairs:
                    gain=1-losses[candidate]/losses[reference]
                    comparisons.append(dict(stock=stock,h=h,beta=beta,sample=sample,n=int(mask.sum()),reference=reference,candidate=candidate,
                        comparison_type='nonnested' if (reference,candidate)==('M2','MD') else 'nested',
                        mse_reference=losses[reference],mse_candidate=losses[candidate],improvement=gain))
                    if (reference,candidate) in pairs[:3]:
                        match=old[(old.stock==stock)&(old.h==h)&(old.beta==beta)&(old['sample']==sample)&(old.narrower==reference)&(old.broader==candidate)]
                        assert len(match)==1
                        row=match.iloc[0]
                        mse_error=max(abs(losses[reference]-float(row.mse_narrow)),abs(losses[candidate]-float(row.mse_broad)))
                        gain_error=abs(gain-float(row.improvement))
                        assert max(mse_error,gain_error)<1e-10,(stock,h,beta,sample,reference,candidate,mse_error,gain_error)
                        validation.append(dict(stock=stock,h=h,beta=beta,sample=sample,reference=reference,candidate=candidate,mse_error=mse_error,improvement_error=gain_error))
            for model,pred in predictions.items():
                for j0,t in enumerate(origins):
                    forecasts.append(dict(stock=stock,h=h,beta=beta,date=int(dates[t]),end_date=int(dates[t+h]),model=model,
                        log_target=float(z[t,j]),log_prediction=float(pred[j0,j]),event_affected=bool(event_affected[j0])))
        print(stock,h,'completed',flush=True)

cmp=pd.DataFrame(comparisons);corr=pd.DataFrame(correlations)
cmp.to_csv(OUT/'comparisons.csv',index=False,encoding='utf-8-sig')
corr.to_csv(OUT/'feature_correlations.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(forecasts).to_csv(OUT/'forecasts_log.csv.gz',index=False,encoding='utf-8-sig',compression={'method':'gzip','mtime':0})
pd.DataFrame(validation).to_csv(OUT/'reproduction_check.csv',index=False,encoding='utf-8-sig')
expected=[
 ('삼성전자',2,5,-.28,.32,.60),('삼성전자',-2,5,-.30,.32,.62),
 ('삼성전자',2,20,-2.04,-.52,1.49),('삼성전자',-2,20,-1.69,.04,1.70),
 ('SK하이닉스',2,5,-1.09,-1.08,.01),('SK하이닉스',-2,5,-1.12,-1.07,.05),
 ('SK하이닉스',2,20,-1.17,-.57,.59),('SK하이닉스',-2,20,-1.18,-.58,.59),
]
max_feedback_difference=0.
for stock,beta,h,*values in expected:
    q=cmp[(cmp.stock==stock)&(cmp.beta==beta)&(cmp.h==h)&(cmp['sample']=='pre_strict')]
    for (ref,cand),value in zip([('MD','MDK'),('M2','MDK'),('M2','MD')],values):
        found=float(q[(q.reference==ref)&(q.candidate==cand)].iloc[0].improvement)*100
        max_feedback_difference=max(max_feedback_difference,abs(found-value))
        assert abs(found-value)<=.0051,(stock,beta,h,ref,cand,found,value)
report={
 'status':'post-hoc symmetric comparison; no significance or equivalence test added',
 'existing_semivariance_comparisons_reproduced':len(validation),
 'max_mse_difference':max(v['mse_error'] for v in validation),
 'max_improvement_difference':max(v['improvement_error'] for v in validation),
 'max_difference_from_feedback_percent_points':max_feedback_difference,
 'input_unchanged':initial_hashes=={'data':sha(DATA),'old_semivariance_results':sha(OLD)},
 'input_sha256':initial_hashes,
 'main_correlation_sample':'Intersection of pre-strict forecast origins for h=5 and h=20. Unique origins per stock, not duplicated by beta or h.',
 'common_origins_n':1874,
 'common_origins_dates':[20180903,20260423],
 'feature_correlation_is_not_a_test_of_predictive_equivalence':True,
}
assert report['input_unchanged']
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
print(corr[corr['sample']=='pre_common_forecast'].to_string(index=False))
print(cmp[(cmp['sample']=='pre_strict')&cmp.reference.isin(['MD','M2'])].to_string(index=False))
