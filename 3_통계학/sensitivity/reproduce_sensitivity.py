"""Post-hoc sensitivity analysis for the bundled statistics manuscript.

The study's original models, sample splits, and predictions are preserved.
Evaluation exclusions are chosen after seeing the data and are not new holdouts.
Use: python3 sensitivity/reproduce_sensitivity.py
Outputs: sensitivity/results/*.csv and validation.json (relative to this script).
"""
from pathlib import Path
import pandas as pd, numpy as np, json, hashlib
import argparse
parser=argparse.ArgumentParser(description="Reproduce the post-hoc evaluation sensitivities in statistics v0.8. Predictions are fitted once; exclusions only change scoring rows.")
parser.add_argument("--stats-dir",type=Path,default=Path(__file__).resolve().parent.parent,help="Directory containing the bundled input CSV and stats_addendum_results.csv")
parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results")
args=parser.parse_args()
SRC=args.stats_dir.resolve()
DST=args.output_dir.resolve()
DST.mkdir(parents=True,exist_ok=True)
input_files=[SRC/'krx_stock_daily_long_20000104_20261113.csv',SRC/'stats_addendum_results.csv']
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
d=pd.read_csv(SRC/'krx_stock_daily_long_20000104_20261113.csv');d['r']=d.close/(d.close-d.change_from_base_price)-1
roll=lambda x,w:pd.Series(x).rolling(w).mean().values
g=lambda b,r:b*np.log1p(r)-np.log1p(b*r)
def gain(e1,e2):return 1-np.mean(e2**2)/np.mean(e1**2)
def nw(f,L):
 n=len(f);u=f-f.mean();v=u@u/n
 for j in range(1,min(L,n-1)+1):v+=2*(1-j/(L+1))*(u[j:]@u[:-j])/n
 return f.mean()/np.sqrt(v/n)
forecast_rows=[];phase_rows=[];month_rows=[];event_rows=[];year_rows=[];summary=[]
for nm,x in d.groupby('stock'):
 x=x[x.date>=20150615].reset_index(drop=True);r=x.r.values;dates=x.date.values;n=len(x);post_index=np.flatnonzero(dates>=20260527)
 for spec in ['raw_post','log_pre']:
  if spec=='log_pre' and nm!='삼성전자':continue
  if spec=='raw_post':
   F=np.column_stack([r**2,roll(r**2,5),roll(r**2,22),roll(r,22),r**3,roll(r**3,5),roll(r**3,22)])
  else:
   v5,v22=roll(r**2,5),roll(r**2,22);F=np.column_stack([np.log(r**2+1e-8),np.log(v5+1e-8),np.log(v22+1e-8),roll(r,22),roll(r**3,5)/(v5+1e-8)**1.5,roll(r**3,22)/(v22+1e-8)**1.5])
  X1=np.column_stack([np.ones(n),F[:,:4]]);X2=np.column_stack([np.ones(n),F])
  for h in [5,20]:
   Y=np.full((n,2),np.nan)
   for j,b in enumerate([2,-2]):
    cs=np.r_[0,np.cumsum(g(b,r))];yy=cs[1+h:n+1]-cs[1:n-h+1];Y[:n-h,j]=yy if spec=='raw_post' else np.log(yy+1e-8)
   if spec=='raw_post':origins=np.arange(max(22+750+h,post_index[0]),n-h)
   else:origins=np.array([t for t in range(22+750+h,n-h) if dates[t+h]<20260527])
   P1=np.full((len(origins),2),np.nan);P2=P1.copy()
   # Fit each origin with only completed targets. The masks below do not refit.
   for q,t in enumerate(origins):
    tr=np.arange(21,t-h+1);P1[q]=X1[t]@np.linalg.lstsq(X1[tr],Y[tr],rcond=None)[0];P2[q]=X2[t]@np.linalg.lstsq(X2[tr],Y[tr],rcond=None)[0]
   for j,b in enumerate([2,-2]):
    e1=Y[origins,j]-P1[:,j];e2=Y[origins,j]-P2[:,j];loss=e1**2-e2**2;cw=loss+(P1[:,j]-P2[:,j])**2
    summary.append(dict(stock=nm,spec=spec,h=h,beta=b,n=len(origins),gain=gain(e1,e2),cw_h1=nw(cw,h-1),cw_40=nw(cw,40),raw_loss_t_h1=nw(loss,h-1),raw_loss_t_40=nw(loss,40),cw_acf1=np.corrcoef(cw[:-1],cw[1:])[0,1]))
    for q,t in enumerate(origins):forecast_rows.append(dict(stock=nm,spec=spec,h=h,beta=b,t=int(t),date=int(dates[t]),end_date=int(dates[t+h]),y=float(Y[t,j]),p1=float(P1[q,j]),p2=float(P2[q,j]),loss_diff=float(loss[q]),cw=float(cw[q])))
    if spec=='raw_post':
     for phase in range(h):
      mm=np.arange(len(origins))%h==phase
      phase_rows.append(dict(stock=nm,h=h,beta=b,phase=phase,n=int(mm.sum()),gain=gain(e1[mm],e2[mm])))
     for month in np.unique(dates[post_index]//100):
      affected=np.array([np.any(dates[t+1:t+h+1]//100==month) for t in origins]);mm=~affected
      month_rows.append(dict(stock=nm,h=h,beta=b,excluded_month=int(month),n_excluded=int(affected.sum()),n_remain=int(mm.sum()),gain=gain(e1[mm],e2[mm])))
     event=post_index[np.argmax(abs(r[post_index]))]
     for mode in ['target_only','target_and_feature22']:
      affected=((origins<event)&(origins+h>=event)) if mode=='target_only' else ((origins+h>=event)&(origins<=event+21))
      mm=~affected;event_rows.append(dict(stock=nm,h=h,beta=b,event_date=int(dates[event]),event_return=float(r[event]),mode=mode,n_excluded=int(affected.sum()),n_remain=int(mm.sum()),gain=gain(e1[mm],e2[mm]),share_loss_removed=float(loss[affected].sum()/loss.sum())))
    else:
     for yr in np.unique(dates[origins]//10000):
      mm=dates[origins]//10000==yr;year_rows.append(dict(stock=nm,h=h,beta=b,year=int(yr),n=int(mm.sum()),gain=gain(e1[mm],e2[mm]),mse1=float(np.mean(e1[mm]**2)),mse2=float(np.mean(e2[mm]**2))))
for name,rows in [('summary',summary),('forecast_rows',forecast_rows),('phase_rows',phase_rows),('month_rows',month_rows),('event_rows',event_rows),('year_rows',year_rows)]:pd.DataFrame(rows).to_csv(DST/(name+'.csv'),index=False)
print('SUMMARY\n',pd.DataFrame(summary).round(5).to_string(index=False))
print('NONOVERLAP PHASE RANGE\n',pd.DataFrame(phase_rows).groupby(['stock','h','beta']).agg(gain_min=('gain','min'),gain_max=('gain','max'),positive=('gain',lambda z:(z>0).sum()),phases=('gain','count')).round(5).to_string())
print('TARGET MONTH REMOVAL\n',pd.DataFrame(month_rows).round(5).to_string(index=False))
print('LARGEST RETURN EVENT REMOVAL\n',pd.DataFrame(event_rows).round(5).to_string(index=False))
print('SAMSUNG MAIN PRE YEARLY\n',pd.DataFrame(year_rows).round(5).to_string(index=False))
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files};assert before==after
# Verify that the full-sample forecasts match the manuscript's existing results.
reference=pd.read_csv(SRC/'stats_addendum_results.csv')
checks=[]
for item in summary:
 raw=item['spec']=='raw_post'
 target=('G' if raw else 'logG')+'('+('+' if item['beta']>0 else '')+str(item['beta'])+')'
 q=reference[(reference.stock==item['stock']) & (reference.h==item['h']) & (reference.target==target) & (reference.variance_block==('raw' if raw else 'log')) & (reference.asym_block==('raw3' if raw else 'norm2')) & (reference['sample']==('post' if raw else 'pre_strict'))]
 assert len(q)==1
 error=abs(item['gain']-float(q.iloc[0].gain_M2_over_M1))
 assert error<1e-9,(item,error)
 checks.append(error)
validation={'comparisons':len(checks),'maximum_absolute_gain_difference':max(checks),'input_sha256':before,'evaluation_exclusions_do_not_refit':True}
(DST/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('VALIDATION',json.dumps(validation,ensure_ascii=False))
