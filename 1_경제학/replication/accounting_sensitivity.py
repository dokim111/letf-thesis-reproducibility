"""Reproduce Table 9 and accounting sensitivities introduced in economics v14 and retained in v17.

Inputs default to this script's folder. KRX CSV files must be obtained separately.
No input is changed. Product exclusions re-aggregate retained products before
absolute values are taken. Date exclusions remove already-constructed daily B/f
observations; they do not rebuild later asset paths. These are post-hoc accounting
diagnostics, not confidence intervals, causal effects or forecasting tests.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description="Post-hoc accounting sensitivities; reads but never edits the thesis package.")
parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent, help="Folder with independently obtained KRX CSV files; defaults to this script folder")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
INPUT = args.data_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
etf = pd.read_csv(INPUT / 'krx_etf_daily_20260527_20261113.csv', encoding='utf-8-sig')
stock = pd.read_csv(INPUT / 'krx_stock_daily_20260401_20261113.csv', encoding='utf-8-sig')
rows = []
alignment = []
for code, g in etf.groupby('short_code'):
    g = g.sort_values('date').reset_index(drop=True)
    b = int(g.beta.iloc[0])
    n = g.net_assets_krw / g.nav
    rel = (n - g.listed_shares.shift(-1)).abs() / g.listed_shares.shift(-1)
    ok = (g.net_assets_krw > 0) & g.listed_shares.shift(-1).notna()
    alignment.append(dict(code=code, name=g.name.iloc[0], n=int(ok.sum()),
        nextday_max_relative_error=float(rel[ok].max()),
        nextday_median_relative_error=float(rel[ok].median()),
        nextday_within_002=int((rel[ok] < .002).sum())))
    for i in range(1, len(g)):
        a0, a1 = g.net_assets_krw[i-1], g.net_assets_krw[i]
        if not (a0 > 0 and a1 > 0):
            continue
        R = g.nav[i] / g.nav[i-1] - 1
        r = R / b
        F = (a1 / g.nav[i] - a0 / g.nav[i-1]) * g.nav[i]
        Fl = (g.listed_shares[i] - g.listed_shares[i-1]) * g.nav[i]
        rows.append(dict(date=int(g.date[i]), code=code, name=g.name[i], und=g.underlying[i],
            beta=b, A=a0, R=R, r=r, B=b*(b-1)*a0*r, F=F, f=b*F,
            f_listed=b*Fl, K=b*(b-1)*a0, RA=R*a0))
panel = pd.DataFrame(rows)
cols = ['A', 'B', 'F', 'f', 'f_listed', 'K', 'RA']
def agg(p, by=['und']):
    return p.groupby(by + ['date'])[cols].sum().reset_index()
def cabs(d, flow='f'):
    denominator = float(d.B.abs().sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Accounting offset ratio requires a positive sum of absolute baseline adjustments')
    return float(1 - (d.B + d[flow]).abs().sum()/denominator)
def phi(d):
    x, y = (d.RA/d.A).to_numpy(), (d.F/d.A).to_numpy()
    return float(-np.linalg.lstsq(np.c_[np.ones(len(x)),x], y, rcond=None)[0][1])
def records(df):
    return json.loads(df.to_json(orient='records', force_ascii=False))

result = {'input_rows':len(etf), 'analysis_rows':len(panel),
    'analysis_days':int(panel.date.nunique()), 'alignment':alignment}

robust = []
aggregate = agg(panel)
for u, d in aggregate.groupby('und'):
    d = d.reset_index(drop=True)
    baseline = cabs(d)
    no_ace = agg(panel[(panel.und == u) & ~panel.name.str.startswith('ACE')])
    loo = [cabs(d.drop(i)) for i in d.index]
    loo_product = []
    for code in panel.loc[panel.und == u,'code'].unique():
        dx = agg(panel[(panel.und == u) & (panel.code != code)])
        loo_product.append(dict(code=code,cabs=cabs(dx)))
    month = d.date // 100
    monthly = [dict(month=int(m),n=len(dx),cabs=cabs(dx)) for m,dx in d.groupby(month)]
    top_b_days = d.loc[d.B.abs().nlargest(5).index,'date'].tolist()
    top_f_days = d.loc[d.f.abs().nlargest(5).index,'date'].tolist()
    crosscheck = []
    for cutoff in [20260619,20260716,20260724,20260730,20260731,20260807,20260819]:
        for label,dx in [('pre',d[d.date<=cutoff]),('post',d[d.date>cutoff])]:
            V = stock[stock.stock==u].set_index('date').trading_value_krw
            crosscheck.append(dict(cutoff=cutoff,period=label,n=len(dx),cabs=cabs(dx),
                mean_K_over_V=float((dx.K.to_numpy()/V.reindex(dx.date).to_numpy()).mean())))
    concentration = []
    delta = (d.B+d.f).abs()-d.B.abs()
    for n in [1,3,5,10]:
        idx = delta.nlargest(n).index
        concentration.append(dict(n=n,excess_share=float(delta[idx].sum()/delta.sum()),
            selected_days=d.date[idx].tolist(), cabs_after_removal=cabs(d.drop(idx))))
    robust.append(dict(underlying=u,baseline=baseline,no_ace=cabs(no_ace),
        listed_shares=cabs(d,'f_listed'),
        leave_one_day_min=min(loo),leave_one_day_max=max(loo),
        min_omitted_day=int(d.date.iloc[int(np.argmin(loo))]),
        max_omitted_day=int(d.date.iloc[int(np.argmax(loo))]),
        no_selected_3days=cabs(d[~d.date.isin([20260713,20260728,20260731])]),
        top5_absB_days=top_b_days,no_top5_absB=cabs(d[~d.date.isin(top_b_days)]),
        top5_absf_days=top_f_days,no_top5_absf=cabs(d[~d.date.isin(top_f_days)]),
        leave_one_product=loo_product,monthly=monthly,boundary=crosscheck,
        positive_excess_concentration=concentration))
result['robustness']=robust

regress=[]
for (u,b), d in agg(panel,['und','beta']).groupby(['und','beta']):
    d=d.reset_index(drop=True)
    loo=[phi(d.drop(i)) for i in d.index]
    thresholds=[]
    for threshold in [.10,.15,.20]:
        dx=d[(d.F/d.A).abs()<=threshold]
        thresholds.append(dict(threshold=threshold,kept=len(dx),phi=phi(dx)))
    influence=[dict(omitted_date=int(d.date[i]),phi=loo[i],change=loo[i]-phi(d))
        for i in np.argsort(np.abs(np.array(loo)-phi(d)))[-3:][::-1]]
    regress.append(dict(underlying=u,beta=int(b),phi=phi(d),n=len(d),
        leave_one_out_min=min(loo),leave_one_out_max=max(loo),most_influential=influence,
        thresholds=thresholds,pre=phi(d[d.date<=20260730]),post=phi(d[d.date>20260730])))
result['regression']=regress
ace_share=etf[(etf.net_assets_krw>0)].assign(ace=lambda x:x.name.str.startswith('ACE'))
ace_bydate=ace_share.groupby(['date','ace']).net_assets_krw.sum().unstack(fill_value=0)
result['ace_asset_share']={'max':float((ace_bydate[True]/ace_bydate.sum(axis=1)).max()),
    'mean':float((ace_bydate[True]/ace_bydate.sum(axis=1)).mean())}
result['disclosure_crosscheck']=records(etf[(etf.short_code.isin(['0193T0','0193W0']))&
    (etf.date.isin([20260730,20260731,20260803]))].assign(implied_shares=lambda x:x.net_assets_krw/x.nav)[
    ['date','short_code','listed_shares','implied_shares','net_assets_krw','nav']])

# Units in the full JSON remain C_abs. Table 9 reports -100*C_abs (percent).
summary=[]
for u in ['삼성전자','SK하이닉스']:
    d=next(x for x in robust if x['underlying']==u)
    for label,n,lo,hi in [
        ('전체 표본',114,-100*d['baseline'],-100*d['baseline']),
        ('ACE 상품 제외',114,-100*d['no_ace'],-100*d['no_ace']),
        ('하루씩 제외한 범위',113,-100*d['leave_one_day_max'],-100*d['leave_one_day_min']),
        ('순설정 수요 절댓값 상위 5일 제외',109,-100*d['no_top5_absf'],-100*d['no_top5_absf'])]:
        summary.append(dict(underlying=u,criterion=label,remaining_days=n,
            increase_percent_min=lo,increase_percent_max=hi))
result['table9_units']='increase in cumulative absolute net exposure adjustment, percent = -100*C_abs'
result['table9_exclusion_rules']={
    'ACE':'Exclude ACE products before summing B and f by underlying and date.',
    'one_day':'Remove each of the 114 aggregated daily B/f pairs in turn. Range of 114 descriptive recalculations, not a confidence interval.',
    'top5_absf':'For each underlying separately, exclude the 5 largest abs(sum_k beta_k*F_kt); 109 daily pairs remain.',
    'date_exclusions':'The original daily accounting paths remain fixed. No later net asset, return or creation path is reconstructed.'}
result['table9']=summary
result['validation']={
    'panel_unique_product_date':bool(not panel.duplicated(['code','date']).any()),
    'days_per_product':{str(k):int(v) for k,v in panel.groupby('code').size().items()},
    'max_relative_nav_identity_residual':float(max(
        abs((g.net_assets_krw.iloc[i]-g.net_assets_krw.iloc[i-1]) -
            (g.net_assets_krw.iloc[i-1]*(g.nav.iloc[i]/g.nav.iloc[i-1]-1)+
             (g.net_assets_krw.iloc[i]/g.nav.iloc[i]-g.net_assets_krw.iloc[i-1]/g.nav.iloc[i-1])*g.nav.iloc[i]))
        / max(1.0,abs(g.net_assets_krw.iloc[i]))
        for _,raw in etf.groupby('short_code')
        for g in [raw.sort_values('date').reset_index(drop=True)]
        for i in range(1,len(g))
        if g.net_assets_krw.iloc[i]>0 and g.net_assets_krw.iloc[i-1]>0)),
    'all_aggregate_K_positive':bool((aggregate.K>0).all()),
    'zero_aggregate_B_days':{u:int((d.B==0).sum()) for u,d in aggregate.groupby('und')},
    'small_B_days_retained_in_period_ratio':{u:int((abs(d.B/d.K)<.01).sum()) for u,d in aggregate.groupby('und')},
    'period_sum_abs_B_positive':{u:bool(d.B.abs().sum()>0) for u,d in aggregate.groupby('und')},
    'turnover_positive_all_analysis_days':{u:bool((stock[stock.stock==u].set_index('date').trading_value_krw.reindex(d.date)>0).all()) for u,d in aggregate.groupby('und')}}
(OUT/'accounting_sensitivity_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(summary).to_csv(OUT/'accounting_sensitivity_summary.csv',index=False,encoding='utf-8-sig')
lines=['Economics v14 Table 9: accounting sensitivities',
       'Increase (%) = -100*C_abs; each retained sample has its own denominator sum(abs(B)).',
       'Date exclusions keep previously reconstructed daily accounting paths fixed.',
       'These are post-hoc diagnostics; no causal effect or confidence interval is estimated.']
for x in summary:
    val=f"{x['increase_percent_min']:.2f}" if x['increase_percent_min']==x['increase_percent_max'] else f"{x['increase_percent_min']:.2f}~{x['increase_percent_max']:.2f}"
    lines.append(f"{x['underlying']} | {x['criterion']} | n={x['remaining_days']} | +{val}%")
for x in robust:
    november=next(m for m in x['monthly'] if m['month']==202611)
    lines.append(f"{x['underlying']} November only | n={november['n']} | increase={-100*november['cabs']:.4f}%")
lines.append('Validation: '+json.dumps(result['validation'],ensure_ascii=False))
output='\n'.join(lines)+'\n'
(OUT/'accounting_sensitivity_output.txt').write_text(output,encoding='utf-8')
print(output,end='')
