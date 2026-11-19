# Replication of Chapter 5 calculations introduced in economics v14 and retained in v17.
# Recomputes Tables 3-8, inputs for Table 10, and the descriptive regression.
# New Table 9 is reproduced by accounting_sensitivity.py.
# Input: three independently obtained KRX CSV files in this folder (not distributed here).
# Run: python3 replicate.py
# Conventions: net assets of 0 on 2026-11-13 are treated as "not yet published" (missing), not as zero.
#   B = beta(beta-1) A_{t-1} r   baseline adjustment (net creations excluded), r = NAV return / beta
#   F = (A_t/NAV_t - A_{t-1}/NAV_{t-1}) NAV_t   estimated net creation, dated by net assets;  f = beta F
#   C_abs = 1 - sum|B+f| / sum|B|   (negative: the absolute adjustment is larger once net creations are included)
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(__file__).resolve().parent
P1_END = 20260730
etf = pd.read_csv(BASE / 'krx_etf_daily_20260527_20261113.csv', encoding='utf-8-sig')
stk = pd.read_csv(BASE / 'krx_stock_daily_20260401_20261113.csv', encoding='utf-8-sig')
fut = pd.read_csv(BASE / 'krx_stock_futures_daily_20260527_20261113.csv', encoding='utf-8-sig')
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

say('== data checks ==')
tot = etf[etf.net_assets_krw > 0].groupby('date').net_assets_krw.sum()
say('total net assets (억): 5/27', round(tot[20260527]/1e8, 1), ' peak', tot.idxmax(), round(tot.max()/1e8, 1), ' 11/12', round(tot[20261112]/1e8, 1))
x = etf[(etf.date == 20260619) & (etf.beta > 0)].groupby('underlying').net_assets_krw.sum() / 1e12
say('6/19 net assets of +2x products (조):', x.round(4).to_dict())
say('products with net assets recorded as 0 on 11/13:', int((etf[etf.date == 20261113].net_assets_krw == 0).sum()))
for code, g in etf.groupby('short_code'):
    g = g.sort_values('date'); ok = g[g.net_assets_krw > 0]
    n_eff = (ok.net_assets_krw / ok.nav).values
    nxt = g.listed_shares.shift(-1).loc[ok.index].values; same = g.listed_shares.loc[ok.index].values
    say(code, g.name.iloc[0], '| A/NAV equals listed shares: same day', int(np.nansum(np.abs(n_eff - same) / same < 0.002)), 'next day', int(np.nansum(np.abs(n_eff - nxt) / nxt < 0.002)), 'of', int(np.sum(~np.isnan(nxt))))

def panel(mode='nav'):
    rows = []
    for code, g in etf.groupby('short_code'):
        g = g.sort_values('date').reset_index(drop=True); b = g.beta.iloc[0]
        for i in range(1, len(g)):
            a0, a1 = g.net_assets_krw[i-1], g.net_assets_krw[i]
            if not (a0 > 0 and a1 > 0): continue
            r = (g.nav[i] / g.nav[i-1] - 1) / b if mode == 'nav' else g.base_index_return_pct[i] / 100
            if pd.isna(r): continue
            F = (a1 / g.nav[i] - a0 / g.nav[i-1]) * g.nav[i]
            F_listed = (g.listed_shares[i] - g.listed_shares[i-1]) * g.nav[i]
            rows.append(dict(date=g.date[i], code=code, und=g.underlying[i], beta=b, A=a0, r=r, B=b*(b-1)*a0*r, F=F, f=b*F, f_listed=b*F_listed, K=b*(b-1)*a0, RA=b*r*a0))
    return pd.DataFrame(rows)
pn_ = panel('nav'); pi_ = panel('idx')
V = stk.pivot(index='date', columns='stock', values='trading_value_krw')
R = stk.pivot(index='date', columns='stock', values='return_pct')
VF = fut.groupby(['date', 'underlying']).trading_value_krw.sum().unstack()
VR = stk.pivot(index='date', columns='stock', values='regular_session_value_krw')   # after-market value is recorded as 0 until 2026-09-11
cols = ['A', 'B', 'F', 'f', 'f_listed', 'K', 'RA']
agg = lambda p, by: p.groupby(by + ['date'])[cols].sum().reset_index()
cabs = lambda B, f: 1 - np.abs(np.asarray(B) + np.asarray(f)).sum() / np.abs(np.asarray(B)).sum()
periods = lambda d: {'ALL': d.date > 0, 'P1': d.date <= P1_END, 'P2': d.date > P1_END}
st = agg(pn_, ['und'])

say('\n== 표 3, 표 4, 표 10: size of the baseline adjustment and K/V ==')
for u, d in st.groupby('und'):
    d = d.reset_index(drop=True); vs = V[u].reindex(d.date).values; vf = VF[u].reindex(d.date).values
    for pn, m in periods(d).items():
        x = d[m]; v = vs[m.values]; w = vf[m.values]
        say(u, pn, 'n', len(x), 'A(조)', round(x.A.mean()/1e12, 2), 'K1%(억)', round(x.K.mean()*0.01/1e8), 'avg|B|(억)', round(x.B.abs().mean()/1e8, 2),
            '|B|/Vs%', round(100*(x.B.abs().values/v).mean(), 3), '|B|/(Vs+Vf)%', round(100*(x.B.abs().values/(v+w)).mean(), 3), 'max|B|/Vs%', round(100*(x.B.abs().values/v).max(), 1),
            'Vs(조)', round(v.mean()/1e12, 2), 'Vf(조)', round(w.mean()/1e12, 2), 'K/Vs', round((x.K.values/v).mean(), 4), 'K/(Vs+Vf)', round((x.K.values/(v+w)).mean(), 4))
    kv = d.K.values / vs; vr = VR[u].reindex(d.date).values; p2 = (d.date > P1_END).values
    say(u, 'K/V with regular-session value only: P1', round((d.K.values/vr)[~p2].mean(), 4), 'P2', round((d.K.values/vr)[p2].mean(), 4))
    say(u, 'daily K/Vs max', round(kv.max(), 3), '| days with 0.5*K/Vs >= 1:', int((0.5*kv >= 1).sum()))

say('\n== 표 5: absolute demand reduction and offset rates (C computed on days with |B_t/K_t| >= 1%) ==')
pn_['grp'] = pn_.und + np.where(pn_.beta > 0, ' +2', ' -2')
def report(name, d):
    line = [name]
    for pn, m in periods(d).items(): line += [pn, round(cabs(d.B[m], d.f[m]), 6)]
    B, f = d.B.values, d.f.values
    big = np.abs(B) / d.K.values >= 0.01; C = -f[big] / B[big]
    return line + ['sum|f|/sum|B|', round(np.abs(f).sum()/np.abs(B).sum(), 3), 'median C', round(float(np.median(C)), 3), 'share C<0', round((C < 0).mean(), 3), 'days', int(big.sum())]
gr = agg(pn_, ['grp'])
for g, d in gr.groupby('grp'): say(*report(g, d.reset_index(drop=True)))
for u, d in st.groupby('und'): say(*report(u + ' 전체', d.reset_index(drop=True)))

say('\n== 5.2절 auxiliary regression: net creation rate on the ETF NAV return (descriptive, not structural) ==')
def fit(y, x, L=5):
    X = np.column_stack([np.ones(len(y)), x]); XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b; n = len(y)
    ols = np.sqrt(e @ e / (n - 2) * XtXi[1, 1])
    S = (X.T * e**2) @ X
    for l in range(1, L + 1):
        G = (X[l:].T * (e[l:] * e[:-l])) @ X[:-l]; S += (1 - l/(L + 1)) * (G + G.T)
    return b[1], ols, np.sqrt((XtXi @ S @ XtXi)[1, 1] * n / (n - 2))   # Newey-West (Bartlett, 5 lags) with n/(n-2)
for g, d in gr.groupby('grp'):
    y = (d.F / d.A).values; x = (d.RA / d.A).values
    b, ols, hac = fit(y, x); keep = np.abs(y) <= 0.15; b2, ols2, _ = fit(y[keep], x[keep])
    say(g, '-slope', round(-b, 4), 'OLS se', round(ols, 4), 'NW(5) se', round(hac, 4), '| excluding |rate|>15%:', round(-b2, 4), 'n', int(keep.sum()), 'dropped', list(d.date[~keep]) if (~keep).sum() <= 3 else int((~keep).sum()))

say('\n== 표 6: up and down days ==')
for u, d in st.groupby('und'):
    d = d.reset_index(drop=True); ret = R[u].reindex(d.date).values
    for pn, m in list(periods(d).items())[1:]:
        for nm, s in [('up', ret > 0), ('down', ret < 0)]:
            x = d[m.values & s]
            say(u, pn, nm, 'n', len(x), 'avgB(억)', round(x.B.mean()/1e8), 'avgf(억)', round(x.f.mean()/1e8), 'C_sum', round(-x.f.sum()/x.B.sum(), 3), 'Cabs', round(cabs(x.B, x.f), 3))
    say(u, 'cumulative net creations sum F (조): P1', round(d.F[d.date <= P1_END].sum()/1e12, 2), 'P2', round(d.F[d.date > P1_END].sum()/1e12, 2))

say('\n== 표 7: selected days (F is the net creation amount, f = sum beta F the exposure demand it creates) ==')
for day in [20260713, 20260728, 20260731]:
    for u, d in st.groupby('und'):
        x = d[d.date == day].iloc[0]
        say(day, u, 'r%', R[u][day], 'B(억)', round(x.B/1e8), 'sum F(억)', round(x.F/1e8), 'f(억)', round(x.f/1e8), 'B+f', round((x.B+x.f)/1e8), '|B|/Vs%', round(100*abs(x.B)/V[u][day], 1), '|B|/(Vs+Vf)%', round(100*abs(x.B)/(V[u][day]+VF[u][day]), 1))

say('\n== 표 8: dating and return definition ==')
sti = agg(pi_, ['und'])
for u, d in st.groupby('und'):
    B, f = d.B.values, d.f.values
    say(u, '| baseline (114)', round(cabs(B, f), 6), '| f moved one day earlier (113)', round(cabs(B[:-1], f[1:]), 6),
        '| change in published listed shares x same-day NAV (114)', round(cabs(B, d.f_listed), 6), '| exchange index return (114)', round(cabs(sti[sti.und == u].B, sti[sti.und == u].f), 6))
open(BASE / 'replication_output.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
