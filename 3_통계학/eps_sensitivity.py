# Sensitivity of the main (log) specification to the constant EPS added inside the logs.  Run: python3 eps_sensitivity.py
# Sample: forecasts whose whole target window ends before listing.  Reports the M2-over-M1 gain and Clark-West t (lag h-1).
import numpy as np, pandas as pd
START, MIN_TRAIN, LIST = 20150615, 750, 20260527
d = pd.read_csv('krx_stock_daily_long_20000104_20261113.csv', encoding='utf-8-sig', dtype={'code': str})
d['r'] = d.close / (d.close - d.change_from_base_price) - 1
g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
roll = lambda x, w: pd.Series(x).rolling(w).mean().values
def nw_t(f, L):
    f = np.asarray(f); n = len(f); u = f - f.mean(); v = u @ u / n
    for l in range(1, L + 1): v += 2*(1 - l/(L + 1))*(u[l:] @ u[:-l])/n
    return f.mean()/np.sqrt(v/n)
rows = []
for EPS in (1e-8, 1e-6, 1e-5, 1e-4):
    for nm, x in d.groupby('stock'):
        x = x[x.date >= START].reset_index(drop=True); r = x.r.values; dates = x.date.values; n = len(r)
        rv5, rv22 = roll(r**2, 5), roll(r**2, 22)
        X1 = np.column_stack([np.ones(n), np.log(r**2 + EPS), np.log(rv5 + EPS), np.log(rv22 + EPS), roll(r, 22)])
        X2 = np.column_stack([X1, roll(r**3, 5)/(rv5 + EPS)**1.5, roll(r**3, 22)/(rv22 + EPS)**1.5])
        for h in (5, 20):
            cs = {b: np.concatenate([[0], np.cumsum(g(b, r))]) for b in (2, -2)}
            Y = np.full((n, 2), np.nan)
            for j, b in enumerate((2, -2)): Y[:n-h, j] = np.log(cs[b][1+h:n+1] - cs[b][1:n-h+1] + EPS)
            P1 = np.full(Y.shape, np.nan); P2 = P1.copy()
            for t in range(22 + MIN_TRAIN + h, n - h):
                tr = np.arange(21, t - h + 1)
                P1[t] = X1[t] @ np.linalg.lstsq(X1[tr], Y[tr], rcond=None)[0]; P2[t] = X2[t] @ np.linalg.lstsq(X2[tr], Y[tr], rcond=None)[0]
            end = np.full(n, 99999999); end[:n-h] = dates[h:]; mm = ~np.isnan(P1[:, 0]) & ~np.isnan(Y[:, 0]) & (end < LIST)
            for j, b in enumerate((2, -2)):
                e1, e2 = (Y - P1)[mm, j], (Y - P2)[mm, j]; cw = e1**2 - (e2**2 - (P1 - P2)[mm, j]**2)
                rows.append(dict(eps=EPS, stock=nm, beta=b, h=h, n=int(mm.sum()), gain_M2_over_M1=1 - np.mean(e2**2)/np.mean(e1**2), cw_t=nw_t(cw, h - 1)))
res = pd.DataFrame(rows); pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda v: '%.4g' % v)
txt = res.pivot_table(index=['stock', 'beta', 'h'], columns='eps', values=['gain_M2_over_M1', 'cw_t']).to_string()
print(txt); open('eps_sensitivity_output.txt', 'w', encoding='utf-8').write(txt + '\n')
