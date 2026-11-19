# Diagnostic for the levels specification after listing: how concentrated is the M2-over-M1 gain?
import numpy as np, pandas as pd
START, MIN_TRAIN = 20150615, 750
d = pd.read_csv('krx_stock_daily_long_20000104_20261113.csv', encoding='utf-8-sig', dtype={'code': str})
d['r'] = d.close / (d.close - d.change_from_base_price) - 1
g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
roll = lambda x, w: pd.Series(x).rolling(w).mean().values
for nm, x in d.groupby('stock'):
    x = x[x.date >= START].reset_index(drop=True); r = x.r.values; dates = x.date.values; n = len(r)
    F = np.column_stack([r**2, roll(r**2, 5), roll(r**2, 22), roll(r, 22), r**3, roll(r**3, 5), roll(r**3, 22)])
    for b in [2, -2]:
        gb = g(b, r)
        for h in [5, 20]:
            y = np.full(n, np.nan); cs = np.concatenate([[0], np.cumsum(gb)]); y[:n-h] = cs[1+h:n+1] - cs[1:n-h+1]
            p1 = np.full(n, np.nan); p2 = p1.copy()
            for t in range(max(22 + MIN_TRAIN + h, np.searchsorted(dates, 20260527)), n - h):
                tr = np.arange(21, t - h + 1)
                X1 = np.column_stack([np.ones(len(tr)), F[tr, :4]]); X2 = np.column_stack([np.ones(len(tr)), F[tr]])
                p1[t] = np.r_[1, F[t, :4]] @ np.linalg.lstsq(X1, y[tr], rcond=None)[0]; p2[t] = np.r_[1, F[t]] @ np.linalg.lstsq(X2, y[tr], rcond=None)[0]
            m = ~np.isnan(p1) & ~np.isnan(y); dd = ((y - p1)**2 - (y - p2)**2)[m]; dt = dates[m]
            o = np.argsort(-np.abs(dd)); tot = dd.sum()
            gain = 1 - ((y - p2)[m]**2).mean()/((y - p1)[m]**2).mean()
            k = 5; rest = np.delete(np.arange(len(dd)), o[:k])
            e1, e2 = (y - p1)[m], (y - p2)[m]
            gain_rest = 1 - (e2[rest]**2).mean()/(e1[rest]**2).mean()
            print(nm, b, h, 'n', m.sum(), 'gain %.3f' % gain, 'share of windows where M2 closer %.3f' % (np.abs(e2) < np.abs(e1)).mean(),
                  'top5 dates', list(dt[o[:k]]), 'top5 share of total diff %.2f' % (dd[o[:k]].sum()/tot), 'gain without top5 %.3f' % gain_rest)
