# Statistics thesis, first pass: does signed third-moment information add to forecasts of leveraged log-decay?
# Run from this folder: python3 stats_experiment.py     (no random numbers)
import numpy as np, pandas as pd
START = 20150615          # +-30% daily price limit in force from this date
MIN_TRAIN = 750           # first forecast needs this many completed training rows
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
d = pd.read_csv('krx_stock_daily_long_20000104_20261113.csv', encoding='utf-8-sig', dtype={'code': str})
d['r'] = d.close / (d.close - d.change_from_base_price) - 1      # return against the exchange base price (handles splits)
g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
def nw_t(f, L):                                                   # t-statistic of the mean with a Newey-West variance
    f = np.asarray(f); n = len(f); u = f - f.mean(); v = u @ u / n
    for l in range(1, L + 1): v += 2*(1 - l/(L + 1))*(u[l:] @ u[:-l])/n
    return f.mean()/np.sqrt(v/n)
roll = lambda x, w: pd.Series(x).rolling(w).mean().values
say('== sample ==')
for nm, x in d.groupby('stock'):
    x = x[x.date >= START]; r = x.r.values
    say(nm, 'days', len(r), 'sd', round(r.std(), 4), 'skew', round(pd.Series(r).skew(), 3), 'max|r|', round(np.abs(r).max(), 4), 'sum r^2', round((r**2).sum(), 4), 'sum r^3', round((r**3).sum(), 5))
say('\n== measurement: realised G over h days against the quadratic and cubic approximations ==')
for nm, x in d.groupby('stock'):
    r = x[x.date >= START].r.values
    for b in [2, -2]:
        c2, c3 = b*(b-1)/2, -b*(b*b-1)/3
        for h in [5, 20]:
            G = roll(g(b, r), h)*h; Q = c2*roll(r**2, h)*h; C = Q + c3*roll(r**3, h)*h; m = ~np.isnan(G)
            e2, e3 = np.abs(Q - G)[m], np.abs(C - G)[m]
            say(nm, 'beta', b, 'h', h, 'mean G', round(G[m].mean(), 5), 'mean |quad err|', round(e2.mean(), 6), 'mean |cubic err|', round(e3.mean(), 6), 'share of windows where cubic is closer', round((e3 < e2).mean(), 3), 'mean signed quad err', round((Q - G)[m].mean(), 6), 'cubic', round((C - G)[m].mean(), 6))
say('\n== forecasting: expanding window, training rows need s + h <= t ==')
rows = []
for nm, x in d.groupby('stock'):
    x = x[x.date >= START].reset_index(drop=True); r = x.r.values; dates = x.date.values; n = len(r)
    F = np.column_stack([r**2, roll(r**2, 5), roll(r**2, 22), roll(r, 22), r**3, roll(r**3, 5), roll(r**3, 22)])
    for b in [2, -2]:
        gb = g(b, r); base22 = roll(gb, 22)
        for h in [5, 20]:
            y = np.full(n, np.nan); cs = np.concatenate([[0], np.cumsum(gb)])
            y[:n-h] = cs[1+h:n+1] - cs[1:n-h+1]                    # y_t = sum_{j=1..h} g(r_{t+j})
            p0 = np.full(n, np.nan); p1 = np.full(n, np.nan); p2 = np.full(n, np.nan)
            for t in range(22 + MIN_TRAIN + h, n - h):
                tr = np.arange(21, t - h + 1)                       # rows s with features available and s + h <= t
                X1 = np.column_stack([np.ones(len(tr)), F[tr, :4]]); X2 = np.column_stack([np.ones(len(tr)), F[tr]])
                b1 = np.linalg.lstsq(X1, y[tr], rcond=None)[0]; b2 = np.linalg.lstsq(X2, y[tr], rcond=None)[0]
                p0[t] = h*base22[t]; p1[t] = np.r_[1, F[t, :4]] @ b1; p2[t] = np.r_[1, F[t]] @ b2
            ok = ~np.isnan(p1) & ~np.isnan(y)
            for lab, m in [('all', ok), ('before listing', ok & (dates < 20260527)), ('after listing', ok & (dates >= 20260527))]:
                if m.sum() < 30: continue
                e0, e1, e2 = (y - p0)[m], (y - p1)[m], (y - p2)[m]
                cw = e1**2 - (e2**2 - (p1 - p2)[m]**2)
                rows.append(dict(stock=nm, beta=b, h=h, sample=lab, n=int(m.sum()), mse0=np.mean(e0**2), mse1=np.mean(e1**2), mse2=np.mean(e2**2),
                                 r2_1_vs_0=1 - np.mean(e1**2)/np.mean(e0**2), r2_2_vs_1=1 - np.mean(e2**2)/np.mean(e1**2), cw_t=nw_t(cw, h - 1), first=int(dates[m][0]), last=int(dates[m][-1])))
res = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: '%.4g' % v)
say(res[['stock', 'beta', 'h', 'sample', 'n', 'first', 'last', 'mse0', 'mse1', 'mse2', 'r2_1_vs_0', 'r2_2_vs_1', 'cw_t']].to_string(index=False))
res.to_csv('stats_experiment_results.csv', index=False, encoding='utf-8-sig')
open('stats_experiment_output.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
