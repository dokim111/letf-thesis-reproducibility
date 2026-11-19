# Statistics thesis, main specification (log). Run from this folder: python3 stats_main_log.py   (no random numbers)
# Targets (h = 5, 20 trading days ahead, beta = +2, -2):
#   Y_b = log G_b,   G_b = sum_{j=1..h} g_b(r_{t+j}),   g_b(r) = b log(1+r) - log(1+b r)
#   D   = log G_{-2} - log G_{+2}   (decay ratio; to first order log 3 + (8/3) * sum r^3 / sum r^2, so the variance level cancels)
# Models: M0 baseline (last 22 days), M1 variance information, M2 = M1 + asymmetry information.
import numpy as np, pandas as pd
START, MIN_TRAIN, EPS, LIST = 20150615, 750, 1e-8, 20260527
out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
d = pd.read_csv('krx_stock_daily_long_20000104_20261113.csv', encoding='utf-8-sig', dtype={'code': str})
d['r'] = d.close / (d.close - d.change_from_base_price) - 1
g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
roll = lambda x, w: pd.Series(x).rolling(w).mean().values
def nw_t(f, L):
    f = np.asarray(f); n = len(f); u = f - f.mean(); v = u @ u / n
    for l in range(1, L + 1): v += 2*(1 - l/(L + 1))*(u[l:] @ u[:-l])/n
    return f.mean()/np.sqrt(v/n)
def ols_nw(X, y, L):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b; S = (X.T * e**2) @ X
    for l in range(1, L + 1):
        G = (X[l:].T * (e[l:]*e[:-l])) @ X[:-l]; S += (1 - l/(L + 1))*(G + G.T)
    return b, np.sqrt(np.diag(XtXi @ S @ XtXi))
rows, coefs, approx = [], [], []
for nm, x in d.groupby('stock'):
    x = x[x.date >= START].reset_index(drop=True); r = x.r.values; dates = x.date.values; n = len(r)
    rv5, rv22 = roll(r**2, 5), roll(r**2, 22)
    F = np.column_stack([np.log(r**2 + EPS), np.log(rv5 + EPS), np.log(rv22 + EPS), roll(r, 22),
                         roll(r**3, 5)/(rv5 + EPS)**1.5, roll(r**3, 22)/(rv22 + EPS)**1.5])
    G = {b: g(b, r) for b in (2, -2)}; cs = {b: np.concatenate([[0], np.cumsum(G[b])]) for b in G}
    c2, c3 = np.concatenate([[0], np.cumsum(r**2)]), np.concatenate([[0], np.cumsum(r**3)])
    for h in (5, 20):
        fut = lambda c: c[1+h:n+1] - c[1:n-h+1]
        Y = {}
        for b in G: Y[b] = np.full(n, np.nan); Y[b][:n-h] = np.log(fut(cs[b]) + EPS)
        Y['D'] = Y[-2] - Y[2]
        kap = np.full(n, np.nan); kap[:n-h] = fut(c3)/(fut(c2) + EPS)
        m = ~np.isnan(Y['D']); lin = np.log(3) + (8/3)*kap
        approx.append(dict(stock=nm, h=h, corr_D_linear=np.corrcoef(Y['D'][m], lin[m])[0, 1], mean_D=Y['D'][m].mean(), log3=np.log(3), sd_D=Y['D'][m].std(), mean_abs_err=np.abs(Y['D'] - lin)[m].mean()))
        base = {2: np.log(h*roll(G[2], 22) + EPS), -2: np.log(h*roll(G[-2], 22) + EPS)}; base['D'] = base[-2] - base[2]
        for key in (2, -2, 'D'):
            y = Y[key]; p0 = np.full(n, np.nan); p1 = p0.copy(); p2 = p0.copy()
            for t in range(22 + MIN_TRAIN + h, n - h):
                tr = np.arange(21, t - h + 1)                                  # training rows: s + h <= t
                X1 = np.column_stack([np.ones(len(tr)), F[tr, :4]]); X2 = np.column_stack([np.ones(len(tr)), F[tr]])
                b1 = np.linalg.lstsq(X1, y[tr], rcond=None)[0]; b2 = np.linalg.lstsq(X2, y[tr], rcond=None)[0]
                p0[t] = base[key][t]; p1[t] = np.r_[1, F[t, :4]] @ b1; p2[t] = np.r_[1, F[t]] @ b2
            ok = ~np.isnan(p1) & ~np.isnan(y)
            for lab, mm in (('before listing', ok & (dates < LIST)), ('after listing', ok & (dates >= LIST))):
                e0, e1, e2 = (y - p0)[mm], (y - p1)[mm], (y - p2)[mm]; cw = e1**2 - (e2**2 - (p1 - p2)[mm]**2)
                rows.append(dict(stock=nm, target='D' if key == 'D' else 'logG(%+d)' % key, h=h, sample=lab, n=int(mm.sum()), mse0=np.mean(e0**2), mse1=np.mean(e1**2), mse2=np.mean(e2**2),
                                 gain_M1_over_M0=1 - np.mean(e1**2)/np.mean(e0**2), gain_M2_over_M1=1 - np.mean(e2**2)/np.mean(e1**2), clark_west_t=nw_t(cw, h - 1)))
            # in-sample coefficients before listing (sign prediction: asymmetry terms should enter log G(+2) and log G(-2) with opposite signs)
            tr = np.array([s for s in range(21, n - h) if dates[s + h] < LIST]); X = np.column_stack([np.ones(len(tr)), F[tr]])
            bb, se = ols_nw(X, y[tr], h - 1)
            coefs.append(dict(stock=nm, target='D' if key == 'D' else 'logG(%+d)' % key, h=h, n=len(tr), skew5=bb[5], t_skew5=bb[5]/se[5], skew22=bb[6], t_skew22=bb[6]/se[6], logrv22=bb[3], t_logrv22=bb[3]/se[3]))
pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: '%.4g' % v)
say('== decay ratio D against its first-order form log 3 + (8/3) sum r^3 / sum r^2 =='); say(pd.DataFrame(approx).to_string(index=False))
res = pd.DataFrame(rows); say('\n== pseudo out-of-sample forecasts (errors in logs) =='); say(res.to_string(index=False))
cf = pd.DataFrame(coefs); say('\n== coefficients on the asymmetry terms, sample before listing (Newey-West t, lag h-1) =='); say(cf.to_string(index=False))
res.to_csv('stats_main_log_results.csv', index=False, encoding='utf-8-sig'); cf.to_csv('stats_main_log_coefficients.csv', index=False, encoding='utf-8-sig')
open('stats_main_log_output.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
