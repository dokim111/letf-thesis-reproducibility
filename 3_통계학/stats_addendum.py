# Statistics thesis, addendum to stats_main_log.py / stats_experiment.py.  Run: python3 stats_addendum.py   (no random numbers)
# (1) Cross-specification: targets {G, log G} x regressor sets, same expanding-window procedure (training rows s + h <= t).
# (2) Component forecasts with the log-specification regressors: log S2 (future sum of squares) and kappa = S3/S2.
# (3) Samples: 'pre_start' = forecast date before listing (as in the first draft); 'pre_strict' = whole target window before listing; 'post'.
# (5) Regressor factorial: variance block {raw, log} x asymmetry block {raw3, raw2, norm2, abs2 (unsigned placebo)}.
# (4) Clark-West with Newey-West lag h-1 and lag 40; M2 against the baseline M0; negative level forecasts.
import numpy as np, pandas as pd
START, MIN_TRAIN, EPS, LIST = 20150615, 750, 1e-8, 20260527
d = pd.read_csv('krx_stock_daily_long_20000104_20261113.csv', encoding='utf-8-sig', dtype={'code': str})
d['r'] = d.close / (d.close - d.change_from_base_price) - 1
g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
roll = lambda x, w: pd.Series(x).rolling(w).mean().values
def nw_t(f, L):
    f = np.asarray(f); n = len(f); u = f - f.mean(); v = u @ u / n
    for l in range(1, min(L, n - 1) + 1): v += 2*(1 - l/(L + 1))*(u[l:] @ u[:-l])/n
    return f.mean()/np.sqrt(v/n)
rows, info = [], []
for nm, x in d.groupby('stock'):
    x = x[x.date >= START].reset_index(drop=True); r = x.r.values; dates = x.date.values; n = len(r)
    info.append('%s days %d zero-return days %d' % (nm, n, int((r == 0).sum())))
    rv5, rv22, m22 = roll(r**2, 5), roll(r**2, 22), roll(r, 22)
    V = {'raw': [r**2, rv5, rv22], 'log': [np.log(r**2 + EPS), np.log(rv5 + EPS), np.log(rv22 + EPS)]}
    A = {'raw3': [r**3, roll(r**3, 5), roll(r**3, 22)], 'raw2': [roll(r**3, 5), roll(r**3, 22)],
         'norm2': [roll(r**3, 5)/(rv5 + EPS)**1.5, roll(r**3, 22)/(rv22 + EPS)**1.5],
         'abs2': [roll(np.abs(r)**3, 5), roll(np.abs(r)**3, 22)]}          # unsigned third moments: scale without sign
    gb = {b: g(b, r) for b in (2, -2)}
    cum = lambda v: np.concatenate([[0], np.cumsum(v)])
    for h in (5, 20):
        fut = lambda v: (lambda c: c[1+h:n+1] - c[1:n-h+1])(cum(v))
        tg = {}
        def put(k, v): tg[k] = np.full(n, np.nan); tg[k][:n-h] = v
        for b in (2, -2): put('G(%+d)' % b, fut(gb[b])); put('logG(%+d)' % b, np.log(fut(gb[b]) + EPS))
        tg['D'] = tg['logG(-2)'] - tg['logG(+2)']
        put('logS2', np.log(fut(r**2) + EPS)); put('kappa', fut(r**3)/(fut(r**2) + EPS))
        base = {'G(+2)': h*roll(gb[2], 22), 'G(-2)': h*roll(gb[-2], 22), 'logS2': np.log(h*rv22 + EPS), 'kappa': roll(r**3, 22)/(rv22 + EPS)}
        base['logG(+2)'] = np.log(base['G(+2)'] + EPS); base['logG(-2)'] = np.log(base['G(-2)'] + EPS); base['D'] = base['logG(-2)'] - base['logG(+2)']
        keys = list(tg); Y = np.column_stack([tg[k] for k in keys])
        end_date = np.full(n, 99999999); end_date[:n-h] = dates[h:]
        for vk in V:
            for ak in A:
                X1 = np.column_stack([np.ones(n)] + V[vk] + [m22]); X2 = np.column_stack([X1] + A[ak])
                P1 = np.full(Y.shape, np.nan); P2 = P1.copy()
                for t in range(22 + MIN_TRAIN + h, n - h):
                    tr = np.arange(21, t - h + 1)
                    P1[t] = X1[t] @ np.linalg.lstsq(X1[tr], Y[tr], rcond=None)[0]; P2[t] = X2[t] @ np.linalg.lstsq(X2[tr], Y[tr], rcond=None)[0]
                ok = ~np.isnan(P1[:, 0]) & ~np.isnan(Y[:, 0])
                for lab, mm in (('pre_start', ok & (dates < LIST)), ('pre_strict', ok & (end_date < LIST)), ('post', ok & (dates >= LIST))):
                    for j, k in enumerate(keys):
                        y = Y[mm, j]; e0, e1, e2 = y - base[k][mm], y - P1[mm, j], y - P2[mm, j]; cw = e1**2 - (e2**2 - (P1[mm, j] - P2[mm, j])**2)
                        rows.append(dict(stock=nm, h=h, target=k, variance_block=vk, asym_block=ak, sample=lab, n=int(mm.sum()),
                                         mse0=np.mean(e0**2), mse1=np.mean(e1**2), mse2=np.mean(e2**2),
                                         gain_M1_over_M0=1 - np.mean(e1**2)/np.mean(e0**2), gain_M2_over_M1=1 - np.mean(e2**2)/np.mean(e1**2), gain_M2_over_M0=1 - np.mean(e2**2)/np.mean(e0**2),
                                         cw_t_lag_h1=nw_t(cw, h - 1), cw_t_lag40=nw_t(cw, 40), share_M2_closer=np.mean(np.abs(e2) < np.abs(e1)),
                                         neg_pred_M1=int((P1[mm, j] < 0).sum()) if k.startswith('G(') else -1, neg_pred_M2=int((P2[mm, j] < 0).sum()) if k.startswith('G(') else -1))
res = pd.DataFrame(rows); res.to_csv('stats_addendum_results.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: '%.4g' % v); pd.set_option('display.max_rows', 500)
out = ['\n'.join(info)]
def show(title, q, cols): out.append('\n== %s ==\n%s' % (title, res.query(q)[cols].to_string(index=False)))
c = ['stock', 'h', 'target', 'variance_block', 'asym_block', 'n', 'gain_M1_over_M0', 'gain_M2_over_M1', 'gain_M2_over_M0', 'cw_t_lag_h1', 'cw_t_lag40', 'share_M2_closer', 'neg_pred_M2']
orig = "((variance_block=='raw' and asym_block=='raw3') or (variance_block=='log' and asym_block=='norm2'))"
show('A. post-listing, original two regressor sets, targets G and log G', "sample=='post' and target in ['G(+2)','G(-2)','logG(+2)','logG(-2)'] and " + orig, c)
show('B. whole target window before listing, log-specification regressors', "sample=='pre_strict' and variance_block=='log' and asym_block=='norm2' and target in ['logG(+2)','logG(-2)','D','logS2','kappa']", c)
show('C. post-listing, log-specification regressors, components', "sample=='post' and variance_block=='log' and asym_block=='norm2' and target in ['D','logS2','kappa']", c)
show('D. post-listing h=5, all six regressor sets', "sample=='post' and h==5 and target in ['G(+2)','G(-2)','logG(+2)','logG(-2)']", c)
show('E. forecast-date split before listing (first-draft sample), original two sets', "sample=='pre_start' and target in ['G(+2)','G(-2)','logG(+2)','logG(-2)','D'] and " + orig, c)
show('F. whole target window before listing, h=5 and 20, all six regressor sets, log G', "sample=='pre_strict' and target in ['logG(+2)','logG(-2)']", c)
show('G. post-listing, variance block raw: signed (raw2) against unsigned (abs2) third moments', "sample=='post' and variance_block=='raw' and asym_block in ['raw2','abs2'] and target in ['G(+2)','G(-2)','logG(+2)','logG(-2)']", c)
show('H. whole target window before listing, unsigned third moments (abs2), log G', "sample=='pre_strict' and asym_block=='abs2' and target in ['logG(+2)','logG(-2)']", c)
open('stats_addendum_output.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out))
