# Builds the LaTeX table bodies of the statistics draft from stats_addendum_results.csv (so no number is typed by hand)
# and prints the ranges quoted in the text.  Usage: python3 make_tables.py [template.tex output.tex]
# Units: LEVEL, COMP, CROSSPOST and CROSSPRE keep gains as ratios (0.01 = 1% improvement);
# EVENT, YEAR and PHASE show gains in percent. Captions/notes in the template state these units.
import pandas as pd, numpy as np, json
r = pd.read_csv('stats_addendum_results.csv')
KS = {'SK하이닉스': 'SK하이닉스', '삼성전자': '삼성전자'}
def f(v, k=3):
    s = ('%.' + str(k) + 'f') % v
    if float(s) == 0: s = s.replace('-', '')
    return ('$-$' + s[1:]) if s.startswith('-') else s
def get(stock, h, target, vb, ab, sample): 
    q = r[(r.stock == stock) & (r.h == h) & (r.target == target) & (r.variance_block == vb) & (r.asym_block == ab) & (r['sample'] == sample)]
    assert len(q) == 1; return q.iloc[0]
T = {}
# table: log G, main specification
L = []
for st in ('SK하이닉스', '삼성전자'):
    first = True
    for h in (5, 20):
        for b in ('+2', '-2'):
            a = get(st, h, 'logG(%s)' % b, 'log', 'norm2', 'pre_strict'); p = get(st, h, 'logG(%s)' % b, 'log', 'norm2', 'post')
            L.append(' & '.join([st if first else '', '$%s$' % b.replace('-', '-'), str(h), f(a.gain_M1_over_M0), f(a.gain_M2_over_M1), f(a.cw_t_lag_h1, 2), f(a.cw_t_lag40, 2),
                                 f(p.gain_M1_over_M0), f(p.gain_M2_over_M1), f(p.gain_M2_over_M0), f(p.cw_t_lag_h1, 2)]) + r'\\'); first = False
T['LEVEL'] = '\n'.join(L)
# table: components
L = []
for st in ('SK하이닉스', '삼성전자'):
    first = True
    for h in (5, 20):
        c = [st if first else '', str(h)]
        for tg in ('logS2', 'kappa', 'D'):
            a = get(st, h, tg, 'log', 'norm2', 'pre_strict'); c += [f(a.gain_M2_over_M1), f(a.cw_t_lag_h1, 2), f(a.cw_t_lag40, 2)]
        L.append(' & '.join(c) + r'\\'); first = False
T['COMP'] = '\n'.join(L)
# table: cross-specification after listing, h = 5
sets = [('raw', 'raw3'), ('raw', 'raw2'), ('raw', 'norm2'), ('log', 'raw2'), ('log', 'norm2')]
L = []
for st in ('SK하이닉스', '삼성전자'):
    first = True
    for b in ('+2', '-2'):
        for tg, lab in (('G(%s)' % b, '$G$'), ('logG(%s)' % b, r'$\log G$')):
            c = [st if first else '', '$%s$' % b, lab] + [f(get(st, 5, tg, v, a, 'post').gain_M2_over_M1) for v, a in sets]
            c += [f(get(st, 5, tg, 'raw', 'raw3', 'post').gain_M2_over_M0, 2), f(get(st, 5, tg, 'log', 'norm2', 'post').gain_M2_over_M0, 2)]
            L.append(' & '.join(c) + r'\\'); first = False
T['CROSSPOST'] = '\n'.join(L)
# table: cross-specification before listing (whole window before listing), log G
L = []
for st in ('SK하이닉스', '삼성전자'):
    first = True
    for h in (5, 20):
        for b in ('+2', '-2'):
            c = [st if first else '', '$%s$' % b, str(h)] + [f(get(st, h, 'logG(%s)' % b, v, a, 'pre_strict').gain_M2_over_M1) for v, a in sets]
            L.append(' & '.join(c) + r'\\'); first = False
T['CROSSPRE'] = '\n'.join(L)
# Post-hoc event, year, and non-overlapping phase tables (percentage units).
from pathlib import Path
sens = Path('sensitivity/results')
summary = pd.read_csv(sens/'summary.csv')
events = pd.read_csv(sens/'event_rows.csv')
years = pd.read_csv(sens/'year_rows.csv')
phases = pd.read_csv(sens/'phase_rows.csv')
L = []
for st in ('삼성전자', 'SK하이닉스'):
    for j, b in enumerate((2, -2)):
        a = summary[(summary.stock == st) & (summary.spec == 'raw_post') & (summary.h == 5) & (summary.beta == b)]
        ee = events[(events.stock == st) & (events.h == 5) & (events.beta == b)]
        q1 = ee[ee['mode'] == 'target_only']; q2 = ee[ee['mode'] == 'target_and_feature22']
        assert len(a) == len(q1) == len(q2) == 1
        assert int(a.iloc[0]['n']) == 111 and int(q1.iloc[0].n_remain) == 106 and int(q2.iloc[0].n_remain) == 84
        row = [st if j == 0 else '', '$%+d$' % b] + [f(100*float(q), 2) for q in (a.iloc[0].gain, q1.iloc[0].gain, q2.iloc[0].gain)]
        L.append(' & '.join(row) + r'\\')
T['EVENT'] = '\n'.join(L)
L = []
for h in (5, 20):
    for b in (2, -2):
        q = years[(years.stock == '삼성전자') & (years.h == h) & (years.beta == b)]
        assert len(q) == 9
        row = [str(h), '$%+d$' % b, '%d--%d' % (q.n.min(), q.n.max()), r'$[%.2f,\ %.2f]$' % (100*q.gain.min(), 100*q.gain.max()), '%d/%d' % ((q.gain > 0).sum(), len(q))]
        L.append(' & '.join(row) + r'\\')
T['YEAR'] = '\n'.join(L)
L = []
for st in ('삼성전자', 'SK하이닉스'):
    first = True
    for h in (5, 20):
        for b in (2, -2):
            q = phases[(phases.stock == st) & (phases.h == h) & (phases.beta == b)]
            assert len(q) == h
            row = [st if first else '', str(h), '$%+d$' % b, '%d--%d' % (q.n.min(), q.n.max()), r'$[%.2f,\ %.2f]$' % (100*q.gain.min(), 100*q.gain.max()), '%d/%d' % ((q.gain > 0).sum(), len(q))]
            L.append(' & '.join(row) + r'\\'); first = False
T['PHASE'] = '\n'.join(L)

json.dump(T, open('tables.json', 'w', encoding='utf-8'), ensure_ascii=False)
import sys
if len(sys.argv) == 3:                                   # python3 make_tables.py <template.tex> <output.tex>
    tex = open(sys.argv[1], encoding='utf-8').read()
    for k, v in T.items():
        assert tex.count('@@%s@@' % k) == 1; tex = tex.replace('@@%s@@' % k, v)
    assert '@@' not in tex; open(sys.argv[2], 'w', encoding='utf-8').write(tex)
# ranges quoted in the text
def rng(q, col): return '%.4f .. %.4f' % (q[col].min(), q[col].max())
main = r[(r.variance_block == 'log') & (r.asym_block == 'norm2')]
for st in ('삼성전자', 'SK하이닉스'):
    q = main[(main.stock == st) & (main['sample'] == 'pre_strict')]
    for tg in (['logG(+2)', 'logG(-2)'], ['logS2'], ['kappa'], ['D']):
        qq = q[q.target.isin(tg)]; print(st, 'pre_strict', tg, 'M2/M1', rng(qq, 'gain_M2_over_M1'), 'CW', rng(qq, 'cw_t_lag_h1'), 'CW40', rng(qq, 'cw_t_lag40'), 'M1/M0', rng(qq, 'gain_M1_over_M0'), 'closer', rng(qq, 'share_M2_closer'))
q = main[(main['sample'] == 'post') & main.target.isin(['logG(+2)', 'logG(-2)'])]; print('post main logG M1/M0', rng(q, 'gain_M1_over_M0'), 'M2/M0', rng(q, 'gain_M2_over_M0'), 'M2/M1', rng(q, 'gain_M2_over_M1'), 'CW', rng(q, 'cw_t_lag_h1'))
q = main[(main['sample'] == 'post') & main.target.isin(['logS2'])]; print('post logS2 M2/M1', q[['stock', 'h', 'gain_M2_over_M1']].values.tolist())
q = main[(main['sample'] == 'post') & main.target.isin(['kappa', 'D'])]; print('post kappa/D M2/M1', rng(q, 'gain_M2_over_M1'))
p5 = r[(r['sample'] == 'post') & (r.h == 5) & r.target.isin(['G(+2)', 'G(-2)', 'logG(+2)', 'logG(-2)'])]
for v, a in sets + [('raw', 'abs2'), ('log', 'raw3'), ('log', 'abs2')]:
    q = p5[(p5.variance_block == v) & (p5.asym_block == a)]
    for pre, lab in (('G(', 'G'), ('logG', 'logG')):
        qq = q[q.target.str.startswith(pre)]; print('post h5', v, a, lab, 'M2/M1', rng(qq, 'gain_M2_over_M1'), 'M2/M0', rng(qq, 'gain_M2_over_M0'), 'M1/M0', rng(qq, 'gain_M1_over_M0'), 'CW', rng(qq, 'cw_t_lag_h1'), 'CW40', rng(qq, 'cw_t_lag40'), 'neg', qq.neg_pred_M2.tolist())
p20 = r[(r['sample'] == 'post') & (r.h == 20) & r.target.isin(['G(+2)', 'G(-2)', 'logG(+2)', 'logG(-2)'])]
for v, a in sets:
    q = p20[(p20.variance_block == v) & (p20.asym_block == a)]; print('post h20', v, a, 'M2/M1', q.gain_M2_over_M1.round(3).tolist())
pp = r[(r['sample'] == 'pre_strict') & r.target.isin(['logG(+2)', 'logG(-2)'])]
for v, a in sets:
    for st in ('삼성전자', 'SK하이닉스'):
        q = pp[(pp.variance_block == v) & (pp.asym_block == a) & (pp.stock == st)]; print('pre', v, a, st, 'M2/M1', rng(q, 'gain_M2_over_M1'))
pg = r[(r['sample'] == 'pre_strict') & r.target.isin(['G(+2)', 'G(-2)']) & (r.variance_block == 'raw') & (r.asym_block == 'raw3')]; print('pre strict G raw/raw3 M2/M1', pg.gain_M2_over_M1.round(4).tolist(), 'M1/M0', pg.gain_M1_over_M0.round(3).tolist())
ab = p5[(p5.variance_block == 'raw') & (p5.asym_block == 'abs2')]; print(ab[['stock', 'target', 'gain_M2_over_M1']].to_string(index=False))
