# Figures and data tables (tables 2 and 3) for the mathematics thesis draft. Usage: python3 math_figs_tables.py <long csv>
import sys, numpy as np, pandas as pd, mpmath as mp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
# Select an installed font with the Hangul glyphs used in these figures.
for family in ('Noto Sans CJK KR', 'Noto Sans KR', 'NanumGothic',
               'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK JP'):
    try:
        font_path = font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        continue
    charmap = font_manager.get_font(font_path).get_charmap()
    if all(ord(ch) in charmap for ch in '일간수익률순감소볼록최솟점연도로그감쇠하이닉스이차경계삼차근사로얻은정확한값'):
        plt.rcParams['font.family'] = family
        break
else:
    raise RuntimeError('한글 그림 글꼴이 없습니다. Noto Sans CJK KR 등 한글 글꼴을 설치한 뒤 다시 실행하세요.')
plt.rcParams['axes.unicode_minus'] = False
Path('fig').mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d9d8d4'
d = pd.read_csv(sys.argv[1], encoding='utf-8-sig', dtype={'code': str}); d['r'] = d.close/(d.close - d.change_from_base_price) - 1
g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
def consts(b, c):
    mp.mp.dps = 30
    gg = lambda r: b*mp.log1p(r) - mp.log1p(b*r); a2, a3 = mp.mpf(b)*(b-1)/2, -mp.mpf(b)*(b*b-1)/3
    h = lambda r: gg(r)/r**2; rho3 = lambda r: (gg(r) - a2*r**2 - a3*r**3)/r**4
    c = mp.mpf(c)
    if b > 1: m, M = h(c), h(-c)
    elif b == -1: m, M = mp.mpf(1), h(c)
    else:
        rs = mp.findroot(lambda r: r*(b*(b-1)*r/((1+r)*(1+b*r))) - 2*gg(r), (mp.mpf('-0.6'), mp.mpf('-0.05')), solver='anderson', tol=1e-20)
        m, M = h(max(rs, -c)), h(c)
    r3 = sorted([rho3(-c), rho3(c)]) if b != -1 else [mp.mpf(1)/2, rho3(c)]
    return float(m), float(M), float(r3[0]), float(r3[1])
def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.6); ax.grid(axis='y', color=GRID, linewidth=0.5); ax.set_axisbelow(True)
# Figure 1: h_beta for beta = 2 and -2
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), dpi=220)
for ax, b, lo, hi, col in [(axs[0], 2, -0.45, 0.8, BLUE), (axs[1], -2, -0.8, 0.45, ORANGE)]:
    x = np.linspace(lo, hi, 2001); x = x[np.abs(x) > 1e-6]; ax.plot(x, g(b, x)/x**2, color=col, linewidth=1.6)
    ax.axvspan(-0.3, 0.3, color=GRID, alpha=0.45, linewidth=0); ax.set_xlabel('일간 수익률 r', fontsize=8.5); ax.set_ylabel('h(r) = g(r) / r²', fontsize=8.5)
    ax.set_title('β = +2: 순감소' if b == 2 else 'β = −2: 볼록, 최솟점 r* ≈ −0.334', fontsize=9, loc='left', color=INK); style(ax)
    ax.set_ylim(0, 6 if b == 2 else 9)
    if b == -2: ax.plot([-0.334483], [2.700938], marker='o', markersize=4, color=INK); ax.annotate('r*', (-0.334483, 2.700938), xytext=(-4, -12), textcoords='offset points', fontsize=8, color=INK)
    ax.annotate('|r| ≤ 0.30', (0, ax.get_ylim()[1]*0.93), ha='center', fontsize=7.5, color=MUTED)
fig.tight_layout(w_pad=1.5); fig.savefig('fig/fig_h.png', facecolor='white'); plt.close(fig)
# Tables
print('constants at c = 0.30:', {b: [round(v, 6) for v in consts(b, 0.30)] for b in (2, -1, -2)})
rows = []
for nm, x in d.groupby('stock'):
    for lab, lo, hi in [('2015.6.15~2026.11.13', 20150615, 20261113), ('2026.5.28~2026.11.13', 20260528, 20261113)]:
        r = x[(x.date >= lo) & (x.date <= hi)].r.values; S2, S3, S4 = (r**2).sum(), (r**3).sum(), (r**4).sum()
        for b in (2, -1, -2):
            m, M, r3lo, r3hi = consts(b, 0.30); a2, a3 = b*(b-1)/2, -b*(b*b-1)/3; G = g(b, r).sum(); Q = a2*S2; C = Q + a3*S3
            rows.append(dict(stock=nm, period=lab, n=len(r), beta=b, S2=S2, S3=S3, S4=S4, G=G, quad=Q, cubic=C, q_lo=m*S2, q_hi=M*S2, c_lo=C + r3lo*S4, c_hi=C + r3hi*S4, maxabs=np.abs(r).max()))
t = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.float_format', lambda v: '%.4f' % v); print(t.to_string(index=False))
assert ((t.G >= t.q_lo) & (t.G <= t.q_hi) & (t.G >= t.c_lo - 1e-12) & (t.G <= t.c_hi + 1e-12)).all()
# Table 3: bounds of the form k*sum(y^2), y = log(1+r) (tangent point at the origin), for the post-listing sample
def ycoef(b, c, ngrid=600001):
    rr = np.linspace(-c, c, ngrid); rr = rr[rr != 0]; k = g(b, rr)/np.log1p(rr)**2
    ends = sorted([g(b, -c)/np.log1p(-c)**2, g(b, c)/np.log1p(c)**2])
    assert abs(k.min() - ends[0]) < 1e-9 and abs(k.max() - ends[1]) < 1e-9      # extrema over |r| <= c are attained at the end points
    return ends
rows3, tex2, tex3 = [], [], []
for nm, x in d.groupby('stock'):
    r = x[(x.date >= 20260528) & (x.date <= 20261113)].r.values; S2, S3, S4, Y2 = (r**2).sum(), (r**3).sum(), (r**4).sum(), (np.log1p(r)**2).sum()
    for b in (2, -2):
        m, M, r3lo, r3hi = consts(b, 0.30); a2, a3 = b*(b-1)/2, -b*(b*b-1)/3; klo, khi = ycoef(b, 0.30); G = g(b, r).sum(); C = a2*S2 + a3*S3
        assert klo*Y2 <= G <= khi*Y2
        rows3.append(dict(stock=nm, beta=b, n=len(r), G=G, r2_lo=m*S2, r2_hi=M*S2, y2_lo=klo*Y2, y2_hi=khi*Y2, c_lo=C + r3lo*S4, c_hi=C + r3hi*S4,
                          ratio_r2=(M - m)*S2/((r3hi - r3lo)*S4), ratio_y2=(khi - klo)*Y2/((r3hi - r3lo)*S4)))
t3 = pd.DataFrame(rows3); pd.set_option('display.float_format', lambda v: '%.6f' % v); print('Table 3 (post-listing sample):'); print(t3.to_string(index=False))
# LaTeX rows for tables 2 and 3 (written to math_tables.json; the -1x rows keep four decimals for the cubic-based bounds)
import json
bs = lambda b: '$%+d$' % b
for st in ('SK하이닉스', '삼성전자'):
    q = t[t.stock == st]
    for i, row in enumerate(q.itertuples()):
        f3 = '[%.4f, %.4f]' if row.beta == -1 else '[%.3f, %.3f]'
        first = st if i == 0 else ''; per = ('2015.6--2026.11' if row.period.startswith('2015') else '2026.5--2026.11') if i % 3 == 0 else ''
        tex2.append(' & '.join([first, per, bs(row.beta), '%.4f' % row.G, '%.4f' % row.quad, '%.4f' % row.cubic, '[%.3f, %.3f]' % (row.q_lo, row.q_hi), f3 % (row.c_lo, row.c_hi)]) + r'\\' + (r'\midrule' if (i == len(q) - 1 and st == 'SK하이닉스') else ''))
    q3 = t3[t3.stock == st]
    for i, row in enumerate(q3.itertuples()):
        tex3.append(' & '.join([st if i == 0 else '', bs(row.beta), '[%.3f, %.3f]' % (row.r2_lo, row.r2_hi), '[%.3f, %.3f]' % (row.y2_lo, row.y2_hi), '[%.3f, %.3f]' % (row.c_lo, row.c_hi)]) + r'\\')
json.dump({'DATA': '\n'.join(tex2), 'COORD': '\n'.join(tex3)}, open('math_tables.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('width ratio quadratic/cubic:', ((t.q_hi - t.q_lo)/(t.c_hi - t.c_lo)).round(1).tolist())
# Figure 2: bounds vs exact G by calendar year, SK hynix, beta = -2 and +2
x = d[(d.stock == 'SK하이닉스') & (d.date >= 20150615)].copy(); x['year'] = x.date // 10000
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), dpi=220)
for ax, b, col in [(axs[0], 2, BLUE), (axs[1], -2, ORANGE)]:
    m, M, r3lo, r3hi = consts(b, 0.30); a2, a3 = b*(b-1)/2, -b*(b*b-1)/3; yrs = sorted(x.year.unique())
    for i, y in enumerate(yrs):
        r = x[x.year == y].r.values; S2, S3, S4 = (r**2).sum(), (r**3).sum(), (r**4).sum(); G = g(b, r).sum(); C = a2*S2 + a3*S3
        ax.plot([i, i], [m*S2, M*S2], color=GRID, linewidth=5, solid_capstyle='butt'); ax.plot([i, i], [C + r3lo*S4, C + r3hi*S4], color=col, linewidth=5, solid_capstyle='butt'); ax.plot([i], [G], marker='_', markersize=9, color=INK, markeredgewidth=1.2)
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels([str(y)[2:] for y in yrs], fontsize=7.5); ax.set_xlabel('연도', fontsize=8.5); ax.set_ylabel('연간 로그감쇠 G', fontsize=8.5)
    ax.set_title('SK하이닉스, β = %s' % ('+2' if b == 2 else '−2'), fontsize=9, loc='left', color=INK); style(ax)
    ax.plot([], [], color=GRID, linewidth=5, label='이차 경계'); ax.plot([], [], color=col, linewidth=5, label='삼차근사로 얻은 G의 경계'); ax.plot([], [], marker='_', linestyle='none', color=INK, label='정확한 값'); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
fig.tight_layout(w_pad=1.5); fig.savefig('fig/fig_bounds.png', facecolor='white'); plt.close(fig)
