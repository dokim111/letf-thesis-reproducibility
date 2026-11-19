# Figures for the statistics thesis draft. Usage: python3 stats_figs.py <long csv>
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from pathlib import Path
from matplotlib import font_manager
# Select an installed font that contains the Hangul labels used in these figures.
# Keep the requested Korean face first; CJK collections may be registered as JP.
for family in ('Noto Sans CJK KR', 'Noto Sans KR', 'NanumGothic',
               'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK JP'):
    try:
        font_path = font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        continue
    charmap = font_manager.get_font(font_path).get_charmap()
    if all(ord(ch) in charmap for ch in '삼성전자하이닉스직전거래일로그감쇠상장'):
        plt.rcParams['font.family'] = family
        break
else:
    raise RuntimeError('한글 그림 글꼴이 없습니다. Noto Sans CJK KR 등 한글 글꼴을 설치한 뒤 다시 실행하세요.')
plt.rcParams['axes.unicode_minus'] = False
Path('fig').mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d9d8d4'
d = pd.read_csv(sys.argv[1], encoding='utf-8-sig', dtype={'code': str}); d['r'] = d.close/(d.close - d.change_from_base_price) - 1
d = d[d.date >= 20150615]; g = lambda b, r: b*np.log1p(r) - np.log1p(b*r)
def style(ax):
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.6); ax.grid(axis='y', color=GRID, linewidth=0.5); ax.set_axisbelow(True)
name = {'SK하이닉스': ORANGE, '삼성전자': BLUE}; h = 20
fig, ax = plt.subplots(figsize=(6.3, 2.7), dpi=220)
for nm, ls in [('SK하이닉스', '--'), ('삼성전자', '-')]:
    x = d[d.stock == nm]; dt = pd.to_datetime(x.date.astype(str)); G = pd.Series(g(2, x.r.values)).rolling(h).sum().values
    ax.plot(dt, G, color=name[nm], linewidth=0.9, linestyle=ls, label=nm)
ax.set_yscale('log'); ax.set_ylabel('직전 20거래일 로그감쇠, β = +2', fontsize=8.5); ax.axvline(pd.Timestamp('2026-05-27'), color=MUTED, linewidth=0.8, linestyle=':')
ax.annotate('2026.5.27\n상장', xy=(pd.Timestamp('2026-05-27'), 0.96), xycoords=('data', 'axes fraction'), xytext=(-42, 0), textcoords='offset points', fontsize=7.5, color=MUTED, va='top')
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); style(ax); ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig('fig/fig_series.png', facecolor='white'); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), dpi=220)
for ax, nm in zip(axs, ['삼성전자', 'SK하이닉스']):
    r = d[d.stock == nm].r.values; n = len(r); c = lambda v: np.concatenate([[0], np.cumsum(v)]); f = lambda cs: cs[h:] - cs[:-h]
    D = np.log(f(c(g(-2, r)))) - np.log(f(c(g(2, r)))); kap = f(c(r**3))/f(c(r**2))
    ax.scatter(kap, D, s=3, color=name[nm], alpha=0.35, linewidth=0)
    xs = np.linspace(kap.min(), kap.max(), 50); ax.plot(xs, np.log(3) + 8/3*xs, color=INK, linewidth=0.9, linestyle='--', label='log 3 + (8/3)κ')
    ax.set_xlabel('κ = Σr³ / Σr² (20거래일)', fontsize=8.5); ax.set_ylabel('D = log G(−2) − log G(+2)', fontsize=8.5); ax.set_title(nm, fontsize=9, loc='left', color=INK); style(ax); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    print(nm, 'corr', round(np.corrcoef(D, kap)[0, 1], 4), 'kappa range', round(kap.min(), 3), round(kap.max(), 3))
fig.tight_layout(w_pad=1.5); fig.savefig('fig/fig_ratio.png', facecolor='white'); plt.close(fig)
