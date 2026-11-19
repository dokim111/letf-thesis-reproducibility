"""Generate the extension tables, including the symmetric downside/cubic comparison."""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "graduation_extension_results"
semis = pd.read_csv(OUT / "semivariance_log_results.csv")
sym = pd.read_csv(ROOT / "symmetric_extension_results/comparisons.csv")
losses = pd.read_csv(OUT / "common_scale_losses.csv")
gains = pd.read_csv(OUT / "common_scale_improvements.csv")


def p(x):
    return f"{100*float(x):.2f}"


def sci(x):
    x = float(x)
    if x < 100:
        return f"{x:.3f}"
    k = math.floor(math.log10(x))
    return rf"${x/10**k:.2f}\times10^{{{k}}}$"


def beta_name(beta):
    return "$+2$" if beta == 2 else "$-2$"


def start(caption, label, columns, headers):
    return [r"\begin{table}[htbp]\centering\small",
            rf"\caption{{{caption}}}\label{{{label}}}",
            rf"\begin{{tabular}}{{{columns}}}\toprule", headers + r"\\\midrule"]


def finish(lines, note):
    lines += [r"\bottomrule\end{tabular}", r"\smallskip", r"\begin{minipage}{0.97\textwidth}\footnotesize " + note + r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


lines = start("하방비중과 삼차 변수의 양방향 예측 비교", "tab:downside", "lrrrrrr", r"종목 & $\beta$ & $h$ & MD$|$M1 & MDK$|$MD & MDK$|$M2 & MD$|$M2")
for sample, sample_label in [("pre_strict", "상장 전: 목표 구간 전체가 상장 전"), ("post", "상장 후: 예측 시점이 상장일 이후")]:
    lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{sample_label}}}}}\\")
    for stock in ("삼성전자", "SK하이닉스"):
        for h in (5, 20):
            for beta in (2, -2):
                q = sym[(sym.stock == stock) & (sym.h == h) & (sym.beta == beta) & (sym["sample"] == sample)].set_index(["reference", "candidate"])
                values = [q.loc[pair, "improvement"] for pair in [("M1", "MD"), ("MD", "MDK"), ("M2", "MDK"), ("M2", "MD")]]
                lines.append(f"{stock} & {beta_name(beta)} & {h} & " + " & ".join(p(v) for v in values) + r"\\")
    if sample == "pre_strict":
        lines.append(r"\midrule")
note = r"개선율 단위는 \%. A$|$B는 B 대비 A의 평균제곱예측오차 개선율이며 음수는 악화다. MDK$|$MD와 MDK$|$M2는 추가 정보의 방향을 바꾼 비교다. MD$|$M2는 비중첩 모형의 직접 비교이며, 이 표에 새로운 유의성 검정은 붙이지 않았다. M2$|$M1은 표 \ref{tab:level} 및 재현 CSV에 있다. 상장 전 표본 수는 $h=5$에서 1,904개, $h=20$에서 1,874개이며 상장 후는 각각 111개와 96개다."
(ROOT / "semivariance_table.tex").write_text(finish(lines, note), encoding="utf-8")

lines = start("같은 양의 수준 예측에 대한 손실함수별 개선율", "tab:common_loss", "lrrrrr", r"종목 & $\beta$ & $h$ & 수준 MSE & 로그 MSE & QLIKE")
for sample, sample_label in [("pre_strict", "상장 전"), ("post", "상장 후")]:
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{sample_label}}}}}\\")
    for stock in ("삼성전자", "SK하이닉스"):
        for h in (5, 20):
            for beta in (2, -2):
                q = gains[(gains.stock == stock) & (gains.h == h) & (gains.beta == beta) & (gains["sample"] == sample) & (gains["floor"] == 1e-8) & (gains.smaller == "log_M1_fit_log_smear") & (gains.larger == "log_M2_fit_log_smear")].iloc[0]
                lines.append(f"{stock} & {beta_name(beta)} & {h} & {p(q.mse_improvement)} & {p(q.log_mse_improvement)} & {p(q.qlike_improvement)}" + r"\\")
    if sample == "pre_strict":
        lines.append(r"\midrule")
note = r"단위는 \%. 기존 로그 설명변수 사양의 M1과 M2를 각각 훈련잔차 기반으로 재변환한 후, 같은 수준 $G$와 같은 양의 예측값으로 세 손실을 계산하였다. 단순 로그회귀 예측 자체를 평가한 기존 표와 대상 예측값이 다르다. 두 모형 모두 이 표의 표본에서 양수 하한에 걸린 예측은 없다."
(ROOT / "common_loss_table.tex").write_text(finish(lines, note), encoding="utf-8")

lines = start("수준·원자료 사양의 상장 후 양수 투영과 QLIKE", "tab:qlike_projection", "lrrrrrr", r"종목 & $\beta$ & $h$ & 음수 예측 & M1 & M2($10^{-8}$) & M2($10^{-6}$)")
for stock in ("삼성전자", "SK하이닉스"):
    for h in (5, 20):
        for beta in (2, -2):
            q = losses[(losses.stock == stock) & (losses.h == h) & (losses.beta == beta) & (losses["sample"] == "post")]
            one = q[(q.model == "raw_M1_fit_level") & (q["floor"] == 1e-8)].iloc[0]
            two = q[(q.model == "raw_M2_fit_level") & (q["floor"] == 1e-8)].iloc[0]
            other = q[(q.model == "raw_M2_fit_level") & (q["floor"] == 1e-6)].iloc[0]
            lines.append(f"{stock} & {beta_name(beta)} & {h} & {int(two.n_nonpositive)} & {sci(one.qlike)} & {sci(two.qlike)} & {sci(other.qlike)}" + r"\\")
note = r"세 손실 열은 평균 QLIKE(무차원)이며 개선율이 아니다. 음수 예측 수는 투영 전 M2의 값이고, M1은 모두 양수다. M2는 $f=\max(\widehat G,\eta)$로 투영한다. 극단적인 QLIKE는 소수의 비양수 예측과 하한 선택에 민감하므로 이를 잠재 예측력의 크기로 해석하지 않는다."
(ROOT / "projection_qlike_table.tex").write_text(finish(lines, note), encoding="utf-8")

print("Generated semivariance_table.tex, common_loss_table.tex, projection_qlike_table.tex")
