# 통계학: 로그감쇠 예측과 사양 민감도 재현

삼성전자와 SK하이닉스 일별 수익률로 계산한 레버리지 로그감쇠를 대상으로, 분산 정보·삼차 변수·하방비중의 예측 성과와 손실함수·평가기간 민감도를 계산합니다. 공개 묶음은 분석 코드, 패키지 버전, 비개인 집계 결과, 표 생성 코드로 구성됩니다.

**원자료는 포함하지 않습니다.** 동봉 집계 CSV에서 표를 다시 만드는 작업은 바로 실행할 수 있습니다. 예측을 처음부터 재계산하려면 아래 KRX 입력을 별도로 준비해야 합니다. 전체 논문 TEX/PDF와 개인별 제출 서류도 포함하지 않습니다.

## 환경과 실행 위치

Python 3.12.14에서 사용한 패키지 버전을 `requirements.txt`에 기록했습니다. 가상환경을 만든 뒤 아래 명령을 실행합니다. 경로가 현재 작업 폴더에 상대적인 기존 스크립트가 있으므로 **이 문서의 명령은 모두 `3_통계학` 안에서 실행**합니다.

```bash
cd 3_통계학
python3 -m pip install -r requirements.txt
```

분석에는 난수나 시드가 없습니다. 반복 최소제곱의 환경 차이를 줄이려면 실행 전 `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`로 설정하십시오. 두 확장 스크립트도 기존 환경변수가 없을 때 이 값을 적용합니다. BLAS 구현에 따라 끝자리 차이가 날 수 있습니다. 그림 생성에만 한글 글꼴(Noto Sans CJK KR 등)이 필요합니다.

## 1. 원자료 없이 집계표 재생성

아래 두 명령은 동봉한 집계 CSV만 읽습니다.

```bash
python3 make_tables.py
python3 make_graduation_extension_tables.py
```

첫 명령은 `tables.json`을 다시 만들고 본문에 사용한 수치 범위를 출력합니다. 두 번째는 `semivariance_table.tex`, `common_loss_table.tex`, `projection_qlike_table.tex`라는 표 조각을 생성합니다. 이 TEX 파일들은 전체 논문이 아니며 독립 컴파일용 문서도 아닙니다. 전체 논문 템플릿은 공개하지 않으므로 `make_tables.py`의 선택적 원고 생성 인수는 이 묶음의 재현 절차에서 사용하지 않습니다.

| 논문의 표·분석 내용 | 원계산 | 집계 결과 / 표 생성 키 |
|---|---|---|
| 로그감쇠비와 일차식의 관계, 삼차 변수 계수 | `stats_main_log.py` | 표준 출력, `stats_main_log_coefficients.csv` |
| 기술통계와 실현 근사오차 | `stats_experiment.py` | 표준 출력 |
| 주 사양의 예측, 분산·삼차 성분, 교차 사양 | `stats_addendum.py` | `stats_addendum_results.csv`; `tables.json`의 `LEVEL`, `COMP`, `CROSSPOST`, `CROSSPRE` |
| 급등일·연도·위상별 민감도 | `sensitivity/reproduce_sensitivity.py` | `sensitivity/results/`; `EVENT`, `YEAR`, `PHASE` |
| 하방비중과 삼차 변수의 양방향 비교 | `stats_symmetric_extension.py` | `symmetric_extension_results/comparisons.csv`, `feature_correlations.csv`; `semivariance_table.tex` |
| 같은 수준 예측에서의 MSE·로그 MSE·QLIKE | `stats_graduation_extension.py` | `graduation_extension_results/common_scale_*.csv`; `common_loss_table.tex` |
| 비양수 예측의 투영과 QLIKE | `stats_graduation_extension.py` | `graduation_extension_results/common_scale_losses.csv`; `projection_qlike_table.tex` |

CSV의 개선율은 기본적으로 비율입니다(`0.01`은 1%). `make_tables.py`의 `LEVEL`, `COMP`, `CROSSPOST`, `CROSSPRE`는 그 비율을 표시하고, `EVENT`, `YEAR`, `PHASE`와 확장표의 개선율은 100을 곱한 백분율을 표시합니다. QLIKE 손실 수준은 개선율과 별개입니다. 표 제목·주와 CSV 열 이름을 함께 확인하십시오.

## 2. 별도로 준비할 원자료

파일명과 위치:

```text
3_통계학/krx_stock_daily_long_20000104_20261113.csv
```

출처는 [KRX 정보데이터시스템](https://data.krx.co.kr/)의 개별종목 시세입니다. 삼성전자(005930)와 SK하이닉스(000660)를 각각 내려받아 다음 열 이름으로 합칩니다. 원자료는 이 저장소가 대신 배포하지 않습니다.

| 열 | 형식·의미 | 계산에 필요 |
|---|---|---|
| `date` | `YYYYMMDD` 정수 | 필수 |
| `stock` | `삼성전자` 또는 `SK하이닉스` | 필수 |
| `code` | `005930` 또는 `000660`, 6자리 문자열 | 식별용 권장 |
| `close` | 당일 종가 | 필수 |
| `change_from_base_price` | 거래소 기준가 대비 부호 있는 가격변화 | 필수 |

원래 입력에는 `open`, `high`, `low`, `change_type_code`, `return_pct_krx`, `volume`, `trading_value_krw`, `listed_shares`도 있지만 이 폴더의 계산은 이 추가 열을 요구하지 않습니다. CSV는 UTF-8(또는 UTF-8 BOM)로 저장하고, `(stock,date)` 중복 없이 **각 종목 안에서 날짜 오름차순**으로 정렬하십시오. 누락 거래일·결측치가 있으면 창과 평가 표본이 달라집니다.

참고 입력은 2000-01-04~2026-11-13의 종목당 6,621행, 총 13,242행입니다. 모든 본 분석은 2015-06-15 이후로 제한하여 종목당 2,802행을 사용합니다. 이 분석구간이 완전하면 그 이전 행은 계산에 쓰이지 않습니다. 참고 자료 수집일은 2026-11-14입니다.

수익률은 다음 식으로 복원합니다.

```python
r = close / (close - change_from_base_price) - 1
```

`change_from_base_price`를 분할 등을 고려하지 않은 전일 종가 차이로 바꾸거나, 반올림된 공표 등락률을 `r`로 대체하면 재현값이 달라질 수 있습니다. 실제 입력 구성은 상위 `DATA_SOURCES.md`와 함께 확인하십시오.

## 3. 원자료에서 전체 계산

원자료를 위 위치에 둔 뒤 아래 순서로 실행합니다. 확장 분석은 앞서 계산한 참조 CSV를 읽습니다.

```bash
python3 stats_main_log.py
python3 stats_experiment.py
python3 stats_addendum.py
python3 eps_sensitivity.py
python3 levels_diag.py
python3 sensitivity/reproduce_sensitivity.py
python3 stats_graduation_extension.py
python3 stats_symmetric_extension.py
python3 make_tables.py
python3 make_graduation_extension_tables.py
```

선택 진단과 그림:

```bash
python3 check_overlap_windows.py . --output overlap_windows.json
python3 stats_figs.py krx_stock_daily_long_20000104_20261113.csv
```

`check_overlap_windows.py`는 확장 분석에서 생성되는 날짜별 예측 파일이 필요하므로 `stats_symmetric_extension.py` 이후 실행합니다. 사건을 제외한 뒤 남은 평가창의 중첩을 확인하며 예측을 다시 적합하지 않습니다. `eps_sensitivity.py`, `levels_diag.py`는 수치 안정성과 수준 사양의 보조 진단입니다.

## 고정된 설계와 해석 범위

- 배율은 `+2`, `-2`, 예측기간은 5·20거래일입니다. 과거 요약창은 1·5·22거래일, 최소 훈련 설정은 750, `EPS=1e-8`입니다.
- 예측시점 `t`에 `s+h <= t`인 완결 목표만 훈련에 사용합니다. 첫 예측 인덱스는 0부터 세어 `22+750+h`입니다.
- 본문 주 결과는 `stats_addendum_results.csv`의 `pre_strict`(미래 목표구간 전체가 2026-05-27 이전)와 `post`(예측시점이 2026-05-27 이후)를 사용합니다. `stats_main_log_results.csv`와 `stats_experiment_results.csv`의 초기 `before listing` 분류 및 `pre_start`는 예측시점만으로 나눈 보조 결과이므로 본문 주 표본과 혼용하지 마십시오.
- M1은 분산 대리변수와 22일 평균 수익률, M2는 M1에 5·22일 정규화 삼차 변수를 추가합니다. MD는 M1에 하방비중을, MDK는 MD에 삼차 변수를 추가합니다. 변수와 사양의 정확한 정의는 해당 스크립트에서 고정되어 있습니다.
- 중첩 로그 모형의 Clark–West 통계량은 Newey–West 시차 `h-1`과 40을 보고합니다. 재변환한 수준 예측·QLIKE·비중첩 직접 비교에 같은 검정을 임의로 붙이지 않습니다.
- 2026-07-31 사건 제외는 이미 생성한 예측의 평가창을 제외하는 사후 민감도 분석입니다. 훈련자료에서 사건을 삭제하거나 독립된 검증표본을 새로 만든 것으로 해석하지 않습니다. 사양도 사후에 정리된 탐색적 분석입니다.
- 로그→수준 재변환은 해당 훈련행의 잔차만 사용합니다. 양수 투영 하한은 `1e-8`, `1e-6`이며, 손실함수별 비교는 같은 수준 목표와 예측값에 적용합니다.

## 공개·제외 파일

동봉 CSV는 모형·기간·종목별 집계 손실, 계수, 상관계수, 민감도 요약입니다. `event_rows.csv`의 `event_return`은 단일 사건(2026-07-31)의 설명값이며 일별 수익률 시계열이 아닙니다. 공개 집계값으로 원래 가격 시계열을 복원할 수는 없습니다.

원자료 CSV, 날짜별 목표·예측 전체, 원자료 해시와 로컬 경로가 기록되는 검증 JSON, 전체 논문·표지·개인정보·내부 검토 기록은 제외했습니다. 전체 실행은 이러한 로컬 진단 파일과 그림을 새로 만들 수 있으나 `.gitignore`로 공개 커밋 대상에서 제외합니다. 특히 다음 파일은 로컬 재현용이며 저장소에 추가하지 않습니다.

- `graduation_extension_results/forecasts.csv.gz`
- `symmetric_extension_results/forecasts_log.csv.gz`
- `sensitivity/results/forecast_rows.csv`
- `overlap_windows.json`, 확장 폴더의 `validation.json`, `design.json`

코드·동봉 집계 결과의 계산 내용은 비공개 작업본과 같습니다. 이 저장소의 집계 CSV 재생성 성공은 원자료에서 예측을 다시 실행했다는 뜻과 구분해야 합니다.
