# 수리과학: 로그감쇠 경계의 수치 검산

유계 일수익률에서의 레버리지 로그감쇠, 정규화 나머지항의 극값, 차수별 전이 임계값을 검산하는 코드입니다. 해석적 증명을 수치 계산으로 대체하지 않으며, 이 저장소에는 전체 논문이나 개인별 제출 서류를 넣지 않았습니다.

**이론 검산은 시장 자료 없이 실행할 수 있습니다.** 두 종목의 수치 예시와 그림을 처음부터 계산하려면 KRX 원자료를 별도로 준비해야 합니다. 원자료는 이 저장소에 포함하지 않습니다.

## 실행 환경

Python 3.12 계열에서 사용한 패키지 버전은 `requirements.txt`에 고정했습니다. 저장소 루트에서 가상환경을 만든 다음 이 폴더로 이동하여 실행합니다.

```bash
cd 2_수리과학
python3 -m pip install -r requirements.txt
```

모든 계산은 결정론적이며 난수와 시드를 사용하지 않습니다. 플랫폼과 수치 라이브러리에 따라 마지막 몇 자리에는 차이가 있을 수 있습니다. 그림에는 Noto Sans CJK KR, NanumGothic 등 한글 글꼴이 필요합니다. 이론 검산에는 글꼴이 필요하지 않습니다.

## 1. 원자료 없이 이론 검산

다음 순서로 실행합니다. 마지막 명령은 바로 앞에서 생성한 `order_thresholds.csv`를 읽습니다.

```bash
python3 theory_checks.py
python3 remainder_checks.py
python3 general_theory_checks.py
python3 order_threshold_checks.py
python3 order_rate_checks.py
```

| 파일 | 계산 내용 | 정밀도·출력 |
|---|---|---|
| `theory_checks.py` | 저차 경계 상수, 음의 배율의 최소점과 초기 전이값 | mpmath 40자리, 표준 출력 |
| `remainder_checks.py` | 적분 나머지항과 저차식의 기호·수치 검산 | mpmath 30자리, SymPy, 표준 출력 |
| `general_theory_checks.py` | 일반 차수의 쌍대성·미분식·극값 구조 | mpmath 40자리, JSON 형식 표준 출력 |
| `order_threshold_checks.py` | 차수별 임계값과 상계, 치환 적분 항등식 | SciPy 적분·Brent 근구하기; 선택 차수를 mpmath 60자리로 대조. `order_thresholds.csv`, `order_threshold_checks_output.json` |
| `order_rate_checks.py` | 임계값 수열의 양측 경계와 점근식 검산 | mpmath 60자리. `order_rate_checks.csv`, `order_rate_checks_output.json` |

`order_thresholds.csv`는 차수 `m=0,...,80`의 `b_m`, 해석적 상계 `b_bar_m`, 잔차 등을 담습니다. `order_rate_checks.csv`는 차수에 따른 수렴률을 확인하는 집계값입니다. 함께 둔 세 JSON은 시장자료나 실행 경로를 포함하지 않는 이론 검산의 참고 출력입니다. `general_theory_checks.py`는 참고 JSON을 자동으로 덮어쓰지 않으므로 새 출력은 터미널에서 확인하거나 별도 파일로 저장하십시오.

## 2. KRX 자료를 준비한 뒤 수치 예시 재현

공통 입력 파일은 아래 위치에 두고 원래 상대경로를 유지합니다.

```text
3_통계학/krx_stock_daily_long_20000104_20261113.csv
```

출처는 [KRX 정보데이터시스템](https://data.krx.co.kr/)의 개별종목 시세입니다. 대상은 삼성전자(005930), SK하이닉스(000660)이며, 참고 입력의 수집 범위는 2000-01-04~2026-11-13입니다. 수리 분석에는 2015-06-15~2026-11-13의 종목당 2,802개 관측치와 2026-05-28~2026-11-13의 종목당 115개 관측치를 사용합니다. 상세 스키마는 상위 폴더의 `DATA_SOURCES.md`와 `../3_통계학/README.md`에 설명합니다.

필수 입력 열은 `date`, `stock`, `close`, `change_from_base_price`이고, 종목코드 `code`도 함께 보관하는 형식을 권합니다. 날짜는 `YYYYMMDD` 정수, 종목명은 정확히 `삼성전자` 또는 `SK하이닉스`로 맞춥니다. 수익률은 `close / (close - change_from_base_price) - 1`로 계산하므로 `change_from_base_price`에는 KRX 기준가 대비 부호 있는 가격변화를 넣어야 합니다. 분할 등을 무시한 전일 종가 차이나 반올림된 등락률로 대체하면 같은 결과를 얻지 못할 수 있습니다.

```bash
python3 math_figs_tables.py ../3_통계학/krx_stock_daily_long_20000104_20261113.csv
```

이 명령은 누적 로그감쇠, 이차·삼차 근사값, 경계와 좌표 비교를 표준 출력으로 제시하고 `math_tables.json`, `fig/fig_h.png`, `fig/fig_bounds.png`를 생성합니다. 상수 계산은 mpmath 30자리입니다. 현재 논문의 표 제목 기준 대응은 다음과 같습니다. 코드의 초기 주석에 적힌 표 번호 대신 이 대응을 사용하십시오.

| 논문의 표·그림 내용 | 생성 코드·출력 |
|---|---|
| 차수에 따른 전이 임계값과 해석적 상계 | `order_threshold_checks.py` → `order_thresholds.csv` |
| `c=0.30`에서의 상수 | `theory_checks.py`, `remainder_checks.py`; `math_figs_tables.py`의 상수 출력 |
| 누적 로그감쇠와 경계 | `math_figs_tables.py` → 표준 출력, `math_tables.json`의 `DATA` |
| 단순·로그수익률 좌표와 삼차 경계 비교 | `math_figs_tables.py` → 표준 출력, `math_tables.json`의 `COORD` |
| 정규화 감쇠 함수와 연도별 누적 경계 그림 | `math_figs_tables.py` → `fig/` |

## 공개 범위

원자료 CSV, 전체 논문 TEX/PDF, 표지·서약서, 개인정보, 내부 검토 기록은 제외했습니다. 코드와 동봉 이론 상수는 비공개 작업본의 계산 내용을 그대로 옮겼습니다. 전체 원자료가 없으면 이론 검산은 가능하지만 실제 종목의 수치 예시까지 처음부터 재현되지는 않습니다.
