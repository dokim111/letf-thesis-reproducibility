# 레버리지 ETF 연구 재현자료 — 이전 통합본 안내

각 졸업논문의 재현자료를 **전공별 독립 저장소**로 분리했습니다. 최신 자료 버전은 **2026-11-19-v23**이며, 해당 논문은 아래 저장소를 인용합니다.

| 논문 | 독립 저장소 | 계산 대상 |
|---|---|---|
| 경제학 | [letf-economics-thesis](https://github.com/dokim111/letf-economics-thesis) | 리밸런싱과 순설정을 반영한 노출 조정액 |
| 수리과학 | [letf-mathematics-thesis](https://github.com/dokim111/letf-mathematics-thesis) | 로그감쇠 나머지항·극값·차수별 전이 |
| 통계학 | [letf-statistics-thesis](https://github.com/dokim111/letf-statistics-thesis) | 비대칭 정보의 추가 예측력과 사양 민감도 |

각 저장소에는 해당 논문의 코드, 실행 안내, 입력 자료 명세, 인용 정보와 체크섬을 넣었습니다. 수리·통계가 공유하는 일별 수익률 표본과 115거래일 예시는 각 README에 밝혔으며, 다른 전공 저장소를 내려받아야 하는 실행 의존성은 없습니다. 원자료부터의 계산에는 KRX 자료를 별도로 확보해야 합니다.

## 이전 링크의 보존

이 통합 저장소의 코드는 **v22 보관본**입니다. 기존 논문에 적힌 고정 커밋 링크가 유지되도록 코드와 이력을 남겨 두었습니다. 이후 변경은 위의 개별 저장소에서 관리합니다. GitHub의 보관 모드 설정을 바꾼 것은 아닙니다.

- [v22 고정 보관본](https://github.com/dokim111/letf-thesis-reproducibility/tree/abb2c9164f4b7c0264867ac56045cf20f4cefc2b)
- [v22 재현 안내](https://github.com/dokim111/letf-thesis-reproducibility/blob/abb2c9164f4b7c0264867ac56045cf20f4cefc2b/README.md)

개별 저장소의 제출 기준 커밋은 각 논문의 재현자료 안내와 아래 링크에 고정했습니다.

- [경제학 v18 — 687af78](https://github.com/dokim111/letf-economics-thesis/tree/687af7804fbc57bb3b5996a6797de6e75678e859)
- [수리과학 v0.4 — 7629077](https://github.com/dokim111/letf-mathematics-thesis/tree/7629077631ba47b8147d659f82bcaa24c53f0a29)
- [통계학 v0.5 — d5dd0f5](https://github.com/dokim111/letf-statistics-thesis/tree/d5dd0f54da661cff7a24b9d2cd69618a42a24e2b)

작성자: 김도연. 이 공개 자료에는 KRX 원자료, 논문 전문, 제출 양식, 학번·연락처와 내부 검토 기록을 포함하지 않습니다. 연구·코드·문서 작성의 AI 보조 범위와 결과 해석의 한계는 각 독립 저장소의 안내를 따릅니다.
