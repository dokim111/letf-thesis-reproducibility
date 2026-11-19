"""Reproduce the endpoint return reported in the economics thesis introduction.

Uses a separately obtained KRX CSV; no source data is distributed here or downloaded by the script. Only Python's
standard library is required. Run from any directory with Python 3.11+.
"""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "krx_stock_daily_20260401_20261113.csv"
OUTPUT = HERE / "intro_drawdown_output.json"
START = "20260701"
END = "20260730"


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(
            (
                row for row in csv.DictReader(handle)
                if row["stock"] == "SK하이닉스" and START <= row["date"] <= END
            ),
            key=lambda row: row["date"],
        )
    if not rows:
        raise ValueError("No observations found for the specified stock and dates.")

    def close(row):
        return float(row["close"].replace(",", ""))

    start = next((row for row in rows if row['date'] == START), None)
    end = next((row for row in rows if row['date'] == END), None)
    if start is None or end is None:
        raise ValueError("Both exact endpoint dates must be observed.")

    decline = close(end) / close(start) - 1.0
    result = {
        "source_file": SOURCE.name,
        "data_provenance": "사용자가 별도로 확보한 KRX 정보데이터시스템 자료; 이 스크립트는 자료를 다운로드하지 않음",
        "stock": "SK하이닉스",
        "window_start": START,
        "window_end": END,
        "observations": len(rows),
        "price_field": "close",
        "start_date": start["date"],
        "start_close_krw": close(start),
        "end_date": end["date"],
        "end_close_krw": close(end),
        "definition": "close(2026-07-30) / close(2026-07-01) - 1",
        "return_decimal": decline,
        "return_percent": decline * 100.0,
        "rounded_return_percent": round(decline * 100.0, 1),
        "scope": "7월 1일 종가부터 7월 30일 종가까지의 수익률; 전체 고점 대비 하락률 또는 7월 한 달 수익률이 아님",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
