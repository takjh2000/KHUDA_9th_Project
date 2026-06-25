"""
Phase 1  데이터 파이프라인 실행
    python run_pipeline.py          # KR + US 전체
    python run_pipeline.py --kr     # KR만
    python run_pipeline.py --us     # US만
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import START_DATE, END_DATE
from data.pipeline import kr_price, us_price, kr_finance, us_finance
from data.build_panel import build as build_panel


def run_kr():
    print("\n[Phase 1-1] KR 가격 데이터")
    price_piv, uni_df = kr_price.build(START_DATE, END_DATE)
    tickers = uni_df["ticker"].unique().tolist()

    print("\n[Phase 1-2] KR 재무 데이터")
    kr_finance.build(tickers, START_DATE, END_DATE)

    print("\n[Phase 1-3] KR 패널 구성")
    build_panel("KR")


def run_us():
    print("\n[Phase 1-1] US 가격 데이터")
    price_piv, uni_df = us_price.build(START_DATE, END_DATE)
    tickers = uni_df["ticker"].unique().tolist()

    print("\n[Phase 1-2] US 재무 데이터")
    us_finance.build(tickers, START_DATE, END_DATE)

    print("\n[Phase 1-3] US 패널 구성")
    build_panel("US")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kr", action="store_true")
    parser.add_argument("--us", action="store_true")
    args = parser.parse_args()

    if not args.kr and not args.us:
        args.kr = args.us = True

    if args.kr:
        run_kr()
    if args.us:
        run_us()

    print("\n[완료] 데이터 파이프라인 종료")


if __name__ == "__main__":
    main()
