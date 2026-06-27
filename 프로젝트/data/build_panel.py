"""
Phase 1-3  패널 데이터 구성
- 재무(분기) → 일별 forward fill → 가격과 병합
- 결측 3분기 이상 종목 제거
- 저장: data/processed/kr_panel.parquet
         data/processed/us_panel.parquet
  컬럼: date, ticker, close, bps, eps, sps, roe, gross_profit, total_assets
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, RAW_DIR


def _finance_to_daily(finance_df: pd.DataFrame,
                      price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    분기별 재무 → 일별 forward fill
    - date 컬럼 = 해당 재무 데이터를 사용 가능한 날 (공시 지연 적용 후)
    - price_index: 전체 거래일 목록
    """
    fin_cols = ["bps", "eps", "sps", "roe", "gross_profit", "total_assets"]
    available_cols = [c for c in fin_cols if c in finance_df.columns]

    finance_df = finance_df.copy()
    finance_df["date"] = pd.to_datetime(finance_df["date"])

    result_frames = []
    for ticker, grp in finance_df.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date")[available_cols]

        # 재무 날짜가 비거래일일 수 있으므로 합집합으로 ffill 후 거래일만 추출
        extended = price_index.union(grp.index).sort_values()
        grp = grp.reindex(extended).ffill().reindex(price_index)
        grp["ticker"] = ticker
        result_frames.append(grp.reset_index().rename(columns={"index": "date"}))

    if not result_frames:
        return pd.DataFrame()
    return pd.concat(result_frames, ignore_index=True)


def build(market: str) -> pd.DataFrame:
    """
    market: "KR" or "US"
    반환: long-form panel DataFrame
    """
    save_path = PROCESSED_DIR / f"{market.lower()}_panel.parquet"
    if save_path.exists():
        print(f"[build_panel] {market} 패널 캐시 로드")
        return pd.read_parquet(save_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 가격 로드
    price_path = RAW_DIR / f"{market.lower()}_price.parquet"
    if not price_path.exists():
        raise FileNotFoundError(f"{price_path} 없음. run_pipeline.py 먼저 실행")
    price_pivot = pd.read_parquet(price_path)   # (date × ticker)
    price_pivot.index = pd.to_datetime(price_pivot.index)
    price_index = price_pivot.index

    # 재무 로드
    finance_path = RAW_DIR / f"{market.lower()}_finance.parquet"
    if not finance_path.exists():
        print(f"  [경고] {finance_path} 없음 - 팩터 계산 불가")
        finance_long = pd.DataFrame()
    else:
        finance_df   = pd.read_parquet(finance_path)
        finance_long = _finance_to_daily(finance_df, price_index)

    # 가격 → long-form
    price_long = (
        price_pivot
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="close")
        .dropna(subset=["close"])
    )

    # 병합
    if not finance_long.empty:
        finance_long["date"] = pd.to_datetime(finance_long["date"])
        panel = price_long.merge(finance_long, on=["date", "ticker"], how="left")
    else:
        panel = price_long

    # 재무 데이터가 단 한 건도 없는 종목만 제거
    fin_cols = [c for c in ["bps", "eps", "sps"] if c in panel.columns]
    if fin_cols:
        ticker_miss = (
            panel.groupby("ticker")[fin_cols[0]]
            .apply(lambda x: x.isna().sum())
        )
        total_rows = panel.groupby("ticker")[fin_cols[0]].count() + ticker_miss
        bad_tickers = ticker_miss[ticker_miss >= total_rows].index
        panel = panel[~panel["ticker"].isin(bad_tickers)]
        print(f"  재무 전무 종목 제거: {len(bad_tickers)}개")

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel.to_parquet(save_path, index=False)
    print(f"[build_panel] {market} 완료: {panel.shape}")
    return panel
