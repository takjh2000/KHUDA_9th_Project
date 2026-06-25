"""
Phase 1-2  US 재무 데이터
- 소스: yfinance 분기별 재무제표 (무료)
- 저장: data/raw/us_finance.parquet
  컬럼: date(사용가능일=분기말+60일), ticker, bps, eps, sps, roe, gross_profit, total_assets
"""
import sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

import yfinance as yf
from config import RAW_DIR, START_DATE, END_DATE, US_FILING_LAG_DAYS


def _get_quarterly_financials(ticker: str) -> pd.DataFrame:
    """yfinance에서 분기별 재무 항목 추출"""
    try:
        t = yf.Ticker(ticker)
        info  = t.info
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares or shares == 0:
            return pd.DataFrame()

        inc  = t.quarterly_income_stmt
        bal  = t.quarterly_balance_sheet
        if inc is None or bal is None or inc.empty or bal.empty:
            return pd.DataFrame()

        rows = []
        for col in inc.columns:                 # col = 분기말 날짜
            if col not in bal.columns:
                continue
            try:
                revenue      = inc.loc["Total Revenue", col]       if "Total Revenue"      in inc.index else np.nan
                gross_profit = inc.loc["Gross Profit", col]        if "Gross Profit"       in inc.index else np.nan
                net_income   = inc.loc["Net Income", col]          if "Net Income"         in inc.index else np.nan
                total_equity = bal.loc["Stockholders Equity", col] if "Stockholders Equity" in bal.index else np.nan
                total_assets = bal.loc["Total Assets", col]        if "Total Assets"       in bal.index else np.nan

                eps = net_income / shares   if pd.notna(net_income)   and shares else np.nan
                bps = total_equity / shares if pd.notna(total_equity) and shares else np.nan
                sps = revenue / shares      if pd.notna(revenue)      and shares else np.nan
                roe = net_income / total_equity if pd.notna(net_income) and pd.notna(total_equity) and total_equity != 0 else np.nan

                rows.append({
                    "date":         col,
                    "ticker":       ticker,
                    "bps":          bps,
                    "eps":          eps,
                    "sps":          sps,
                    "roe":          roe,
                    "gross_profit": gross_profit,
                    "total_assets": total_assets,
                })
            except Exception:
                pass

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def build(tickers: list[str],
          start: str = START_DATE,
          end: str = END_DATE) -> pd.DataFrame:
    """
    US 분기별 재무 데이터 구축 (분기말 + US_FILING_LAG_DAYS 후 사용 가능)
    """
    save_path = RAW_DIR / "us_finance.parquet"
    if save_path.exists():
        print("[us_finance] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[us_finance] {len(tickers)}개 종목 재무 수집 중...")

    frames = []
    for i, ticker in enumerate(tickers, 1):
        df = _get_quarterly_financials(ticker)
        if not df.empty:
            frames.append(df)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}")
        time.sleep(0.1)

    if not frames:
        print("  [경고] US 재무 데이터 수집 실패")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]) + pd.Timedelta(days=US_FILING_LAG_DAYS)

    # 기간 필터
    result = result[(result["date"] >= start) & (result["date"] <= end)]
    result.to_parquet(save_path, index=False)
    print(f"[us_finance] 완료: {len(result)}행")
    return result
