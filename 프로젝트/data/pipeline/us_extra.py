"""
US 추가 재무/분류 데이터 수집
- cashflow_op, cash, operating_income, debt, ebitda, equity
- sharesout, sga_expense, operating_expense, goodwill
- sector, industry (분류)
- volume (일별 거래량)
저장: data/raw/us_extra_finance.parquet
      data/raw/us_sector.parquet
      data/raw/us_price_vol.parquet  (close + volume)
"""
import sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

import yfinance as yf
from config import RAW_DIR, START_DATE, END_DATE, US_FILING_LAG_DAYS


def _get_extra_financials(ticker: str) -> tuple[list[dict], dict]:
    """
    yfinance에서 추가 재무 항목 추출
    반환: (rows_list, sector_info)
    """
    rows = []
    sector_info = {"ticker": ticker, "sector": None, "industry": None}
    try:
        t = yf.Ticker(ticker)
        info = t.info
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        sector_info["sector"]   = info.get("sector")
        sector_info["industry"] = info.get("industry")

        inc  = t.income_stmt
        bal  = t.balance_sheet
        cf   = t.cashflow

        if inc is None or bal is None or cf is None:
            return rows, sector_info
        if inc.empty or bal.empty or cf.empty:
            return rows, sector_info

        for col in inc.columns:
            row = {"date": col, "ticker": ticker}

            def _get(df, *keys):
                for k in keys:
                    matches = [i for i in df.index if k.lower() in str(i).lower()]
                    if matches and col in df.columns:
                        try:
                            v = df.loc[matches[0], col]
                            if pd.notna(v):
                                return float(v)
                        except Exception:
                            pass
                return np.nan

            row["operating_income"]   = _get(inc, "Operating Income", "EBIT")
            row["ebitda"]             = _get(inc, "EBITDA", "Normalized EBITDA")
            row["sga_expense"]        = _get(inc, "Selling General And Administration", "SGA")
            row["operating_expense"]  = _get(inc, "Operating Expense", "Total Operating Expenses")

            if col in bal.columns:
                row["cash"]        = _get(bal, "Cash And Cash Equivalents", "Cash Cash Equivalents")
                row["debt"]        = _get(bal, "Total Debt", "Long Term Debt And Capital Lease")
                row["equity"]      = _get(bal, "Stockholders Equity", "Total Equity")
                row["goodwill"]    = _get(bal, "Goodwill", "Goodwill And Other Intangible Assets")

            if col in cf.columns:
                row["cashflow_op"] = _get(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating")

            row["sharesout"] = shares if shares else np.nan
            rows.append(row)

    except Exception:
        pass

    return rows, sector_info


def build_extra_finance(tickers: list[str]) -> pd.DataFrame:
    save_path = RAW_DIR / "us_extra_finance.parquet"
    if save_path.exists():
        print("[us_extra_finance] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[us_extra_finance] {len(tickers)}개 종목 추가 재무 수집 중...")

    all_rows = []
    sector_rows = []

    for i, ticker in enumerate(tickers, 1):
        rows, sector_info = _get_extra_financials(ticker)
        all_rows.extend(rows)
        sector_rows.append(sector_info)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}")
        time.sleep(0.15)

    # 섹터 저장
    sector_df = pd.DataFrame(sector_rows)
    sector_df.to_parquet(RAW_DIR / "us_sector.parquet", index=False)
    print(f"[us_sector] 저장: {len(sector_df)}개 종목")

    if not all_rows:
        print("  [경고] 추가 재무 데이터 수집 실패")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=US_FILING_LAG_DAYS)
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df.to_parquet(save_path, index=False)
    print(f"[us_extra_finance] 완료: {len(df)}행")
    return df


def build_price_with_volume(tickers: list[str]) -> pd.DataFrame:
    """close + volume 일별 데이터"""
    save_path = RAW_DIR / "us_price_vol.parquet"
    if save_path.exists():
        print("[us_price_vol] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[us_price_vol] {len(tickers)}개 종목 거래량 다운로드 중...")

    vol_frames = {}
    batch = 50
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        try:
            raw = yf.download(chunk, start=START_DATE, end=END_DATE,
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                vol = raw["Volume"]
            else:
                vol = raw[["Volume"]].rename(columns={"Volume": chunk[0]})
            for col in vol.columns:
                vol_frames[col] = vol[col]
        except Exception:
            pass
        print(f"  거래량: {min(i + batch, len(tickers))}/{len(tickers)}")
        time.sleep(0.3)

    if not vol_frames:
        return pd.DataFrame()

    vol_panel = pd.DataFrame(vol_frames)
    vol_panel.index = pd.to_datetime(vol_panel.index)
    vol_panel.index.name = "date"
    vol_panel.to_parquet(save_path)
    print(f"[us_price_vol] 완료: {vol_panel.shape}")
    return vol_panel


if __name__ == "__main__":
    uni = pd.read_parquet(RAW_DIR / "us_universe.parquet")
    tickers = uni["ticker"].unique().tolist()
    build_extra_finance(tickers)
    build_price_with_volume(tickers)
