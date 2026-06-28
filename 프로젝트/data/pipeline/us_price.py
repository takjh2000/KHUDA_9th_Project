"""
Phase 1-1  US 가격 데이터
- 유니버스: S&P 500 (과거 구성종목 CSV 우선, 없으면 현재 Wikipedia 목록)
- 소스: yfinance
- 저장: data/raw/us_price.parquet  (date × ticker 피벗)
        data/raw/us_universe.parquet (date, ticker)

과거 S&P 500 구성종목 CSV 사용 시:
    data/raw/sp500_historical.csv  컬럼: date(YYYY-MM-DD), ticker
"""
import sys, time, warnings
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

import yfinance as yf
from config import RAW_DIR, START_DATE, END_DATE


# ── 유니버스 ────────────────────────────────────────────────────────────────

def get_historical_sp500(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    분기별 S&P 500 구성종목 이력 반환
    우선순위: (1) 사전 준비 CSV  (2) Wikipedia 현재 목록
    CSV 형식: date, ticker  (ticker 컬럼은 콤마로 구분된 티커 문자열)
    """
    hist_csv = RAW_DIR / "sp500_historical.csv"
    if hist_csv.exists():
        hist = pd.read_csv(hist_csv, parse_dates=["date"])
        rows = []
        for d in dates:
            mask = hist["date"] <= d
            if mask.any():
                latest_row = hist[mask].sort_values("date").iloc[-1]
                ticker_str = str(latest_row["ticker"])
                tickers = [t.strip() for t in ticker_str.split(",") if t.strip()]
                rows.extend([{"date": d, "ticker": t} for t in tickers])
        return pd.DataFrame(rows)

    # Wikipedia fallback
    current = _get_current_sp500()
    print("  [경고] sp500_historical.csv 없음 → 현재 S&P 500 구성종목 사용 (생존편향 주의)")
    rows = [{"date": d, "ticker": t} for d in dates for t in current]
    return pd.DataFrame(rows)


def _get_current_sp500() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        # 대표 종목 샘플 (접근 불가 시)
        return [
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","UNH",
            "LLY","XOM","MA","HD","PG","JNJ","AVGO","MRK","CVX","ABBV",
            "COST","PEP","KO","ADBE","CRM","WMT","MCD","CSCO","ABT","NKE",
            "TMO","TXN","DHR","NEE","PM","QCOM","IBM","GS","CAT","LOW",
            "SPGI","HON","UNP","LIN","AMGN","ORCL","BMY","RTX","ELV","MDT",
        ]


# ── 가격 ────────────────────────────────────────────────────────────────────

def download_prices(tickers: list[str],
                    start: str = START_DATE,
                    end: str = END_DATE) -> pd.DataFrame:
    """yfinance 배치 다운로드 → (date × ticker) 수정종가 피벗"""
    frames = []
    batch = 50
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        raw = yf.download(chunk, start=start, end=end,
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": chunk[0]})
        frames.append(close)
        print(f"  US 가격: {min(i+batch, len(tickers))}/{len(tickers)}")
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1)
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    panel.columns = panel.columns.astype(str)
    return panel.sort_index()


# ── 메인 ────────────────────────────────────────────────────────────────────

def build(start: str = START_DATE, end: str = END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_path    = RAW_DIR / "us_price.parquet"
    universe_path = RAW_DIR / "us_universe.parquet"

    if price_path.exists() and universe_path.exists():
        print("[us_price] 캐시 로드")
        return pd.read_parquet(price_path), pd.read_parquet(universe_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    quarter_dates = pd.date_range(start, end, freq="QE")
    print(f"[us_price] {len(quarter_dates)}개 분기 유니버스 조회 중...")
    universe_df = get_historical_sp500(quarter_dates)
    universe_df.to_parquet(universe_path, index=False)

    all_tickers = universe_df["ticker"].unique().tolist()
    print(f"[us_price] 총 {len(all_tickers)}개 종목 가격 다운로드 중...")
    price_pivot = download_prices(all_tickers, start, end)
    price_pivot.to_parquet(price_path)

    print(f"[us_price] 완료: {price_pivot.shape}")
    return price_pivot, universe_df


if __name__ == "__main__":
    build()
