"""
Phase 1-1  KR 가격 데이터
- 유니버스: 코스피200 (과거 시점 구성종목 → 생존편향 제거)
- 소스: FinanceDataReader (가격) / pykrx (과거 구성종목)
- 저장: data/raw/kr_price.parquet  (date × ticker 피벗)
        data/raw/kr_universe.parquet (date, ticker — 각 분기 유니버스)
"""
import sys, time, warnings
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr
from config import RAW_DIR, START_DATE, END_DATE


# ── 유니버스: 과거 코스피200 구성종목 ───────────────────────────────────

def get_historical_kospi200(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    분기별 코스피200 구성종목 이력 반환
    pykrx 성공 시 실제 과거 구성종목, 실패 시 현재 구성종목으로 대체
    반환: DataFrame(date, ticker)
    """
    rows = []
    for d in dates:
        tickers = _fetch_kospi200_at(d)
        for t in tickers:
            rows.append({"date": d, "ticker": t})
    return pd.DataFrame(rows)


_FDR_KRX_CACHE: pd.DataFrame | None = None

def _get_fdr_kospi200() -> list[str]:
    """FDR KRX 목록에서 KOSPI 시총 상위 200종목 반환 (캐시)"""
    global _FDR_KRX_CACHE
    if _FDR_KRX_CACHE is None:
        df = fdr.StockListing("KRX")
        _FDR_KRX_CACHE = df[df["Market"] == "KOSPI"].copy()
    top200 = _FDR_KRX_CACHE.nlargest(200, "Marcap")
    return top200["Code"].tolist()


def _fetch_kospi200_at(date: pd.Timestamp) -> list[str]:
    """특정 날짜의 코스피200 구성종목 반환"""
    d_str = date.strftime("%Y%m%d")

    # 1순위: pykrx 과거 구성종목
    try:
        from pykrx import stock as krx
        df = krx.get_index_portfolio_deposit_file("1028", d_str)
        if hasattr(df, "empty") and not df.empty:
            col = "티커" if "티커" in df.columns else df.columns[0]
            return df[col].tolist()
    except Exception:
        pass

    # 2순위: FinanceDataReader KRX 시총 상위 200 (생존편향 존재하나 대체 불가 시 사용)
    try:
        tickers = _get_fdr_kospi200()
        if tickers:
            print(f"  [경고] {d_str} pykrx 조회 실패 → FDR KOSPI 시총 상위 200 사용 (생존편향 주의)")
            return tickers
    except Exception:
        pass

    return []


# ── 가격 데이터 ────────────────────────────────────────────────────────────

def download_prices(tickers: list[str],
                    start: str = START_DATE,
                    end: str = END_DATE) -> pd.DataFrame:
    """
    FinanceDataReader로 종목별 수정종가 다운로드
    반환: (date × ticker) 피벗
    """
    frames = {}
    for i, code in enumerate(tickers, 1):
        try:
            df = fdr.DataReader(code, start, end)
            if not df.empty and "Close" in df.columns:
                frames[code] = df["Close"]
        except Exception:
            pass
        if i % 50 == 0:
            print(f"  가격 다운로드: {i}/{len(tickers)}")
        time.sleep(0.03)

    if not frames:
        return pd.DataFrame()

    panel = pd.DataFrame(frames)
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    return panel.sort_index()


# ── 메인 ────────────────────────────────────────────────────────────────────

def build(start: str = START_DATE, end: str = END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    코스피200 가격 패널 + 분기별 유니버스 구성
    반환: (price_pivot, universe_df)
    """
    price_path   = RAW_DIR / "kr_price.parquet"
    universe_path = RAW_DIR / "kr_universe.parquet"

    if price_path.exists() and universe_path.exists():
        print("[kr_price] 캐시 로드")
        return pd.read_parquet(price_path), pd.read_parquet(universe_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 분기말 날짜 목록
    quarter_dates = pd.date_range(start, end, freq="QE")
    print(f"[kr_price] {len(quarter_dates)}개 분기 유니버스 조회 중...")
    universe_df = get_historical_kospi200(quarter_dates)
    universe_df.to_parquet(universe_path, index=False)

    if universe_df.empty or "ticker" not in universe_df.columns:
        print("[kr_price] 오류: 유니버스 데이터를 가져오지 못했습니다.")
        return pd.DataFrame(), universe_df

    all_tickers = universe_df["ticker"].unique().tolist()
    print(f"[kr_price] 총 {len(all_tickers)}개 종목 가격 다운로드 중...")
    price_pivot = download_prices(all_tickers, start, end)
    price_pivot.to_parquet(price_path)

    print(f"[kr_price] 완료: {price_pivot.shape}")
    return price_pivot, universe_df


if __name__ == "__main__":
    build()
