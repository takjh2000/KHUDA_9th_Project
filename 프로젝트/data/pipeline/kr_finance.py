"""
Phase 1-2  KR 재무 데이터
- 소스 우선순위:
    1. OpenDartReader (DART_API_KEY 환경변수 필요) → BPS, EPS, SPS, ROE, GP, TA
    2. pykrx get_market_fundamental_by_date → BPS, EPS (SPS/GP/TA 없음)
- 저장: data/raw/kr_finance.parquet
  컬럼: date(공시일), ticker, bps, eps, sps, roe, gross_profit, total_assets

DART API 키 발급: https://opendart.fss.or.kr  (무료)
발급 후 환경변수 설정:
    setx DART_API_KEY "your_key_here"  (Windows)
"""
import os, sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

from config import RAW_DIR, START_DATE, END_DATE, KR_FILING_LAG_DAYS


# ── OpenDartReader 경로 ────────────────────────────────────────────────────

def _build_via_dart(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """OpenDartReader로 분기별 재무 수집"""
    import OpenDartReader as dart
    api_key = os.environ.get("DART_API_KEY", "")
    if not api_key:
        raise EnvironmentError("DART_API_KEY 환경변수 없음")

    d = dart.OpenDartReader(api_key)
    rows = []
    for i, ticker in enumerate(tickers, 1):
        try:
            corp = d.corp_code(ticker)
            for year in pd.date_range(start, end, freq="YE").year:
                for q in [11011, 11012, 11013, 11014]:   # 1Q~4Q
                    try:
                        fs = d.finstate(corp, year, q)
                        if fs is None or fs.empty:
                            continue
                        row = _parse_dart_fs(fs, ticker, year, q)
                        if row:
                            rows.append(row)
                    except Exception:
                        pass
        except Exception:
            pass
        if i % 50 == 0:
            print(f"  DART 재무: {i}/{len(tickers)}")
        time.sleep(0.2)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_dart_fs(fs: pd.DataFrame, ticker: str, year: int, q: int) -> dict | None:
    """DART 재무제표에서 필요 항목 추출"""
    def get_val(account: str) -> float:
        row = fs[fs["account_nm"].str.contains(account, na=False)]
        if row.empty:
            return np.nan
        try:
            return float(str(row.iloc[0]["thstrm_amount"]).replace(",", ""))
        except Exception:
            return np.nan

    quarter_map = {11011: "Q1", 11012: "H1", 11013: "Q3", 11014: "FY"}
    return {
        "ticker":       ticker,
        "year":         year,
        "quarter":      quarter_map.get(q, ""),
        "revenue":      get_val("매출"),
        "gross_profit": get_val("매출총이익"),
        "net_income":   get_val("당기순이익"),
        "total_equity": get_val("자본총계"),
        "total_assets": get_val("자산총계"),
    }


# ── pykrx 폴백 경로 ─────────────────────────────────────────────────────────

def _build_via_pykrx(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """pykrx로 월별 BPS / EPS 수집 (SPS / GP / TA 없음)"""
    from pykrx import stock as krx

    def _s(d):
        return pd.Timestamp(d).strftime("%Y%m%d")

    frames = []
    for i, ticker in enumerate(tickers, 1):
        try:
            df = krx.get_market_fundamental_by_date(_s(start), _s(end), ticker)
            if df.empty:
                continue
            df = df[["BPS", "EPS"]].rename(columns={"BPS": "bps", "EPS": "eps"})
            df.index.name = "date"
            df["ticker"] = ticker
            df["sps"] = np.nan
            df["roe"] = np.nan
            df["gross_profit"] = np.nan
            df["total_assets"] = np.nan
            frames.append(df.reset_index())
        except Exception:
            pass
        if i % 50 == 0:
            print(f"  pykrx 재무: {i}/{len(tickers)}")
        time.sleep(0.1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── 메인 ────────────────────────────────────────────────────────────────────

def build(tickers: list[str],
          start: str = START_DATE,
          end: str = END_DATE) -> pd.DataFrame:
    """
    KR 분기별 재무 데이터 구축
    반환: DataFrame — 컬럼: date(사용가능일), ticker, bps, eps, sps, roe, gross_profit, total_assets
    """
    save_path = RAW_DIR / "kr_finance.parquet"
    if save_path.exists():
        print("[kr_finance] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1순위: DART
    df = pd.DataFrame()
    if os.environ.get("DART_API_KEY"):
        print("[kr_finance] OpenDartReader로 수집 중...")
        try:
            df = _build_via_dart(tickers, start, end)
        except Exception as e:
            print(f"  DART 실패: {e}")

    # 2순위: pykrx
    if df.empty:
        print("[kr_finance] pykrx로 수집 중... (BPS/EPS만 가능)")
        try:
            df = _build_via_pykrx(tickers, start, end)
        except Exception as e:
            print(f"  pykrx 실패: {e}")

    if df.empty:
        print("  [경고] KR 재무 데이터 수집 실패. data/raw/kr_finance.parquet 직접 제공 필요")
        print("         컬럼: date, ticker, bps, eps, sps, roe, gross_profit, total_assets")
        return pd.DataFrame()

    # 공시 지연 적용
    df["date"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=KR_FILING_LAG_DAYS)
    df.to_parquet(save_path, index=False)
    print(f"[kr_finance] 완료: {len(df)}행")
    return df
