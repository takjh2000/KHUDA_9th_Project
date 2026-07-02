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
    """OpenDartReader로 연간(FY) 재무 수집 — 2015~END_DATE"""
    from opendartreader import OpenDartReader
    api_key = os.environ.get("DART_API_KEY", "")
    if not api_key:
        raise EnvironmentError("DART_API_KEY 환경변수 없음")

    d = OpenDartReader(api_key)

    # stock_code(6자리 종목코드) → corp_code(8자리 DART 코드) 매핑
    corp_codes_df = d.corp_codes
    code_map = corp_codes_df.dropna(subset=["stock_code"]).set_index("stock_code")["corp_code"].to_dict()

    start_year = max(pd.Timestamp(start).year, 2015)
    end_year   = pd.Timestamp(end).year

    rows = []
    for i, ticker in enumerate(tickers, 1):
        corp = code_map.get(ticker)
        if not corp:
            continue

        for year in range(start_year, end_year + 1):
            try:
                # 11011 = 사업보고서(연간), CFS = 연결재무제표
                fs = d.finstate_all(corp, year, reprt_code="11011", fs_div="CFS")
                if fs is None or fs.empty:
                    # 연결재무제표 없으면 별도재무제표 시도
                    fs = d.finstate_all(corp, year, reprt_code="11011", fs_div="OFS")
                if fs is None or fs.empty:
                    continue
                row = _parse_dart_fs(fs, ticker, year)
                if row:
                    rows.append(row)
            except Exception:
                pass
            time.sleep(0.05)

        if i % 10 == 0:
            print(f"  DART 재무: {i}/{len(tickers)}")
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # EPS + 당기순이익으로 주식수 추정
    eps_mask = df["eps"].notna() & df["_net_income"].notna() & (df["eps"] != 0)
    df.loc[eps_mask, "_shares"] = df.loc[eps_mask, "_net_income"] / df.loc[eps_mask, "eps"]

    # BPS가 직접 있으면 자본총계 기반으로 재추정 (더 정확)
    bps_mask = df["bps"].notna() & df["_total_equity"].notna() & (df["bps"] != 0)
    df.loc[bps_mask, "_shares"] = df.loc[bps_mask, "_total_equity"] / df.loc[bps_mask, "bps"]

    def _per_share(total_col):
        return (df[total_col] / df["_shares"]).where(df["_shares"].notna() & (df["_shares"] > 0))

    df["sps"]          = _per_share("_revenue")
    df["ops"]          = _per_share("_op_income")
    df["shares"]       = df["_shares"]
    df["gross_profit"] = df["_gross_profit"]
    df["total_assets"] = df["_total_assets"]

    # BPS가 없으면 자본총계 / 주식수로 계산
    no_bps = df["bps"].isna() & df["_total_equity"].notna() & df["_shares"].notna() & (df["_shares"] > 0)
    df.loc[no_bps, "bps"] = df.loc[no_bps, "_total_equity"] / df.loc[no_bps, "_shares"]

    keep = ["date", "ticker", "bps", "eps", "sps", "roe", "ops",
            "shares", "gross_profit", "total_assets"]
    return df[[c for c in keep if c in df.columns]]


def _parse_dart_fs(fs: pd.DataFrame, ticker: str, year: int) -> dict | None:
    """DART finstate_all 결과에서 주요 항목 추출 (금액 단위: 원)"""
    def get(pattern: str, sj_divs=None) -> float:
        names = fs["account_nm"].str.replace(" ", "", regex=False)
        mask = names.str.contains(pattern.replace(" ", ""), na=False, regex=False)
        if sj_divs:
            mask &= fs["sj_div"].isin(sj_divs)
        for _, r in fs[mask].iterrows():
            try:
                val = str(r.get("thstrm_amount", "")).replace(",", "").strip()
                if val and val not in ("-", ""):
                    return float(val)
            except Exception:
                pass
        return np.nan

    # 손익계산서 항목 (IS 또는 CIS — 회사에 따라 포괄손익계산서로 태깅됨)
    revenue      = get("매출액", ["IS", "CIS"])
    if np.isnan(revenue):
        revenue  = get("영업수익", ["IS", "CIS"])       # 금융업/삼성전자 등 대체 계정명
    op_income    = get("영업이익", ["IS", "CIS"])
    net_income   = get("당기순이익", ["IS", "CIS"])
    gross_profit = get("매출총이익", ["IS", "CIS"])

    # 재무상태표 항목 (BS)
    total_equity = get("자본총계", ["BS"])
    total_assets = get("자산총계", ["BS"])

    # 주당 항목 (IS 또는 CIS 안에 포함됨)
    bps = get("주당순자산", ["BS", "IS", "CIS"])
    eps = get("기본주당이익", ["IS", "CIS"])
    if np.isnan(eps):
        eps = get("주당순이익", ["IS", "CIS"])
    if np.isnan(eps):
        eps = get("주당이익", ["IS", "CIS"])

    # BPS가 없으면 자본총계로 나중에 계산
    roe = (net_income / total_equity) if (
        pd.notna(net_income) and pd.notna(total_equity) and total_equity != 0
    ) else np.nan

    return {
        "date":          pd.Timestamp(f"{year}-12-31"),
        "ticker":        ticker,
        "bps":           bps,
        "eps":           eps,
        "roe":           roe,
        "_revenue":      revenue,
        "_op_income":    op_income,
        "_net_income":   net_income,
        "_total_equity": total_equity,
        "_total_assets": total_assets,
        "_gross_profit": gross_profit,
    }


# ── 네이버 금융 폴백 경로 ────────────────────────────────────────────────────

def _build_via_naver(tickers: list[str]) -> pd.DataFrame:
    """네이버 금융 메인 페이지에서 연간 EPS/BPS/ROE/SPS 수집 (최근 3년)"""
    import requests
    from bs4 import BeautifulSoup
    from io import StringIO

    def _fetch_one(code):
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.encoding = "euc-kr"
        tables = pd.read_html(StringIO(r.text))
        if len(tables) < 7:
            return []

        t = tables[4]
        t.columns = t.columns.get_level_values(1)

        # 연간 컬럼만 (E 예상치·분기 제외), iloc 기반으로 중복 컬럼 방지
        all_cols = t.columns.tolist()
        annual_indices = [i for i, c in enumerate(all_cols)
                          if isinstance(c, str) and len(c) == 7
                          and "." in c and "(E)" not in c
                          and all_cols.index(c) == i]  # 첫 등장만
        if not annual_indices:
            return []

        # 주식수 (tables[6])
        shares = None
        for val in tables[6].values.flatten():
            try:
                v = int(str(val).replace(",", ""))
                if v > 1_000_000:
                    shares = v
                    break
            except Exception:
                pass

        rows = []
        for idx in annual_indices:
            col_name = all_cols[idx]
            try:
                date = pd.Timestamp(col_name.replace(".", "-") + "-01") + pd.offsets.MonthEnd(0)
                eps  = t.iloc[9, idx]
                bps  = t.iloc[11, idx]
                roe  = t.iloc[5, idx]
                rev  = t.iloc[0, idx]   # 억원
                sps  = (float(rev) * 1e8 / shares) if shares and pd.notna(rev) else np.nan
                # 영업이익 (row 1, 억원) → per share
                op_raw = t.iloc[1, idx] if t.shape[0] > 1 else np.nan
                op_total = float(op_raw) * 1e8 if pd.notna(op_raw) else np.nan
                ops = op_total / shares if (shares and pd.notna(op_total)) else np.nan
                rows.append({
                    "date":         date,
                    "ticker":       code,
                    "eps":          float(eps) if pd.notna(eps) else np.nan,
                    "bps":          float(bps) if pd.notna(bps) else np.nan,
                    "roe":          float(roe) / 100 if pd.notna(roe) else np.nan,
                    "sps":          sps,
                    "ops":          ops,
                    "shares":       float(shares) if shares else np.nan,
                    "gross_profit": np.nan,
                    "total_assets": np.nan,
                })
            except Exception:
                pass
        return rows

    all_rows = []
    for i, code in enumerate(tickers, 1):
        try:
            all_rows.extend(_fetch_one(code))
        except Exception:
            pass
        if i % 20 == 0:
            print(f"  네이버 재무: {i}/{len(tickers)}")
        time.sleep(0.3)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


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

    # 3순위: 네이버 금융 스크래핑 (최근 3년 연간, EPS/BPS/ROE/SPS)
    if df.empty:
        print("[kr_finance] 네이버 금융으로 수집 중... (최근 3년 연간)")
        try:
            df = _build_via_naver(tickers)
        except Exception as e:
            print(f"  네이버 실패: {e}")

    if df.empty:
        print("  [경고] KR 재무 데이터 수집 실패. data/raw/kr_finance.parquet 직접 제공 필요")
        print("         컬럼: date, ticker, bps, eps, sps, roe, gross_profit, total_assets")
        return pd.DataFrame()

    # 공시 지연 적용
    df["date"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=KR_FILING_LAG_DAYS)
    df.to_parquet(save_path, index=False)
    print(f"[kr_finance] 완료: {len(df)}행")
    return df
