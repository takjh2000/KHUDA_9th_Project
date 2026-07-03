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

# kr_extra.py와 공유하는 DART API 키 (환경변수 없을 때 fallback)
_DART_KEY_FALLBACK = "86e09bb1fc58dcbbef296479044ebdfe7ff0c29f"


# ── OpenDartReader 경로 ────────────────────────────────────────────────────

def _build_via_dart(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """OpenDartReader로 연간(FY) 재무 수집 — 2015~END_DATE"""
    import OpenDartReader
    api_key = os.environ.get("DART_API_KEY", "") or _DART_KEY_FALLBACK
    if not api_key:
        raise EnvironmentError("DART_API_KEY 없음")

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

    df["_shares"] = np.nan

    # 1순위: DART에서 직접 파싱한 발행주식수
    if "_issued_shares" in df.columns:
        issued_ok = df["_issued_shares"].notna() & (df["_issued_shares"] > 0)
        df.loc[issued_ok, "_shares"] = df.loc[issued_ok, "_issued_shares"]

    # 2순위: EPS + 당기순이익으로 주식수 역산
    eps_ok = (df["_shares"].isna() & df["eps"].notna()
              & df["_net_income"].notna() & (df["eps"] != 0))
    df.loc[eps_ok, "_shares"] = df.loc[eps_ok, "_net_income"] / df.loc[eps_ok, "eps"]

    # 3순위: BPS + 자본총계로 주식수 역산 (더 정확)
    bps_ok = (df["bps"].notna() & df["_total_equity"].notna() & (df["bps"] != 0))
    df.loc[bps_ok, "_shares"] = df.loc[bps_ok, "_total_equity"] / df.loc[bps_ok, "bps"]

    def _per_share(total_col):
        return (df[total_col] / df["_shares"]).where(df["_shares"].notna() & (df["_shares"] > 0))

    df["sps"]          = _per_share("_revenue")
    df["ops"]          = _per_share("_op_income")
    df["shares"]       = df["_shares"]
    df["gross_profit"] = df["_gross_profit"]
    df["total_assets"] = df["_total_assets"]

    # BPS 없으면 자본총계/주식수로 파생
    no_bps = df["bps"].isna() & df["_total_equity"].notna() & df["_shares"].notna() & (df["_shares"] > 0)
    df.loc[no_bps, "bps"] = df.loc[no_bps, "_total_equity"] / df.loc[no_bps, "_shares"]

    # EPS 없으면 당기순이익/주식수로 파생
    no_eps = df["eps"].isna() & df["_net_income"].notna() & df["_shares"].notna() & (df["_shares"] > 0)
    df.loc[no_eps, "eps"] = df.loc[no_eps, "_net_income"] / df.loc[no_eps, "_shares"]

    keep = ["date", "ticker", "bps", "eps", "sps", "roe", "ops",
            "shares", "gross_profit", "total_assets"]
    return df[[c for c in keep if c in df.columns]]


def _parse_dart_fs(fs: pd.DataFrame, ticker: str, year: int) -> dict | None:
    """DART finstate_all 결과에서 주요 항목 추출 (금액 단위: 원)"""
    def get(pattern: str, sj_divs=None) -> float:
        mask = fs["account_nm"].str.contains(pattern, na=False, regex=False)
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

    # 손익계산서 항목 (IS)
    revenue      = get("매출액", ["IS"])
    if np.isnan(revenue):
        revenue  = get("영업수익", ["IS"])       # 금융업/삼성전자 등 대체 계정명
    op_income    = get("영업이익", ["IS"])
    net_income   = get("당기순이익", ["IS", "CIS"])
    gross_profit = get("매출총이익", ["IS"])

    # 재무상태표 항목 (BS)
    total_equity = get("자본총계", ["BS"])
    total_assets = get("자산총계", ["BS"])

    # 주당 항목 — IS·CIS 모두 검색, 다양한 계정명 시도
    for bps_nm in ["주당순자산", "1주당순자산", "주당 순자산"]:
        bps = get(bps_nm, ["BS", "IS", "CIS"])
        if not np.isnan(bps):
            break

    for eps_nm in ["기본주당이익", "기본주당순이익", "주당순이익", "주당이익", "1주당순이익"]:
        eps = get(eps_nm, ["IS", "CIS"])
        if not np.isnan(eps):
            break

    # 발행주식수 직접 파싱 — EPS/BPS 파생에 사용
    issued_shares = np.nan
    for sh_nm in ["보통주발행주식수", "발행주식수", "주식수", "유통주식수"]:
        issued_shares = get(sh_nm, None)   # sj_div 무관하게 검색
        if not np.isnan(issued_shares):
            break

    # BPS가 없으면 자본총계로 나중에 계산
    roe = (net_income / total_equity) if (
        pd.notna(net_income) and pd.notna(total_equity) and total_equity != 0
    ) else np.nan

    return {
        "date":             pd.Timestamp(f"{year}-12-31"),
        "ticker":           ticker,
        "bps":              bps,
        "eps":              eps,
        "roe":              roe,
        "_revenue":         revenue,
        "_op_income":       op_income,
        "_net_income":      net_income,
        "_total_equity":    total_equity,
        "_total_assets":    total_assets,
        "_gross_profit":    gross_profit,
        "_issued_shares":   issued_shares,
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


# ── DART 재조회로 BPS/EPS 보완 ──────────────────────────────────────────────

def _fill_bps_eps_via_dart(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """BPS/EPS NaN 종목을 DART API 재조회 + 개선된 파싱으로 채우기"""
    nan_tickers = df[df["bps"].isna() | df["eps"].isna()]["ticker"].unique().tolist()
    if not nan_tickers:
        return df

    print(f"  DART 재조회 대상: {len(nan_tickers)}개 종목 (BPS/EPS NaN)")
    new_data = _build_via_dart(nan_tickers, start, end)
    if new_data.empty:
        return df

    # df는 lag 적용 후 날짜, new_data는 12-31(lag 전) → lag 제거 후 회계연도로 매칭
    df = df.copy()
    df["_year"] = (pd.to_datetime(df["date"]) - pd.Timedelta(days=KR_FILING_LAG_DAYS)).dt.year
    new_data = new_data.copy()
    new_data["_year"] = pd.to_datetime(new_data["date"]).dt.year

    for col in ["bps", "eps"]:
        if col not in new_data.columns:
            continue
        ann = (
            new_data[new_data[col].notna()][["ticker", "_year", col]]
            .rename(columns={col: f"_{col}_new"})
        )
        merged = df.merge(ann, on=["ticker", "_year"], how="left")
        df[col] = df[col].combine_first(merged[f"_{col}_new"])

    df = df.drop(columns=["_year"])
    after_nn = df["bps"].notna().mean()
    print(f"  DART 재조회 후 BPS non-null: {after_nn:.1%}")
    return df


# ── 메인 ────────────────────────────────────────────────────────────────────

def build(tickers: list[str],
          start: str = START_DATE,
          end: str = END_DATE) -> pd.DataFrame:
    """
    KR 분기별 재무 데이터 구축
    반환: DataFrame — 컬럼: date(사용가능일), ticker, bps, eps, sps, roe, gross_profit, total_assets
    """
    save_path = RAW_DIR / "kr_finance.parquet"

    # 기존 캐시 로드 후 누락 종목만 보완
    existing_df = pd.DataFrame()
    tickers_to_fetch = list(tickers)

    if save_path.exists():
        existing_df = pd.read_parquet(save_path)
        cached_tickers = set(existing_df["ticker"].unique())
        tickers_to_fetch = [t for t in tickers if t not in cached_tickers]

        if not tickers_to_fetch:
            bps_nan = existing_df["bps"].isna().mean() if "bps" in existing_df.columns else 1.0
            if bps_nan > 0.3:
                # 캐시 BPS NaN이 30% 초과 → pykrx로 자동 보완
                print(f"[kr_finance] 캐시 BPS NaN {bps_nan:.0%} → pykrx 보완 실행")
                existing_df = _fill_bps_eps_via_dart(existing_df, start, end)
                existing_df.to_parquet(save_path, index=False)
            else:
                print("[kr_finance] 캐시 로드")
            return existing_df

        print(f"[kr_finance] 캐시 로드 + 누락 {len(tickers_to_fetch)}개 종목 보완")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame()

    # 1순위: DART (환경변수 없으면 fallback 키 사용) — 5개 팩터 전체 커버
    print(f"[kr_finance] DART로 {len(tickers_to_fetch)}개 종목 수집 중...")
    try:
        new_df = _build_via_dart(tickers_to_fetch, start, end)
    except Exception as e:
        print(f"  DART 실패: {e}")

    # 2순위: pykrx — DART 미수집 또는 BPS/EPS NaN 종목 보완
    if not new_df.empty:
        # BPS/EPS 둘 다 없는 종목 → pykrx 대상에 재포함
        no_bps_eps = set(new_df[new_df["bps"].isna() & new_df["eps"].isna()]["ticker"].unique())
        dart_ok = set(new_df["ticker"].unique()) - no_bps_eps
    else:
        dart_ok = set()
        no_bps_eps = set()
    pykrx_targets = [t for t in tickers_to_fetch if t not in dart_ok]

    if pykrx_targets:
        print(f"[kr_finance] DART 재조회로 {len(pykrx_targets)}개 종목 BPS/EPS 보완 중...")
        try:
            if not new_df.empty:
                new_df = _fill_bps_eps_via_dart(new_df, start, end)
            # DART에 행 자체가 없는 종목 → pykrx 시도 후 실패 시 스킵
            dart_missing = [t for t in pykrx_targets if not new_df.empty
                            and t not in set(new_df["ticker"])]
            if dart_missing:
                try:
                    pykrx_df = _build_via_pykrx(dart_missing, start, end)
                    if not pykrx_df.empty:
                        new_df = pd.concat([new_df, pykrx_df], ignore_index=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"  DART 재조회 실패: {e}")

    # 3순위: 네이버 금융 — 여전히 미수집 종목 보완 (BPS/EPS/ROE/SPS)
    covered_so_far = set(new_df["ticker"].unique()) if not new_df.empty else set()
    naver_targets = [t for t in tickers_to_fetch if t not in covered_so_far]

    if naver_targets:
        print(f"[kr_finance] 네이버 금융으로 {len(naver_targets)}개 종목 보완 중... (최근 3년)")
        try:
            naver_df = _build_via_naver(naver_targets)
            if not naver_df.empty:
                new_df = pd.concat([new_df, naver_df], ignore_index=True) if not new_df.empty else naver_df
        except Exception as e:
            print(f"  네이버 실패: {e}")

    if new_df.empty and existing_df.empty:
        print("  [경고] KR 재무 데이터 수집 실패. data/raw/kr_finance.parquet 직접 제공 필요")
        print("         컬럼: date, ticker, bps, eps, sps, roe, gross_profit, total_assets")
        return pd.DataFrame()

    if new_df.empty:
        return existing_df

    # 신규 수집 데이터에만 공시 지연 적용 (기존 캐시는 이미 적용됨)
    new_df["date"] = pd.to_datetime(new_df["date"]) + pd.Timedelta(days=KR_FILING_LAG_DAYS)

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="first")
    else:
        combined = new_df

    combined.to_parquet(save_path, index=False)
    print(f"[kr_finance] 완료: {len(combined)}행 ({combined['ticker'].nunique()}개 종목)")
    return combined
