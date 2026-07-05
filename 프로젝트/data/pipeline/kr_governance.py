"""
KR 지배구조/주주환원 데이터 수집 (DART API)
- 관계기업투자(equity_method_investment), 투자부동산(investment_property): 재무상태표 계정과목
- 배당성향(dividend_payout_ratio), 현금배당수익률(dividend_yield): alotMatter API (배당에 관한 사항)
- 최대주주등 지분율(major_shareholder_ratio): hyslrSttus API (최대주주 현황, 특수관계인 포함 합계)
저장: data/raw/kr_governance.parquet
  컬럼: date, ticker, equity_method_investment, investment_property,
        dividend_payout_ratio, dividend_yield, major_shareholder_ratio
"""
import os, sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

from config import RAW_DIR, START_DATE, END_DATE, KR_FILING_LAG_DAYS

DART_KEY = os.environ.get("DART_API_KEY", "")


def _to_float(v) -> float:
    try:
        v = str(v).replace(",", "").strip()
        if v and v not in ("-", ""):
            return float(v)
    except Exception:
        pass
    return np.nan


def _get_bs_items(d, corp: str, year: int) -> dict:
    """관계기업투자 / 투자부동산 — 재무상태표 계정과목"""
    try:
        fs = d.finstate_all(corp, year, reprt_code="11011", fs_div="CFS")
        if fs is None or fs.empty:
            fs = d.finstate_all(corp, year, reprt_code="11011", fs_div="OFS")
        if fs is None or fs.empty:
            return {}
    except Exception:
        return {}

    def get(pattern: str, sj_divs=None) -> float:
        names = fs["account_nm"].str.replace(" ", "", regex=False)
        mask = names.str.contains(pattern.replace(" ", ""), na=False, regex=False)
        if sj_divs:
            mask &= fs["sj_div"].isin(sj_divs)
        for _, r in fs[mask].iterrows():
            val = _to_float(r.get("thstrm_amount", ""))
            if not np.isnan(val):
                return val
        return np.nan

    eq_method = get("관계기업및공동기업투자", ["BS"])
    if np.isnan(eq_method):
        eq_method = get("관계기업투자", ["BS"])
    if np.isnan(eq_method):
        eq_method = get("지분법적용투자주식", ["BS"])

    inv_property = get("투자부동산", ["BS"])

    return {
        "equity_method_investment": eq_method,
        "investment_property": inv_property,
    }


def _get_dividend(corp: str, year: int) -> dict:
    """배당성향 / 현금배당수익률 — alotMatter API"""
    try:
        r = requests.get(
            "https://opendart.fss.or.kr/api/alotMatter.json",
            params={"crtfc_key": DART_KEY, "corp_code": corp,
                    "bsns_year": str(year), "reprt_code": "11011"},
            timeout=10,
        )
        data = r.json()
        if data.get("status") != "000" or not data.get("list"):
            return {}
    except Exception:
        return {}

    payout, yld = np.nan, np.nan
    for row in data["list"]:
        se = str(row.get("se", "")).replace(" ", "")
        val = _to_float(row.get("thstrm", ""))
        if "현금배당성향" in se and np.isnan(payout):
            payout = val
        elif "현금배당수익률" in se and np.isnan(yld):
            yld = val
    return {"dividend_payout_ratio": payout, "dividend_yield": yld}


def _get_major_shareholder(corp: str, year: int) -> dict:
    """최대주주 및 특수관계인 지분율 합계 — hyslrSttus API
    DART가 종목종류(stock_knd)별로 "계"(합계) 행을 별도로 내려주므로,
    보통주 "계" 행 값을 그대로 사용 (개별 주주 행과 중복 합산 방지)
    """
    try:
        r = requests.get(
            "https://opendart.fss.or.kr/api/hyslrSttus.json",
            params={"crtfc_key": DART_KEY, "corp_code": corp,
                    "bsns_year": str(year), "reprt_code": "11011"},
            timeout=10,
        )
        data = r.json()
        if data.get("status") != "000" or not data.get("list"):
            return {}
    except Exception:
        return {}

    total = np.nan
    for row in data["list"]:
        nm = str(row.get("nm", "")).replace(" ", "").replace("\n", "")
        stock_knd = str(row.get("stock_knd", ""))
        if nm == "계" and "보통" in stock_knd:
            total = _to_float(row.get("trmend_posesn_stock_qota_rt", ""))
            break
    if np.isnan(total):
        # "계" 행이 없으면(단일 주주만 있는 경우 등) 개별 행 합산으로 폴백
        rows = [row for row in data["list"] if str(row.get("nm", "")).strip() != "계"]
        vals = [_to_float(r.get("trmend_posesn_stock_qota_rt", "")) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        total = sum(vals) if vals else np.nan
    return {"major_shareholder_ratio": total}


def build(tickers: list[str], start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    save_path = RAW_DIR / "kr_governance.parquet"
    if save_path.exists():
        print("[kr_governance] 캐시 로드")
        return pd.read_parquet(save_path)

    if not DART_KEY:
        print("  [경고] DART_API_KEY 없음. kr_governance 수집 불가")
        return pd.DataFrame()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[kr_governance] DART 지배구조/배당 데이터 수집 중... ({len(tickers)}개 종목)")

    import OpenDartReader
    d = OpenDartReader(DART_KEY)
    corp_codes_df = d.corp_codes
    code_map = corp_codes_df.dropna(subset=["stock_code"]).set_index("stock_code")["corp_code"].to_dict()

    start_year = max(pd.Timestamp(start).year, 2015)
    end_year = pd.Timestamp(end).year

    rows = []
    for i, ticker in enumerate(tickers, 1):
        corp = code_map.get(ticker)
        if not corp:
            continue
        for year in range(start_year, end_year + 1):
            row = {"date": pd.Timestamp(f"{year}-12-31"), "ticker": ticker}
            row.update(_get_bs_items(d, corp, year))
            time.sleep(0.05)
            row.update(_get_dividend(corp, year))
            time.sleep(0.05)
            row.update(_get_major_shareholder(corp, year))
            time.sleep(0.05)
            rows.append(row)

        if i % 10 == 0:
            print(f"  {i}/{len(tickers)}")
        time.sleep(0.1)

    if not rows:
        print("  [경고] kr_governance 수집 실패")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=KR_FILING_LAG_DAYS)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df.to_parquet(save_path, index=False)
    print(f"[kr_governance] 저장: {len(df)}행")
    for col in ["equity_method_investment", "investment_property",
                "dividend_payout_ratio", "dividend_yield", "major_shareholder_ratio"]:
        print(f"  {col}: {df[col].notna().mean():.1%}")
    return df


if __name__ == "__main__":
    uni = pd.read_parquet(RAW_DIR / "kr_universe.parquet")
    tickers = uni["ticker"].unique().tolist()
    build(tickers)
