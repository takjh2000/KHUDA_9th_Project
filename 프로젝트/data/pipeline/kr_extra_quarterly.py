"""
KR 추가 재무 데이터 - 분기 버전 (DART API)
- 연 1회(사업보고서)가 아니라 연 4회(1분기/반기/3분기/사업보고서) 수집
- 손익계산서(IS/CIS) 항목: thstrm_amount가 이미 "해당 분기 단독" 값
  (반기보고서/3분기보고서는 thstrm_amount=단독, thstrm_add_amount=누적으로 DART가 이미 분리 제공)
  단, 4분기 단독값은 별도 보고서가 없어 "사업보고서 전체(thstrm_amount) - 3분기보고서 누적(thstrm_add_amount)"로 역산
- 재무상태표(BS) 항목: 시점 잔액이라 각 보고서 값 그대로 사용 (누적 문제 없음)
- 현금흐름표(CF) 항목: thstrm_add_amount 컬럼이 없고 thstrm_amount 자체가 "연초~해당분기말 누적"이므로
  직전 분기 누적값과의 차감으로 단독 분기값 역산
저장: data/raw/kr_extra_finance_quarterly.parquet
  컬럼: date(분기말), ticker, operating_income, sga_expense, equity, debt, goodwill, cashflow_op
"""
import os, sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

from config import RAW_DIR, KR_FILING_LAG_DAYS

DART_KEY = os.environ.get("DART_API_KEY", "")

# reprt_code -> (분기말 월/일, 분기 순번)
REPORTS = [("11013", 3, 31, 1), ("11012", 6, 30, 2), ("11014", 9, 30, 3), ("11011", 12, 31, 4)]


def _to_float(v) -> float:
    try:
        v = str(v).replace(",", "").strip()
        if v and v not in ("-", ""):
            return float(v)
    except Exception:
        pass
    return np.nan


def _get(fs: pd.DataFrame, pattern: str, sj_divs=None, col="thstrm_amount") -> float:
    names = fs["account_nm"].str.replace(" ", "", regex=False)
    mask = names.str.contains(pattern.replace(" ", ""), na=False, regex=False)
    if sj_divs:
        mask &= fs["sj_div"].isin(sj_divs)
    for _, r in fs[mask].iterrows():
        val = _to_float(r.get(col, ""))
        if not np.isnan(val):
            return val
    return np.nan


def _parse_report(fs: pd.DataFrame) -> dict:
    """한 보고서(fs)에서 IS/BS 단독값 + CF 누적값(thstrm_amount)을 뽑는다"""
    operating_income = _get(fs, "영업이익", ["IS", "CIS"])
    sga_expense      = _get(fs, "판매비와관리비", ["IS", "CIS"])
    equity           = _get(fs, "자본총계", ["BS"])
    total_debt       = _get(fs, "부채총계", ["BS"])
    goodwill         = _get(fs, "영업권", ["BS"])
    if np.isnan(goodwill):
        goodwill     = _get(fs, "무형자산", ["BS"])

    # 3분기보고서/사업보고서에서 IS 항목의 누적치(4분기 역산용) - 손익계산서 항목 전부 필요
    oi_cum  = _get(fs, "영업이익", ["IS", "CIS"], col="thstrm_add_amount")
    sga_cum = _get(fs, "판매비와관리비", ["IS", "CIS"], col="thstrm_add_amount")

    cashflow_op_cum = _get(fs, "영업활동현금흐름", ["CF"])
    if np.isnan(cashflow_op_cum):
        cashflow_op_cum = _get(fs, "영업활동으로인한현금흐름", ["CF"])

    return {
        "operating_income": operating_income,
        "sga_expense": sga_expense,
        "equity": equity,
        "debt": total_debt,
        "goodwill": goodwill,
        "cashflow_op_cum": cashflow_op_cum,  # CF는 항상 누적치, 나중에 차감
        "operating_income_cum": oi_cum,       # 4분기 역산용
        "sga_expense_cum": sga_cum,           # 4분기 역산용
    }


def build_ticker_year(d, corp: str, ticker: str, year: int) -> list[dict]:
    quarters = {}
    for code, mm, dd, q in REPORTS:
        try:
            fs = d.finstate_all(corp, year, reprt_code=code, fs_div="CFS")
            if fs is None or fs.empty:
                fs = d.finstate_all(corp, year, reprt_code=code, fs_div="OFS")
            if fs is None or fs.empty:
                continue
            parsed = _parse_report(fs)
        except Exception:
            continue
        quarters[q] = {"date": pd.Timestamp(year=year, month=mm, day=dd), **parsed}
        time.sleep(0.05)

    if not quarters:
        return []

    rows = []
    prev_cf_cum = 0.0
    for q in [1, 2, 3, 4]:
        if q not in quarters:
            prev_cf_cum = np.nan
            continue
        cur = quarters[q]
        row = {
            "date": cur["date"], "ticker": ticker,
            "operating_income": cur["operating_income"],
            "sga_expense": cur["sga_expense"],
            "equity": cur["equity"],
            "debt": cur["debt"],
            "goodwill": cur["goodwill"],
        }
        # 4분기 단독 = 사업보고서 전체(연간) - 3분기 누적 (손익계산서 항목 전부 동일하게 적용)
        if q == 4 and 3 in quarters:
            oi_3q_cum = quarters[3].get("operating_income_cum", np.nan)
            if not np.isnan(oi_3q_cum):
                row["operating_income"] = cur["operating_income"] - oi_3q_cum
            sga_3q_cum = quarters[3].get("sga_expense_cum", np.nan)
            if not np.isnan(sga_3q_cum):
                row["sga_expense"] = cur["sga_expense"] - sga_3q_cum

        # CF: 단독 = 이번 누적 - 직전 누적 (1분기는 누적=단독)
        cf_cum = cur["cashflow_op_cum"]
        if np.isnan(cf_cum) or np.isnan(prev_cf_cum):
            row["cashflow_op"] = np.nan
        else:
            row["cashflow_op"] = cf_cum - prev_cf_cum
        prev_cf_cum = cf_cum

        rows.append(row)
    return rows


def build(tickers: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    save_path = RAW_DIR / "kr_extra_finance_quarterly.parquet"
    if save_path.exists():
        print("[kr_extra_finance_quarterly] 캐시 로드")
        return pd.read_parquet(save_path)

    if not DART_KEY:
        print("  [경고] DART_API_KEY 없음")
        return pd.DataFrame()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    partial_path = RAW_DIR / "kr_extra_finance_quarterly_partial.parquet"
    all_rows = []
    done_tickers = set()
    if partial_path.exists():
        prev = pd.read_parquet(partial_path)
        all_rows = prev.to_dict("records")
        done_tickers = set(prev["ticker"].unique())
        print(f"[kr_extra_finance_quarterly] 이어받기: 이미 완료된 {len(done_tickers)}개 종목 로드")

    print(f"[kr_extra_finance_quarterly] DART 분기 재무 수집 중... ({len(tickers)}개 종목 x {end_year-start_year+1}년)")

    import OpenDartReader
    d = OpenDartReader(DART_KEY)
    code_map = d.corp_codes.dropna(subset=["stock_code"]).set_index("stock_code")["corp_code"].to_dict()

    def _save_partial():
        tmp_df = pd.DataFrame(all_rows)
        tmp_df.to_parquet(partial_path, index=False)

    for i, ticker in enumerate(tickers, 1):
        if ticker in done_tickers:
            continue
        corp = code_map.get(ticker)
        if not corp:
            continue
        for year in range(start_year, end_year + 1):
            all_rows.extend(build_ticker_year(d, corp, ticker, year))
        if i % 10 == 0:
            print(f"  {i}/{len(tickers)}", flush=True)
            _save_partial()
        time.sleep(0.1)

    _save_partial()

    if not all_rows:
        print("  [경고] 수집 실패")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=KR_FILING_LAG_DAYS)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(save_path, index=False)
    print(f"[kr_extra_finance_quarterly] 저장: {len(df)}행")
    for col in ["operating_income", "sga_expense", "equity", "debt", "goodwill", "cashflow_op"]:
        print(f"  {col}: {df[col].notna().mean():.1%}")
    return df


if __name__ == "__main__":
    tickers = pd.read_csv(RAW_DIR / "_kr_universe_2019_2023_tickers.csv", header=None)[0].astype(str).str.zfill(6).tolist()
    build(tickers, 2019, 2023)
