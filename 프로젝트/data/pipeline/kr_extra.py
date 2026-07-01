"""
KR 추가 재무 + 업종 계층 데이터 수집 (DART API)
- 추가 재무: cashflow_op, operating_income, equity, debt, goodwill, sga_expense
- 업종 계층: sector(대분류) / industry(중분류) / subindustry(소분류) — KSIC 기반
저장:
  data/raw/kr_extra_finance.parquet
  data/raw/kr_sector_hierarchy.parquet  (ticker, sector, industry, subindustry)
"""
import os, sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")

from config import RAW_DIR, START_DATE, END_DATE, KR_FILING_LAG_DAYS

DART_KEY = os.environ.get("DART_API_KEY", "")

# ── KSIC 대분류 → GICS 스타일 Sector 매핑 ──────────────────────────────────
KSIC_SECTION_TO_SECTOR = {
    "A": "Consumer Staples",    # 농업, 임업, 어업
    "B": "Energy",              # 광업
    "C": "Industrials",         # 제조업 (기본, 세부는 아래서 override)
    "D": "Utilities",           # 전기, 가스, 증기
    "E": "Utilities",           # 수도, 하수, 환경
    "F": "Real Estate",         # 건설업
    "G": "Consumer Discretionary",  # 도소매업
    "H": "Industrials",         # 운수, 창고
    "I": "Communication Services",  # 숙박, 음식
    "J": "Communication Services",  # 정보통신업
    "K": "Financials",          # 금융, 보험
    "L": "Real Estate",         # 부동산업
    "M": "IT",                  # 전문, 과학, 기술
    "N": "Industrials",         # 사업시설 관리
    "O": "Industrials",         # 공공행정
    "P": "Consumer Staples",    # 교육
    "Q": "Healthcare",          # 보건, 사회복지
    "R": "Communication Services",  # 예술, 스포츠
    "S": "Consumer Discretionary",  # 협회, 수리
    "T": "Consumer Staples",    # 가구내 고용
    "U": "Industrials",         # 국제기구
}

# KSIC 제조업(C) 세부 2자리 코드 → 더 정확한 Sector
KSIC_MFG_DIV_TO_SECTOR = {
    "10": "Consumer Staples",   # 식료품
    "11": "Consumer Staples",   # 음료
    "12": "Consumer Staples",   # 담배
    "13": "Consumer Discretionary",  # 섬유
    "14": "Consumer Discretionary",  # 의복
    "15": "Consumer Discretionary",  # 가죽
    "16": "Materials",          # 목재
    "17": "Consumer Staples",   # 펄프, 종이
    "18": "Communication Services",  # 인쇄
    "19": "Energy",             # 코크스, 석유
    "20": "Materials",          # 화학
    "21": "Healthcare",         # 의약품
    "22": "Materials",          # 고무, 플라스틱
    "23": "Materials",          # 비금속
    "24": "Materials",          # 1차금속
    "25": "Industrials",        # 금속가공
    "26": "IT",                 # 전자부품, 반도체
    "27": "IT",                 # 의료, 정밀기기
    "28": "IT",                 # 전기장비
    "29": "Industrials",        # 기계
    "30": "Consumer Discretionary",  # 자동차
    "31": "Industrials",        # 기타운송
    "32": "Consumer Discretionary",  # 가구
    "33": "Industrials",        # 기타제조
}


def _ksic_to_sector(code: str) -> str:
    if not code or len(code) < 2:
        return "Other"
    section = code[0].upper()
    div2 = code[:2] if len(code) >= 2 else ""
    if section == "C" and div2 in KSIC_MFG_DIV_TO_SECTOR:
        return KSIC_MFG_DIV_TO_SECTOR[div2]
    return KSIC_SECTION_TO_SECTOR.get(section, "Other")


# ── DART 업종명 매핑 ────────────────────────────────────────────────────────

def _build_ksic_names() -> dict:
    """KSIC 코드 → 업종명 (간략 버전, 핵심 코드만)"""
    return {
        # 섹션 수준 (1자리)
        "A": "농림어업", "B": "광업", "C": "제조업", "D": "전기가스업",
        "E": "환경업", "F": "건설업", "G": "도소매업", "H": "운수창고업",
        "I": "숙박음식업", "J": "정보통신업", "K": "금융보험업",
        "L": "부동산업", "M": "전문과학기술", "N": "사업지원", "Q": "보건복지업",
        # 제조업 중분류 (2자리)
        "10": "식료품제조", "11": "음료제조", "13": "섬유제조", "14": "의복제조",
        "20": "화학제품", "21": "의약품제조", "22": "고무플라스틱", "23": "비금속광물",
        "24": "1차금속", "25": "금속가공", "26": "전자부품반도체", "27": "의료정밀기기",
        "28": "전기장비", "29": "기계장비", "30": "자동차", "31": "기타운송장비",
        # 정보통신 중분류
        "58": "출판", "59": "영상음향", "60": "방송", "61": "통신업",
        "62": "컴퓨터프로그래밍", "63": "정보서비스",
        # 금융 중분류
        "64": "금융업", "65": "보험업", "66": "금융보조",
        # 전문과학
        "70": "연구개발", "71": "전문서비스", "72": "건축기술",
        # 보건
        "86": "보건업", "87": "사회복지",
    }


KSIC_NAMES = _build_ksic_names()


def _code_to_name(code: str, level: int) -> str:
    """KSIC 코드 → 업종명 (level: 1=섹션, 2=중분류, 3=소분류)"""
    if not code:
        return "Unknown"
    key = code[:level].upper() if level == 1 else code[:level]
    return KSIC_NAMES.get(key, f"업종{key}")


# ── DART 기업 정보 수집 ────────────────────────────────────────────────────

def _get_corp_map() -> dict:
    """stock_code → {corp_code, induty_code} — DART API 직접 호출"""
    import requests
    from opendartreader import OpenDartReader
    d = OpenDartReader(DART_KEY)
    corp_df = d.corp_codes
    corp_df = corp_df.dropna(subset=["stock_code"])

    result = {}
    for _, row in corp_df.iterrows():
        sc = str(row["stock_code"]).zfill(6)
        result[sc] = {"corp_code": row["corp_code"], "induty_code": ""}

    # induty_code는 company API에서 개별 조회 (배치 불가)
    # 전체 조회는 너무 느리므로 corp_codes에 있는 종목만 처리
    # 샘플로 확인 후 필요시 전체 조회
    print(f"  corp_codes 총 {len(result)}개 종목")
    return result


def build_sector_hierarchy(tickers: list[str]) -> pd.DataFrame:
    save_path = RAW_DIR / "kr_sector_hierarchy.parquet"
    if save_path.exists():
        print("[kr_sector_hierarchy] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[kr_sector_hierarchy] DART 기업개황 API로 업종코드 수집 중... ({len(tickers)}개)")

    import requests
    from opendartreader import OpenDartReader
    d = OpenDartReader(DART_KEY)
    corp_df = d.corp_codes.dropna(subset=["stock_code"])
    ticker_to_corp = {str(r["stock_code"]).zfill(6): r["corp_code"]
                      for _, r in corp_df.iterrows()}

    rows = []
    for i, ticker in enumerate(tickers, 1):
        corp_code = ticker_to_corp.get(ticker, "")
        induty_code = ""

        if corp_code:
            try:
                r = requests.get(
                    "https://opendart.fss.or.kr/api/company.json",
                    params={"crtfc_key": DART_KEY, "corp_code": corp_code},
                    timeout=5,
                )
                data = r.json()
                if data.get("status") == "000":
                    induty_code = str(data.get("induty_code", "")).strip()
            except Exception:
                pass
            time.sleep(0.1)

        sector      = _ksic_to_sector(induty_code)
        industry    = _code_to_name(induty_code, 2) if len(induty_code) >= 2 else _code_to_name(induty_code, 1)
        subindustry = _code_to_name(induty_code, 3) if len(induty_code) >= 3 else industry

        rows.append({
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "subindustry": subindustry,
            "ksic_code": induty_code,
        })

        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}")

    df = pd.DataFrame(rows)
    df.to_parquet(save_path, index=False)
    print(f"[kr_sector_hierarchy] 저장: {len(df)}개 종목")
    print(f"  sector: {df['sector'].nunique()}개  industry: {df['industry'].nunique()}개  "
          f"subindustry: {df['subindustry'].nunique()}개")
    return df


# ── DART 추가 재무 수집 ────────────────────────────────────────────────────

def _parse_extra(fs: pd.DataFrame, ticker: str, year: int) -> dict:
    """DART finstate_all에서 추가 항목 추출"""
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

    operating_income = get("영업이익", ["IS"])
    sga_expense      = get("판매비와관리비", ["IS"])
    equity           = get("자본총계", ["BS"])
    total_debt       = get("부채총계", ["BS"])
    goodwill         = get("영업권", ["BS"])
    if np.isnan(goodwill):
        goodwill     = get("무형자산", ["BS"])  # 영업권 없으면 무형자산

    # 현금흐름표
    cashflow_op = get("영업활동현금흐름", ["CF"])
    if np.isnan(cashflow_op):
        cashflow_op = get("영업활동으로인한현금흐름", ["CF"])

    return {
        "date":             pd.Timestamp(f"{year}-12-31"),
        "ticker":           ticker,
        "operating_income": operating_income,
        "sga_expense":      sga_expense,
        "equity":           equity,
        "debt":             total_debt,
        "goodwill":         goodwill,
        "cashflow_op":      cashflow_op,
    }


def build_extra_finance(tickers: list[str]) -> pd.DataFrame:
    save_path = RAW_DIR / "kr_extra_finance.parquet"
    if save_path.exists():
        print("[kr_extra_finance] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[kr_extra_finance] DART 추가 재무 수집 중... ({len(tickers)}개 종목)")

    from opendartreader import OpenDartReader
    d = OpenDartReader(DART_KEY)
    corp_codes_df = d.corp_codes
    code_map = corp_codes_df.dropna(subset=["stock_code"]).set_index("stock_code")["corp_code"].to_dict()

    start_year = max(pd.Timestamp(START_DATE).year, 2015)
    end_year   = pd.Timestamp(END_DATE).year

    all_rows = []
    for i, ticker in enumerate(tickers, 1):
        corp = code_map.get(ticker)
        if not corp:
            continue
        for year in range(start_year, end_year + 1):
            try:
                fs = d.finstate_all(corp, year, reprt_code="11011", fs_div="CFS")
                if fs is None or fs.empty:
                    fs = d.finstate_all(corp, year, reprt_code="11011", fs_div="OFS")
                if fs is None or fs.empty:
                    continue
                row = _parse_extra(fs, ticker, year)
                all_rows.append(row)
            except Exception:
                pass
            time.sleep(0.05)

        if i % 20 == 0:
            print(f"  {i}/{len(tickers)}")
        time.sleep(0.15)

    if not all_rows:
        print("  [경고] 추가 재무 수집 실패")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=KR_FILING_LAG_DAYS)
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df.to_parquet(save_path, index=False)
    print(f"[kr_extra_finance] 저장: {len(df)}행")
    return df


if __name__ == "__main__":
    import pandas as pd
    uni = pd.read_parquet(RAW_DIR / "kr_universe.parquet")
    tickers = uni["ticker"].unique().tolist()

    print("=== 업종 계층 수집 ===")
    hier = build_sector_hierarchy(tickers)
    print(hier["sector"].value_counts().to_string())

    print("\n=== 추가 재무 수집 ===")
    extra = build_extra_finance(tickers)
    if not extra.empty:
        for col in ["operating_income", "cashflow_op", "debt", "equity", "goodwill", "sga_expense"]:
            pct = extra[col].notna().mean() if col in extra.columns else 0
            print(f"  {col}: {pct:.1%}")
