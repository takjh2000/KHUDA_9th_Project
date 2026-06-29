"""
KR 종목 sector/industry 수집
KRX KIND → 업종 분류 (161개 업종 → 11개 GICS-스타일 섹터로 매핑)
저장: data/raw/kr_sector.parquet  (ticker, sector, industry)
"""
import sys, requests, warnings
from pathlib import Path
from io import StringIO
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
warnings.filterwarnings("ignore")
from config import RAW_DIR

# 업종 키워드 → GICS 섹터 매핑
SECTOR_MAP = {
    "반도체": "IT",
    "전자": "IT",
    "소프트웨어": "IT",
    "통신": "IT",
    "디스플레이": "IT",
    "IT": "IT",
    "정보": "IT",
    "인터넷": "IT",
    "게임": "IT",
    "제약": "Healthcare",
    "바이오": "Healthcare",
    "의료": "Healthcare",
    "헬스": "Healthcare",
    "병원": "Healthcare",
    "화학": "Materials",
    "소재": "Materials",
    "철강": "Materials",
    "금속": "Materials",
    "비철": "Materials",
    "플라스틱": "Materials",
    "섬유": "Materials",
    "자동차": "Consumer Discretionary",
    "의류": "Consumer Discretionary",
    "패션": "Consumer Discretionary",
    "가구": "Consumer Discretionary",
    "가전": "Consumer Discretionary",
    "레저": "Consumer Discretionary",
    "여행": "Consumer Discretionary",
    "호텔": "Consumer Discretionary",
    "유통": "Consumer Discretionary",
    "백화점": "Consumer Discretionary",
    "식품": "Consumer Staples",
    "음료": "Consumer Staples",
    "담배": "Consumer Staples",
    "농업": "Consumer Staples",
    "생활": "Consumer Staples",
    "은행": "Financials",
    "금융": "Financials",
    "보험": "Financials",
    "증권": "Financials",
    "투자": "Financials",
    "자산": "Financials",
    "건설": "Real Estate",
    "부동산": "Real Estate",
    "주택": "Real Estate",
    "에너지": "Energy",
    "석유": "Energy",
    "가스": "Energy",
    "전력": "Utilities",
    "전기": "Utilities",
    "수도": "Utilities",
    "환경": "Utilities",
    "운송": "Industrials",
    "물류": "Industrials",
    "항공": "Industrials",
    "해운": "Industrials",
    "기계": "Industrials",
    "조선": "Industrials",
    "방산": "Industrials",
    "포장": "Industrials",
    "인쇄": "Industrials",
    "광고": "Communication Services",
    "미디어": "Communication Services",
    "방송": "Communication Services",
    "출판": "Communication Services",
}


def _map_sector(industry: str) -> str:
    for keyword, sector in SECTOR_MAP.items():
        if keyword in str(industry):
            return sector
    return "Other"


def _fetch(market_type: str) -> pd.DataFrame:
    url = "http://kind.krx.co.kr/corpgeneral/corpList.do"
    params = {"method": "download", "searchType": "13", "marketType": market_type}
    r = requests.get(url, params=params, timeout=15)
    r.encoding = "euc-kr"
    df = pd.read_html(StringIO(r.text), encoding="euc-kr")[0]
    return df


def build() -> pd.DataFrame:
    save_path = RAW_DIR / "kr_sector.parquet"
    if save_path.exists():
        print("[kr_sector] 캐시 로드")
        return pd.read_parquet(save_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for mkt in ["stockMkt", "kosdaqMkt"]:
        try:
            df = _fetch(mkt)
            frames.append(df)
            print(f"  {mkt}: {len(df)}개")
        except Exception as e:
            print(f"  {mkt} 실패: {e}")

    if not frames:
        print("[kr_sector] 수집 실패")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = raw.columns.str.strip()

    # 종목코드 / 업종 컬럼 찾기
    ticker_col = next((c for c in raw.columns if "코드" in c), None)
    industry_col = next((c for c in raw.columns if "업종" in c and "구분" not in c), None)

    if ticker_col is None:
        print("[kr_sector] 종목코드 컬럼 없음")
        return pd.DataFrame()

    result = pd.DataFrame()
    result["ticker"] = raw[ticker_col].astype(str).str.zfill(6)

    if industry_col:
        result["industry"] = raw[industry_col].astype(str).str.strip()
    else:
        result["industry"] = "Unknown"

    result["sector"] = result["industry"].apply(_map_sector)
    result = result.drop_duplicates("ticker").reset_index(drop=True)

    result.to_parquet(save_path, index=False)
    print(f"[kr_sector] 저장: {len(result)}개 종목")
    print(f"  sector 분포:\n{result['sector'].value_counts().to_string()}")
    return result


if __name__ == "__main__":
    build()
