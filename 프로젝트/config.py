from pathlib import Path

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"

# ── 기간 ──────────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

# ── 팩터 / 시장 ───────────────────────────────────────────────────────────
FACTORS = ["BP", "EP", "SP", "ROE", "GPA"]   # 5개 펀더멘털 팩터
MARKETS = ["KR", "US"]

# ── 전략 ──────────────────────────────────────────────────────────────────
LONG_PCT      = 0.20        # 상위 20% 롱
SHORT_PCT     = 0.20        # 하위 20% 숏
REBALANCE     = "QE"        # 분기말 리밸런싱

# ── 팩터 전처리 ───────────────────────────────────────────────────────────
WINSORIZE_PCT = 0.01        # 상하위 1% 윈소라이즈

# ── 재무 공시 지연 ────────────────────────────────────────────────────────
US_FILING_LAG_DAYS = 60     # 분기말 + 60일 후 사용
KR_FILING_LAG_DAYS = 0      # 네이버 금융 스크래핑 시 0 (DART 사용 시 45로 복원)
