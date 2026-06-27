"""
Phase 2  팩터 계산
- 5개 펀더멘털 팩터: BP, EP, SP, ROE, GPA
- 전처리: 상하위 1% 윈소라이즈 → 횡단면 z-score
- 입력: panel DataFrame (date, ticker, close, bps, eps, sps, roe, gross_profit, total_assets)
- 반환: (date × ticker) 피벗 dict {"BP": df, "EP": df, ...}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mstats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WINSORIZE_PCT


# ── 팩터 정의 ─────────────────────────────────────────────────────────────

FACTOR_DEFS = {
    "BP":  ("bps",          "close",        "div"),   # BPS / 주가
    "EP":  ("eps",          "close",        "div"),   # EPS / 주가
    "SP":  ("sps",          "close",        "div"),   # SPS / 주가
    "ROE": ("roe",          None,           "raw"),   # ROE 직접 사용
    "GPA": ("gross_profit", "total_assets", "div"),   # GP / TA
}


def _compute_raw(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    단일 팩터 raw 값 계산 → (date × ticker) 피벗
    """
    num_col, den_col, mode = FACTOR_DEFS[factor]

    p = panel.copy()

    # 필요 컬럼 확인
    if num_col not in p.columns:
        raise KeyError(f"패널에 '{num_col}' 컬럼 없음 -- 재무 데이터 확인 필요")

    if mode == "div":
        if den_col is None:
            raise ValueError
        if den_col not in p.columns:
            raise KeyError(f"패널에 '{den_col}' 컬럼 없음")
        denom = p[den_col].replace(0, np.nan)
        p["_raw"] = p[num_col] / denom
    else:
        p["_raw"] = p[num_col]

    pivot = p.pivot_table(index="date", columns="ticker", values="_raw")
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()


def _winsorize(series: pd.Series, pct: float = WINSORIZE_PCT) -> pd.Series:
    """횡단면 상하위 pct% 윈소라이즈"""
    arr = mstats.winsorize(series.dropna(), limits=[pct, pct])
    result = series.copy()
    result[series.notna()] = arr
    return result


def _zscore(series: pd.Series) -> pd.Series:
    """횡단면 z-score 정규화"""
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def _normalize(pivot: pd.DataFrame) -> pd.DataFrame:
    """행(날짜)별 윈소라이즈 + z-score"""
    return pivot.apply(lambda row: _zscore(_winsorize(row)), axis=1)


# ── 공개 API ─────────────────────────────────────────────────────────────

def compute_all(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    5개 팩터 계산 + 정규화
    반환: {"BP": (date × ticker), "EP": ..., "SP": ..., "ROE": ..., "GPA": ...}
          값이 클수록 좋은 종목 (롱 후보)
    """
    result = {}
    for fname in ["BP", "EP", "SP", "ROE", "GPA"]:
        try:
            raw   = _compute_raw(panel, fname)
            normed = _normalize(raw)
            result[fname] = normed
            print(f"  팩터 {fname}: {normed.notna().sum().sum():,}개 유효 값")
        except KeyError as e:
            print(f"  팩터 {fname} 건너뜀: {e}")
        except Exception as e:
            print(f"  팩터 {fname} 오류: {e}")
    return result


def compute_ic(factor_pivot: pd.DataFrame,
               fwd_ret_pivot: pd.DataFrame,
               period: int = 1) -> pd.Series:
    """
    IC (Information Coefficient) 계산
    = 각 날짜별 팩터 점수 vs 미래 period일 수익률의 Spearman 상관계수
    """
    ic_vals = []
    dates   = factor_pivot.index.intersection(fwd_ret_pivot.index)
    for d in dates:
        f = factor_pivot.loc[d].dropna()
        r = fwd_ret_pivot.loc[d].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        ic = f[common].corr(r[common], method="spearman")
        ic_vals.append({"date": d, "ic": ic})
    return pd.DataFrame(ic_vals).set_index("date")["ic"] if ic_vals else pd.Series(dtype=float)
