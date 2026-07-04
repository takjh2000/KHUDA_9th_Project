"""
Phase 4  성과 집계
- PnL, Return(CAGR), Sharpe, MDD, Turnover
- 팩터 × 시장 10개 조합 요약 테이블
"""
import numpy as np
import pandas as pd
from backtest.engine import BacktestResult


def cagr(returns: pd.Series) -> float:
    """연환산 복리 수익률"""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    total = (1 + r).prod()
    years = len(r) / 252
    return total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan


def sharpe(returns: pd.Series) -> float:
    """연환산 샤프 (rf=0, 롱숏 시장 중립 가정)"""
    r = returns.dropna()
    if r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(252)


def mdd(returns: pd.Series) -> float:
    """최대 낙폭"""
    cum = (1 + returns.dropna()).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def margin(returns: pd.Series, turnover: float) -> float:
    """Margin (bps) = 평균 일간 수익률 / 평균 회전율 — 거래 1단위당 수익 (WQ Brain 스타일)"""
    r = returns.dropna()
    if len(r) == 0 or turnover is None or pd.isna(turnover) or turnover == 0:
        return np.nan
    return (r.mean() / turnover) * 10000


def annual_returns(returns: pd.Series) -> pd.Series:
    """연도별 수익률"""
    r = returns.dropna()
    r.index = pd.DatetimeIndex(r.index)
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def summarize(result: BacktestResult, label: str = "") -> dict:
    """단일 백테스트 결과 → 지표 딕셔너리"""
    ls = result.ls_returns
    return {
        "label":    label,
        "CAGR":     cagr(ls),
        "Sharpe":   sharpe(ls),
        "MDD":      mdd(ls),
        "Turnover": result.avg_turnover,
    }


def build_summary_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    팩터 × 시장 10개 조합 요약 테이블
    results: {"KR_BP": BacktestResult, "US_BP": ..., ...}
    """
    rows = [summarize(v, k) for k, v in results.items()]
    df = pd.DataFrame(rows).set_index("label")
    df["CAGR"]   = df["CAGR"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    df["Sharpe"]  = df["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    df["MDD"]    = df["MDD"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    df["Turnover"] = df["Turnover"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    return df
