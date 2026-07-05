"""
Phase 4  성과 집계 (WQ Brain 스타일 지표 정의)
- Book Size = $20,000,000 (롱 $10M + 숏 $10M), daily_return = daily_PnL / (Book Size/2)
- Sharpe, Returns, Turnover, Drawdown, Margin, Fitness
- 팩터 × 시장 10개 조합 요약 테이블
"""
import numpy as np
import pandas as pd
from backtest.engine import BacktestResult
from config import BOOK_SIZE

CAPITAL = BOOK_SIZE / 2   # 롱/숏 각 사이드에 배분되는 자본 ($10M)


def cagr(returns: pd.Series) -> float:
    """(참고용) 복리 연환산 수익률 — 아래 returns_metric()과는 다른 개념(compounding)"""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    total = (1 + r).prod()
    years = len(r) / 252
    return total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan


def returns_metric(returns: pd.Series) -> float:
    """Returns = mean(daily_PnL) × 252 / (Book Size/2) = mean(daily_return) × 252
    ls_ret(비중 정규화된 롱숏 일간수익률)이 곧 daily_return과 같으므로 그대로 사용"""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return r.mean() * 252


def sharpe(returns: pd.Series) -> float:
    """Sharpe = mean(daily_return) / std(daily_return) × √252  (rf=0)"""
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(252)


def mdd(returns: pd.Series) -> float:
    """Drawdown = max(cumulative_PnL[t] - cumulative_PnL[s]) for t<s
    cumulative_PnL은 달러 PnL의 누적 '합'(가산식) — 복리(compounding) 아님.
    반환값은 Book Size/2 대비 비율(음수, 예: -0.15 = -15%)."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    cum_pnl = (r * CAPITAL).cumsum()
    drawdown_dollar = (cum_pnl.cummax() - cum_pnl).max()
    if pd.isna(drawdown_dollar):
        return np.nan
    return -drawdown_dollar / CAPITAL


def margin(returns: pd.Series, turnover: float) -> float:
    """Margin (bps) = Total_PnL / Total_Dollar_Volume_Traded × 10,000
    = [mean(daily_return)×CAPITAL×N] / [mean(daily_turnover)×BOOK_SIZE×N] × 10,000
    = (mean(daily_return) / mean(daily_turnover)) × (CAPITAL/BOOK_SIZE) × 10,000"""
    r = returns.dropna()
    if len(r) == 0 or turnover is None or pd.isna(turnover) or turnover == 0:
        return np.nan
    return (r.mean() / turnover) * (CAPITAL / BOOK_SIZE) * 10000


def fitness(sharpe_val: float, returns_val: float, turnover_val: float) -> float:
    """Fitness = Sharpe × √|Returns| / max(Turnover, 0.125)"""
    if pd.isna(sharpe_val) or pd.isna(returns_val) or pd.isna(turnover_val):
        return np.nan
    denom = max(turnover_val, 0.125)
    return sharpe_val * np.sqrt(abs(returns_val)) / denom


def annual_returns(returns: pd.Series) -> pd.Series:
    """연도별 수익률"""
    r = returns.dropna()
    r.index = pd.DatetimeIndex(r.index)
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def summarize(result: BacktestResult, label: str = "") -> dict:
    """단일 백테스트 결과 → 지표 딕셔너리 (WQ Brain 스타일)"""
    ls = result.ls_returns
    to = result.avg_turnover
    ret = returns_metric(ls)
    sh  = sharpe(ls)
    return {
        "label":    label,
        "Returns":  ret,
        "Sharpe":   sh,
        "Turnover": to,
        "Drawdown": mdd(ls),
        "Margin":   margin(ls, to),
        "Fitness":  fitness(sh, ret, to),
    }


def build_summary_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    팩터 × 시장 10개 조합 요약 테이블
    results: {"KR_BP": BacktestResult, "US_BP": ..., ...}
    """
    rows = [summarize(v, k) for k, v in results.items()]
    df = pd.DataFrame(rows).set_index("label")
    df["Returns"]  = df["Returns"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    df["Sharpe"]   = df["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    df["Turnover"] = df["Turnover"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    df["Drawdown"] = df["Drawdown"].map(lambda x: f"{abs(x):.2%}" if pd.notna(x) else "N/A")
    df["Margin"]   = df["Margin"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    df["Fitness"]  = df["Fitness"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    return df
