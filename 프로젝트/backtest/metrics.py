"""
Phase 4  성과 집계
- WQ Brain 스타일 지표: Returns, CAGR, Sharpe, MDD, Drawdown, Turnover, Fitness, Margin
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
    """연환산 샤프 (rf=0, 롱숏 시장 중립 가정)
    Sharpe_annual = √252 × (μ/σ)
    """
    r = returns.dropna()
    if r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(252)


def mdd(returns: pd.Series) -> float:
    """최대 낙폭 (전체 기간 누적 기준)"""
    cum = (1 + returns.dropna()).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()


def annual_returns(returns: pd.Series) -> pd.Series:
    """연도별 수익률"""
    r = returns.dropna()
    r.index = pd.DatetimeIndex(r.index)
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


# ── WQ Brain 스타일 지표 ─────────────────────────────────────────────────────

def returns_total(returns: pd.Series) -> float:
    """총 누적 수익률 (비연환산)
    Returns = ∏(1 + R_t) - 1
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return (1 + r).prod() - 1


def drawdown_wq(returns: pd.Series) -> float:
    """WQ Brain Drawdown: 연도별 최대 낙폭 중 최악값
    Drawdown = max_y (Annual Max Loss_y)
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    r = r.copy()
    r.index = pd.DatetimeIndex(r.index)
    worst = 0.0
    for _, group in r.groupby(r.index.year):
        cum = (1 + group).cumprod()
        dd  = ((cum - cum.cummax()) / cum.cummax()).min()
        worst = min(worst, dd)
    return worst


def fitness(returns: pd.Series, result: BacktestResult) -> float:
    """WQ Brain Fitness = Sharpe × √|Returns| / max(Daily Turnover, 0.125)
    높을수록 좋음. 분기 리밸런싱 전략은 turnover 플로어(0.125) 적용
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    s  = sharpe(r)
    rt = abs(returns_total(r))
    dt = result.daily_turnover
    if any(pd.isna(x) for x in [s, rt, dt]):
        return np.nan
    return s * np.sqrt(rt) / max(dt, 0.125)


def margin(returns: pd.Series, result: BacktestResult) -> float:
    """총 PnL / 총 거래량 (bps) — WQ Brain Margin 근사
    m = P / D × 10000  (WQ 최소 기준: 5 bps)
    왕복(round-trip) 기준으로 나눔
    """
    r = returns.dropna()
    if len(r) == 0 or result.turn_df.empty:
        return np.nan
    total_ret    = returns_total(r)
    total_traded = result.turn_df["turnover"].sum() * 2  # 왕복 거래량
    if total_traded == 0:
        return np.nan
    return (total_ret / total_traded) * 10000


def summarize(result: BacktestResult, label: str = "") -> dict:
    """단일 백테스트 결과 → 지표 딕셔너리 (WQ Brain 전체 지표)"""
    ls = result.ls_returns
    return {
        "label":    label,
        "Returns":  returns_total(ls),
        "CAGR":     cagr(ls),
        "Sharpe":   sharpe(ls),
        "MDD":      mdd(ls),
        "Drawdown": drawdown_wq(ls),
        "Turnover": result.avg_turnover,
        "Fitness":  fitness(ls, result),
        "Margin":   margin(ls, result),
    }


def build_summary_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    팩터 × 시장 조합 요약 테이블 (WQ Brain 스타일 전체 지표)
    results: {"KR_BP": BacktestResult, "US_BP": ..., ...}
    """
    rows = [summarize(v, k) for k, v in results.items()]
    df = pd.DataFrame(rows).set_index("label")
    fmt_map = {
        "Returns":  "{:.2%}",
        "CAGR":     "{:.2%}",
        "Sharpe":   "{:.2f}",
        "MDD":      "{:.2%}",
        "Drawdown": "{:.2%}",
        "Turnover": "{:.2%}",
        "Fitness":  "{:.2f}",
        "Margin":   "{:.1f}",
    }
    for col, fmt in fmt_map.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else "N/A")
    return df
