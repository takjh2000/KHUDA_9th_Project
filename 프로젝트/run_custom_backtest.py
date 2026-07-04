"""
15개 커스텀 전략 백테스트 (US 시장)
전략 8, 9, 13은 WQ Brain 독점 데이터 필요 → 건너뜀
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from config import PROCESSED_DIR, RAW_DIR
from backtest.engine import LongShortBacktester, Config
from backtest.metrics import cagr, sharpe, mdd
from factors.wq_ops import (
    rank, zscore, ts_mean, ts_std, ts_rank, ts_zscore,
    ts_delta, ts_delay, ts_corr, ts_backfill,
    signed_power, group_neutralize_dynamic, group_rank,
    group_zscore, df_max, trade_when, bucket,
    group_neutralize,
)

RESULTS_DIR = Path("results/custom")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_us():
    panel = pd.read_parquet(PROCESSED_DIR / "us_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # 추가 재무 데이터 병합 (연간 데이터 → 일별 ffill)
    extra_path = RAW_DIR / "us_extra_finance.parquet"
    if extra_path.exists():
        extra = pd.read_parquet(extra_path)
        extra["date"] = pd.to_datetime(extra["date"])
        extra_cols = [c for c in extra.columns if c not in ["date", "ticker"]]
        panel = panel.merge(extra, on=["date", "ticker"], how="left")
        # 연간 데이터 forward fill (ticker별로)
        panel = panel.sort_values(["ticker", "date"])
        panel[extra_cols] = panel.groupby("ticker")[extra_cols].ffill(limit=252)
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        print(f"  추가 재무 컬럼: {extra_cols}")

    price_pivot = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()

    # 섹터 데이터
    sector_map = {}
    sector_path = RAW_DIR / "us_sector.parquet"
    if sector_path.exists():
        sec_df = pd.read_parquet(sector_path)
        sector_map = sec_df.set_index("ticker")["sector"].to_dict()

    # 거래량 데이터
    vol_pivot = None
    vol_path = RAW_DIR / "us_price_vol.parquet"
    if vol_path.exists():
        vol_pivot = pd.read_parquet(vol_path)
        vol_pivot.index = pd.to_datetime(vol_pivot.index)

    universe = None
    uni_path = RAW_DIR / "us_universe.parquet"
    if uni_path.exists():
        universe = pd.read_parquet(uni_path)
        universe["date"] = pd.to_datetime(universe["date"])

    return price_pivot, panel, universe, sector_map, vol_pivot


def to_pivot(panel, col):
    if col not in panel.columns:
        return None
    p = panel.pivot_table(index="date", columns="ticker", values=col).sort_index()
    if p.empty:
        return None
    # 연간 데이터는 최대 252일 forward fill
    p = p.ffill(limit=252)
    return p


def coalesce_piv(p_extra, p_base):
    """p_extra의 NaN을 p_base로 채움 (extra 데이터가 없는 기간 보완용)"""
    if p_extra is None:
        return p_base
    if p_base is None:
        return p_extra
    base_aligned = p_base.reindex(index=p_extra.index, columns=p_extra.columns)
    return p_extra.combine_first(base_aligned)


def sector_groups(panel, sector_map):
    """ticker → sector Series (패널 컬럼 기준)"""
    tickers = panel["ticker"].unique()
    return pd.Series({t: sector_map.get(t, "Unknown") for t in tickers})


# ── 전략 정의 ──────────────────────────────────────────────────────────────────

def s01_low_accrual(panel, price_pivot, sector_map, vol_pivot):
    """Low Accrual Clean"""
    roe_p  = to_pivot(panel, "roe")
    gp_p   = to_pivot(panel, "gross_profit")
    ta_p   = to_pivot(panel, "total_assets")
    cf_p   = to_pivot(panel, "cashflow_op")
    eps_p  = to_pivot(panel, "eps")
    if roe_p is None or ta_p is None:
        return None

    if eps_p is not None:
        # cashflow_op 없는 기간은 roe*total_assets 로 대체 (같은 단위로 근사)
        roe_ta = (roe_p * ta_p.replace(0, np.nan)) if roe_p is not None else None
        cf_filled = coalesce_piv(cf_p, roe_ta)
        if cf_filled is not None:
            cf_aligned = cf_filled.reindex(index=eps_p.index, columns=eps_p.columns).ffill(limit=60)
            accrual = rank(-(eps_p - cf_aligned) / ta_p.replace(0, np.nan))
        else:
            gpa = gp_p / ta_p.replace(0, np.nan) if gp_p is not None else None
            accrual = rank(-(roe_p - gpa)) if gpa is not None else rank(-roe_p)
    else:
        gpa = gp_p / ta_p.replace(0, np.nan) if gp_p is not None else None
        accrual = rank(-(roe_p - gpa)) if gpa is not None else rank(-roe_p)

    cap_bucket = bucket(rank(price_pivot), n=5)
    cap_neut = group_neutralize_dynamic(accrual, cap_bucket)
    returns = price_pivot.pct_change()
    mom = ts_rank(returns, 252)
    f = cap_neut - mom
    return zscore(f)


def s02_oey(panel, price_pivot, sector_map, vol_pivot):
    """Operating Income Earnings Yield"""
    oi_p = to_pivot(panel, "operating_income")
    eps_p = to_pivot(panel, "eps")
    # operating_income 없는 기간(2021 이전)은 eps로 대체
    combined = coalesce_piv(oi_p, eps_p)
    if combined is None:
        return None
    ep_base = combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    ep_base = ep_base / price_pivot.replace(0, np.nan)

    oey = ts_rank(ep_base, 126)
    f = rank(oey)
    return zscore(f)


def s03_cash_cf_divergence(panel, price_pivot, sector_map, vol_pivot):
    """Cash & CF Divergence"""
    cash_p = to_pivot(panel, "cash")
    cf_p   = to_pivot(panel, "cashflow_op")
    gp_p   = to_pivot(panel, "gross_profit")
    ta_p   = to_pivot(panel, "total_assets")
    sps_p  = to_pivot(panel, "sps")

    if ta_p is None or sps_p is None:
        return None

    # cash 없는 기간은 gross_profit/total_assets(수익성 proxy)로 대체
    gpa_p = gp_p / ta_p.replace(0, np.nan) if gp_p is not None else None
    cash_combined = coalesce_piv(cash_p, gpa_p)
    # cashflow_op 없는 기간은 sps로 대체
    cf_combined = coalesce_piv(cf_p, sps_p)

    if cash_combined is None or cf_combined is None:
        return None

    cash_piv = cash_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    cf_piv   = cf_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)

    f = -ts_corr(ts_mean(cash_piv, 5), ts_mean(cf_piv, 5), 252)
    return zscore(f)


def s04_growth_weighted_lt(panel, price_pivot, sector_map, vol_pivot):
    """Growth-Weighted LT Investment"""
    sps_p = to_pivot(panel, "sps")
    ta_p  = to_pivot(panel, "total_assets")
    if sps_p is None or ta_p is None:
        return None

    revenue_growth = rank(ts_mean(ts_delta(sps_p, 63), 5))
    lt_invest = ta_p.rolling(252, min_periods=126).sum()
    lt_invest = lt_invest.reindex(index=revenue_growth.index, columns=revenue_growth.columns).ffill(limit=60)
    gwi = lt_invest * revenue_growth
    slope = ts_delta(gwi, 756) / 756
    f = signed_power(slope, 1.5)
    return zscore(f)


def s05_industry_neutral_cf_yield(panel, price_pivot, sector_map, vol_pivot):
    """Industry-Neutral CF Yield"""
    cf_p = to_pivot(panel, "cashflow_op")
    gp_p = to_pivot(panel, "gross_profit")
    ta_p = to_pivot(panel, "total_assets")

    if ta_p is None:
        return None

    # cashflow_op 없는 기간은 gross_profit/total_assets로 대체
    gpa_p = gp_p / ta_p.replace(0, np.nan) if gp_p is not None else None
    cf_combined = coalesce_piv(cf_p, gpa_p)
    if cf_combined is None:
        return None

    cf_piv = cf_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    # cf total → yield (price와 단위 맞춤은 rank/zscore로 흡수)
    cf_yield = cf_piv / price_pivot.replace(0, np.nan)

    z = ts_zscore(cf_yield, 63)
    groups = sector_groups(panel, sector_map)
    common_tickers = z.columns.intersection(groups.index)
    if len(common_tickers) > 10:
        f = group_rank(z[common_tickers], groups[common_tickers])
    else:
        f = rank(z)
    return zscore(f)


def s06_debt_spike_reversal(panel, price_pivot, sector_map, vol_pivot):
    """Debt Spike Reversal Momentum"""
    debt_p = to_pivot(panel, "debt")
    ta_p   = to_pivot(panel, "total_assets")
    # debt 없는 기간은 total_assets로 대체
    debt_piv = coalesce_piv(debt_p, ta_p)
    if debt_piv is None:
        return None

    debt_piv = debt_piv.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    x = -ts_zscore(debt_piv, 252)
    alpha = signed_power(x, 4)
    condition = debt_piv > ts_mean(debt_piv, 63)
    alt = pd.DataFrame(-1.0, index=alpha.index, columns=alpha.columns)
    f = trade_when(condition, alpha, alt)
    return zscore(f)


def s07_aggressive_dual_value(panel, price_pivot, sector_map, vol_pivot):
    """Aggressive Dual Value Blend with Momentum Neutralization"""
    eps_p = to_pivot(panel, "eps")
    bps_p = to_pivot(panel, "bps")
    ebitda_p = to_pivot(panel, "ebitda")
    eq_p  = to_pivot(panel, "equity")

    if eps_p is None or bps_p is None:
        return None

    # ebitda 없는 기간은 eps로 대체
    ebitda_combined = coalesce_piv(ebitda_p, eps_p)
    ebitda_piv = ebitda_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    underrated = ebitda_piv / price_pivot.replace(0, np.nan)

    # equity 없는 기간은 bps로 대체
    eq_combined = coalesce_piv(eq_p, bps_p)
    eq_piv = eq_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    low_pbr = eq_piv / price_pivot.replace(0, np.nan)

    underrated_adj  = rank(underrated)
    low_pbr_recent  = ts_rank(low_pbr, 63)
    signal = df_max(underrated_adj, low_pbr_recent)

    returns = price_pivot.pct_change()
    mom_group = bucket(rank(ts_mean(returns, 240)), n=10)
    f = group_neutralize_dynamic(signal, mom_group)
    return zscore(f)


def s08_deferred_revenue(panel, price_pivot, sector_map, vol_pivot):
    """Unrecognized Deferred Revenue — WQ Brain 독점 데이터(fnd6_drc) 필요, 건너뜀"""
    return None


def s09_tax_adjusted_es(panel, price_pivot, sector_map, vol_pivot):
    """Tax-Adjusted Earnings Surprise — WQ Brain 독점 데이터(fn_*) 필요, 건너뜀"""
    return None


def s10_profitable_buyback(panel, price_pivot, sector_map, vol_pivot):
    """Profitable Buyback & Cash Flow Distortion"""
    oi_p  = to_pivot(panel, "operating_income")
    ta_p  = to_pivot(panel, "total_assets")
    gp_p  = to_pivot(panel, "gross_profit")
    eps_p = to_pivot(panel, "eps")

    if ta_p is None or eps_p is None:
        return None

    # operating_income 없는 기간은 gross_profit으로 대체
    oi_combined = coalesce_piv(oi_p, gp_p)
    if oi_combined is not None:
        oi_piv = oi_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
        prof2 = ts_backfill(oi_piv / ta_p.replace(0, np.nan), 252)
    else:
        prof2 = None

    # 자사주 신호: eps 성장 (sharesout 없으므로 eps 희석 효과로 대체)
    eps_signal = -(ts_delta(eps_p, 252) / eps_p.shift(252).abs().replace(0, np.nan))
    eps_signal = -eps_signal  # buyback → eps 증가 → 롱

    if prof2 is not None:
        signal = eps_signal * (1 + rank(zscore(prof2)))
    else:
        signal = eps_signal

    # 변동성 공포 필터 (implied_vol 없으므로 가격 변동성으로 대체)
    ret = price_pivot.pct_change()
    fear = ts_rank(ts_std(ret, 21), 252)
    f = trade_when(rank(fear) < 0.8, signal, pd.DataFrame(np.nan, index=signal.index, columns=signal.columns))
    f = f.ffill(limit=5)
    return zscore(f)


def s11_sga_driven(panel, price_pivot, sector_map, vol_pivot):
    """SG&A-Driven Marketing Efficiency Dynamic"""
    sga_p = to_pivot(panel, "sga_expense")
    oe_p  = to_pivot(panel, "operating_expense")
    sps_p = to_pivot(panel, "sps")

    if sps_p is None:
        return None

    # sga/oe 없는 기간은 매출 성장 모멘텀으로 대체
    rev_growth = ts_delta(sps_p, 252) / sps_p.shift(252).abs().replace(0, np.nan)
    if sga_p is not None and oe_p is not None:
        sga_piv = sga_p.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
        oe_piv  = oe_p.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
        signal_raw = sga_piv / oe_piv.replace(0, np.nan)
        ratio_signal = signal_raw / ts_delay(signal_raw, 252).replace(0, np.nan)
        signal = coalesce_piv(ratio_signal, rev_growth)
    else:
        signal = rev_growth

    cond = sps_p > ts_mean(sps_p, 252)
    a = zscore(signal)
    f = trade_when(cond, a, -a)
    return zscore(f)


def s12_goodwill_overvaluation(panel, price_pivot, sector_map, vol_pivot):
    """Goodwill Overvaluation & Financial Accrued Risk"""
    gw_p  = to_pivot(panel, "goodwill")
    sps_p = to_pivot(panel, "sps")
    eps_p = to_pivot(panel, "eps")

    ta_p = to_pivot(panel, "total_assets")
    if sps_p is None or ta_p is None:
        return None

    # goodwill 없는 기간은 total_assets로 대체 (고자산 저효율 = 리스크)
    gw_combined = coalesce_piv(gw_p, ta_p)
    gw_piv = gw_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    ratio  = gw_piv / sps_p.replace(0, np.nan)
    goodwill_sales = -ts_backfill(zscore(ratio), 63)

    if eps_p is not None:
        is_high_accrued = rank(eps_p) > 0.5
        f = goodwill_sales.where(~is_high_accrued, goodwill_sales * 2)
    else:
        f = goodwill_sales

    return zscore(f)


def s13_corporate_transparency(panel, price_pivot, sector_map, vol_pivot):
    """Corporate Transparency — WQ Brain 독점 데이터(fnd6_acdo, pv13_*) 필요, 건너뜀"""
    return None


def s14_volatility_regime_debt(panel, price_pivot, sector_map, vol_pivot):
    """Volatility-Regime Filtered Debt Decay Momentum"""
    debt_p = to_pivot(panel, "debt")
    ta_p   = to_pivot(panel, "total_assets")
    # debt 없는 기간은 total_assets로 대체
    debt_piv = coalesce_piv(debt_p, ta_p)
    if debt_piv is None:
        return None

    debt_piv = debt_piv.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    f_raw = signed_power(-ts_zscore(debt_piv, 63), 1.8)

    # 변동성 레짐: vwap*volume 대신 price*volume 사용
    ret = price_pivot.pct_change()
    if vol_pivot is not None:
        vol_aligned = vol_pivot.reindex(index=price_pivot.index, columns=price_pivot.columns).fillna(0)
        pv = price_pivot * (1 + ts_rank(vol_aligned, 21))
        regime_raw = ts_zscore(ts_std(pv, 21), 63)
    else:
        regime_raw = ts_zscore(ts_std(ret, 21), 63)

    nan_df = pd.DataFrame(np.nan, index=f_raw.index, columns=f_raw.columns)
    f = trade_when(regime_raw < -0.1, f_raw, nan_df)
    f = f.where(~(regime_raw > 0.8), other=np.nan)
    f = f.ffill(limit=5)
    return zscore(f)


def s15_book_to_cap_momentum(panel, price_pivot, sector_map, vol_pivot):
    """Book to Cap Momentum"""
    bps_p = to_pivot(panel, "bps")
    eq_p  = to_pivot(panel, "equity")

    # equity 없는 기간은 bps로 대체
    eq_combined = coalesce_piv(eq_p, bps_p)
    if eq_combined is None:
        return None
    eq_piv = eq_combined.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    bp = eq_piv / price_pivot.replace(0, np.nan)

    d1 = zscore(ts_delta(bp, 21))
    d3 = zscore(ts_delta(ts_delay(bp, 21), 42))
    f  = d1 + d3
    return zscore(f)


# ── 전략 목록 ──────────────────────────────────────────────────────────────────

STRATEGIES = {
    "S01_LowAccrual":          s01_low_accrual,
    "S02_OEY":                 s02_oey,
    "S03_CashCFDivergence":    s03_cash_cf_divergence,
    "S04_GrowthWeightedLT":    s04_growth_weighted_lt,
    "S05_IndustryNeutralCFY":  s05_industry_neutral_cf_yield,
    "S06_DebtSpikeReversal":   s06_debt_spike_reversal,
    "S07_AggressiveDualValue": s07_aggressive_dual_value,
    "S08_DeferredRevenue":     s08_deferred_revenue,
    "S09_TaxAdjustedES":       s09_tax_adjusted_es,
    "S10_ProfitableBuyback":   s10_profitable_buyback,
    "S11_SGADriven":           s11_sga_driven,
    "S12_GoodwillOverval":     s12_goodwill_overvaluation,
    "S13_CorpTransparency":    s13_corporate_transparency,
    "S14_VolRegimeDebt":       s14_volatility_regime_debt,
    "S15_BookToCapMom":        s15_book_to_cap_momentum,
}

STRATEGY_NAMES = {
    "S01_LowAccrual":          "Low Accrual Clean",
    "S02_OEY":                 "Operating Income Earnings Yield",
    "S03_CashCFDivergence":    "Cash & CF Divergence",
    "S04_GrowthWeightedLT":    "Growth-Weighted LT Investment",
    "S05_IndustryNeutralCFY":  "Industry-Neutral CF Yield",
    "S06_DebtSpikeReversal":   "Debt Spike Reversal",
    "S07_AggressiveDualValue": "Aggressive Dual Value Blend",
    "S08_DeferredRevenue":     "Unrecognized Deferred Revenue (건너뜀)",
    "S09_TaxAdjustedES":       "Tax-Adjusted Earnings Surprise (건너뜀)",
    "S10_ProfitableBuyback":   "Profitable Buyback & CF Distortion",
    "S11_SGADriven":           "SG&A-Driven Marketing Efficiency",
    "S12_GoodwillOverval":     "Goodwill Overvaluation Risk",
    "S13_CorpTransparency":    "Corporate Transparency (건너뜀)",
    "S14_VolRegimeDebt":       "Volatility-Regime Debt Decay",
    "S15_BookToCapMom":        "Book to Cap Momentum",
}


# ── 결과 저장 ──────────────────────────────────────────────────────────────────

def plot_single(result, name, label):
    ls  = result.ls_returns.dropna()
    if len(ls) == 0:
        return
    cum = (1 + ls).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(cum.index, cum.values, color="#1565C0", lw=1.5)
    ax1.fill_between(cum.index, cum.values, 1, alpha=0.1, color="#1565C0")
    ax1.axhline(1, color="gray", lw=0.5, linestyle="--")
    ax1.set_title(f"{name}  롱숏 PnL", fontsize=13, fontweight="bold")
    ax1.set_ylabel("누적 배수")
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    metrics_str = (
        f"CAGR {cagr(ls):.2%}  |  Sharpe {sharpe(ls):.2f}  |  "
        f"MDD {mdd(ls):.2%}  |  Turnover {result.avg_turnover:.2%}"
    )
    ax1.set_xlabel(metrics_str, fontsize=10)

    ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#D32F2F")
    ax2.set_ylabel("DD")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    fig.tight_layout()
    path = RESULTS_DIR / f"pnl_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


def plot_summary(summary_rows):
    df = pd.DataFrame(summary_rows).set_index("전략")
    df_num = df[["CAGR", "Sharpe", "MDD"]].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(df) * 0.5 + 2)))
    colors_cagr  = ["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["CAGR"]]
    colors_sharpe = ["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["Sharpe"]]
    colors_mdd   = ["#D32F2F"] * len(df_num)

    axes[0].barh(df_num.index, df_num["CAGR"] * 100, color=colors_cagr)
    axes[0].set_title("CAGR (%)", fontweight="bold")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(df_num.index, df_num["Sharpe"], color=colors_sharpe)
    axes[1].set_title("Sharpe Ratio", fontweight="bold")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].grid(alpha=0.3, axis="x")

    axes[2].barh(df_num.index, df_num["MDD"] * 100, color=colors_mdd)
    axes[2].set_title("MDD (%)", fontweight="bold")
    axes[2].grid(alpha=0.3, axis="x")

    fig.suptitle("US 커스텀 전략 15개 성과 요약", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = RESULTS_DIR / "summary_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


def plot_cumulative_all(all_results):
    """전략별 누적 수익률 한 장에"""
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(all_results))
    for i, (label, res) in enumerate(all_results.items()):
        ls = res.ls_returns.dropna()
        if len(ls) == 0:
            continue
        cum = (1 + ls).cumprod()
        ax.plot(cum.index, cum.values, lw=1.2, label=label, color=cmap(i))

    ax.axhline(1, color="gray", lw=0.5, linestyle="--")
    ax.set_title("US 커스텀 전략 누적 수익률 비교", fontsize=13, fontweight="bold")
    ax.set_ylabel("누적 배수")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()
    path = RESULTS_DIR / "cumulative_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    print("데이터 로드 중...")
    price_pivot, panel, universe, sector_map, vol_pivot = load_us()
    print(f"  종목 수: {price_pivot.shape[1]}, 기간: "
          f"{price_pivot.index[0].date()} ~ {price_pivot.index[-1].date()}")

    cfg = Config()
    all_results = {}
    summary_rows = []

    for label, fn in STRATEGIES.items():
        name = STRATEGY_NAMES[label]
        print(f"\n[{label}] {name}")

        factor = fn(panel, price_pivot, sector_map, vol_pivot)
        if factor is None:
            print(f"  → 건너뜀 (데이터 없음 또는 구현 불가)")
            continue

        valid = factor.notna().sum().sum()
        if valid == 0:
            print(f"  → 유효 팩터값 없음, 건너뜀")
            continue
        print(f"  팩터 유효값: {valid:,}개")

        bt = LongShortBacktester(
            price=price_pivot,
            factor=factor,
            universe=universe,
            config=cfg,
        )
        res = bt.run()
        if len(res.ls_returns) == 0:
            print(f"  → 수익률 없음")
            continue

        ls = res.ls_returns
        c = cagr(ls)
        s = sharpe(ls)
        m = mdd(ls)
        t = res.avg_turnover
        print(f"  CAGR {c:.2%}  Sharpe {s:.2f}  MDD {m:.2%}  Turnover {t:.2%}")

        all_results[label] = res
        summary_rows.append({
            "전략": label,
            "전략명": name,
            "CAGR": c,
            "Sharpe": s,
            "MDD": m,
            "Turnover": t,
        })

        plot_single(res, name, label)

    if not all_results:
        print("\n실행된 전략 없음.")
        return

    # 비교 차트
    print("\n비교 차트 생성 중...")
    plot_cumulative_all(all_results)
    plot_summary(summary_rows)

    # CSV 저장
    summary_df = pd.DataFrame(summary_rows)
    summary_df["CAGR"]     = summary_df["CAGR"].map(lambda x: f"{x:.2%}")
    summary_df["Sharpe"]   = summary_df["Sharpe"].map(lambda x: f"{x:.2f}")
    summary_df["MDD"]      = summary_df["MDD"].map(lambda x: f"{x:.2%}")
    summary_df["Turnover"] = summary_df["Turnover"].map(lambda x: f"{x:.2%}")
    csv_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n요약 CSV 저장: {csv_path}")

    print(f"\n[완료] 결과: {RESULTS_DIR}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
