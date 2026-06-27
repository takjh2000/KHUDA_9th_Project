"""
Phase 5  비교 분석 및 시각화
- KR vs US PnL 겹쳐서 플롯 (팩터별)
- 10개 조합 요약 테이블
- 연도별 수익률 히트맵 (팩터 × 연도)
- 개인투자자 비중 × 팩터 IC 상관 분석
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import FACTORS, RESULTS_DIR
from backtest.engine import BacktestResult
from backtest.metrics import cagr, sharpe, mdd, annual_returns, build_summary_table

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


# ── PnL 비교 플롯 ───────────────────────────────────────────────────────────

def plot_pnl_comparison(results: dict[str, BacktestResult],
                        factors: list[str] = FACTORS,
                        save: bool = True):
    """
    팩터별로 KR vs US 누적 수익률 겹쳐서 플롯
    5개 서브플롯 → results/pnl_comparison.png
    """
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 5),
                              sharey=False)
    if len(factors) == 1:
        axes = [axes]

    for ax, fname in zip(axes, factors):
        for market, color, ls in [("KR", "#D32F2F", "-"), ("US", "#1565C0", "--")]:
            key = f"{market}_{fname}"
            if key not in results:
                continue
            ret = results[key].ls_returns.dropna()
            cum = (1 + ret).cumprod()
            ax.plot(cum.index, cum.values,
                    label=market, color=color, linestyle=ls, lw=1.5)

        ax.set_title(fname, fontsize=12, fontweight="bold")
        ax.set_ylabel("누적 수익률")
        ax.axhline(1, color="gray", lw=0.6, linestyle=":")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle("팩터별 롱숏 PnL: KR vs US", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "pnl_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  저장: {path}")
    return fig


# ── 연도별 히트맵 ───────────────────────────────────────────────────────────

def plot_annual_heatmap(results: dict[str, BacktestResult],
                        market: str = "KR",
                        save: bool = True):
    """
    팩터 × 연도 수익률 히트맵
    """
    ann_rows = {}
    for fname in FACTORS:
        key = f"{market}_{fname}"
        if key not in results:
            continue
        ann = annual_returns(results[key].ls_returns)
        ann_rows[fname] = ann

    if not ann_rows:
        return None

    heat = pd.DataFrame(ann_rows).T * 100   # percent

    if heat.empty or heat.shape[1] == 0:
        return None

    valid_vals = heat.values[~np.isnan(heat.values)]
    if len(valid_vals) == 0:
        return None

    vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    fig, ax = plt.subplots(figsize=(max(8, heat.shape[1] * 1.1), 4))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(heat.values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=45, fontsize=9)
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(heat.index)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 15 else "black")

    plt.colorbar(im, ax=ax, label="%")
    ax.set_title(f"{market} 팩터별 연도별 수익률 히트맵", fontsize=12)
    fig.tight_layout()

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"heatmap_{market}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  저장: {path}")
    return fig


# ── 개별 PnL + DD 플롯 ──────────────────────────────────────────────────────

def plot_single(result: BacktestResult,
                title: str = "",
                save_path: str | None = None):
    """단일 전략 PnL + 낙폭 차트"""
    ls = result.ls_returns.dropna()
    cum = (1 + ls).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(cum.index, cum.values, color="#1565C0", lw=1.5)
    ax1.fill_between(cum.index, cum.values, 1, alpha=0.1, color="#1565C0")
    ax1.axhline(1, color="gray", lw=0.5, linestyle="--")
    ax1.set_title(title or "롱숏 PnL", fontsize=13)
    ax1.set_ylabel("누적 배수")
    ax1.grid(alpha=0.25)

    metrics_text = (
        f"CAGR {cagr(ls):.2%}  |  "
        f"Sharpe {sharpe(ls):.2f}  |  "
        f"MDD {mdd(ls):.2%}  |  "
        f"Turnover {result.avg_turnover:.2%}"
    )
    ax1.set_xlabel(metrics_text, fontsize=10)

    ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#D32F2F")
    ax2.set_ylabel("DD")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    if save_path:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  저장: {save_path}")
    return fig


# ── IC × 개인투자자 비중 상관 분석 ───────────────────────────────────────────

def analyze_ic_vs_individual(ic_series: pd.Series,
                              indiv_ratio: pd.Series,
                              factor_name: str = "",
                              save: bool = True):
    """
    팩터 IC와 개인투자자 순매수 비중의 상관 분석
    ic_series: 날짜별 IC (Spearman)
    indiv_ratio: 날짜별 개인투자자 비중 (시장 전체 기준)
    """
    aligned = ic_series.align(indiv_ratio, join="inner")
    ic_a, ind_a = aligned[0].dropna(), aligned[1].dropna()
    common = ic_a.index.intersection(ind_a.index)

    if len(common) < 10:
        print("  [IC 분석] 데이터 부족으로 건너뜀")
        return None

    corr = ic_a[common].corr(ind_a[common])
    print(f"  [{factor_name}] IC × 개인비중 상관: {corr:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ic_a.index, ic_a.values, color="#1565C0", lw=1)
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].set_title(f"{factor_name} IC 시계열")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(ind_a[common], ic_a[common], alpha=0.4, s=10)
    axes[1].set_xlabel("개인투자자 비중")
    axes[1].set_ylabel("IC")
    axes[1].set_title(f"IC vs 개인비중  (r={corr:.3f})")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"ic_vs_indiv_{factor_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  저장: {path}")
    return fig


# ── 전체 결과 출력 ────────────────────────────────────────────────────────────

def print_full_report(results: dict[str, BacktestResult]):
    """10개 조합 요약 테이블 출력"""
    print("\n" + "=" * 60)
    print("  팩터 × 시장 성과 요약")
    print("=" * 60)
    tbl = build_summary_table(results)
    print(tbl.to_string())
    return tbl
