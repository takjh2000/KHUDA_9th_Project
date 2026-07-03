"""
WQ Brain 전략 전체 구현 (11개)
불가: 4(fnd6_newqv1300), 8(fnd6_drc), 9(fn_profit_loss_q), 13(fnd6_acdo)
구현: 1,2,3,5,6,7,10,11,12,14,15

실행: python run_wq_all.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from config import PROCESSED_DIR, RESULTS_DIR
from backtest.engine import LongShortBacktester, Config
from backtest import metrics as _m
from factors.wq_ops import (
    rank, zscore, ts_mean, ts_std, ts_rank, ts_delta, ts_delay,
    ts_zscore, ts_corr, ts_backfill, bucket,
    group_neutralize, group_rank, group_zscore,
    group_neutralize_dynamic, df_max, signed_power, trade_when
)

# ── 폰트 ─────────────────────────────────────────────────────────────────────
for f in fm.findSystemFonts():
    if "malgun" in f.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FF9800",
          "#00BCD4","#E91E63","#607D8B","#8BC34A","#795548","#03A9F4"]

# ── 성과 지표 ─────────────────────────────────────────────────────────────────
def calc_metrics(ls):
    return {"cagr": _m.cagr(ls), "sharpe": _m.sharpe(ls), "mdd": _m.mdd(ls)}

def yearly_returns(ls):
    return ls.groupby(ls.index.year).apply(lambda r: (1 + r).prod() - 1)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_all():
    panel = pd.read_parquet(PROCESSED_DIR / "kr_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    def piv(col):
        return panel.pivot_table(index="date", columns="ticker", values=col) if col in panel.columns else None
    close  = piv("close")
    bps    = piv("bps")
    ops    = piv("ops")
    eps    = piv("eps")
    sps    = piv("sps")
    roe    = piv("roe")
    gp     = piv("gross_profit")    # 절대값 (원)
    ta     = piv("total_assets")    # 절대값 (원)
    shares = piv("shares")
    ret    = close.pct_change()

    # 파생 지표
    te  = bps * shares          # total equity (원)
    ni  = eps * shares          # net income (원)
    rev = sps * shares          # revenue (원)
    oi  = ops * shares          # operating income (원)
    debt = (ta - te).clip(lower=0)   # total liabilities proxy (원)
    # per-share debt
    debt_ps = debt / shares.replace(0, np.nan)

    return dict(close=close, bps=bps, ops=ops, eps=eps, sps=sps, roe=roe,
                gp=gp, ta=ta, shares=shares, ret=ret,
                te=te, ni=ni, rev=rev, oi=oi, debt=debt, debt_ps=debt_ps)

def load_industry():
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")
        sc = next((c for c in df.columns if c.lower() in ("sector","industry","섹터","업종")), None)
        cc = next((c for c in df.columns if c.lower() in ("code","symbol","ticker")), "Code")
        if sc:
            s = df.set_index(cc)[sc].dropna()
            return s[s != ""]
    except Exception:
        pass
    return pd.Series(dtype=str)

def load_kospi():
    try:
        import FinanceDataReader as fdr
        return fdr.DataReader("KS11","2015-01-01","2024-12-31")["Close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)

# ── 전략 구현 ─────────────────────────────────────────────────────────────────

# 전략 1 — Low Accrual Clean
# accrual = -(income - cashflow_op) / assets
# 근사: -(eps - ops) / bps  (주당 현금 외 이익 / 주당 장부가)
def strat_1(d, industry):
    accrual = -(d["eps"] - d["ops"]) / d["bps"].replace(0, np.nan)
    accrual_r = rank(accrual)
    cap_bucket = bucket(rank(d["close"]), 5)
    cap_neut   = group_neutralize_dynamic(accrual_r, cap_bucket)
    mom        = ts_rank(d["ret"], 252)
    return (cap_neut - mom).dropna(how="all")

# 전략 2 — Operating Income EY
def strat_2(d, industry):
    oey = d["ops"] / d["close"].replace(0, np.nan)
    oey_ts = ts_rank(oey, 126)
    if industry.empty:
        return rank(oey_ts).dropna(how="all")
    common = industry.index.intersection(oey_ts.columns)
    return group_rank(oey_ts[common], industry[common]).dropna(how="all")

# 전략 3 — Cash & Cash Flow Divergence
# ts_corr(ts_mean(cash, 5), ts_mean(cashflow, 5), 252)
# 근사: 현금(bps 변화) vs 현금흐름(eps) 1년 rolling 상관 — 낮을수록 매수
def strat_3(d, industry):
    cash_proxy = ts_mean(d["bps"], 5)          # BPS 변화 ~ 자산 내 현금 proxy
    cf_proxy   = ts_mean(d["eps"], 5)          # EPS ~ 영업현금흐름 proxy
    corr = ts_corr(cash_proxy, cf_proxy, 252)  # 상관이 낮을수록 정보 비효율
    return (-corr).dropna(how="all")           # 낮은 상관 = 매수

# 전략 5 — Industry-Neutral Cash Flow Yield
# f = group_rank(ts_zscore(cashflow_op / cap, 63), industry)
# 근사: cashflow_op ≈ ops (영업이익을 현금흐름 proxy로 사용)
def strat_5(d, industry):
    cf_yield = d["ops"] / d["close"].replace(0, np.nan)
    cf_zs    = ts_zscore(cf_yield, 63)
    if industry.empty:
        return rank(cf_zs).dropna(how="all")
    common = industry.index.intersection(cf_zs.columns)
    return group_rank(cf_zs[common], industry[common]).dropna(how="all")

# 전략 6 — Debt Spike Reversal Momentum
# x = -ts_zscore(debt, 252); alpha = signed_power(x, 4)
# condition = debt > ts_mean(debt, 63) → trade_when(condition, alpha, -1)
# 근사: debt_ps = (total_assets - bps*shares) / shares
def strat_6(d, industry):
    dp  = d["debt_ps"].replace(0, np.nan)
    dp  = ts_backfill(dp, 252)
    x   = -ts_zscore(dp, 252)
    alpha     = signed_power(x, 4)
    condition = dp > ts_mean(dp, 63)
    return trade_when(condition, alpha, -1).dropna(how="all")

# 전략 7 — Aggressive Dual Value Blend
def strat_7(d, industry):
    ec  = d["bps"]  / d["close"].replace(0, np.nan)
    opc = d["ops"]  / d["close"].replace(0, np.nan)
    common = ec.columns.intersection(opc.columns)
    ec, opc = ec[common], opc[common]
    ind_sub = industry.reindex(common).fillna("Unknown") if not industry.empty else pd.Series("All", index=common)
    signal  = df_max(rank(group_neutralize(opc, ind_sub)),
                     group_rank(ts_rank(ec, 63), ind_sub))
    mom_bkt = bucket(rank(ts_mean(d["ret"][common], 240)), 10)
    return group_neutralize_dynamic(signal, mom_bkt).dropna(how="all")

# 전략 10 — Profitable Buyback (fear filter 제외)
# prof2 = operating_income / assets
# f = -sharesout / ts_delay(sharesout, 252)  (주식수 감소 = 자사주 매입 = 롱)
# signal = f * (1 + rank(group_zscore(prof2, industry)))
def strat_10(d, industry):
    shares_bf = ts_backfill(d["shares"], 252)
    f = -(shares_bf / ts_delay(shares_bf, 252).replace(0, np.nan))
    prof2 = ts_backfill(d["oi"] / d["ta"].replace(0, np.nan), 252)
    if not industry.empty:
        common = industry.index.intersection(f.columns)
        f      = f[common]
        prof2  = prof2[common]
        ind_sub = industry.reindex(common).fillna("Unknown")
        gz      = group_zscore(prof2, ind_sub)
    else:
        gz = zscore(prof2)
    signal = f * (1 + rank(gz).fillna(0))
    return signal.dropna(how="all")

# 전략 11 — SG&A-Driven Marketing Efficiency
# signal = sga_expense / operating_expense; f = signal / ts_delay(signal, 252)
# trade_when(revenue > ts_mean(revenue, 252), zscore(f), not_cond)
# 근사: sga_per_share ≈ sps - gross_profit/shares - ops (per-share SGA)
#        total_opex   ≈ sps - ops
def strat_11(d, industry):
    gp_ps = d["gp"] / d["shares"].replace(0, np.nan)
    sga   = (d["sps"] - gp_ps - d["ops"]).clip(lower=0)    # SGA per share proxy
    opex  = (d["sps"] - d["ops"]).replace(0, np.nan)        # total opex per share
    signal = sga / opex
    signal = signal.replace([np.inf, -np.inf], np.nan)
    f   = signal / ts_delay(signal, 252).replace(0, np.nan)
    f   = ts_backfill(f, 5)
    cond = d["sps"] > ts_mean(d["sps"], 252)
    a    = f.clip(-10, 10)    # hump 근사
    out  = trade_when(cond, zscore(a), -zscore(a))
    return out.dropna(how="all")

# 전략 12 — Goodwill Overvaluation
# goodwill_sales_ratio = -zscore(goodwill / sales)
# 근사: intangible premium = total_assets/shares / bps - 1 (시장 장부가 대비 초과자산 비율)
#        → 높을수록 무형자산 과대 계상 가능성 → 숏
# accrued_liab 없으므로 단순 goodwill proxy만 사용
def strat_12(d, industry):
    ta_ps         = d["ta"] / d["shares"].replace(0, np.nan)
    intangible_pr = (ta_ps / d["bps"].replace(0, np.nan) - 1).clip(lower=0)
    ratio         = ts_backfill(intangible_pr / d["sps"].replace(0, np.nan), 63)
    signal        = -ts_zscore(ratio, 63)   # 높은 무형자산 비율 = 숏
    return signal.dropna(how="all")

# 전략 14 — Volatility-Regime Filtered Debt Decay Momentum
# f = signed_power(-ts_zscore(debt, 63), 1.8)
# regime_raw = ts_zscore(ts_std_dev(vwap*(1+ts_rank(volume,21)), 21), 63)
# trade_when(regime < -0.1, f, regime > 0.8)
# 근사: vwap → close, volume 없으므로 price return volatility로 regime 대체
def strat_14(d, industry):
    dp  = d["debt_ps"].replace(0, np.nan)
    dp  = ts_backfill(dp, 252)
    f   = signed_power(-ts_zscore(dp, 63), 1.8)
    # regime: price vol z-score
    ret_std    = ts_std(d["ret"], 21)
    regime_raw = ts_zscore(ret_std, 63)
    low_vol  = regime_raw < -0.1
    high_vol = regime_raw >  0.8
    out = f.copy()
    out[high_vol] = np.nan    # 고변동성 구간 포지션 청산
    out[~low_vol & ~high_vol] = np.nan  # 저변동성 구간에서만 신호 사용
    return out.dropna(how="all")

# 전략 15 — Book to Cap Momentum
def strat_15(d, industry):
    ec = d["bps"] / d["close"].replace(0, np.nan)
    d1 = zscore(ts_delta(ec, 21))
    d3 = zscore(ts_delta(ts_delay(ec, 21), 42))
    return (d1 + d3).dropna(how="all")

# ── 시장 필터 ─────────────────────────────────────────────────────────────────
def apply_market_filter(factor, kospi, lookback=252):
    if kospi.empty:
        return factor
    kc  = (1 + kospi).cumprod()
    mom = kc / kc.shift(lookback) - 1
    bull = (mom > 0).reindex(factor.index, method="ffill")
    result = factor.copy()
    result[~bull.values.reshape(-1, 1) * np.ones((1, factor.shape[1]), dtype=bool)] = np.nan
    return result

# ── 백테스트 ─────────────────────────────────────────────────────────────────
def run_bt(name, factor, close):
    valid = factor.notna().sum().sum()
    if valid == 0:
        print(f"  [{name}] 신호 없음")
        return None
    bt  = LongShortBacktester(price=close, factor=factor, config=Config())
    res = bt.run()
    if res.ls_returns.empty:
        print(f"  [{name}] 수익률 없음")
        return None
    m = calc_metrics(res.ls_returns)
    print(f"  [{name}]  CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  MDD {m['mdd']:.2%}")
    return res, m

# ── PDF 생성 ──────────────────────────────────────────────────────────────────
def make_strategy_page(pdf, name, res, m, note=""):
    ls  = res.ls_returns
    cum = (1 + ls).cumprod()
    dd  = cum / cum.cummax() - 1
    yr  = yearly_returns(ls)

    fig = plt.figure(figsize=(11.69, 6))
    fig.suptitle(name, fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(cum.index, cum.values, color="#2196F3", linewidth=1.5)
    ax1.fill_between(cum.index, cum.values, 1,
                     where=(cum.values >= 1), alpha=0.15, color="#4CAF50")
    ax1.fill_between(cum.index, cum.values, 1,
                     where=(cum.values <  1), alpha=0.15, color="#F44336")
    ax1.axhline(1, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("누적 수익률")
    ax1.grid(alpha=0.3)

    ax_t = fig.add_subplot(gs[0, 2])
    ax_t.axis("off")
    ax_t.text(0.05, 0.95,
              f"CAGR:    {m['cagr']:+.2%}\n"
              f"Sharpe:  {m['sharpe']:.2f}\n"
              f"MDD:     {m['mdd']:.2%}\n"
              f"Turnover:{res.avg_turnover:.2%}\n\n{note}",
              transform=ax_t.transAxes, fontsize=9, va="top",
              bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.5)
    ax2.set_title("Drawdown")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 2])
    cols = ["#4CAF50" if v >= 0 else "#F44336" for v in yr.values]
    ax3.bar(yr.index.astype(str), yr.values * 100, color=cols, alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("연도별 수익률")
    ax3.set_ylabel("%")
    ax3.tick_params(axis="x", rotation=45, labelsize=7)
    ax3.grid(alpha=0.3, axis="y")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_metrics_explanation_page(pdf):
    """성과 지표 설명 페이지"""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("성과 지표 설명 — CAGR · Sharpe · MDD · Turnover · Drawdown",
                 fontsize=14, fontweight="bold", y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.04)

    HEADER_COLOR = "#1a237e"
    BOX_COLOR    = "#E8EAF6"

    # ── ① CAGR ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_facecolor(BOX_COLOR)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax1.transAxes,
                                facecolor=BOX_COLOR, zorder=0))
    ax1.text(0.5, 0.96, "CAGR  (연평균 복리 수익률)",
             ha="center", va="top", fontsize=11, fontweight="bold",
             color=HEADER_COLOR, transform=ax1.transAxes)
    ax1.text(0.5, 0.86, "Compound Annual Growth Rate",
             ha="center", va="top", fontsize=8, color="gray",
             transform=ax1.transAxes)
    body1 = (
        "백테스트 전 기간을 복리로 환산했을 때\n"
        "매년 평균 몇 % 수익을 냈는지 나타내는 지표\n\n"
        "계산:  (최종자산 / 초기자산)^(1/연수) − 1\n\n"
        "예시\n"
        "  +11.27% (전략7) → 1억 투자 시 9년 후 약 2.6억\n"
        "  −4.27%  (전략10) → 매년 4%씩 복리로 손실\n\n"
        "단순 총수익률과 달리 기간 차이를\n"
        "보정하므로 전략 간 비교에 적합"
    )
    ax1.text(0.05, 0.73, body1, ha="left", va="top", fontsize=8.5,
             linespacing=1.6, transform=ax1.transAxes)

    # ── ② Sharpe Ratio ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                                facecolor="#E8F5E9", zorder=0))
    ax2.text(0.5, 0.96, "Sharpe Ratio  (샤프 비율)",
             ha="center", va="top", fontsize=11, fontweight="bold",
             color="#1B5E20", transform=ax2.transAxes)
    ax2.text(0.5, 0.86, "위험 1단위당 수익률",
             ha="center", va="top", fontsize=8, color="gray",
             transform=ax2.transAxes)
    body2 = (
        "계산:  (전략 수익률 − 무위험이자율) / 수익률 표준편차\n\n"
        "해석 기준\n"
    )
    ax2.text(0.05, 0.73, body2, ha="left", va="top", fontsize=8.5,
             linespacing=1.6, transform=ax2.transAxes)
    # 해석 테이블
    table_data = [
        ["범위",      "해석"],
        ["< 0",       "무위험 자산보다 못함"],
        ["0 ~ 0.5",   "보통"],
        ["0.5 ~ 1.0", "양호"],
        ["> 1.0",     "우수"],
    ]
    tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                    bbox=[0.04, 0.28, 0.92, 0.34], cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j in range(2):
        tbl[0, j].set_facecolor("#1B5E20")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    row_colors = ["#FFCDD2", "#FFF9C4", "#C8E6C9", "#A5D6A7"]
    for i in range(1, 5):
        for j in range(2):
            tbl[i, j].set_facecolor(row_colors[i - 1])
    ax2.text(0.05, 0.22,
             "  CAGR이 높아도 Sharpe가 낮으면 변동성이 너무\n"
             "  커서 투자자가 실제로 버티기 어렵다는 의미",
             ha="left", va="top", fontsize=8, linespacing=1.5,
             color="#1B5E20", transform=ax2.transAxes)

    # ── ③ MDD ───────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes,
                                facecolor="#FFF3E0", zorder=0))
    ax3.text(0.5, 0.96, "MDD  (최대 낙폭)",
             ha="center", va="top", fontsize=11, fontweight="bold",
             color="#BF360C", transform=ax3.transAxes)
    ax3.text(0.5, 0.86, "Maximum Drawdown",
             ha="center", va="top", fontsize=8, color="gray",
             transform=ax3.transAxes)
    body3 = (
        "계산:  (저점 − 직전 고점) / 직전 고점 × 100\n\n"
        "고점 이후 가장 많이 하락한 최대 손실 폭\n\n"
        "예시\n"
        "  −50.65% (전략10) → 어느 시점 진입 시\n"
        "              절반 이상 손실 가능\n"
        "  −27.07% (전략6)  → 상대적으로 방어적\n\n"
        "Drawdown 차트 = 매 시점의 고점 대비\n"
        "현재 손실률 추이 (0에 가까울수록 고점 근처)\n\n"
        "MDD가 크면 실전에서 심리적으로 버티기\n"
        "어려워 중도 청산 위험이 높아짐"
    )
    ax3.text(0.05, 0.73, body3, ha="left", va="top", fontsize=8.5,
             linespacing=1.6, transform=ax3.transAxes)

    # ── ④ Turnover ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax4.transAxes,
                                facecolor="#F3E5F5", zorder=0))
    ax4.text(0.5, 0.96, "Turnover  (회전율)",
             ha="center", va="top", fontsize=11, fontweight="bold",
             color="#4A148C", transform=ax4.transAxes)
    ax4.text(0.5, 0.86, "포트폴리오 일평균 종목 교체 비율",
             ha="center", va="top", fontsize=8, color="gray",
             transform=ax4.transAxes)
    body4 = (
        "계산:  하루에 포트폴리오의 몇 %를 교체하는지\n\n"
        "예시\n"
        "  13.75% (전략10) → 거의 안 바꿈\n"
        "              거래비용 낮음, 신호 반응 느림\n"
        "  82.15% (전략5)  → 자주 교체\n"
        "              슬리피지·수수료 영향 큼\n\n"
        "주의\n"
        "  백테스트 수치는 거래비용 미반영\n"
        "  Turnover가 높을수록 실전 수익률은\n"
        "  백테스트 대비 더 낮아질 수 있음\n\n"
        "  ※ 3대 지표 요약\n"
        "  좋은 전략 = CAGR↑ + Sharpe↑ + MDD↓"
    )
    ax4.text(0.05, 0.73, body4, ha="left", va="top", fontsize=8.5,
             linespacing=1.6, transform=ax4.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_summary_page(pdf, all_results):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 6))
    fig.suptitle("WQ Brain 전략 11개 전체 성과 비교", fontsize=14, fontweight="bold")

    ax = axes[0]
    for i, (name, (res, m)) in enumerate(all_results.items()):
        ls = (1 + res.ls_returns).cumprod()
        ax.plot(ls.index, ls.values, label=f"{name} ({m['cagr']:+.1%})",
                color=COLORS[i % len(COLORS)], linewidth=1.2, alpha=0.85)
    ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("전략별 누적 수익률")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for name, (res, m) in all_results.items():
        yr = yearly_returns(res.ls_returns)
        rows.append([name,
                     f"{m['cagr']:+.2%}",
                     f"{m['sharpe']:.2f}",
                     f"{m['mdd']:.2%}",
                     f"{yr.min():+.1%}({int(yr.idxmin())})",
                     f"{yr.max():+.1%}({int(yr.idxmax())})"])
    cols = ["전략", "CAGR", "Sharpe", "MDD", "최악연도", "최고연도"]
    tbl = ax2.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 2.0)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1a237e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#E8EAF6" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)
    ax2.set_title("전략별 성과 요약", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_yearly_heatmap(pdf, all_results):
    yearly = {n: yearly_returns(r.ls_returns) for n, (r, _) in all_results.items()}
    df_yr  = pd.DataFrame(yearly)

    fig, ax = plt.subplots(figsize=(11.69, max(5, len(df_yr.columns) * 0.7 + 2)))
    fig.suptitle("전략 × 연도별 수익률 히트맵", fontsize=13, fontweight="bold")
    data = df_yr.T.values * 100
    im   = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-40, vmax=60)
    ax.set_xticks(range(len(df_yr.index)))
    ax.set_xticklabels(df_yr.index.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(len(df_yr.columns)))
    ax.set_yticklabels([n.replace("_", " ") for n in df_yr.columns], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.0f}%", ha="center", va="center",
                        fontsize=7.5,
                        color="black" if abs(data[i, j]) < 30 else "white")
    plt.colorbar(im, ax=ax, label="수익률 (%)")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_title_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#0D47A1")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0D47A1")
    ax.axis("off")
    ax.text(0.5, 0.82, "WorldQuant Brain 전략", fontsize=28,
            color="white", ha="center", fontweight="bold")
    ax.text(0.5, 0.70, "한국 시장 적용 분석 및 저하 구간 원인 연구",
            fontsize=17, color="#90CAF9", ha="center")
    ax.text(0.5, 0.60, "KHUDA 금융트랙 | 백테스트 기간: 2016 – 2024",
            fontsize=13, color="#BBDEFB", ha="center")
    strat_list = (
        "[ 구현 완료 11개 전략 ]\n"
        "전략 1: Low Accrual Clean (근사)          전략 2: Operating Income EY\n"
        "전략 3: Cash & CF Divergence (근사)       전략 5: Industry-Neutral CF Yield (근사)\n"
        "전략 6: Debt Spike Reversal                전략 7: Aggressive Dual Value Blend\n"
        "전략 10: Profitable Buyback (공포필터 제외) 전략 11: SGA Efficiency (근사)\n"
        "전략 12: Goodwill Overvaluation (근사)    전략 14: Debt Decay + Vol Regime (근사)\n"
        "전략 15: Book to Cap Momentum\n\n"
        "[ 리포트 구성 ]\n"
        "① 전략 성과 요약  ② 한국 시장 적용 분석 (vs KOSPI)\n"
        "③ 저하 구간 원인 분석  ④ 시장 필터 적용 전략 개선\n"
        "⑤ 전략별 상세 (11개)  ⑥ 결론 및 시사점\n\n"
        "※ 불가 전략(4개): 전략 4, 8, 9, 13 — WQ 독점 데이터 사용"
    )
    ax.text(0.5, 0.30, strat_list, fontsize=9.5, color="#E3F2FD",
            ha="center", va="center", linespacing=1.8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_market_analysis(pdf, core_results, kospi):
    """KOSPI vs 전략 수익률 비교 — 한국 시장 적용 분석"""
    if kospi.empty:
        return

    kospi_yr = yearly_returns(kospi)

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("한국 시장 적용 분석 — KOSPI 대비 성과", fontsize=14, fontweight="bold")

    # ① KOSPI 연도별 수익률 bar
    ax = axes[0, 0]
    bc = ["#4CAF50" if v >= 0 else "#F44336" for v in kospi_yr.values]
    ax.bar(kospi_yr.index.astype(str), kospi_yr.values * 100, color=bc, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("KOSPI 연도별 수익률")
    ax.set_ylabel("수익률 (%)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    for i, (yr, v) in enumerate(kospi_yr.items()):
        ax.text(i, v * 100 + (1.5 if v >= 0 else -3), f"{v:.1%}",
                ha="center", fontsize=7)

    # ② 전략 vs KOSPI 누적 수익률
    ax2 = axes[0, 1]
    kc  = (1 + kospi).cumprod()
    ax2.plot(kc.index, kc.values / kc.iloc[0], color="gray", linewidth=2,
             label="KOSPI", linestyle="--")
    for i, (name, (res, m)) in enumerate(core_results.items()):
        ls = (1 + res.ls_returns).cumprod()
        ax2.plot(ls.index, ls.values, label=f"{name} ({m['cagr']:+.1%})",
                 color=COLORS[i % len(COLORS)], linewidth=1.5)
    ax2.axhline(1, color="black", linewidth=0.4, linestyle=":")
    ax2.set_title("전략 vs KOSPI 누적 수익률")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(alpha=0.3)

    # ③ 전략 vs KOSPI 연도별 산점도
    ax3 = axes[1, 0]
    for i, (name, (res, _)) in enumerate(core_results.items()):
        yr  = yearly_returns(res.ls_returns)
        common = yr.index.intersection(kospi_yr.index)
        ax3.scatter(kospi_yr[common] * 100, yr[common] * 100,
                    label=name, color=COLORS[i % len(COLORS)], alpha=0.8, s=60)
        # 연도 레이블
        for y in common:
            ax3.annotate(str(y), (kospi_yr[y] * 100, yr[y] * 100),
                         fontsize=6, alpha=0.6)
    ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax3.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax3.set_xlabel("KOSPI 연간 수익률 (%)")
    ax3.set_ylabel("전략 연간 수익률 (%)")
    ax3.set_title("전략 수익률 vs KOSPI (연도별 산점도)")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # ④ 연도별 비교 bar (KOSPI vs 주요 전략)
    ax4 = axes[1, 1]
    yrs  = sorted(kospi_yr.index.tolist())
    x    = np.arange(len(yrs))
    n_s  = len(core_results)
    w    = 0.7 / (n_s + 1)
    ax4.bar(x - w * n_s / 2, [kospi_yr.get(y, 0) * 100 for y in yrs],
            w, label="KOSPI", color="gray", alpha=0.6)
    for i, (name, (res, _)) in enumerate(core_results.items()):
        yr  = yearly_returns(res.ls_returns)
        vals = [yr.get(y, np.nan) * 100 for y in yrs]
        ax4.bar(x - w * n_s / 2 + w * (i + 1), vals,
                w, label=name, color=COLORS[i % len(COLORS)], alpha=0.85)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(y) for y in yrs], rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("수익률 (%)")
    ax4.set_title("KOSPI vs 주요 전략 연도별 비교")
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_underperform_analysis(pdf, core_results, kospi):
    """저하 구간 원인 심층 분석"""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("저하 구간 원인 심층 분석", fontsize=14, fontweight="bold")

    kospi_yr = yearly_returns(kospi) if not kospi.empty else pd.Series(dtype=float)

    # ① 저하 구간 연도별 성과 비교 (핵심 전략 4개)
    ax = axes[0, 0]
    bad_years = [2017, 2018, 2020, 2021]
    name_list = list(core_results.keys())[:4]
    x = np.arange(len(bad_years))
    w = 0.8 / (len(name_list) + 1)
    for i, name in enumerate(name_list):
        yr   = yearly_returns(core_results[name][0].ls_returns)
        vals = [yr.get(y, np.nan) * 100 for y in bad_years]
        ax.bar(x + i * w - w * len(name_list) / 2, vals, w,
               label=name, color=COLORS[i], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in bad_years])
    ax.set_ylabel("수익률 (%)")
    ax.set_title("저하 구간 연도별 전략 수익률")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # ② KOSPI vs 저하 전략 (저하 구간 집중)
    ax2 = axes[0, 1]
    if not kospi.empty:
        kc = (1 + kospi).cumprod()
        ax2.plot(kc.index, kc.values / kc.iloc[0], color="gray",
                 linewidth=2, label="KOSPI", linestyle="--")
    for i, name in enumerate(name_list):
        ls = (1 + core_results[name][0].ls_returns).cumprod()
        ax2.plot(ls.index, ls.values, color=COLORS[i], linewidth=1.3,
                 label=name, alpha=0.85)
    # 저하 구간 음영
    for yr_s, yr_e in [(2017, 2017), (2018, 2018), (2020, 2021)]:
        ax2.axvspan(pd.Timestamp(f"{yr_s}-01-01"),
                    pd.Timestamp(f"{yr_e}-12-31"),
                    alpha=0.12, color="red")
    ax2.set_title("저하 구간(음영) 기간 수익률 추이")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.axhline(1, color="black", linewidth=0.4, linestyle=":")

    # ③ 저하 원인 분석표
    ax3 = axes[1, 0]
    ax3.axis("off")
    cause_data = [
        ["연도", "KOSPI", "주요 저하 전략", "원인"],
        ["2017", "+22%", "전략1(-16%), 전략2(-14%)", "대형 성장주 랠리\n가치 팩터 전반 약화"],
        ["2018", "-17%", "전략7(-16%)", "미중 무역분쟁·금리인상\n소형가치주 급락"],
        ["2020", "+31%", "전략2(-23%)", "코로나 유동성장세\n성장주 급등·영업이익 무력화"],
        ["2021", "+4%",  "전략2(-13%)", "메타버스·바이오 테마장\n펀더멘털 팩터 작동 안함"],
    ]
    tbl = ax3.table(cellText=cause_data[1:], colLabels=cause_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 3.0)
    for j in range(4):
        tbl[0, j].set_facecolor("#B71C1C")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    colors_row = ["#FFCDD2", "#EF9A9A", "#FFCDD2", "#EF9A9A"]
    for i in range(1, 5):
        for j in range(4):
            tbl[i, j].set_facecolor(colors_row[i - 1])
    ax3.set_title("저하 구간 원인 분석", fontsize=11, fontweight="bold", pad=8)

    # ④ 구조적 원인 — 팩터 환경 분석
    ax4 = axes[1, 1]
    ax4.axis("off")
    text = (
        "[ 구조적 저하 원인 ]\n\n"
        "■ Value 팩터 공통 약화 (2017, 2020~2021)\n"
        "  - 저PBR·저PER 기업이 언더퍼폼하는\n"
        "    'Value Trap' 국면 반복\n"
        "  - 성장주 프리미엄 급등 시 발생\n\n"
        "■ 모멘텀 역전 (2018)\n"
        "  - 전년도(2017) 상승 종목 급반전\n"
        "  - 소형주 중심 패닉셀 → 가치주 직격\n\n"
        "■ 유동성 장세 vs 펀더멘털 (2020)\n"
        "  - 영업이익 무관 테마·성장주 급등\n"
        "  - 수익성 기반 팩터 신호 완전 역전\n\n"
        "■ 공통 시사점\n"
        "  → 시장 모멘텀 필터 추가 필요\n"
        "  → 팩터 분산 (Value + Quality + Momentum)"
    )
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
             fontsize=9.5, va="top", linespacing=1.7,
             bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.9))
    ax4.set_title("구조적 저하 원인 요약", fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_filter_page(pdf, core_results, filt_results, core_names):
    """시장 필터 적용 전후 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.5))
    fig.suptitle("전략 개선 — KOSPI 12개월 모멘텀 필터 적용",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for i, name in enumerate(core_names):
        if name in core_results:
            ls = (1 + core_results[name][0].ls_returns).cumprod()
            ax.plot(ls.index, ls.values, color=COLORS[i], linewidth=1.2,
                    linestyle="--", label=f"{name} (원본)", alpha=0.6)
        fk = name + "_f"
        if fk in filt_results:
            ls = (1 + filt_results[fk][0].ls_returns).cumprod()
            ax.plot(ls.index, ls.values, color=COLORS[i], linewidth=2,
                    label=f"{name} (필터)")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title("필터 전후 누적 수익률")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for name in core_names:
        fk   = name + "_f"
        o_m  = core_results[name][1] if name in core_results else {}
        f_m  = filt_results[fk][1]   if fk in filt_results  else {}
        d_cagr = (f_m.get("cagr", 0) - o_m.get("cagr", 0))
        d_mdd  = (f_m.get("mdd",  0) - o_m.get("mdd",  0))
        rows.append([
            name,
            f"{o_m.get('cagr',0):+.2%}", f"{f_m.get('cagr',0):+.2%}",
            f"{d_cagr:+.2%}",
            f"{o_m.get('sharpe',0):.2f}", f"{f_m.get('sharpe',0):.2f}",
            f"{o_m.get('mdd',0):.2%}",   f"{f_m.get('mdd',0):.2%}",
            f"{d_mdd:+.2%}",
        ])
    cols = ["전략", "CAGR\n원본", "CAGR\n필터", "CAGR\n변화",
            "Sharpe\n원본", "Sharpe\n필터", "MDD\n원본", "MDD\n필터", "MDD\n변화"]
    tbl = ax2.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 2.8)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1a237e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            if j in (2, 5, 7):
                tbl[i, j].set_facecolor("#E8F5E9")
            elif j in (3, 8):
                val_str = rows[i - 1][j]
                try:
                    val = float(val_str.replace("%", "").replace("+", ""))
                    tbl[i, j].set_facecolor("#C8E6C9" if val > 0 else "#FFCDD2")
                except Exception:
                    pass
            else:
                tbl[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")
    ax2.set_title("필터 적용 전후 성과 비교", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_conclusion(pdf, all_results, filt_results):
    fig, ax = plt.subplots(figsize=(11.69, 7.5))
    ax.axis("off")
    fig.suptitle("결론 및 시사점", fontsize=16, fontweight="bold", y=0.97)

    best = sorted(all_results.items(), key=lambda x: x[1][1]["sharpe"], reverse=True)

    text = f"""
[ 1. 한국 시장 적용 결과 ]

  • 11개 전략 중 유효 알파 확인: 4개 (전략 2·5·7·15)
    - 전략 7 (Dual Value Blend): CAGR +11.27%, Sharpe 0.68  ← 최우수
    - 전략 5 (CF Yield 근사):   CAGR +8.45%,  Sharpe 0.53
    - 전략 2 (OpIncome EY):     CAGR +6.15%,  Sharpe 0.41
    - 전략 15 (BookCap Mom):    CAGR +5.34%,  Sharpe 0.36

  • 신호 약함 (근사 한계): 전략 1·3·6·10·11·12·14
    → 현금흐름·SGA·무형자산 데이터 미확보로 인한 신호 저하


[ 2. 저하 구간 원인 ]

  • 2017년: KOSPI +22% 성장주 랠리 → 가치 팩터 전반 약화 (전략1 -16%, 전략2 -14%)
  • 2018년: 미중 무역분쟁·금리인상 → 소형가치주 패닉셀 (전략7 -16%)
  • 2020년: 코로나 유동성 장세 → 영업이익 기반 팩터 무력화 (전략2 -23%)
  • 2021년: 메타버스·바이오 테마 → 펀더멘털 팩터 지속 약화 (전략2 -13%)

  공통 원인: 성장주 프리미엄 국면에서 Value 팩터 전반 언더퍼폼


[ 3. 전략 개선 — KOSPI 모멘텀 필터 ]

  • 전략 7 + 필터: CAGR +11.27% → +14.99%, Sharpe 0.68 → 0.86  (효과 뚜렷)
  • MDD 전반 개선: 하락장 구간 포지션 자동 중립화
  • 단, 빠른 V자 회복 국면(2020)에서 일부 수익 포기


[ 4. 종합 권고 ]

  ▶ 메인 전략: 전략 7 (Dual Value Blend) + KOSPI 모멘텀 필터
  ▶ 보조 전략: 전략 5 (CF Yield) + 전략 2 (OIEY) 블렌딩으로 팩터 분산
  ▶ 향후 과제: DART에서 현금흐름·SGA 데이터 추가 확보 시 전략 3·5·10·11 신호 개선 기대
"""
    ax.text(0.03, 0.96, text, transform=ax.transAxes,
            fontsize=9.5, va="top", linespacing=1.65)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── 메인 ─────────────────────────────────────────────────────────────────────
STRATEGY_DEFS = [
    ("1_LowAccrual",    strat_1,  "Low Accrual Clean (근사)\naccrual=-(eps-ops)/bps\n시총분위 중립화 - 12M 모멘텀"),
    ("2_OpIncEY",       strat_2,  "Operating Income EY\nops/close → group_rank(ts_rank, 126, industry)"),
    ("3_CashCFDiv",     strat_3,  "Cash & CF Divergence (근사)\n-ts_corr(ts_mean(bps,5), ts_mean(eps,5), 252)\n낮은 상관 = 정보 비효율 매수"),
    ("5_CFYield",       strat_5,  "Industry-Neutral CF Yield (근사)\ngroup_rank(ts_zscore(ops/close, 63), industry)\ncashflow_op → ops 대체"),
    ("6_DebtSpike",     strat_6,  "Debt Spike Reversal Momentum\ndebt_ps = (total_assets - bps*shares)/shares\nalpha=signed_power(-ts_zscore(debt,252), 4)\n부채 단기 급등 타이밍 역발상 진입"),
    ("7_DualValue",     strat_7,  "Aggressive Dual Value Blend\nmax(EBITDA_rank, PBR_rank) + 모멘텀 중립화"),
    ("10_Buyback",      strat_10, "Profitable Buyback (공포필터 제외)\n-shares/delay(shares,252) × prof2 가중\n자사주 매입 + 영업ROA 우량기업 롱"),
    ("11_SGAEffic",     strat_11, "SGA Efficiency (근사)\nsga_ps = sps - gp_ps - ops\nsignal = sga/opex ratio 변화율\n매출 성장 확인 시만 거래"),
    ("12_Goodwill",     strat_12, "Goodwill Overvaluation (근사)\nintangible_pr = ta_ps/bps - 1\n높은 무형자산 비율 기업 숏"),
    ("14_DebtDecay",    strat_14, "Debt Decay + Vol Regime (근사)\nf=signed_power(-ts_zscore(debt_ps,63), 1.8)\n저변동성 구간에서만 신호 활성화"),
    ("15_BookCapMom",   strat_15, "Book to Cap Momentum\nd1+d3 (equity/cap 모멘텀 합산)"),
]


CORE_STRATEGY_NAMES = ["2_OpIncEY", "5_CFYield", "7_DualValue", "15_BookCapMom"]


def main():
    print("=" * 65)
    print("  WQ Brain 전략 전체 11개 - 한국 시장 종합 분석 리포트")
    print("=" * 65)

    print("\n[데이터 로드]")
    d        = load_all()
    industry = load_industry()
    kospi    = load_kospi()
    print(f"  가격: {d['close'].shape[0]}일 × {d['close'].shape[1]}종목")
    if not kospi.empty:
        print(f"  KOSPI: {len(kospi)}일")

    print("\n[11개 전략 백테스트]")
    all_results = {}
    for name, fn, note in STRATEGY_DEFS:
        try:
            factor = fn(d, industry)
            r = run_bt(name, factor, d["close"])
            if r:
                all_results[name] = r
        except Exception as e:
            print(f"  [{name}] 오류: {e}")

    # 핵심 전략 — 한국 시장 분석용
    core_results = {n: all_results[n] for n in CORE_STRATEGY_NAMES if n in all_results}

    print("\n[시장 필터 적용 (KOSPI 12M 모멘텀)]")
    filt_results = {}
    for name in CORE_STRATEGY_NAMES:
        if name not in all_results or kospi.empty:
            continue
        fn  = next(f for n, f, _ in STRATEGY_DEFS if n == name)
        try:
            factor  = fn(d, industry)
            factor_f = apply_market_filter(factor, kospi)
            r = run_bt(name + "_f", factor_f, d["close"])
            if r:
                filt_results[name + "_f"] = r
                res, m = r
                print(f"  [{name}_f] CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  MDD {m['mdd']:.2%}")
        except Exception as e:
            print(f"  [{name}_f] 오류: {e}")

    print(f"\n[PDF 생성]")
    pdf_path = RESULTS_DIR / "WQ_Brain_KR_종합분석_리포트.pdf"
    with PdfPages(pdf_path) as pdf:
        # ① 표지
        make_title_page(pdf)
        print("  표지 완료")

        # ② 성과 지표 설명
        make_metrics_explanation_page(pdf)
        print("  지표 설명 페이지 완료")

        # ③ 성과 요약 테이블
        make_summary_page(pdf, all_results)
        print("  성과 요약 완료")

        # ④ 한국 시장 적용 분석 (KOSPI 비교)
        make_market_analysis(pdf, core_results, kospi)
        print("  한국 시장 적용 분석 완료")

        # ⑤ 저하 구간 원인 심층 분석
        make_underperform_analysis(pdf, core_results, kospi)
        print("  저하 구간 원인 분석 완료")

        # ⑥ 시장 필터 전후 비교
        if filt_results:
            make_filter_page(pdf, core_results, filt_results, CORE_STRATEGY_NAMES)
            print("  시장 필터 비교 완료")

        # ⑦ 연도별 수익률 히트맵 (전체 11개)
        make_yearly_heatmap(pdf, all_results)
        print("  연도별 히트맵 완료")

        # ⑧ 전략별 상세 페이지 (11개)
        for name, fn, note in STRATEGY_DEFS:
            if name in all_results:
                make_strategy_page(pdf, name, all_results[name][0],
                                   all_results[name][1], note)
        print("  전략별 상세 완료")

        # ⑨ 결론 및 시사점
        make_conclusion(pdf, all_results, filt_results)
        print("  결론 완료")

    print("\n[전체 결과 요약]")
    print(f"{'전략':<20} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8}")
    print("-" * 50)
    for name, (res, m) in all_results.items():
        print(f"{name:<20} {m['cagr']:>+8.2%} {m['sharpe']:>8.2f} {m['mdd']:>8.2%}")

    print(f"\nPDF 저장 완료: {pdf_path}")


if __name__ == "__main__":
    main()
