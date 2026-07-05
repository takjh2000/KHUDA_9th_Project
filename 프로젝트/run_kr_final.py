"""
KR 11개 전략 최종 백테스트 (2019-01-01 ~ 2023-12-31)
- 데이터 커버리지/구현 불가 문제로 제외된 S04, S08, S09, S13은 아예 포함하지 않음
- 중립화: Market/Sector/Industry 각 전략 스펙대로
- Decay: decay_linear (선형 가중 이동평균)
- 롱숏 상위/하위 50%, factor-score 비중, 전략별 truncation
- 지표: Returns, Sharpe, Margin, Turnover, Drawdown, Fitness (WQ Brain 스타일, Book Size $20M 기준)
- 출력: results/kr_final/report.pdf, summary.csv
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
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from config import PROCESSED_DIR, RAW_DIR
from backtest.engine import LongShortBacktester, Config
from backtest.metrics import returns_metric, sharpe, mdd, margin, fitness
from factors.wq_ops import (
    rank, zscore, ts_mean, ts_std, ts_rank, ts_zscore,
    ts_delta, ts_delay, ts_corr, ts_backfill,
    signed_power, group_neutralize_dynamic, group_rank,
    group_zscore, df_max, trade_when, bucket,
    group_neutralize, decay_linear, truncate, hump,
)

RESULTS_DIR = Path("results/kr_final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_START = "2019-01-01"
BACKTEST_END   = "2023-12-31"


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_kr():
    panel = pd.read_parquet(PROCESSED_DIR / "kr_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    extra_path = RAW_DIR / "kr_extra_finance.parquet"
    if extra_path.exists():
        extra = pd.read_parquet(extra_path)
        extra["date"] = pd.to_datetime(extra["date"])
        extra_cols = [c for c in extra.columns if c not in ["date", "ticker"]]
        panel_s = panel.sort_values("date")
        extra_s = extra.sort_values("date")
        panel = pd.merge_asof(
            panel_s, extra_s[["date", "ticker"] + extra_cols],
            on="date", by="ticker", direction="backward"
        )
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

        if "equity" in panel.columns and "debt" in panel.columns:
            mask = panel["total_assets"].isna()
            panel.loc[mask, "total_assets"] = (
                panel.loc[mask, "equity"].fillna(0) + panel.loc[mask, "debt"].fillna(0)
            ).replace(0, np.nan)
        if "operating_income" in panel.columns and "sga_expense" in panel.columns:
            mask = panel["gross_profit"].isna()
            panel.loc[mask, "gross_profit"] = (
                panel.loc[mask, "operating_income"].fillna(0)
                + panel.loc[mask, "sga_expense"].fillna(0)
            ).replace(0, np.nan)

    # KOSPI200 point-in-time 유니버스로 필터링
    uni_path = RAW_DIR / "kr_universe.parquet"
    universe = None
    if uni_path.exists():
        universe = pd.read_parquet(uni_path)
        universe["date"] = pd.to_datetime(universe["date"])
        panel = panel.merge(universe.assign(_in_uni=True), on=["date", "ticker"], how="inner")
        panel = panel.drop(columns=["_in_uni"])

    price_pivot = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()

    sector_s   = pd.Series(dtype=str)
    industry_s = pd.Series(dtype=str)

    hier_path = RAW_DIR / "kr_sector_hierarchy.parquet"
    sec_path  = RAW_DIR / "kr_sector.parquet"
    load_path = hier_path if hier_path.exists() else sec_path if sec_path.exists() else None

    if load_path:
        sec_df = pd.read_parquet(load_path)
        INDUSTRY_KW = {
            "Healthcare":              ["의약품", "의약물질", "제약", "바이오", "의료기기", "보건", "의료", "병원", "사회복지"],
            "IT":                      ["소프트웨어", "컴퓨터", "정보", "통신", "반도체", "전자", "인터넷", "데이터", "게임", "IT", "전지", "배터리", "축전지", "케이블", "절연선"],
            "Financials":              ["은행", "증권", "보험", "금융", "투자", "저축", "신탁", "캐피탈"],
            "Materials":               ["화학", "철강", "금속", "비금속", "고무", "플라스틱", "섬유", "종이", "목재", "광업", "유리", "시멘트"],
            "Consumer Discretionary":  ["자동차", "의류", "봉제", "가구", "가전", "오락", "여행", "호텔", "숙박", "소매", "백화점", "면세", "가죽", "신발"],
            "Consumer Staples":        ["식품", "음료", "주류", "담배", "음식", "식료품", "농업", "축산", "수산", "도매", "낙농", "유지", "곡물"],
            "Industrials":             ["기계", "조선", "선박", "보트", "항공", "운수", "물류", "건설", "중공업", "방위", "무기", "포장", "인쇄", "서비스", "경영", "컨설팅", "인력"],
            "Energy":                  ["석유", "가스", "에너지", "석탄", "발전", "태양광", "풍력"],
            "Utilities":               ["전기", "수도", "가스공급", "환경", "폐기물", "열공급"],
            "Real Estate":             ["부동산", "임대", "리츠", "건물관리"],
            "Communication Services":  ["방송", "광고", "미디어", "영화", "음반", "출판", "스포츠", "공연"],
        }
        def _fix_sector(row):
            if row["sector"] not in ("Other", "Unknown", ""):
                return row["sector"]
            ind = str(row.get("industry", ""))
            for sector, kws in INDUSTRY_KW.items():
                if any(kw in ind for kw in kws):
                    return sector
            return "Other"

        sec_df["sector"] = sec_df.apply(_fix_sector, axis=1)
        sector_s   = sec_df.set_index("ticker")["sector"]
        industry_s = sec_df.set_index("ticker")["industry"]

    return price_pivot, panel, universe, sector_s, industry_s


def to_pivot(panel, col):
    if col not in panel.columns:
        return None
    p = panel.pivot_table(index="date", columns="ticker", values=col).sort_index()
    if p.empty:
        return None
    return p.ffill(limit=252)


def make_groups(panel, group_s: pd.Series) -> pd.Series:
    tickers = panel["ticker"].unique()
    return pd.Series({t: group_s.get(t, "Unknown") for t in tickers})


def neutralize(f, level, sector_s, industry_s, panel):
    if level == "Market":
        mu = f.mean(axis=1)
        return f.sub(mu, axis=0)
    elif level == "Sector":
        groups = make_groups(panel, sector_s)
        common = f.columns.intersection(groups.index)
        return group_neutralize(f[common], groups[common])
    elif level == "Industry":
        groups = make_groups(panel, industry_s)
        common = f.columns.intersection(groups.index)
        return group_neutralize(f[common], groups[common])
    return f


def apply_decay(f, d):
    return f if d <= 0 else decay_linear(f, d)


def build_factor(raw_f, decay_n, neutralization, truncation_pct, sector_s, industry_s, panel):
    f = apply_decay(raw_f, decay_n)
    f = neutralize(f, neutralization, sector_s, industry_s, panel)
    if truncation_pct > 0:
        f = truncate(f, truncation_pct)
    return zscore(f)


# ── 전략 정의 (11개만) ─────────────────────────────────────────────────────────

def s01_raw(panel, price_pivot):
    roe_p, eq_p, cf_op_p, ta_p, shares_p = (
        to_pivot(panel, c) for c in ["roe", "equity", "cashflow_op", "total_assets", "shares"]
    )
    if roe_p is None or eq_p is None or cf_op_p is None or ta_p is None or shares_p is None:
        return None
    net_income = roe_p * eq_p.reindex_like(roe_p).ffill(limit=252)
    cf = cf_op_p.reindex_like(roe_p).ffill(limit=252)
    ta = ta_p.reindex_like(roe_p).ffill(limit=252).replace(0, np.nan)
    accrual = rank(-(net_income - cf) / ta)
    cap = price_pivot * shares_p.reindex_like(price_pivot).ffill(limit=252)
    cap_bucket = bucket(rank(cap), n=5)
    cap_neut = group_neutralize_dynamic(accrual, cap_bucket)
    mom = ts_rank(price_pivot.pct_change(), 252)
    return cap_neut - mom


def s02_raw(panel, price_pivot):
    oi_p, shares_p, eps_p = (to_pivot(panel, c) for c in ["operating_income", "shares", "eps"])
    if oi_p is not None and shares_p is not None:
        cap = price_pivot * shares_p.reindex_like(price_pivot).ffill(limit=252)
        oey = oi_p.reindex_like(price_pivot).ffill(limit=252) / cap.replace(0, np.nan)
    elif eps_p is not None:
        oey = eps_p / price_pivot.replace(0, np.nan)
    else:
        return None
    return ts_rank(oey, 126)


def s03_raw(panel, price_pivot):
    cf_op_p, eq_p, ta_p = (to_pivot(panel, c) for c in ["cashflow_op", "equity", "total_assets"])
    if cf_op_p is None or eq_p is None or ta_p is None:
        return None
    ta = ta_p.ffill(limit=252).replace(0, np.nan)
    cash_ratio = eq_p.reindex_like(ta).ffill(limit=252) / ta
    cf_ratio = cf_op_p.reindex_like(ta).ffill(limit=252) / ta
    return -ts_corr(ts_mean(cash_ratio, 5), ts_mean(cf_ratio, 5), 252)


def s05_raw(panel, price_pivot, industry_s):
    cf_op_p, shares_p = to_pivot(panel, "cashflow_op"), to_pivot(panel, "shares")
    if cf_op_p is None:
        return None
    if shares_p is not None:
        cap = price_pivot * shares_p.reindex_like(price_pivot).ffill(limit=252)
        cf_yield = cf_op_p.reindex_like(price_pivot).ffill(limit=252) / cap.replace(0, np.nan)
    else:
        ta_p = to_pivot(panel, "total_assets")
        if ta_p is None:
            return None
        cf_yield = cf_op_p.reindex_like(price_pivot).ffill(limit=252) / \
                   ta_p.reindex_like(price_pivot).ffill(limit=252).replace(0, np.nan)
    z = ts_zscore(cf_yield, 63)
    groups = make_groups(panel, industry_s)
    common = z.columns.intersection(groups.index)
    return group_rank(z[common], groups[common])


def s06_raw(panel, price_pivot):
    debt_p, ta_p = to_pivot(panel, "debt"), to_pivot(panel, "total_assets")
    src = debt_p if debt_p is not None else ta_p
    if src is None:
        return None
    debt_piv = src.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    x = -ts_zscore(debt_piv, 252)
    alpha = signed_power(x, 4)
    condition = debt_piv > ts_mean(debt_piv, 63)
    f = trade_when(condition, alpha, -1)
    return hump(f, 0.003)


def s07_raw(panel, price_pivot, industry_s):
    eps_p, bps_p = to_pivot(panel, "eps"), to_pivot(panel, "bps")
    if eps_p is None or bps_p is None:
        return None
    underrated = eps_p / price_pivot.replace(0, np.nan)
    low_pbr = bps_p / price_pivot.replace(0, np.nan)
    groups = make_groups(panel, industry_s)
    common1 = underrated.columns.intersection(groups.index)
    underrated_adj = rank(group_neutralize(underrated[common1], groups[common1]))
    low_pbr_ts = ts_rank(low_pbr, 63)
    common2 = low_pbr_ts.columns.intersection(groups.index)
    low_pbr_recent = group_rank(low_pbr_ts[common2], groups[common2])
    signal = df_max(underrated_adj, low_pbr_recent)
    returns = price_pivot.pct_change()
    mom_group = bucket(rank(ts_mean(returns, 240)), n=10)
    return group_neutralize_dynamic(signal, mom_group)


def s10_raw(panel, price_pivot):
    oi_p, ta_p, shares_p = (
        to_pivot(panel, c) for c in ["operating_income", "total_assets", "shares"]
    )
    if ta_p is None or shares_p is None:
        return None
    if oi_p is not None:
        prof2 = ts_backfill(oi_p.reindex_like(ta_p).ffill(limit=252) / ta_p.replace(0, np.nan), 252)
    else:
        prof2 = None
    shares_aligned = shares_p.reindex_like(ta_p).ffill(limit=252)
    shares_prev = shares_aligned.shift(252)
    buyback = -(shares_aligned / shares_prev.replace(0, np.nan))
    if prof2 is not None:
        signal = buyback * (1 + rank(prof2.reindex_like(buyback)))
    else:
        signal = buyback
    return signal.ffill(limit=5)


def s11_raw(panel, price_pivot):
    sga_p, oi_p, sps_p, shares_p = (
        to_pivot(panel, c) for c in ["sga_expense", "operating_income", "sps", "shares"]
    )
    if sga_p is None or oi_p is None or sps_p is None or shares_p is None:
        return None
    sga = sga_p.ffill(limit=252)
    oi = oi_p.reindex_like(sga).ffill(limit=252)
    revenue = (sps_p * shares_p).reindex_like(sga).ffill(limit=252)
    operating_expense = (revenue - oi).replace(0, np.nan)
    signal = sga / operating_expense
    f = signal / ts_delay(signal, 252).replace(0, np.nan)
    cond = revenue > ts_mean(revenue, 252)
    a = hump(f, 0.001)
    return trade_when(cond, zscore(a), ~cond)


def s12_raw(panel, price_pivot):
    gw_p, ta_p, sps_p, eps_p, shares_p = (
        to_pivot(panel, c) for c in ["goodwill", "total_assets", "sps", "eps", "shares"]
    )
    if sps_p is None or shares_p is None:
        return None
    sales = (sps_p * shares_p).reindex_like(sps_p).ffill(limit=252)
    if gw_p is not None:
        gw_aligned = gw_p.reindex_like(sales).ffill(limit=60)
        ratio = gw_aligned / sales.replace(0, np.nan)
    elif ta_p is not None:
        ratio = ta_p.reindex_like(sales) / sales.replace(0, np.nan)
    else:
        return None
    gw_signal = -ts_backfill(zscore(ratio), 63)
    if eps_p is not None:
        high_accrued = rank(eps_p) > 0.5
        f = gw_signal.where(~high_accrued, gw_signal * 2)
    else:
        f = gw_signal
    return f


def s14_raw(panel, price_pivot):
    debt_p, ta_p = to_pivot(panel, "debt"), to_pivot(panel, "total_assets")
    src = debt_p if debt_p is not None else ta_p
    if src is None:
        return None
    debt_piv = src.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    f_raw = signed_power(-ts_zscore(debt_piv, 63), 1.8)
    ret = price_pivot.pct_change()
    regime_raw = ts_zscore(ts_std(ret, 21), 63)
    f = trade_when(regime_raw < -0.1, f_raw, regime_raw > 0.8)
    return f.ffill(limit=5)


def s15_raw(panel, price_pivot):
    bps_p = to_pivot(panel, "bps")
    if bps_p is None:
        return None
    bp = bps_p / price_pivot.replace(0, np.nan)
    d1 = zscore(ts_delta(bp, 21))
    d3 = zscore(ts_delta(ts_delay(bp, 21), 42))
    return d1 + d3


# ── 전략 메타데이터 (11개만) ────────────────────────────────────────────────────

STRATEGIES = {
    "S01_LowAccrual":          (s01_raw, 6,  "Market",   0.08),
    "S02_OEY":                 (s02_raw, 4,  "Industry", 0.08),
    "S03_CashCFDivergence":    (s03_raw, 4,  "Industry", 0.04),
    "S05_IndustryNeutralCFY":  (s05_raw, 4,  "Sector",   0.08),
    "S06_DebtSpikeReversal":   (s06_raw, 4,  "Sector",   0.01),
    "S07_AggressiveDualValue": (s07_raw, 4,  "Market",   0.01),
    "S10_ProfitableBuyback":   (s10_raw, 1,  "Market",   0.08),
    "S11_SGADriven":           (s11_raw, 2,  "Market",   0.08),
    "S12_GoodwillOverval":     (s12_raw, 4,  "Industry", 0.04),
    "S14_VolRegimeDebt":       (s14_raw, 16, "Market",   0.01),
    "S15_BookToCapMom":        (s15_raw, 5,  "Industry", 0.04),
}

STRATEGY_NAMES = {
    "S01_LowAccrual":          "Low Accrual Clean",
    "S02_OEY":                 "Operating Income EY",
    "S03_CashCFDivergence":    "Cash & CF Divergence",
    "S05_IndustryNeutralCFY":  "Industry-Neutral CF Yield",
    "S06_DebtSpikeReversal":   "Debt Spike Reversal",
    "S07_AggressiveDualValue": "Aggressive Dual Value",
    "S10_ProfitableBuyback":   "Profitable Buyback",
    "S11_SGADriven":           "SG&A Marketing Efficiency",
    "S12_GoodwillOverval":     "Goodwill Overvaluation",
    "S14_VolRegimeDebt":       "Vol-Regime Debt Decay",
    "S15_BookToCapMom":        "Book to Cap Momentum",
}


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    print("KR 데이터 로드 중...", flush=True)
    price_pivot, panel, universe, sector_s, industry_s = load_kr()
    bt_price_pivot = price_pivot.loc[BACKTEST_START:BACKTEST_END]
    print(f"  백테스트 기간: {BACKTEST_START} ~ {BACKTEST_END}", flush=True)

    all_results = {}
    rows = []

    for label, (fn, decay_n, neut_level, trunc) in STRATEGIES.items():
        name = STRATEGY_NAMES[label]
        print(f"\n[{label}] {name}", flush=True)

        if label in ("S05_IndustryNeutralCFY", "S07_AggressiveDualValue"):
            raw_f = fn(panel, price_pivot, industry_s)
        else:
            raw_f = fn(panel, price_pivot)
        if raw_f is None:
            print("  raw signal 없음, 건너뜀", flush=True)
            continue

        factor = build_factor(raw_f, decay_n, neut_level, trunc, sector_s, industry_s, panel)

        cfg = Config(long_pct=0.5, short_pct=0.5, truncation=trunc, score_weight=True)
        bt = LongShortBacktester(price=bt_price_pivot, factor=factor, universe=universe, config=cfg)
        res = bt.run()
        ls = res.ls_returns
        if len(ls) == 0:
            print("  수익률 없음, 건너뜀", flush=True)
            continue

        ret, s, m, t = returns_metric(ls), sharpe(ls), mdd(ls), res.avg_turnover
        mg = margin(ls, t)
        fit = fitness(s, ret, t)
        print(f"  Returns {ret:7.2%}  Sharpe {s:6.2f}  Margin {mg:7.1f}bp  "
              f"Turnover {t:6.2%}  MDD {m:7.2%}  Fitness {fit:6.2f}", flush=True)

        all_results[label] = res
        rows.append({
            "전략": label, "전략명": name,
            "Returns": ret, "Sharpe": s, "Margin": mg,
            "Turnover": t, "Drawdown": m, "Fitness": fit,
        })

    if not rows:
        print("\n결과 없음. 종료.", flush=True)
        return

    summary = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장: {csv_path}", flush=True)
    print(summary.to_string(index=False), flush=True)

    generate_pdf(summary, all_results)


def generate_pdf(summary, all_results):
    pdf_path = RESULTS_DIR / "report.pdf"
    with PdfPages(pdf_path) as pdf:

        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.08, 0.94, "KR 11개 전략 최종 백테스트 리포트", fontsize=20, fontweight="bold")
        fig.text(0.08, 0.905,
                  f"기간 {BACKTEST_START} ~ {BACKTEST_END}   |   "
                  f"long/short 50%/50%, 전략별 truncation, KOSPI200 유니버스   |   "
                  f"S04/S08/S09/S13 제외(데이터 불가/버그)",
                  fontsize=10.5, color="#555")

        disp = summary.set_index("전략")[
            ["Returns", "Sharpe", "Margin", "Turnover", "Drawdown", "Fitness"]
        ].copy()
        disp["Returns"]       = disp["Returns"].map(lambda x: f"{x:.2%}")
        disp["Sharpe"]        = disp["Sharpe"].map(lambda x: f"{x:.2f}")
        disp["Margin"]    = disp["Margin"].map(lambda x: f"{x:.1f}")
        disp["Turnover"]      = disp["Turnover"].map(lambda x: f"{x:.2%}")
        disp["Drawdown"] = disp["Drawdown"].map(lambda x: f"{x:.2%}")
        disp["Fitness"]       = disp["Fitness"].map(lambda x: f"{x:.2f}")

        ax = fig.add_axes([0.05, 0.08, 0.9, 0.72])
        ax.axis("off")
        cell_text = [[idx] + list(row) for idx, row in zip(disp.index, disp.values)]
        col_labels = ["전략", "Returns", "Sharpe", "Margin", "Turnover", "Drawdown", "Fitness"]
        tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#1565C0")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        for label, res in all_results.items():
            ls = res.ls_returns.dropna()
            cum = (1 + ls).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            row = summary[summary["전략"] == label].iloc[0]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5),
                                            gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(cum.index, cum.values, color="#1565C0", lw=1.4)
            ax1.fill_between(cum.index, cum.values, 1, alpha=0.1, color="#1565C0")
            ax1.axhline(1, color="gray", lw=0.5, linestyle="--")
            ax1.set_title(f"{label}  ({STRATEGY_NAMES[label]})", fontsize=13, fontweight="bold")
            ax1.set_ylabel("누적 배수")
            ax1.grid(alpha=0.25)
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            metrics_str = (f"Returns {row['Returns']:.2%}  |  Sharpe {row['Sharpe']:.2f}  |  "
                           f"Margin {row['Margin']:.1f}bp  |  Turnover {row['Turnover']:.2%}  |  "
                           f"MDD {row['Drawdown']:.2%}  |  Fitness {row['Fitness']:.2f}")
            ax1.set_xlabel(metrics_str, fontsize=9.5)

            ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#D32F2F")
            ax2.set_ylabel("DD")
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax2.grid(alpha=0.25)
            ax2.xaxis.set_major_locator(mdates.YearLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPDF 저장: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
