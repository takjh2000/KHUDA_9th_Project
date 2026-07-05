"""
KR 11개 전략 최종 백테스트 - 분기 재무 데이터 버전 (2019-01-01 ~ 2023-12-31)
run_kr_final.py와 100% 동일한 로직/전략이며, 유일한 차이는
연 1회(kr_extra_finance.parquet) 대신 연 4회(kr_extra_finance_quarterly.parquet)
재무 데이터를 병합한다는 점뿐.
- 출력: results/kr_final_quarterly/report.pdf, summary.csv
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

matplotlib.rcParams["font.family"] = "AppleGothic"
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
from run_kr_final import (
    s01_raw, s02_raw, s03_raw, s05_raw, s06_raw, s07_raw,
    s10_raw, s11_raw, s12_raw, s14_raw, s15_raw,
    to_pivot, make_groups, neutralize, apply_decay, build_factor,
    STRATEGIES, STRATEGY_NAMES,
)

RESULTS_DIR = Path("results/kr_final_quarterly")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_START = "2019-01-01"
BACKTEST_END   = "2023-12-31"


# ── 데이터 로드 (분기 재무 병합) ────────────────────────────────────────────────

def load_kr():
    panel = pd.read_parquet(PROCESSED_DIR / "kr_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    extra_path = RAW_DIR / "kr_extra_finance_quarterly.parquet"
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


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    print("KR 데이터 로드 중 (분기 재무)...", flush=True)
    price_pivot, panel, universe, sector_s, industry_s = load_kr()
    bt_price_pivot = price_pivot.loc[BACKTEST_START:BACKTEST_END]
    print(f"  백테스트 기간: {BACKTEST_START} ~ {BACKTEST_END}", flush=True)

    all_results = {}
    rows = []

    for label, (fn, decay_n, neut_level, trunc) in STRATEGIES.items():
        name = STRATEGY_NAMES[label]
        print(f"\n[{label}] {name}", flush=True)

        if fn in (s05_raw, s07_raw):
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
        fig.text(0.08, 0.94, "KR 11개 전략 최종 백테스트 리포트 (분기 재무 데이터)", fontsize=18, fontweight="bold")
        fig.text(0.08, 0.905,
                  f"기간 {BACKTEST_START} ~ {BACKTEST_END}   |   "
                  f"long/short 50%/50%, 전략별 truncation, KOSPI200 유니버스   |   "
                  f"재무데이터 연 4회(분기) 갱신   |   S04/S08/S09/S13 제외",
                  fontsize=10.5, color="#555")

        disp = summary.set_index("전략")[
            ["Returns", "Sharpe", "Margin", "Turnover", "Drawdown", "Fitness"]
        ].copy()
        disp["Returns"]       = disp["Returns"].map(lambda x: f"{x:.2%}")
        disp["Sharpe"]        = disp["Sharpe"].map(lambda x: f"{x:.2f}")
        disp["Margin"]    = disp["Margin"].map(lambda x: f"{x:.1f}")
        disp["Turnover"]      = disp["Turnover"].map(lambda x: f"{x:.2%}")
        disp["Drawdown"] = disp["Drawdown"].map(lambda x: f"{abs(x):.2%}")
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

            # 2,000,000원(2000 단위 × k=1000) 초기 투자금 기준 누적 손익
            profit = (2000 * cum - 2000) * 1000

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                            gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(profit.index, profit.values, color="#59CDD5", lw=1.4)
            ax1.axhline(0, color="gray", lw=0.5, linestyle="--")
            ax1.set_title(f"{label}  ({STRATEGY_NAMES[label]})", fontsize=13, fontweight="bold")
            ax1.set_ylabel("누적 손익")
            ax1.grid(alpha=0.25)
            ax1.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: f"{x / 1000:,.0f}K")
            )
            ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            for tick in ax1.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

            metrics_str = (f"Returns {row['Returns']:.2%}  |  Sharpe {row['Sharpe']:.2f}  |  "
                           f"Margin {row['Margin']:.1f}bp\n"
                           f"Turnover {row['Turnover']:.2%}  |  Drawdown {abs(row['Drawdown']):.2%}  |  "
                           f"Fitness {row['Fitness']:.2f}")
            ax1.set_xlabel(metrics_str, fontsize=9)

            ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="#D32F2F")
            ax2.set_ylabel("DD")
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax2.grid(alpha=0.25)
            ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            for tick in ax2.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPDF 저장: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
