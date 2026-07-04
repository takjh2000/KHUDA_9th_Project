"""
11개 전략 백테스트 (KR 시장, 2019-01-01 ~ 2023-12-31)
- 원래 run_kr_backtest.py의 Config 그대로 사용: long/short 50%/50%, 전략별 truncation
- 팩터는 전체 기간 데이터로 계산 (lookback 보존), 실거래만 이 구간으로 제한
- 지표: CAGR(Return), Sharpe, Margin(bps), Turnover, MDD(Drawdown)
- S04, S08, S09, S13: 제외 (데이터 불가 / 버그, KNOWN_ISSUES.md 참조)
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from backtest.engine import LongShortBacktester, Config
from backtest.metrics import cagr, sharpe, mdd, margin

from run_kr_backtest import load_kr, STRATEGIES, STRATEGY_NAMES, build_factor

RESULTS_DIR = Path("results/kr_20192023")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_START = "2019-01-01"
BACKTEST_END   = "2023-12-31"

FEASIBLE = [
    "S01_LowAccrual", "S02_OEY", "S03_CashCFDivergence", "S05_IndustryNeutralCFY",
    "S06_DebtSpikeReversal", "S07_AggressiveDualValue", "S10_ProfitableBuyback",
    "S11_SGADriven", "S12_GoodwillOverval", "S14_VolRegimeDebt", "S15_BookToCapMom",
]


def main():
    print("KR 데이터 로드 중...", flush=True)
    price_pivot, panel, universe, sector_s, industry_s, subindustry_s = load_kr()
    bt_price_pivot = price_pivot.loc[BACKTEST_START:BACKTEST_END]
    print(f"  백테스트 기간: {BACKTEST_START} ~ {BACKTEST_END}", flush=True)

    all_results = {}
    rows = []

    for label in FEASIBLE:
        fn, decay, neut, trunc = STRATEGIES[label]
        name = STRATEGY_NAMES[label]
        print(f"\n[{label}] {name}", flush=True)

        raw = fn(panel, price_pivot)
        if raw is None:
            print("  raw signal 없음, 건너뜀", flush=True)
            continue

        factor = build_factor(raw, decay, neut, trunc, sector_s, industry_s, panel)

        cfg = Config(long_pct=0.5, short_pct=0.5, truncation=trunc, score_weight=True)
        bt = LongShortBacktester(price=bt_price_pivot, factor=factor, universe=universe, config=cfg)
        res = bt.run()
        ls = res.ls_returns
        if len(ls) == 0:
            print("  수익률 없음, 건너뜀", flush=True)
            continue

        c, s, m, t = cagr(ls), sharpe(ls), mdd(ls), res.avg_turnover
        mg = margin(ls, t)
        print(f"  CAGR {c:7.2%}  Sharpe {s:6.2f}  Margin {mg:7.1f}bp  "
              f"Turnover {t:6.2%}  MDD {m:7.2%}", flush=True)

        all_results[label] = res
        rows.append({
            "전략": label, "전략명": name,
            "Return(CAGR)": c, "Sharpe": s, "Margin(bp)": mg,
            "Turnover": t, "Drawdown(MDD)": m,
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

        # 1p: 타이틀 + 요약 표
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.08, 0.94, "KR 11개 전략 백테스트 리포트", fontsize=20, fontweight="bold")
        fig.text(0.08, 0.905,
                  f"기간 {BACKTEST_START} ~ {BACKTEST_END}   |   "
                  f"Config: long/short 50%/50%, 전략별 truncation, KOSPI200 유니버스",
                  fontsize=11, color="#555")

        disp = summary.set_index("전략")[
            ["Return(CAGR)", "Sharpe", "Margin(bp)", "Turnover", "Drawdown(MDD)"]
        ].copy()
        disp["Return(CAGR)"]  = disp["Return(CAGR)"].map(lambda x: f"{x:.2%}")
        disp["Sharpe"]        = disp["Sharpe"].map(lambda x: f"{x:.2f}")
        disp["Margin(bp)"]    = disp["Margin(bp)"].map(lambda x: f"{x:.1f}")
        disp["Turnover"]      = disp["Turnover"].map(lambda x: f"{x:.2%}")
        disp["Drawdown(MDD)"] = disp["Drawdown(MDD)"].map(lambda x: f"{x:.2%}")

        ax = fig.add_axes([0.05, 0.08, 0.9, 0.72])
        ax.axis("off")
        cell_text = [[idx] + list(row) for idx, row in zip(disp.index, disp.values)]
        col_labels = ["전략", "Return(CAGR)", "Sharpe", "Margin(bp)", "Turnover", "Drawdown(MDD)"]
        tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#1565C0")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # 2p~: 전략별 PnL + Drawdown 차트
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

            metrics_str = (f"CAGR {row['Return(CAGR)']:.2%}  |  Sharpe {row['Sharpe']:.2f}  |  "
                           f"Margin {row['Margin(bp)']:.1f}bp  |  Turnover {row['Turnover']:.2%}  |  "
                           f"MDD {row['Drawdown(MDD)']:.2%}")
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
