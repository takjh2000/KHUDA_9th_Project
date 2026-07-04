"""
가설 그룹 G 검증 (KR 시장, 11개 전략 × 3개 가설)

G-1 1개월 리버설 결합
    - 기존 raw fundamental 신호는 그대로 두고, 1개월 가격 리버설 신호를 z-score 합으로 결합
G-2 SubIndustry 중립화
    - f(i,t) - mean[f(j,t) | j ∈ SubIndustry(i)]  (전략별 기존 Market/Sector/Industry 중립화 대신
      SubIndustry(161개 세분류) 단위로 교체)
G-3 롱숏 10%/10% 농축
    - long_pct/short_pct를 20%/20% → 10%/10%로 좁혀 신호가 확실한 종목만 남김 (팩터/중립화는 baseline과 동일)

데이터 커버리지 재확인 결과(2026-07-04), S04/S08/S09/S13을 제외한 11개 전략 모두
2016~2024년 다년간 백테스트 가능:
S01_LowAccrual, S02_OEY, S03_CashCFDivergence, S05_IndustryNeutralCFY,
S06_DebtSpikeReversal, S07_AggressiveDualValue, S10_ProfitableBuyback,
S11_SGADriven, S12_GoodwillOverval, S14_VolRegimeDebt, S15_BookToCapMom
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
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from backtest.engine import LongShortBacktester, Config
from backtest.metrics import cagr, sharpe, mdd
from factors.wq_ops import zscore, group_neutralize, truncate

from run_kr_backtest import (
    load_kr, STRATEGIES, apply_decay, build_factor, make_groups,
)

RESULTS_DIR = Path("results/kr_hypothesis3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEASIBLE = [
    "S01_LowAccrual",
    "S02_OEY",
    "S03_CashCFDivergence",
    "S05_IndustryNeutralCFY",
    "S06_DebtSpikeReversal",
    "S07_AggressiveDualValue",
    "S10_ProfitableBuyback",
    "S11_SGADriven",
    "S12_GoodwillOverval",
    "S14_VolRegimeDebt",
    "S15_BookToCapMom",
]

HYPOTHESES = ["G1", "G2", "G3"]
HYP_LABEL = {
    "G1": "G-1 1개월 리버설 결합",
    "G2": "G-2 SubIndustry 중립화",
    "G3": "G-3 롱숏 10%/10% 농축",
}


# ── G-1: 1개월 리버설 신호 ────────────────────────────────────────────────

def build_reversal_1m(price_pivot):
    rev = -(price_pivot / price_pivot.shift(21) - 1)
    return zscore(rev)


def blend(base_factor, overlay_signal):
    common_idx = base_factor.index.intersection(overlay_signal.index)
    common_cols = base_factor.columns.intersection(overlay_signal.columns)
    b = base_factor.loc[common_idx, common_cols]
    o = overlay_signal.loc[common_idx, common_cols]
    combined = zscore(b) + zscore(o)
    return zscore(combined)


# ── G-2: SubIndustry 중립화로 교체 ───────────────────────────────────────

def build_factor_subindustry(raw_f, decay_n, truncation_pct, subindustry_s, panel):
    f = apply_decay(raw_f, decay_n)
    groups = make_groups(panel, subindustry_s)
    common = f.columns.intersection(groups.index)
    f = group_neutralize(f[common], groups[common])
    if truncation_pct > 0:
        f = truncate(f, truncation_pct)
    return zscore(f)


# ── 백테스트 실행 ─────────────────────────────────────────────────────────

def run_one(price_pivot, factor, universe, long_pct=0.20, short_pct=0.20):
    cfg = Config(long_pct=long_pct, short_pct=short_pct)
    bt = LongShortBacktester(price=price_pivot, factor=factor, universe=universe, config=cfg)
    res = bt.run()
    ls = res.ls_returns
    if len(ls) == 0:
        return None
    return {
        "CAGR": cagr(ls), "Sharpe": sharpe(ls), "MDD": mdd(ls),
        "Turnover": res.avg_turnover, "res": res,
    }


def main():
    print("KR 데이터 로드 중...", flush=True)
    price_pivot, panel, universe, sector_s, industry_s, subindustry_s = load_kr()

    print("G-1 1개월 리버설 신호 계산 중...", flush=True)
    rev_signal = build_reversal_1m(price_pivot)

    rows = []
    results = {}   # label -> {"base":res, "G1":res, "G2":res, "G3":res}

    for label in FEASIBLE:
        fn, decay, neut, trunc = STRATEGIES[label]
        print(f"\n[{label}] (Neut={neut})", flush=True)
        raw = fn(panel, price_pivot)
        if raw is None:
            print("  raw signal 없음, 건너뜀", flush=True)
            continue

        base_f = build_factor(raw, decay, neut, trunc, sector_s, industry_s, panel)
        base = run_one(price_pivot, base_f, universe, 0.20, 0.20)
        if base is None:
            print("  baseline 수익률 계산 불가, 건너뜀", flush=True)
            continue

        entry = {"base": base}
        row_tags = [("Baseline", base)]
        print(f"  BASE  CAGR {base['CAGR']:7.2%}  Sharpe {base['Sharpe']:6.2f}  "
              f"MDD {base['MDD']:7.2%}  Turnover {base['Turnover']:6.2%}", flush=True)

        # G-1
        g1_f = blend(base_f, rev_signal)
        g1 = run_one(price_pivot, g1_f, universe, 0.20, 0.20)
        if g1 is not None:
            entry["G1"] = g1
            row_tags.append(("G1", g1))
            print(f"  G1    CAGR {g1['CAGR']:7.2%}  Sharpe {g1['Sharpe']:6.2f}  "
                  f"MDD {g1['MDD']:7.2%}  Turnover {g1['Turnover']:6.2%}", flush=True)
        else:
            print("  G1    수익률 계산 불가, 건너뜀", flush=True)

        # G-2
        g2_f = build_factor_subindustry(raw, decay, trunc, subindustry_s, panel)
        g2 = run_one(price_pivot, g2_f, universe, 0.20, 0.20)
        if g2 is not None:
            entry["G2"] = g2
            row_tags.append(("G2", g2))
            print(f"  G2    CAGR {g2['CAGR']:7.2%}  Sharpe {g2['Sharpe']:6.2f}  "
                  f"MDD {g2['MDD']:7.2%}  Turnover {g2['Turnover']:6.2%}", flush=True)
        else:
            print("  G2    수익률 계산 불가, 건너뜀", flush=True)

        # G-3 (팩터 동일, long/short pct만 10%/10%)
        g3 = run_one(price_pivot, base_f, universe, 0.10, 0.10)
        if g3 is not None:
            entry["G3"] = g3
            row_tags.append(("G3", g3))
            print(f"  G3    CAGR {g3['CAGR']:7.2%}  Sharpe {g3['Sharpe']:6.2f}  "
                  f"MDD {g3['MDD']:7.2%}  Turnover {g3['Turnover']:6.2%}", flush=True)
        else:
            print("  G3    수익률 계산 불가, 건너뜀", flush=True)

        for tag, r in row_tags:
            rows.append({
                "전략": label, "구성": tag,
                "CAGR": r["CAGR"], "Sharpe": r["Sharpe"],
                "MDD": r["MDD"], "Turnover": r["Turnover"],
            })

        results[label] = entry

    if not rows:
        print("\n결과 없음. 종료.", flush=True)
        return

    summary = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "summary_by_hypothesis.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장: {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)

    generate_pdf(summary, results)


# ── PDF 리포트 ────────────────────────────────────────────────────────────

def generate_pdf(summary, results):
    pdf_path = RESULTS_DIR / "hypothesis_report.pdf"
    with PdfPages(pdf_path) as pdf:

        # 1p: 타이틀 + 방법론
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.08, 0.94, "KR 전략 가설 검증 리포트", fontsize=22, fontweight="bold")
        fig.text(0.08, 0.905, "가설 그룹 G — 11개 전략 × 3개 가설, 개별 기여도 분석",
                  fontsize=13, color="#555")
        body = (
            "G-1  1개월 리버설 결합 — 기존 raw fundamental 신호에 1개월 가격 리버설 신호를\n"
            "     z-score 합으로 결합 (개인투자자 비중이 높은 한국 시장의 단기 과잉반응 되돌림 가설)\n\n"
            "G-2  SubIndustry 중립화 — f(i,t) - mean[f(j,t) | j∈SubIndustry(i)].\n"
            "     전략별 기존 Market/Sector/Industry 중립화를 SubIndustry(161개 세분류)로 교체.\n"
            "     ※ 현재 데이터의 subindustry 컬럼은 industry와 100% 동일(별도 세분류 없음) —\n"
            "     이미 Industry로 중립화하는 전략(S02,S03,S12,S15)은 baseline과 결과 동일함.\n\n"
            "G-3  롱숏 10%/10% 농축 — long_pct/short_pct를 20%/20% → 10%/10%로 좁힘.\n"
            "     팩터·중립화는 baseline과 동일, 포지션 선정 비율만 변경 (WQ Brain 표준 농축).\n\n"
            "대상 전략 (2026-07-04 재검증 — S04/S08/S09/S13만 제외, 11개 전부 다년간 백테스트 가능)\n"
            "     S01_LowAccrual, S02_OEY, S03_CashCFDivergence, S05_IndustryNeutralCFY,\n"
            "     S06_DebtSpikeReversal, S07_AggressiveDualValue, S10_ProfitableBuyback,\n"
            "     S11_SGADriven, S12_GoodwillOverval, S14_VolRegimeDebt, S15_BookToCapMom"
        )
        fig.text(0.08, 0.06, body, fontsize=10.5, va="bottom", family="Malgun Gothic")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # 2p~: 전략별 요약 표
        for label in FEASIBLE:
            if label not in results:
                continue
            sub = summary[summary["전략"] == label].set_index("구성")
            disp = sub[["CAGR", "Sharpe", "MDD", "Turnover"]].copy()
            disp["CAGR"] = disp["CAGR"].map(lambda x: f"{x:.2%}")
            disp["Sharpe"] = disp["Sharpe"].map(lambda x: f"{x:.2f}")
            disp["MDD"] = disp["MDD"].map(lambda x: f"{x:.2%}")
            disp["Turnover"] = disp["Turnover"].map(lambda x: f"{x:.2%}")

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.axis("off")
            ax.set_title(f"{label}  —  Baseline vs G-1/G-2/G-3", fontsize=15, fontweight="bold", pad=15)

            cell_text = [[idx] + list(row) for idx, row in zip(disp.index, disp.values)]
            col_labels = ["구성", "CAGR", "Sharpe", "MDD", "Turnover"]
            tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                            loc="center", cellLoc="center", bbox=[0.05, 0.05, 0.9, 0.85])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)

            base_sharpe = sub.loc["Baseline", "Sharpe"]
            for i, tag in enumerate(disp.index):
                improved = sub.loc[tag, "Sharpe"] > base_sharpe if tag != "Baseline" else None
                color = "#FFFFFF" if tag == "Baseline" else ("#E8F5E9" if improved else "#FFEBEE")
                for j in range(len(col_labels)):
                    tbl[(i + 1, j)].set_facecolor(color)
                    tbl[(i + 1, j)].set_text_props(fontweight="bold" if tag == "Baseline" else "normal")

            pdf.savefig(fig)
            plt.close(fig)

        # 전략별 3장: baseline vs G1 / G2 / G3
        colors = {"base": "#90A4AE"}
        hyp_colors = {"G1": "#EF6C00", "G2": "#0277BD", "G3": "#2E7D32"}

        for label in FEASIBLE:
            if label not in results:
                continue
            r = results[label]
            base_ls = r["base"]["res"].ls_returns.dropna()
            base_cum = (1 + base_ls).cumprod()

            for tag in HYPOTHESES:
                if tag not in r:
                    continue
                hyp_ls = r[tag]["res"].ls_returns.dropna()
                hyp_cum = (1 + hyp_ls).cumprod()

                fig, ax = plt.subplots(figsize=(11, 6))
                ax.plot(base_cum.index, base_cum.values, color=colors["base"], lw=1.3, label="Baseline")
                ax.plot(hyp_cum.index, hyp_cum.values, color=hyp_colors[tag], lw=1.3,
                        label=f"{HYP_LABEL[tag]} 적용")
                ax.axhline(1, color="gray", lw=0.5, linestyle="--")

                b_m, h_m = r["base"], r[tag]
                subtitle = (f"Sharpe {b_m['Sharpe']:.2f}→{h_m['Sharpe']:.2f}   "
                            f"CAGR {b_m['CAGR']:.1%}→{h_m['CAGR']:.1%}   "
                            f"MDD {b_m['MDD']:.1%}→{h_m['MDD']:.1%}   "
                            f"Turnover {b_m['Turnover']:.1%}→{h_m['Turnover']:.1%}")
                ax.set_title(f"{label}  —  {HYP_LABEL[tag]}\n{subtitle}", fontsize=12, fontweight="bold")
                ax.legend()
                ax.grid(alpha=0.3)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nPDF 저장: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
