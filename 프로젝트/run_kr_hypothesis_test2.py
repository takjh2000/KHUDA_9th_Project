"""
가설 그룹 F 검증 (KR 시장, 신호(factor) 레벨 오버레이)
F-1~F-5 각각을 baseline과 단독 비교

F-1 12-1 모멘텀 단독
    - 직전 1개월 제외 12개월 수익률
F-2 1개월 리버설 단독
    - 최근 1개월 수익률의 반대 방향 (단기 되돌림)
F-3 거래대금 급증 신호
    - 거래대금(amount)의 20일 z-score. 수급이 몰리는 종목 포착
F-4 거래량 가중 모멘텀
    - 21일 가격 모멘텀에 최근 거래량 순위를 곱해 "거래량 실린 상승"만 강조
F-5 코리안 디스카운트 대응 (재평가 속도)
    - Book-to-Price(BPS/Price)의 252일 rolling z-score에서 최근 63일간 변화(재평가 속도)를 결합.
      저평가 상태에서 벗어나는(재평가가 진행 중인) 종목을 롱 방향으로 가중

각 가설은 raw fundamental 신호는 그대로 두고, baseline 팩터와 z-score 합으로 결합.

데이터 커버리지 문제로 KNOWN_ISSUES.md 에 따라 아래 5개 전략만 대상으로 함:
S03_CashCFDivergence, S06_DebtSpikeReversal, S10_ProfitableBuyback,
S11_SGADriven, S14_VolRegimeDebt
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

from config import RAW_DIR
from backtest.engine import LongShortBacktester, Config
from backtest.metrics import cagr, sharpe, mdd
from factors.wq_ops import zscore, ts_zscore, ts_delta, ts_rank

from run_kr_backtest import (
    load_kr, STRATEGIES, build_factor, to_pivot,
)

RESULTS_DIR = Path("results/kr_hypothesis2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEASIBLE = [
    "S03_CashCFDivergence",
    "S06_DebtSpikeReversal",
    "S10_ProfitableBuyback",
    "S11_SGADriven",
    "S14_VolRegimeDebt",
]

HYPOTHESES = ["F1", "F2", "F3", "F4", "F5"]
HYP_LABEL = {
    "F1": "F-1 12-1 모멘텀",
    "F2": "F-2 1개월 리버설",
    "F3": "F-3 거래대금 급증",
    "F4": "F-4 거래량 가중 모멘텀",
    "F5": "F-5 코리안 디스카운트(재평가 속도)",
}


# ── kr_price_panel.parquet에서 거래량/거래대금 로드 ──────────────────────────

def load_price_panel_field(price_pivot, col):
    path = RAW_DIR / "kr_price_panel.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["date", "ticker", col])
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="date", columns="ticker", values=col).sort_index()
    piv = piv.reindex(index=price_pivot.index, columns=price_pivot.columns)
    return piv


# ── F-1: 12-1 모멘텀 단독 ────────────────────────────────────────────────

def build_mom_12_1(price_pivot):
    mom = price_pivot.shift(21) / price_pivot.shift(252) - 1
    return zscore(mom)


# ── F-2: 1개월 리버설 단독 ───────────────────────────────────────────────

def build_reversal_1m(price_pivot):
    rev = -(price_pivot / price_pivot.shift(21) - 1)
    return zscore(rev)


# ── F-3: 거래대금 급증 신호 ──────────────────────────────────────────────

def build_amount_spike(price_pivot, amount_pivot):
    if amount_pivot is None:
        return None
    spike = ts_zscore(amount_pivot, 20)
    return zscore(spike)


# ── F-4: 거래량 가중 모멘텀 ──────────────────────────────────────────────

def build_volume_weighted_momentum(price_pivot, volume_pivot):
    if volume_pivot is None:
        return None
    mom_21 = price_pivot.pct_change(21)
    vol_confirm = ts_rank(volume_pivot, 60)
    f = mom_21 * vol_confirm
    return zscore(f)


# ── F-5: 코리안 디스카운트 재평가 속도 신호 ──────────────────────────────────

def build_korea_discount_signal(panel, price_pivot):
    bps_p = to_pivot(panel, "bps")
    if bps_p is None:
        return None
    bp = bps_p.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=252)
    bp = bp / price_pivot.replace(0, np.nan)          # Book-to-Price (클수록 저평가)
    bp_z = ts_zscore(bp, 252)
    reval_speed = ts_delta(bp_z, 63)                  # 최근 분기 저평가 수준 변화
    return zscore(-reval_speed)                       # 저평가가 좁혀지는(재평가 중인) 종목 롱


# ── 베이스 팩터 + 오버레이 신호 결합 ─────────────────────────────────────────

def blend(base_factor, overlay_signal):
    common_idx = base_factor.index.intersection(overlay_signal.index)
    common_cols = base_factor.columns.intersection(overlay_signal.columns)
    b = base_factor.loc[common_idx, common_cols]
    o = overlay_signal.loc[common_idx, common_cols]
    combined = zscore(b) + zscore(o)
    return zscore(combined)


# ── 백테스트 실행 ─────────────────────────────────────────────────────────

def run_one(price_pivot, factor, universe):
    cfg = Config()
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

    print("F-1/F-2 가격 모멘텀/리버설 신호 계산 중...", flush=True)
    mom_signal = build_mom_12_1(price_pivot)
    rev_signal = build_reversal_1m(price_pivot)

    print("F-3 거래대금 로드 중...", flush=True)
    amount_pivot = load_price_panel_field(price_pivot, "amount")
    amount_signal = build_amount_spike(price_pivot, amount_pivot)
    if amount_signal is None:
        print("  amount 데이터 없음 → F-3 건너뜀", flush=True)

    print("F-4 거래량 로드 중...", flush=True)
    volume_pivot = load_price_panel_field(price_pivot, "volume")
    vol_mom_signal = build_volume_weighted_momentum(price_pivot, volume_pivot)
    if vol_mom_signal is None:
        print("  volume 데이터 없음 → F-4 건너뜀", flush=True)

    print("F-5 코리안 디스카운트 재평가 속도 신호 계산 중...", flush=True)
    discount_signal = build_korea_discount_signal(panel, price_pivot)
    if discount_signal is None:
        print("  bps 데이터 없음 → F-5 건너뜀", flush=True)

    signals = {
        "F1": mom_signal, "F2": rev_signal, "F3": amount_signal,
        "F4": vol_mom_signal, "F5": discount_signal,
    }

    rows = []
    results = {}   # label -> {"base":res, "F1":res, ...}

    for label in FEASIBLE:
        fn, decay, neut, trunc = STRATEGIES[label]
        print(f"\n[{label}]", flush=True)
        raw = fn(panel, price_pivot)
        if raw is None:
            print("  raw signal 없음, 건너뜀", flush=True)
            continue

        base_f = build_factor(raw, decay, neut, trunc, sector_s, industry_s, panel)
        base = run_one(price_pivot, base_f, universe)
        if base is None:
            print("  baseline 수익률 계산 불가, 건너뜀", flush=True)
            continue

        entry = {"base": base}
        row_tags = [("Baseline", base)]
        print(f"  BASE  CAGR {base['CAGR']:7.2%}  Sharpe {base['Sharpe']:6.2f}  "
              f"MDD {base['MDD']:7.2%}  Turnover {base['Turnover']:6.2%}", flush=True)

        for tag in HYPOTHESES:
            sig = signals.get(tag)
            if sig is None:
                continue
            f_hyp = blend(base_f, sig)
            r = run_one(price_pivot, f_hyp, universe)
            if r is None:
                print(f"  {tag:5s} 수익률 계산 불가, 건너뜀", flush=True)
                continue
            entry[tag] = r
            row_tags.append((tag, r))
            print(f"  {tag:5s} CAGR {r['CAGR']:7.2%}  Sharpe {r['Sharpe']:6.2f}  "
                  f"MDD {r['MDD']:7.2%}  Turnover {r['Turnover']:6.2%}", flush=True)

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
        fig.text(0.08, 0.905, "가설 그룹 F — 신호(factor) 레벨 오버레이, 가설별 개별 기여도 분석",
                  fontsize=13, color="#555")
        body = (
            "적용 가설 (기존 raw fundamental 신호는 그대로 두고, z-score 결합으로 오버레이 신호를 "
            "더함. 5개 가설을 각각 baseline과 단독 비교)\n\n"
            "F-1  12-1 모멘텀 단독 — 직전 1개월 제외 12개월 수익률\n"
            "F-2  1개월 리버설 단독 — 최근 1개월 수익률의 반대 방향\n"
            "F-3  거래대금 급증 신호 — 거래대금(amount) 20일 z-score, 수급 쏠림 포착\n"
            "F-4  거래량 가중 모멘텀 — 21일 가격 모멘텀 × 최근 거래량 순위\n"
            "F-5  코리안 디스카운트 대응(재평가 속도) — BPS/Price의 252일 z-score에서 최근 63일\n"
            "     변화. 저평가가 좁혀지는(재평가 진행 중) 종목에 롱 가중치\n\n"
            "공통 가설: 기존 전략이 fundamental만 쓰다 보니 한국 시장 가격을 실제로 움직이는 힘\n"
            "(외국인 수급, 테마 쏠림, 단기 수급, 밸류 트랩)을 못 잡아서 PnL 개형이 망가진다.\n"
            "→ 가격/거래량/재평가 신호를 더하면 진입·청산 타이밍이 보정되어 PnL이 안정될 것이다.\n\n"
            "대상 전략 (데이터 커버리지 문제로 5개만 검증 — 상세는 KNOWN_ISSUES.md)\n"
            "     S03_CashCFDivergence, S06_DebtSpikeReversal, S10_ProfitableBuyback,\n"
            "     S11_SGADriven, S14_VolRegimeDebt"
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

            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.axis("off")
            ax.set_title(f"{label}  —  Baseline vs F-1~F-5", fontsize=15, fontweight="bold", pad=15)

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

        # 전략별 5장: baseline vs F1 / F2 / F3 / F4 / F5
        colors = {"base": "#90A4AE"}
        hyp_colors = {"F1": "#EF6C00", "F2": "#0277BD", "F3": "#2E7D32", "F4": "#AD1457", "F5": "#6A1B9A"}

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
