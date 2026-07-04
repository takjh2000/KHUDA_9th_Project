"""
가설 그룹 E 검증 (KR 시장, 포트폴리오 구성 단계 공통 오버레이)
가설별 개별 기여도 분석 — E-1/E-2/E-3 각각을 baseline과 단독 비교

E-1 글로벌 리스크오프 국면 익스포저 축소
    - KR 유니버스 등가중 지수의 21일 변동성이 역사적 상위 구간일 때 포지션 축소
    - 신호(factor) 자체는 baseline과 동일, 백테스트 단계에서 position_scale만 적용
E-2 업종/시총 이중 중립화
    - 기존 Market/Sector/Industry 중립화 위에 실제 시가총액(kr_price_panel.parquet) 기준
      사이즈 버킷 중립화를 추가
E-3 보유기간 연장 (차익거래 제약 반영)
    - decay_linear 기간을 2배로 늘려 신호 전환 속도를 늦춤

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

from config import PROCESSED_DIR, RAW_DIR
from backtest.engine import LongShortBacktester, Config
from backtest.metrics import cagr, sharpe, mdd
from factors.wq_ops import rank, zscore, bucket, group_neutralize_dynamic, truncate

from run_kr_backtest import (
    load_kr, STRATEGIES, apply_decay, neutralize,
)

RESULTS_DIR = Path("results/kr_hypothesis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEASIBLE = [
    "S03_CashCFDivergence",
    "S06_DebtSpikeReversal",
    "S10_ProfitableBuyback",
    "S11_SGADriven",
    "S14_VolRegimeDebt",
]

HYPOTHESES = ["E1", "E2", "E3"]
HYP_LABEL = {
    "E1": "E-1 리스크오프 축소",
    "E2": "E-2 시총 중립화",
    "E3": "E-3 보유기간 연장",
}


# ── E-2용 실제 시가총액 로드 ──────────────────────────────────────────────

def load_market_cap(price_pivot):
    path = RAW_DIR / "kr_price_panel.parquet"
    if not path.exists():
        print("  kr_price_panel.parquet 없음 → E-2 건너뜀")
        return None
    df = pd.read_parquet(path, columns=["date", "ticker", "market_cap"])
    df["date"] = pd.to_datetime(df["date"])
    cap = df.pivot_table(index="date", columns="ticker", values="market_cap").sort_index()
    cap = cap.reindex(index=price_pivot.index, columns=price_pivot.columns)
    cap = cap.ffill(limit=252)
    return cap


# ── E-1용 리스크오프 국면 배율 ────────────────────────────────────────────

def build_regime_scale(price_pivot):
    mkt_ret = price_pivot.pct_change().mean(axis=1)
    mkt_vol = mkt_ret.rolling(21, min_periods=10).std()
    roll_mu = mkt_vol.rolling(252, min_periods=60).mean()
    roll_sd = mkt_vol.rolling(252, min_periods=60).std()
    vol_z = (mkt_vol - roll_mu) / roll_sd.replace(0, np.nan)

    scale = pd.Series(1.0, index=vol_z.index)
    scale[vol_z > 1.0] = 0.5
    scale[vol_z > 2.0] = 0.0
    scale = scale.fillna(1.0)
    return scale, vol_z


# ── E-2 적용 중립화 (기존 neutralize 위에 사이즈 버킷 추가) ──────────────────

def apply_size_neutral(f: pd.DataFrame, cap_pivot: pd.DataFrame) -> pd.DataFrame:
    if cap_pivot is None:
        return f
    common_idx = f.index.intersection(cap_pivot.index)
    common_cols = f.columns.intersection(cap_pivot.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return f
    cap_sub = cap_pivot.loc[common_idx, common_cols]
    cap_bucket = bucket(rank(cap_sub), n=5)
    f_sub = f.loc[common_idx, common_cols]
    f_neut = group_neutralize_dynamic(f_sub, cap_bucket)
    result = f.copy()
    result.loc[common_idx, common_cols] = f_neut
    return result


# ── 팩터 빌드: baseline / E-2 / E-3 (E-1은 신호가 아니라 backtest 단계에서 적용) ──

def build_factor_baseline(raw_f, decay_n, neutralization, truncation_pct,
                           sector_s, industry_s, panel):
    f = apply_decay(raw_f, decay_n)
    f = neutralize(f, neutralization, sector_s, industry_s, panel)
    if truncation_pct > 0:
        f = truncate(f, truncation_pct)
    return zscore(f)


def build_factor_e2(raw_f, decay_n, neutralization, truncation_pct,
                     sector_s, industry_s, panel, cap_pivot):
    f = apply_decay(raw_f, decay_n)
    f = neutralize(f, neutralization, sector_s, industry_s, panel)
    f = apply_size_neutral(f, cap_pivot)
    if truncation_pct > 0:
        f = truncate(f, truncation_pct)
    return zscore(f)


def build_factor_e3(raw_f, decay_n, neutralization, truncation_pct,
                     sector_s, industry_s, panel):
    decay_boost = max(decay_n * 2, 2) if decay_n > 0 else 10
    f = apply_decay(raw_f, decay_boost)
    f = neutralize(f, neutralization, sector_s, industry_s, panel)
    if truncation_pct > 0:
        f = truncate(f, truncation_pct)
    return zscore(f)


# ── 백테스트 실행 ─────────────────────────────────────────────────────────

def run_one(price_pivot, factor, universe, position_scale=None):
    cfg = Config(position_scale=position_scale)
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
    print("KR 데이터 로드 중...")
    price_pivot, panel, universe, sector_s, industry_s, subindustry_s = load_kr()

    print("E-1 리스크오프 국면 배율 계산 중...")
    regime_scale, vol_z = build_regime_scale(price_pivot)
    print(f"  축소(0.5x) 구간: {(regime_scale == 0.5).sum()}일, "
          f"중립화(0x) 구간: {(regime_scale == 0.0).sum()}일 / 전체 {len(regime_scale)}일")

    print("E-2 실제 시가총액 로드 중...")
    cap_pivot = load_market_cap(price_pivot)

    rows = []
    results = {}   # label -> {"base":res, "E1":res, "E2":res, "E3":res}

    for label in FEASIBLE:
        fn, decay, neut, trunc = STRATEGIES[label]
        print(f"\n[{label}]")
        raw = fn(panel, price_pivot)
        if raw is None:
            print("  raw signal 없음, 건너뜀")
            continue

        base_f = build_factor_baseline(raw, decay, neut, trunc, sector_s, industry_s, panel)
        e2_f   = build_factor_e2(raw, decay, neut, trunc, sector_s, industry_s, panel, cap_pivot)
        e3_f   = build_factor_e3(raw, decay, neut, trunc, sector_s, industry_s, panel)

        base = run_one(price_pivot, base_f, universe, position_scale=None)
        e1   = run_one(price_pivot, base_f, universe, position_scale=regime_scale)  # 신호 동일, scale만 적용
        e2   = run_one(price_pivot, e2_f,   universe, position_scale=None)
        e3   = run_one(price_pivot, e3_f,   universe, position_scale=None)

        if any(r is None for r in [base, e1, e2, e3]):
            print("  수익률 계산 불가, 건너뜀")
            continue

        for tag, r in [("BASE", base), ("E1", e1), ("E2", e2), ("E3", e3)]:
            print(f"  {tag:5s} CAGR {r['CAGR']:7.2%}  Sharpe {r['Sharpe']:6.2f}  "
                  f"MDD {r['MDD']:7.2%}  Turnover {r['Turnover']:6.2%}")

        for tag, r in [("Baseline", base), ("E1", e1), ("E2", e2), ("E3", e3)]:
            rows.append({
                "전략": label, "구성": tag,
                "CAGR": r["CAGR"], "Sharpe": r["Sharpe"],
                "MDD": r["MDD"], "Turnover": r["Turnover"],
            })

        results[label] = {"base": base, "E1": e1, "E2": e2, "E3": e3}

    if not rows:
        print("\n결과 없음. 종료.")
        return

    summary = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / "summary_by_hypothesis.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장: {summary_path}")
    print(summary.to_string(index=False))

    generate_pdf(summary, results, regime_scale, vol_z)


# ── PDF 리포트 ────────────────────────────────────────────────────────────

def generate_pdf(summary, results, regime_scale, vol_z):
    pdf_path = RESULTS_DIR / "hypothesis_report.pdf"
    with PdfPages(pdf_path) as pdf:

        # 1p: 타이틀 + 방법론
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.08, 0.94, "KR 전략 가설 검증 리포트", fontsize=22, fontweight="bold")
        fig.text(0.08, 0.905, "가설 그룹 E — 가설별 개별 기여도 분석", fontsize=13, color="#555")
        body = (
            "적용 가설 (US 전략 코드는 그대로 두고, 포트폴리오 구성 단계에만 적용. 3개 가설을 각각 "
            "baseline과 단독 비교)\n\n"
            "E-1  글로벌 리스크오프 국면 익스포저 축소\n"
            "     KR 유니버스 등가중 수익률의 21일 변동성이 역사적 평균 대비 +1시그마 이상이면\n"
            "     포지션 50% 축소, +2시그마 이상이면 완전 중립화. (신호는 baseline과 동일)\n\n"
            "E-2  업종/시총 이중 중립화\n"
            "     기존 Market/Sector/Industry 중립화 위에 실제 시가총액(kr_price_panel.parquet)\n"
            "     기준 5분위 사이즈 버킷 중립화를 추가.\n\n"
            "E-3  보유기간 연장 (차익거래 제약 반영)\n"
            "     전략별 decay_linear 스무딩 기간을 2배로 연장 (예: 4일→8일).\n\n"
            "대상 전략 (데이터 커버리지 문제로 5개만 검증 — 상세는 KNOWN_ISSUES.md)\n"
            "     S03_CashCFDivergence, S06_DebtSpikeReversal, S10_ProfitableBuyback,\n"
            "     S11_SGADriven, S14_VolRegimeDebt\n\n"
            "제외 전략: S01, S02, S05, S07, S12, S15 (재무데이터 결합 커버리지가 2024년부터만\n"
            "형성되어 다년간 백테스트 불가), S04 (신호 자체가 계산되지 않음)"
        )
        fig.text(0.08, 0.08, body, fontsize=10.5, va="bottom", family="Malgun Gothic")
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # 2p~: 전략별 요약 표 (한 페이지에 전략 1개씩, row 4개 - Baseline/E1/E2/E3)
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
            ax.set_title(f"{label}  —  Baseline vs E-1/E-2/E-3", fontsize=15, fontweight="bold", pad=15)

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

        # 다음: 리스크오프 국면 배율 시계열 (E-1 설명용)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(regime_scale.index, regime_scale.values, color="#D32F2F", lw=1)
        ax.set_title("E-1 리스크오프 국면 포지션 배율 (1.0=풀 익스포저)", fontsize=12, fontweight="bold")
        ax.set_ylabel("scale", rotation=0, ha="right", labelpad=15)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 전략별 3장: baseline vs E1 / baseline vs E2 / baseline vs E3
        colors = {"base": "#90A4AE"}
        hyp_colors = {"E1": "#D32F2F", "E2": "#1565C0", "E3": "#2E7D32"}

        for label in FEASIBLE:
            if label not in results:
                continue
            r = results[label]
            base_ls = r["base"]["res"].ls_returns.dropna()
            base_cum = (1 + base_ls).cumprod()

            for tag in HYPOTHESES:
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

    print(f"\nPDF 저장: {pdf_path}")


if __name__ == "__main__":
    main()
