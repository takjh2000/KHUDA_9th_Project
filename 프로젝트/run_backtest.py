"""
Phase 2~5  팩터 계산 → 백테스팅 → 비교 분석
    python run_backtest.py              # KR + US 전체
    python run_backtest.py --market KR  # KR만
    python run_backtest.py --market US  # US만
    python run_backtest.py --factor BP  # 특정 팩터만
"""
import sys, argparse, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import pandas as pd
from config import FACTORS, MARKETS, PROCESSED_DIR, RAW_DIR
from factors.compute import compute_all
from backtest.engine import LongShortBacktester, Config
from backtest.metrics import build_summary_table
from analysis.compare import (
    plot_pnl_comparison, plot_annual_heatmap,
    plot_single, print_full_report
)


def load_panel(market: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    패널 로드
    반환: (price_pivot, panel_df, universe_df)
    """
    panel_path = PROCESSED_DIR / f"{market.lower()}_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} 없음. python run_pipeline.py --{market.lower()} 먼저 실행"
        )
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    price_pivot = panel.pivot_table(
        index="date", columns="ticker", values="close"
    ).sort_index()

    universe = None
    uni_path = RAW_DIR / f"{market.lower()}_universe.parquet"
    if uni_path.exists():
        universe = pd.read_parquet(uni_path)
        universe["date"] = pd.to_datetime(universe["date"])

    return price_pivot, panel, universe


def run_market(market: str,
               target_factors: list[str]) -> dict[str, object]:
    """단일 시장의 모든 팩터 백테스트 실행"""
    print(f"\n{'='*55}")
    print(f"  {market} 시장  팩터: {target_factors}")
    print(f"{'='*55}")

    price_pivot, panel, universe = load_panel(market)
    print(f"  종목 수: {price_pivot.shape[1]},  기간: "
          f"{price_pivot.index[0].date()} ~ {price_pivot.index[-1].date()}")

    # Phase 2: 팩터 계산
    print("\n[Phase 2] 팩터 계산")
    all_factors = compute_all(panel)

    # Phase 3: 백테스팅
    print("\n[Phase 3] 백테스팅")
    results = {}
    cfg = Config()

    for fname in target_factors:
        if fname not in all_factors:
            print(f"  {fname}: 팩터 데이터 없음 — 건너뜀")
            continue

        factor_pivot = all_factors[fname]
        print(f"  [{market}_{fname}] 실행 중...")
        bt = LongShortBacktester(
            price=price_pivot,
            factor=factor_pivot,
            universe=universe,
            config=cfg,
        )
        res = bt.run()
        results[f"{market}_{fname}"] = res

        if len(res.ls_returns) == 0:
            print(f"    수익률 데이터 없음")
            continue

        from backtest.metrics import cagr, sharpe, mdd
        ls = res.ls_returns
        print(f"    CAGR {cagr(ls):.2%}  Sharpe {sharpe(ls):.2f}  "
              f"MDD {mdd(ls):.2%}  Turnover {res.avg_turnover:.2%}")

        # 개별 차트 저장
        plot_single(
            res,
            title=f"{market} {fname}  롱숏 PnL",
            save_path=str(Path("results") / f"pnl_{market}_{fname}.png"),
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["KR", "US", "ALL"], default="ALL")
    parser.add_argument("--factor", default=None,
                        help=f"특정 팩터만 ({', '.join(FACTORS)})")
    args = parser.parse_args()

    markets = MARKETS if args.market == "ALL" else [args.market]
    target_factors = [args.factor] if args.factor else FACTORS

    # Phase 3 실행
    all_results = {}
    for market in markets:
        try:
            res = run_market(market, target_factors)
            all_results.update(res)
        except FileNotFoundError as e:
            print(f"\n[오류] {e}")

    if not all_results:
        print("\n실행 가능한 결과 없음. run_pipeline.py 먼저 실행하세요.")
        return

    # Phase 4: 요약 테이블
    print("\n[Phase 4] 성과 집계")
    print_full_report(all_results)

    # Phase 5: 비교 시각화
    print("\n[Phase 5] 비교 분석")
    if len(markets) > 1:
        plot_pnl_comparison(all_results, factors=target_factors)

    for market in markets:
        plot_annual_heatmap(all_results, market=market)

    print("\n[완료] 모든 결과가 results/ 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
