"""
15개 커스텀 전략 백테스트 (KR 시장)
- 중립화: Market/Sector/Industry 각 전략 스펙대로
- Decay: decay_linear (선형 가중 이동평균)
- Truncation: engine Config에 전달
- 롱숏 상위/하위 50%, factor-score 비중
- S08, S09, S13: WQ Brain 독점 데이터 필요 → 건너뜀
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
    group_neutralize, ts_sum, decay_linear, truncate, hump,
    ts_regression_slope,
)

RESULTS_DIR = Path("results/kr_custom")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_kr():
    panel = pd.read_parquet(PROCESSED_DIR / "kr_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # extra finance 병합 (cashflow_op, debt, equity, goodwill, sga_expense 등)
    # extra는 연간 데이터(연도말 + filing lag), panel은 일별 → merge_asof 사용
    extra_path = RAW_DIR / "kr_extra_finance.parquet"
    if extra_path.exists():
        extra = pd.read_parquet(extra_path)
        extra["date"] = pd.to_datetime(extra["date"])
        extra_cols = [c for c in extra.columns if c not in ["date", "ticker"]]
        # merge_asof: ticker별로 panel 날짜 이전의 가장 최근 extra 데이터 매핑
        panel_s = panel.sort_values("date")
        extra_s = extra.sort_values("date")
        panel = pd.merge_asof(
            panel_s, extra_s[["date", "ticker"] + extra_cols],
            on="date", by="ticker", direction="backward"
        )
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        notnull_rates = {c: panel[c].notna().mean() for c in extra_cols}
        print(f"  추가 재무 컬럼 (non-null율): "
              + ", ".join(f"{c}:{v:.0%}" for c, v in notnull_rates.items()))

        # total_assets / gross_profit 파생 (DART extra로 보완)
        if "equity" in panel.columns and "debt" in panel.columns:
            mask = panel["total_assets"].isna()
            panel.loc[mask, "total_assets"] = (
                panel.loc[mask, "equity"].fillna(0) + panel.loc[mask, "debt"].fillna(0)
            ).replace(0, np.nan)
            print(f"  total_assets 파생 후: {panel['total_assets'].notna().mean():.0%}")
        if "operating_income" in panel.columns and "sga_expense" in panel.columns:
            mask = panel["gross_profit"].isna()
            panel.loc[mask, "gross_profit"] = (
                panel.loc[mask, "operating_income"].fillna(0)
                + panel.loc[mask, "sga_expense"].fillna(0)
            ).replace(0, np.nan)
            print(f"  gross_profit 파생 후: {panel['gross_profit'].notna().mean():.0%}")

    # KOSPI200 point-in-time 유니버스로 필터링
    # (그날 실제로 KOSPI200에 속했던 종목만 남김 → 이후 rank/zscore 등
    #  횡단면 연산이 전체 패널이 아니라 그날의 200종목 안에서만 계산됨)
    uni_path = RAW_DIR / "kr_universe.parquet"
    universe = None
    if uni_path.exists():
        universe = pd.read_parquet(uni_path)
        universe["date"] = pd.to_datetime(universe["date"])
        before_rows = len(panel)
        before_dates = panel["date"].nunique()
        panel = panel.merge(universe.assign(_in_uni=True), on=["date", "ticker"], how="inner")
        panel = panel.drop(columns=["_in_uni"])
        print(f"  KOSPI200 마스크 적용: {before_rows:,}행 → {len(panel):,}행 "
              f"({before_dates}일 → {panel['date'].nunique()}일)")

    price_pivot = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()

    # 섹터 계층 로드 + industry 키워드로 Other 보완
    sector_s      = pd.Series(dtype=str)
    industry_s    = pd.Series(dtype=str)
    subindustry_s = pd.Series(dtype=str)

    hier_path = RAW_DIR / "kr_sector_hierarchy.parquet"
    sec_path  = RAW_DIR / "kr_sector.parquet"
    load_path = hier_path if hier_path.exists() else sec_path if sec_path.exists() else None

    if load_path:
        sec_df = pd.read_parquet(load_path)
        # industry 키워드 → sector 보완 매핑
        INDUSTRY_KW = {
            "Healthcare":              ["의약품", "제약", "바이오", "의료기기", "보건", "의료", "병원", "사회복지"],
            "IT":                      ["소프트웨어", "컴퓨터", "정보", "통신", "반도체", "전자", "인터넷", "데이터", "게임", "IT"],
            "Financials":              ["은행", "증권", "보험", "금융", "투자", "저축", "신탁", "캐피탈"],
            "Materials":               ["화학", "철강", "금속", "비금속", "고무", "플라스틱", "섬유", "종이", "목재", "광업", "유리", "시멘트"],
            "Consumer Discretionary":  ["자동차", "의류", "봉제", "가구", "가전", "오락", "여행", "호텔", "숙박", "소매", "백화점", "면세"],
            "Consumer Staples":        ["식품", "음료", "주류", "담배", "음식", "식료품", "농업", "축산", "수산", "도매"],
            "Industrials":             ["기계", "조선", "항공", "운수", "물류", "건설", "중공업", "방위", "포장", "인쇄", "서비스", "경영", "컨설팅", "인력"],
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

        sector_s      = sec_df.set_index("ticker")["sector"]
        industry_s    = sec_df.set_index("ticker")["industry"]
        if "subindustry" in sec_df.columns:
            subindustry_s = sec_df.set_index("ticker")["subindustry"]
        else:
            subindustry_s = industry_s

        tickers_panel = panel["ticker"].unique()
        n_other = sum(1 for t in tickers_panel if sector_s.get(t, "Other") == "Other")
        print(f"  sector 보완 후 Other: {n_other}/{len(tickers_panel)}")

    return price_pivot, panel, universe, sector_s, industry_s, subindustry_s


def to_pivot(panel, col):
    if col not in panel.columns:
        return None
    p = panel.pivot_table(index="date", columns="ticker", values=col).sort_index()
    if p.empty:
        return None
    p = p.ffill(limit=252)
    return p


def make_groups(panel, group_s: pd.Series) -> pd.Series:
    """ticker → group Series (패널 내 ticker 기준)"""
    tickers = panel["ticker"].unique()
    return pd.Series({t: group_s.get(t, "Unknown") for t in tickers})


def neutralize(f: pd.DataFrame, level: str,
               sector_s, industry_s, panel) -> pd.DataFrame:
    """중립화 적용"""
    if level == "Market":
        # 전체 유니버스 평균 제거
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


def apply_decay(f: pd.DataFrame, d: int) -> pd.DataFrame:
    if d <= 0:
        return f
    return decay_linear(f, d)


def build_factor(raw_f, decay_n, neutralization, truncation_pct,
                 sector_s, industry_s, panel):
    """decay → neutralize → truncate → zscore"""
    f = apply_decay(raw_f, decay_n)
    f = neutralize(f, neutralization, sector_s, industry_s, panel)
    if truncation_pct > 0:
        f = truncate(f, truncation_pct)
    return zscore(f)


# ── 전략 정의 ──────────────────────────────────────────────────────────────────
# 각 전략은 raw signal 반환, build_factor에서 decay/neut/trunc 처리

def s01_raw(panel, price_pivot):
    """Low Accrual Clean (근사)
    WQ: accrual = rank(-(net_income - cashflow_op) / total_assets)
        cap_neut = group_neutralize(accrual, bucket(rank(cap), 5))
        mom = ts_rank(returns, 252)
        f = cap_neut - mom
    근사: net_income = roe × equity  (DART; 같은 단위)
    """
    roe_p   = to_pivot(panel, "roe")
    eq_p    = to_pivot(panel, "equity")
    cf_op_p = to_pivot(panel, "cashflow_op")
    ta_p    = to_pivot(panel, "total_assets")
    if roe_p is None or eq_p is None or cf_op_p is None or ta_p is None:
        return None

    # net_income (원) = roe × equity  — 단위 통일
    net_income = roe_p * eq_p.reindex_like(roe_p).ffill(limit=252)
    cf         = cf_op_p.reindex_like(roe_p).ffill(limit=252)
    ta         = ta_p.reindex_like(roe_p).ffill(limit=252).replace(0, np.nan)

    accrual    = rank(-(net_income - cf) / ta)

    # cap proxy: price rank (시가총액 대신 주가 순위)
    cap_bucket = bucket(rank(price_pivot), n=5)
    cap_neut   = group_neutralize_dynamic(accrual, cap_bucket)
    mom        = ts_rank(price_pivot.pct_change(), 252)
    return cap_neut - mom


def s02_raw(panel, price_pivot):
    """Operating Income EY
    WQ: oey = operating_income / market_cap
        f   = ts_rank(oey, 126)
    근사: 우선 operating_income/cap, 없으면 eps/close
    """
    oi_p  = to_pivot(panel, "operating_income")
    eq_p  = to_pivot(panel, "equity")
    bps_p = to_pivot(panel, "bps")
    eps_p = to_pivot(panel, "eps")

    if oi_p is not None and eq_p is not None and bps_p is not None:
        # shares ≈ equity / bps  (단위: equity=원, bps=원/주 → 주수)
        shares = eq_p.reindex_like(price_pivot).ffill(limit=252) / \
                 bps_p.reindex_like(price_pivot).ffill(limit=252).replace(0, np.nan)
        cap = price_pivot * shares
        oey = oi_p.reindex_like(price_pivot).ffill(limit=252) / cap.replace(0, np.nan)
    elif eps_p is not None:
        oey = eps_p / price_pivot.replace(0, np.nan)
    else:
        return None

    return ts_rank(oey, 126)


def s03_raw(panel, price_pivot):
    """Cash & CF Divergence (근사)
    WQ: -ts_corr(ts_mean(cash_balance, 5), ts_mean(cashflow_op, 5), 252)
    근사: cash_balance ≈ equity/total_assets (자기자본비율)
          cashflow_op/total_assets (CF집약도)
          둘 다 DART 연간 데이터 → ffill → 비교 가능
    """
    cf_op_p = to_pivot(panel, "cashflow_op")
    eq_p    = to_pivot(panel, "equity")
    ta_p    = to_pivot(panel, "total_assets")
    if cf_op_p is None or eq_p is None or ta_p is None:
        return None

    ta = ta_p.ffill(limit=252).replace(0, np.nan)
    cash_ratio = eq_p.reindex_like(ta).ffill(limit=252) / ta          # equity ratio
    cf_ratio   = cf_op_p.reindex_like(ta).ffill(limit=252) / ta       # CF intensity

    return -ts_corr(ts_mean(cash_ratio, 5), ts_mean(cf_ratio, 5), 252)


def s04_raw(panel, price_pivot):
    """Growth-Weighted LT Investment
    원래: ts_regression_slope(gwi, 756) — 연간데이터로 3년치 필요
    대체: 자산증가율 × 매출성장률 — 성장하면서 투자도 늘리는 기업
    YoY asset growth weighted by revenue growth
    """
    sps_p = to_pivot(panel, "sps")
    ta_p  = to_pivot(panel, "total_assets")
    if sps_p is None or ta_p is None:
        return None
    # YoY 자산 증가율 (장기투자 지표)
    ta_growth   = ts_delta(ta_p, 252) / ta_p.shift(252).abs().replace(0, np.nan)
    # YoY 매출 성장률 (성장성 지표)
    rev_growth  = rank(ts_delta(sps_p, 252) / sps_p.shift(252).abs().replace(0, np.nan))
    gwi = ta_growth.reindex_like(rev_growth) * rev_growth
    return signed_power(gwi, 1.5)


def s05_raw(panel, price_pivot):
    """Industry-Neutral CF Yield (근사)
    WQ: group_rank(ts_zscore(cashflow_op / market_cap, 63), industry)
    근사: cap = price × shares (shares = equity/bps)
    """
    cf_op_p = to_pivot(panel, "cashflow_op")
    eq_p    = to_pivot(panel, "equity")
    bps_p   = to_pivot(panel, "bps")
    if cf_op_p is None:
        return None

    if eq_p is not None and bps_p is not None:
        shares = eq_p.reindex_like(price_pivot).ffill(limit=252) / \
                 bps_p.reindex_like(price_pivot).ffill(limit=252).replace(0, np.nan)
        cap = price_pivot * shares
        cf_yield = cf_op_p.reindex_like(price_pivot).ffill(limit=252) / cap.replace(0, np.nan)
    else:
        # fallback: cf / total_assets
        ta_p = to_pivot(panel, "total_assets")
        if ta_p is None:
            return None
        cf_yield = cf_op_p.reindex_like(price_pivot).ffill(limit=252) / \
                   ta_p.reindex_like(price_pivot).ffill(limit=252).replace(0, np.nan)

    return ts_zscore(cf_yield, 63)


def s06_raw(panel, price_pivot):
    """Debt Spike Reversal Momentum
    x = -ts_zscore(debt, 252)
    alpha = signed_power(x, 4)
    condition = debt > ts_mean(debt, 63)
    hump(trade_when(condition, alpha, -1), hump=0.003)
    """
    debt_p = to_pivot(panel, "debt")   # DART extra: 부채총계
    ta_p   = to_pivot(panel, "total_assets")
    src = debt_p if debt_p is not None else ta_p
    if src is None:
        return None
    debt_piv = src.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    x = -ts_zscore(debt_piv, 252)
    alpha = signed_power(x, 4)
    condition = debt_piv > ts_mean(debt_piv, 63)
    alt = pd.DataFrame(-1.0, index=alpha.index, columns=alpha.columns)
    f = trade_when(condition, alpha, alt)
    return hump(f, 0.003)


def s07_raw(panel, price_pivot):
    """Aggressive Dual Value Blend with Momentum Neutralization
    underrated = ebitda/cap → eps/close proxy
    underrated_adj = rank(group_neutralize(underrated, industry))
    low_pbr = equity/cap → bps/close proxy
    low_pbr_recent = group_rank(ts_rank(low_pbr, 63), industry)
    signal = max(underrated_adj, low_pbr_recent)
    momentum_group = bucket(rank(ts_mean(returns, 240)), 10)
    f = group_neutralize(signal, momentum_group)
    """
    eps_p = to_pivot(panel, "eps")
    bps_p = to_pivot(panel, "bps")
    if eps_p is None or bps_p is None:
        return None

    underrated = eps_p / price_pivot.replace(0, np.nan)
    low_pbr = bps_p / price_pivot.replace(0, np.nan)

    underrated_adj = rank(underrated)
    low_pbr_recent = ts_rank(low_pbr, 63)
    signal = df_max(underrated_adj, low_pbr_recent)

    returns = price_pivot.pct_change()
    mom_group = bucket(rank(ts_mean(returns, 240)), n=10)
    return group_neutralize_dynamic(signal, mom_group)


def s08_raw(panel, price_pivot):
    return None  # fnd6_drc 독점 데이터


def s09_raw(panel, price_pivot):
    return None  # fn_* 독점 데이터


def s10_raw(panel, price_pivot):
    """Profitable Buyback (공포필터 제외)
    WQ: prof2 = ts_backfill(operating_income / total_assets, 252)
        buyback = -shares_t / shares_t-252  (주식수 감소 = 자사주 매입)
        signal  = buyback * (1 + rank(group_zscore(prof2, industry)))
        [fear filter 제외]
    근사: buyback ≈ -cashflow_op YoY 변화 / equity  (잉여현금 → 주주환원)
          또는 equity YoY 감소율  (자본 감소 = 자사주 소각)
    """
    oi_p  = to_pivot(panel, "operating_income")
    ta_p  = to_pivot(panel, "total_assets")
    eq_p  = to_pivot(panel, "equity")
    cf_p  = to_pivot(panel, "cashflow_op")

    if ta_p is None or eq_p is None:
        return None

    # prof2: operating_income / total_assets (수익성)
    if oi_p is not None:
        prof2 = ts_backfill(
            oi_p.reindex_like(ta_p).ffill(limit=252) / ta_p.replace(0, np.nan), 252
        )
    else:
        prof2 = None

    # buyback proxy: equity YoY 감소율 (자본 줄어들면 배당/자사주매입)
    eq_prev  = eq_p.shift(252)
    buyback  = -(eq_p - eq_prev) / eq_prev.abs().replace(0, np.nan)  # 감소 = positive

    if prof2 is not None:
        signal = buyback * (1 + rank(prof2.reindex_like(buyback)))
    else:
        signal = buyback

    return signal.ffill(limit=5)


def s11_raw(panel, price_pivot):
    """SGA Efficiency (근사)
    WQ: signal = sga_expense / operating_expenses  (판관비 비중)
        f      = signal / ts_delay(signal, 252)   (YoY 비중 변화)
        cond   = revenue > ts_mean(revenue, 252)
        hump(trade_when(cond, zscore(f), -zscore(f)), 0.001)
    근사: operating_expenses ≈ sga + operating_income
          revenue proxy ≈ sps (매출액 per share)
    """
    sga_p = to_pivot(panel, "sga_expense")
    oi_p  = to_pivot(panel, "operating_income")
    eq_p  = to_pivot(panel, "equity")
    if sga_p is None:
        return None

    sga = sga_p.ffill(limit=252)

    if oi_p is not None:
        # sga / (sga + operating_income) = SGA 비중
        oi   = oi_p.reindex_like(sga).ffill(limit=252)
        opex = (sga + oi.abs()).replace(0, np.nan)
        signal_raw = sga / opex
    else:
        # fallback: sga / equity (자본 대비 판관비 비중)
        if eq_p is None:
            return None
        eq = eq_p.reindex_like(sga).ffill(limit=252).replace(0, np.nan)
        signal_raw = sga / eq

    # YoY 변화율 (낮아질수록 효율적)
    prev = signal_raw.shift(252)
    f_ratio = signal_raw / prev.replace(0, np.nan)

    # revenue proxy: equity YoY 변화 (성장 환경)
    if eq_p is not None:
        eq2 = eq_p.reindex_like(sga).ffill(limit=252)
        cond = eq2 > ts_mean(eq2, 252)
    else:
        cond = pd.DataFrame(True, index=sga.index, columns=sga.columns)

    a = zscore(-f_ratio)  # SGA 비중 감소 = positive
    result = trade_when(cond, a, -a)
    return hump(result, 0.001)


def s12_raw(panel, price_pivot):
    """Goodwill Overvaluation & Financial Accrued Risk
    goodwill_sales_ratio = -ts_backfill(zscore(goodwill / sales), 63)
    is_high_accrued = rank(eps) > 0.5
    if_else(high_accrued, ratio * 2, ratio)
    """
    gw_p  = to_pivot(panel, "goodwill")   # DART extra: 영업권/무형자산
    ta_p  = to_pivot(panel, "total_assets")
    sps_p = to_pivot(panel, "sps")
    eps_p = to_pivot(panel, "eps")
    if sps_p is None:
        return None

    if gw_p is not None:
        gw_aligned = gw_p.reindex_like(sps_p).ffill(limit=60)
        ratio = gw_aligned / sps_p.replace(0, np.nan)
    elif ta_p is not None:
        ratio = ta_p / sps_p.replace(0, np.nan)
    else:
        return None

    gw_signal = -ts_backfill(zscore(ratio), 63)
    if eps_p is not None:
        high_accrued = rank(eps_p) > 0.5
        f = gw_signal.where(~high_accrued, gw_signal * 2)
    else:
        f = gw_signal
    return f


def s13_raw(panel, price_pivot):
    return None  # fnd6_acdo, pv13_* 독점 데이터


def s14_raw(panel, price_pivot):
    """Volatility-Regime Filtered Debt Decay Momentum
    f = signed_power(-ts_zscore(debt, 63), 1.8)
    regime_raw = ts_zscore(ts_std(price * (1 + ts_rank(volume, 21)), 21), 63)
    trade_when(regime_raw < -0.1, f, nan) & filter(regime_raw > 0.8 → nan)
    """
    debt_p = to_pivot(panel, "debt")   # DART extra: 부채총계
    ta_p   = to_pivot(panel, "total_assets")
    src = debt_p if debt_p is not None else ta_p
    if src is None:
        return None
    debt_piv = src.reindex(index=price_pivot.index, columns=price_pivot.columns).ffill(limit=60)
    f_raw = signed_power(-ts_zscore(debt_piv, 63), 1.8)

    ret = price_pivot.pct_change()
    regime_raw = ts_zscore(ts_std(ret, 21), 63)

    nan_df = pd.DataFrame(np.nan, index=f_raw.index, columns=f_raw.columns)
    f = trade_when(regime_raw < -0.1, f_raw, nan_df)
    f = f.where(~(regime_raw > 0.8), other=np.nan)
    return f.ffill(limit=5)


def s15_raw(panel, price_pivot):
    """Book to Cap Momentum
    d1 = zscore(ts_delta(equity / cap, 21))
    d3 = zscore(ts_delta(ts_delay(equity / cap, 21), 42))
    f = d1 + d3
    """
    bps_p = to_pivot(panel, "bps")
    if bps_p is None:
        return None
    bp = bps_p / price_pivot.replace(0, np.nan)
    d1 = zscore(ts_delta(bp, 21))
    d3 = zscore(ts_delta(ts_delay(bp, 21), 42))
    return d1 + d3


# ── 전략 메타데이터 ────────────────────────────────────────────────────────────

STRATEGIES = {
    "S01_LowAccrual":          (s01_raw, 6,  "Market",   0.08),
    "S02_OEY":                 (s02_raw, 4,  "Industry", 0.08),
    "S03_CashCFDivergence":    (s03_raw, 4,  "Industry", 0.04),
    "S04_GrowthWeightedLT":    (s04_raw, 0,  "Industry", 0.10),
    "S05_IndustryNeutralCFY":  (s05_raw, 4,  "Sector",   0.08),
    "S06_DebtSpikeReversal":   (s06_raw, 4,  "Sector",   0.01),
    "S07_AggressiveDualValue": (s07_raw, 4,  "Market",   0.01),
    "S08_DeferredRevenue":     (s08_raw, 0,  "Industry", 0.08),
    "S09_TaxAdjustedES":       (s09_raw, 0,  "Industry", 0.08),
    "S10_ProfitableBuyback":   (s10_raw, 1,  "Market",   0.08),
    "S11_SGADriven":           (s11_raw, 2,  "Market",   0.08),
    "S12_GoodwillOverval":     (s12_raw, 4,  "Industry", 0.04),
    "S13_CorpTransparency":    (s13_raw, 4,  "Market",   0.08),
    "S14_VolRegimeDebt":       (s14_raw, 16, "Market",   0.01),
    "S15_BookToCapMom":        (s15_raw, 5,  "Industry", 0.04),
}

STRATEGY_NAMES = {
    "S01_LowAccrual":          "Low Accrual Clean",
    "S02_OEY":                 "Operating Income EY",
    "S03_CashCFDivergence":    "Cash & CF Divergence",
    "S04_GrowthWeightedLT":    "Growth-Weighted LT Invest",
    "S05_IndustryNeutralCFY":  "Industry-Neutral CF Yield",
    "S06_DebtSpikeReversal":   "Debt Spike Reversal",
    "S07_AggressiveDualValue": "Aggressive Dual Value",
    "S08_DeferredRevenue":     "Deferred Revenue (건너뜀)",
    "S09_TaxAdjustedES":       "Tax-Adjusted ES (건너뜀)",
    "S10_ProfitableBuyback":   "Profitable Buyback",
    "S11_SGADriven":           "SG&A Marketing Efficiency",
    "S12_GoodwillOverval":     "Goodwill Overvaluation",
    "S13_CorpTransparency":    "Corp Transparency (건너뜀)",
    "S14_VolRegimeDebt":       "Vol-Regime Debt Decay",
    "S15_BookToCapMom":        "Book to Cap Momentum",
}


# ── 결과 저장 ──────────────────────────────────────────────────────────────────

def plot_single(result, name, label):
    ls = result.ls_returns.dropna()
    if len(ls) == 0:
        return
    cum = (1 + ls).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(cum.index, cum.values, color="#1565C0", lw=1.5)
    ax1.fill_between(cum.index, cum.values, 1, alpha=0.1, color="#1565C0")
    ax1.axhline(1, color="gray", lw=0.5, linestyle="--")
    ax1.set_title(f"[KR] {name}  롱숏 PnL", fontsize=13, fontweight="bold")
    ax1.set_ylabel("누적 배수")
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    from backtest.metrics import cagr, sharpe, mdd
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
    from backtest.metrics import cagr, sharpe, mdd
    df = pd.DataFrame(summary_rows).set_index("전략")
    df_num = df[["CAGR", "Sharpe", "MDD"]].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(df) * 0.5 + 2)))
    axes[0].barh(df_num.index, df_num["CAGR"] * 100,
                 color=["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["CAGR"]])
    axes[0].set_title("CAGR (%)", fontweight="bold")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(df_num.index, df_num["Sharpe"],
                 color=["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["Sharpe"]])
    axes[1].set_title("Sharpe Ratio", fontweight="bold")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].grid(alpha=0.3, axis="x")

    axes[2].barh(df_num.index, df_num["MDD"] * 100, color="#D32F2F")
    axes[2].set_title("MDD (%)", fontweight="bold")
    axes[2].grid(alpha=0.3, axis="x")

    fig.suptitle("KR 커스텀 전략 성과 요약", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = RESULTS_DIR / "summary_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


def plot_cumulative_all(all_results):
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(all_results))
    for i, (label, res) in enumerate(all_results.items()):
        ls = res.ls_returns.dropna()
        if len(ls) == 0:
            continue
        cum = (1 + ls).cumprod()
        ax.plot(cum.index, cum.values, lw=1.2, label=label, color=cmap(i))

    ax.axhline(1, color="gray", lw=0.5, linestyle="--")
    ax.set_title("KR 커스텀 전략 누적 수익률 비교", fontsize=13, fontweight="bold")
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
    from backtest.metrics import cagr, sharpe, mdd

    print("KR 데이터 로드 중...")
    price_pivot, panel, universe, sector_s, industry_s, subindustry_s = load_kr()
    print(f"  종목 수: {price_pivot.shape[1]}, 기간: "
          f"{price_pivot.index[0].date()} ~ {price_pivot.index[-1].date()}")
    print(f"  sector 있는 종목: {len(sector_s)}, industry 있는 종목: {len(industry_s)}")
    print(f"  panel 컬럼: {list(panel.columns)}")

    all_results = {}
    summary_rows = []

    for label, (fn, decay_n, neut_level, trunc) in STRATEGIES.items():
        name = STRATEGY_NAMES[label]
        print(f"\n[{label}] {name}  (Decay={decay_n}, Neut={neut_level}, Trunc={trunc})")

        try:
            raw_f = fn(panel, price_pivot)
        except Exception as e:
            import traceback
            print(f"  → 예외 발생: {e}")
            traceback.print_exc()
            continue
        if raw_f is None:
            print(f"  → 건너뜀 (데이터 없음 또는 구현 불가)")
            continue

        factor = build_factor(raw_f, decay_n, neut_level, trunc,
                              sector_s, industry_s, panel)

        valid = factor.notna().sum().sum()
        if valid == 0:
            print(f"  → 유효 팩터값 없음, 건너뜀")
            continue
        print(f"  팩터 유효값: {valid:,}개")

        cfg = Config(long_pct=0.5, short_pct=0.5,
                     truncation=trunc, score_weight=True)
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
            "전략": label, "전략명": name,
            "CAGR": c, "Sharpe": s, "MDD": m, "Turnover": t,
        })
        plot_single(res, name, label)

    if not all_results:
        print("\n실행된 전략 없음.")
        return

    print("\n비교 차트 생성 중...")
    plot_cumulative_all(all_results)
    plot_summary(summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    for col, fmt in [("CAGR", "{:.2%}"), ("Sharpe", "{:.2f}"),
                     ("MDD", "{:.2%}"), ("Turnover", "{:.2%}")]:
        summary_df[col] = summary_df[col].map(lambda x, f=fmt: f.format(x))
    csv_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n요약 CSV 저장: {csv_path}")
    print(f"\n[완료] 결과: {RESULTS_DIR}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
