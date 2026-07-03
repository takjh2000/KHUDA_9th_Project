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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from config import PROCESSED_DIR, RAW_DIR
from backtest.engine import LongShortBacktester, Config
from backtest.metrics import (
    cagr, sharpe, mdd,
    returns_total, drawdown_wq, fitness, margin,
)
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

# ── 백테스트 기간 ─────────────────────────────────────────────────────────────
BT_START = "2019-01-01"
BT_END   = "2023-12-31"


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
        subindustry_s = sec_df.get("subindustry", industry_s)
        if isinstance(subindustry_s, pd.DataFrame):
            subindustry_s = industry_s

        tickers_panel = panel["ticker"].unique()
        n_other = sum(1 for t in tickers_panel if sector_s.get(t, "Other") == "Other")
        print(f"  sector 보완 후 Other: {n_other}/{len(tickers_panel)}")

    universe = None
    uni_path = RAW_DIR / "kr_universe.parquet"
    if uni_path.exists():
        universe = pd.read_parquet(uni_path)
        universe["date"] = pd.to_datetime(universe["date"])

    # KOSPI 200 유니버스 마스크 적용
    # 팩터 계산(rank/zscore)도 KOSPI 200 구성종목 내에서만 이뤄지도록 제한
    if universe is not None and not universe.empty:
        kospi200_all = set(universe["ticker"])
        # price_pivot: KOSPI 200 종목 컬럼만 유지
        valid_cols = [c for c in price_pivot.columns if c in kospi200_all]
        price_pivot = price_pivot[valid_cols]
        # 분기별 구성종목 마스크: 해당 분기 비구성종목은 NaN (시점별 멤버십 반영)
        uni_pivot = (
            universe.assign(member=1)
            .pivot_table(index="date", columns="ticker", values="member", aggfunc="first")
            .reindex(columns=price_pivot.columns)
        )
        uni_mask = uni_pivot.reindex(price_pivot.index, method="ffill").notna()
        price_pivot = price_pivot.where(uni_mask)
        # panel: KOSPI 200 종목만 유지
        panel = panel[panel["ticker"].isin(kospi200_all)].copy().reset_index(drop=True)
        n_tickers = price_pivot.notna().any().sum()
        print(f"  KOSPI 200 필터 적용: {n_tickers}개 종목 (분기별 멤버십 반영)")

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
               sector_s, industry_s, panel,
               subindustry_s=None) -> pd.DataFrame:
    """중립화 적용"""
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
    elif level == "SubIndustry":
        # H1: 재벌 계열사 동조화 — SubIndustry 수준 중립화
        src = subindustry_s if (subindustry_s is not None and not subindustry_s.empty) else industry_s
        groups = make_groups(panel, src)
        common = f.columns.intersection(groups.index)
        return group_neutralize(f[common], groups[common])
    return f


def apply_decay(f: pd.DataFrame, d: int) -> pd.DataFrame:
    if d <= 0:
        return f
    return decay_linear(f, d)


def build_factor(raw_f, decay_n, neutralization, truncation_pct,
                 sector_s, industry_s, panel,
                 subindustry_s=None, cap_df=None):
    """decay → neutralize → truncate → zscore"""
    f = apply_decay(raw_f, decay_n)
    if neutralization == "CapBucket":
        # H3: 외국인 패시브 시총 편향 — 시총 5분위 동적 중립화
        if cap_df is not None:
            cap_sub = cap_df.reindex(columns=f.columns)
            cap_bkt = bucket(rank(cap_sub), 5).reindex(f.index, method="ffill")
            f = group_neutralize_dynamic(f, cap_bkt)
        else:
            mu = f.mean(axis=1)
            f = f.sub(mu, axis=0)
    else:
        f = neutralize(f, neutralization, sector_s, industry_s, panel, subindustry_s)
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
    WQ: ts_corr(ts_mean(cash_balance, 5), ts_mean(cashflow_op, 5), 252)
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

    return ts_corr(ts_mean(cash_ratio, 5), ts_mean(cf_ratio, 5), 252)


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

# ── 가설 검증 전략 ─────────────────────────────────────────────────────────────
# H1: SubIndustry 중립화 — 재벌 계열사 동조화 제거 (모든 전략 대상)
# H2: Decay 분석 — 기존 STRATEGIES 결과 활용 (코드 변경 없음)
# H3: CapBucket 중립화  — 외국인 패시브 시총 편향 제거 (모든 전략 대상)
HYPOTHESIS_STRATEGIES = {
    # H2: 전 전략에 단축 Decay 적용 (개인투자자 단기 노이즈 → 짧은 decay가 유리한가?)
    # S04(decay=0), S10(decay=1) 은 이미 최소 → 건너뜀
    "H2_S01_LowAccrual":          (s01_raw, 2,  "Market",   0.08),   # 6→2
    "H2_S02_OEY":                 (s02_raw, 1,  "Industry", 0.08),   # 4→1
    "H2_S03_CashCFDivergence":    (s03_raw, 1,  "Industry", 0.04),   # 4→1
    "H2_S05_IndustryNeutralCFY":  (s05_raw, 1,  "Sector",   0.08),   # 4→1
    "H2_S06_DebtSpikeReversal":   (s06_raw, 1,  "Sector",   0.01),   # 4→1
    "H2_S07_AggressiveDualValue": (s07_raw, 1,  "Market",   0.01),   # 4→1
    "H2_S11_SGADriven":           (s11_raw, 1,  "Market",   0.08),   # 2→1
    "H2_S12_GoodwillOverval":     (s12_raw, 1,  "Industry", 0.04),   # 4→1
    "H2_S14_VolRegimeDebt":       (s14_raw, 4,  "Market",   0.01),   # 16→4
    "H2_S15_BookToCapMom":        (s15_raw, 2,  "Industry", 0.04),   # 5→2
    # H1: 전 전략에 SubIndustry 중립화 적용
    "H1_S01_LowAccrual":          (s01_raw, 6,  "SubIndustry", 0.08),
    "H1_S02_OEY":                 (s02_raw, 4,  "SubIndustry", 0.08),
    "H1_S03_CashCFDivergence":    (s03_raw, 4,  "SubIndustry", 0.04),
    "H1_S05_IndustryNeutralCFY":  (s05_raw, 4,  "SubIndustry", 0.08),
    "H1_S06_DebtSpikeReversal":   (s06_raw, 4,  "SubIndustry", 0.01),
    "H1_S07_AggressiveDualValue": (s07_raw, 4,  "SubIndustry", 0.01),
    "H1_S10_ProfitableBuyback":   (s10_raw, 1,  "SubIndustry", 0.08),
    "H1_S11_SGADriven":           (s11_raw, 2,  "SubIndustry", 0.08),
    "H1_S12_GoodwillOverval":     (s12_raw, 4,  "SubIndustry", 0.04),
    "H1_S14_VolRegimeDebt":       (s14_raw, 16, "SubIndustry", 0.01),
    "H1_S15_BookToCapMom":        (s15_raw, 5,  "SubIndustry", 0.04),
    # H3: 전 전략에 CapBucket 중립화 적용
    "H3_S01_LowAccrual":          (s01_raw, 6,  "CapBucket",   0.08),
    "H3_S02_OEY":                 (s02_raw, 4,  "CapBucket",   0.08),
    "H3_S03_CashCFDivergence":    (s03_raw, 4,  "CapBucket",   0.04),
    "H3_S05_IndustryNeutralCFY":  (s05_raw, 4,  "CapBucket",   0.08),
    "H3_S06_DebtSpikeReversal":   (s06_raw, 4,  "CapBucket",   0.01),
    "H3_S07_AggressiveDualValue": (s07_raw, 4,  "CapBucket",   0.01),
    "H3_S10_ProfitableBuyback":   (s10_raw, 1,  "CapBucket",   0.08),
    "H3_S11_SGADriven":           (s11_raw, 2,  "CapBucket",   0.08),
    "H3_S12_GoodwillOverval":     (s12_raw, 4,  "CapBucket",   0.04),
    "H3_S14_VolRegimeDebt":       (s14_raw, 16, "CapBucket",   0.01),
    "H3_S15_BookToCapMom":        (s15_raw, 5,  "CapBucket",   0.04),
    # H4: 롱숏 집중도 강화 (WQ Brain 표준 상하위 20%)
    # 현상: 현재 50/50 포트폴리오가 신호 희석 → 상하위 20%로 팩터 강도 증폭
    "H4_S01_LowAccrual":          (s01_raw, 6,  "Market",   0.08, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S02_OEY":                 (s02_raw, 4,  "Industry", 0.08, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S03_CashCFDivergence":    (s03_raw, 4,  "Industry", 0.04, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S05_IndustryNeutralCFY":  (s05_raw, 4,  "Sector",   0.08, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S06_DebtSpikeReversal":   (s06_raw, 4,  "Sector",   0.01, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S07_AggressiveDualValue": (s07_raw, 4,  "Market",   0.01, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S10_ProfitableBuyback":   (s10_raw, 1,  "Market",   0.08, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S11_SGADriven":           (s11_raw, 2,  "Market",   0.08, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S12_GoodwillOverval":     (s12_raw, 4,  "Industry", 0.04, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S14_VolRegimeDebt":       (s14_raw, 16, "Market",   0.01, {"long_pct": 0.2, "short_pct": 0.2}),
    "H4_S15_BookToCapMom":        (s15_raw, 5,  "Industry", 0.04, {"long_pct": 0.2, "short_pct": 0.2}),
    # H5: 반기 리밸런싱 (한국 반기보고서 발행 주기와 일치)
    # 현상: 분기 리밸런싱이 한국 기업 반기 공시 주기와 불일치 → Turnover 과다
    "H5_S01_LowAccrual":          (s01_raw, 6,  "Market",   0.08, {"rebalance": "6ME"}),
    "H5_S02_OEY":                 (s02_raw, 4,  "Industry", 0.08, {"rebalance": "6ME"}),
    "H5_S03_CashCFDivergence":    (s03_raw, 4,  "Industry", 0.04, {"rebalance": "6ME"}),
    "H5_S05_IndustryNeutralCFY":  (s05_raw, 4,  "Sector",   0.08, {"rebalance": "6ME"}),
    "H5_S06_DebtSpikeReversal":   (s06_raw, 4,  "Sector",   0.01, {"rebalance": "6ME"}),
    "H5_S07_AggressiveDualValue": (s07_raw, 4,  "Market",   0.01, {"rebalance": "6ME"}),
    "H5_S10_ProfitableBuyback":   (s10_raw, 1,  "Market",   0.08, {"rebalance": "6ME"}),
    "H5_S11_SGADriven":           (s11_raw, 2,  "Market",   0.08, {"rebalance": "6ME"}),
    "H5_S12_GoodwillOverval":     (s12_raw, 4,  "Industry", 0.04, {"rebalance": "6ME"}),
    "H5_S14_VolRegimeDebt":       (s14_raw, 16, "Market",   0.01, {"rebalance": "6ME"}),
    "H5_S15_BookToCapMom":        (s15_raw, 5,  "Industry", 0.04, {"rebalance": "6ME"}),
    # H6: 대형주 유니버스 (시총 중위수 이상 — KOSDAQ 소형주 유동성 노이즈 제거)
    # 현상: 저유동성 소형주가 팩터 신호를 왜곡 → 시총 상위 50%로 실현 가능 알파 확보
    "H6_S01_LowAccrual":          (s01_raw, 6,  "Market",   0.08, {"universe": "largecap"}),
    "H6_S02_OEY":                 (s02_raw, 4,  "Industry", 0.08, {"universe": "largecap"}),
    "H6_S03_CashCFDivergence":    (s03_raw, 4,  "Industry", 0.04, {"universe": "largecap"}),
    "H6_S05_IndustryNeutralCFY":  (s05_raw, 4,  "Sector",   0.08, {"universe": "largecap"}),
    "H6_S06_DebtSpikeReversal":   (s06_raw, 4,  "Sector",   0.01, {"universe": "largecap"}),
    "H6_S07_AggressiveDualValue": (s07_raw, 4,  "Market",   0.01, {"universe": "largecap"}),
    "H6_S10_ProfitableBuyback":   (s10_raw, 1,  "Market",   0.08, {"universe": "largecap"}),
    "H6_S11_SGADriven":           (s11_raw, 2,  "Market",   0.08, {"universe": "largecap"}),
    "H6_S12_GoodwillOverval":     (s12_raw, 4,  "Industry", 0.04, {"universe": "largecap"}),
    "H6_S14_VolRegimeDebt":       (s14_raw, 16, "Market",   0.01, {"universe": "largecap"}),
    "H6_S15_BookToCapMom":        (s15_raw, 5,  "Industry", 0.04, {"universe": "largecap"}),
    # H7: Truncation 표준화 (전략별 0.01~0.10 편차 → 0.05 일괄 적용)
    # 현상: 전략별 상이한 Truncation으로 집중 리스크 불균등 → 0.05로 표준화
    "H7_S01_LowAccrual":          (s01_raw, 6,  "Market",   0.05),
    "H7_S02_OEY":                 (s02_raw, 4,  "Industry", 0.05),
    "H7_S03_CashCFDivergence":    (s03_raw, 4,  "Industry", 0.05),
    "H7_S05_IndustryNeutralCFY":  (s05_raw, 4,  "Sector",   0.05),
    "H7_S06_DebtSpikeReversal":   (s06_raw, 4,  "Sector",   0.05),
    "H7_S07_AggressiveDualValue": (s07_raw, 4,  "Market",   0.05),
    "H7_S10_ProfitableBuyback":   (s10_raw, 1,  "Market",   0.05),
    "H7_S11_SGADriven":           (s11_raw, 2,  "Market",   0.05),
    "H7_S12_GoodwillOverval":     (s12_raw, 4,  "Industry", 0.05),
    "H7_S14_VolRegimeDebt":       (s14_raw, 16, "Market",   0.05),
    "H7_S15_BookToCapMom":        (s15_raw, 5,  "Industry", 0.05),
}


# ── H7 그리드서치 ──────────────────────────────────────────────────────────────

def _find_h7_truncations(all_results, panel, price_pivot, universe,
                          sector_s, industry_s, subindustry_s, cap_df):
    """H7: 전략별 최적 Truncation 그리드서치 — Sharpe 최대화"""
    CANDIDATES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    h7_trunc = {}
    h7_entries = {k: v for k, v in HYPOTHESIS_STRATEGIES.items() if k.startswith("H7")}

    print("\n  [H7 그리드서치] 전략별 최적 Truncation 탐색 중...")
    for label, config_entry in h7_entries.items():
        fn, decay_n, neut_level = config_entry[0], config_entry[1], config_entry[2]
        base_label  = label[3:]  # "H7_S01_LowAccrual" → "S01_LowAccrual"
        base_res    = all_results.get(base_label)
        base_sharpe = sharpe(base_res.ls_returns) if base_res else -np.inf

        best_trunc  = 0.05
        best_sharpe = -np.inf

        try:
            raw_f = fn(panel, price_pivot)
        except Exception:
            h7_trunc[label] = best_trunc
            continue
        if raw_f is None:
            h7_trunc[label] = best_trunc
            continue

        for trunc in CANDIDATES:
            try:
                factor = build_factor(raw_f, decay_n, neut_level, trunc,
                                      sector_s, industry_s, panel, subindustry_s, cap_df)
                if factor.notna().sum().sum() == 0:
                    continue
                cfg = Config(long_pct=0.5, short_pct=0.5, truncation=trunc, score_weight=True)
                bt  = LongShortBacktester(price=price_pivot, factor=factor,
                                          universe=universe, config=cfg)
                res = bt.run()
                if len(res.ls_returns) == 0:
                    continue
                s = sharpe(res.ls_returns)
                if s > best_sharpe:
                    best_sharpe = s
                    best_trunc  = trunc
            except Exception:
                pass

        print(f"    {base_label}: trunc {config_entry[3]:.2f} → {best_trunc:.2f}  "
              f"(Sharpe {base_sharpe:.2f} → {best_sharpe:.2f}, Δ{best_sharpe - base_sharpe:+.2f})")
        h7_trunc[label] = best_trunc

    return h7_trunc


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

    fit = fitness(ls, result)
    mg  = margin(ls, result)
    rt  = returns_total(ls)
    fit_str = f"{fit:.2f}" if pd.notna(fit) else "N/A"
    mg_str  = f"{mg:.1f}"  if pd.notna(mg)  else "N/A"
    metrics_str = (
        f"Returns {rt:.2%}  |  Sharpe {sharpe(ls):.2f}  |  "
        f"Turnover {result.avg_turnover:.2%}  |  Fitness {fit_str}  |  Margin {mg_str}bps"
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
    df = pd.DataFrame(summary_rows).set_index("전략")
    df_num = df[["Sharpe", "Fitness", "Margin"]].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(df) * 0.5 + 2)))
    axes[0].barh(df_num.index, df_num["Sharpe"],
                 color=["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["Sharpe"]])
    axes[0].set_title("Sharpe Ratio", fontweight="bold")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(df_num.index, df_num["Fitness"],
                 color=["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["Fitness"]])
    axes[1].set_title("Fitness", fontweight="bold")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].grid(alpha=0.3, axis="x")

    axes[2].barh(df_num.index, df_num["Margin"],
                 color=["#1565C0" if v >= 0 else "#D32F2F" for v in df_num["Margin"]])
    axes[2].set_title("Margin (bps)", fontweight="bold")
    axes[2].axvline(0, color="black", lw=0.8)
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


# ── 가설 검증 PDF 생성 ───────────────────────────────────────────────────────

_C_BASE = "#9E9E9E"
_C_HYP  = "#1565C0"
_C_POS  = "#C8E6C9"
_C_NEG  = "#FFCDD2"
_C_HEAD = "#1a237e"


def _yearly(ls: pd.Series) -> pd.Series:
    return ls.groupby(ls.index.year).apply(lambda r: (1 + r).prod() - 1)


def _hyp_title_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#0D47A1")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0D47A1")
    ax.axis("off")
    ax.text(0.5, 0.88, "KR 시장 전략 저하 원인 — 7대 가설 검증 리포트",
            fontsize=22, color="white", ha="center", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.81, "WQ Brain 전략 적용 | KR 시장 백테스트 기간: 2019~2023",
            fontsize=13, color="#90CAF9", ha="center", transform=ax.transAxes)
    body = (
        "[ H1 ]  재벌 계열사 동조화  →  SubIndustry 수준 중립화로 그룹 공통 노출 제거\n"
        "[ H2 ]  개인투자자 단기 노이즈  →  Decay 단축으로 신호 반감기 조정\n"
        "[ H3 ]  외국인 패시브 시총 편향  →  Cap Bucket 5분위 중립화로 수급 편향 제거\n"
        "[ H4 ]  신호 희석 (과도한 분산)  →  롱숏 상하위 20% 농축 (WQ Brain 표준)\n"
        "[ H5 ]  분기 리밸런싱 불일치  →  반기(6개월) 리밸런싱으로 공시 주기 정렬\n"
        "[ H6 ]  소형주 유동성 노이즈  →  시총 상위 50% 대형주 유니버스 제한\n"
        "[ H7 ]  Truncation 불균등  →  전략별 최적 Truncation 교정으로 집중 리스크 제거"
    )
    ax.text(0.5, 0.46, body, fontsize=11, color="#E3F2FD",
            ha="center", va="center", linespacing=2.0,
            transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _comparison_page(pdf, hyp_label, neut_desc, base_label,
                     hyp_res, base_res, hyp_m, base_m):
    """원본 vs 가설 비교 페이지 (2x2 레이아웃)"""
    ls_b = base_res.ls_returns.dropna()
    ls_h = hyp_res.ls_returns.dropna()
    cum_b = (1 + ls_b).cumprod()
    cum_h = (1 + ls_h).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"{hyp_label}  vs  {base_label}  |  {neut_desc}",
                 fontsize=12, fontweight="bold", y=0.98)

    # 누적 수익률
    ax = axes[0, 0]
    ax.plot(cum_b.index, cum_b.values, color=_C_BASE, lw=1.5, linestyle="--",
            label=f"원본 Sharpe {base_m['sharpe']:+.2f}", alpha=0.85)
    ax.plot(cum_h.index, cum_h.values, color=_C_HYP, lw=2,
            label=f"가설 Sharpe {hyp_m['sharpe']:+.2f}")
    ax.axhline(1, color="black", lw=0.4, linestyle=":")
    ax.set_title("누적 수익률")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)

    # 연도별 수익률
    ax2 = axes[0, 1]
    yr_b = _yearly(ls_b)
    yr_h = _yearly(ls_h)
    years = sorted(set(yr_b.index) | set(yr_h.index))
    x = np.arange(len(years))
    w = 0.38
    ax2.bar(x - w/2, [yr_b.get(y, np.nan)*100 for y in years], w,
            color=_C_BASE, alpha=0.85, label="원본")
    ax2.bar(x + w/2, [yr_h.get(y, np.nan)*100 for y in years], w,
            color=_C_HYP, alpha=0.85, label="가설")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in years], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("수익률 (%)")
    ax2.set_title("연도별 수익률 비교")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    # Drawdown
    ax3 = axes[1, 0]
    dd_b = cum_b / cum_b.cummax() - 1
    dd_h = cum_h / cum_h.cummax() - 1
    ax3.fill_between(dd_b.index, dd_b.values, 0, alpha=0.2, color=_C_BASE, label="원본")
    ax3.plot(dd_h.index, dd_h.values, color=_C_HYP, lw=1.5, label="가설")
    ax3.set_title("Drawdown 비교")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, fontsize=7)

    # 지표 테이블 (WQ Brain 지표)
    ax4 = axes[1, 1]
    ax4.axis("off")
    ds  = hyp_m["sharpe"]   - base_m["sharpe"]
    drt = hyp_m["returns"]  - base_m["returns"]
    ddq = hyp_m["drawdown"] - base_m["drawdown"]
    dto = hyp_m["turnover"] - base_m["turnover"]
    bf  = base_m.get("fitness", np.nan)
    hf  = hyp_m.get("fitness",  np.nan)
    df_ = hf - bf if (pd.notna(bf) and pd.notna(hf)) else np.nan
    bm_ = base_m.get("margin", np.nan)
    hm_ = hyp_m.get("margin",  np.nan)
    dm_ = hm_ - bm_ if (pd.notna(bm_) and pd.notna(hm_)) else np.nan
    rows = [
        ["지표",      "원본",                       "가설",                       "변화"],
        ["Returns",  f"{base_m['returns']:+.2%}",  f"{hyp_m['returns']:+.2%}",  f"{drt:+.2%}"],
        ["Sharpe",   f"{base_m['sharpe']:.2f}",   f"{hyp_m['sharpe']:.2f}",    f"{ds:+.2f}"],
        ["Drawdown", f"{base_m['drawdown']:.2%}",  f"{hyp_m['drawdown']:.2%}",  f"{ddq:+.2%}"],
        ["Turnover", f"{base_m['turnover']:.2%}",  f"{hyp_m['turnover']:.2%}",  f"{dto:+.2%}"],
        ["Fitness",
         f"{bf:.2f}" if pd.notna(bf) else "N/A",
         f"{hf:.2f}" if pd.notna(hf) else "N/A",
         f"{df_:+.2f}" if pd.notna(df_) else "-"],
        ["Margin",
         f"{bm_:.1f}bps" if pd.notna(bm_) else "N/A",
         f"{hm_:.1f}bps" if pd.notna(hm_) else "N/A",
         f"{dm_:+.1f}bps" if pd.notna(dm_) else "-"],
    ]
    tbl = ax4.table(cellText=rows[1:], colLabels=rows[0],
                    bbox=[0.0, 0.05, 1.0, 0.90], cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j in range(4):
        tbl[0, j].set_facecolor(_C_HEAD)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    goods = [drt > 0, ds > 0, ddq > 0, dto < 0,
             (pd.notna(df_) and df_ > 0), (pd.notna(dm_) and dm_ > 0)]
    for i, is_pos in enumerate(goods, 1):
        tbl[i, 3].set_facecolor(_C_POS if is_pos else _C_NEG)
        for j in range(3):
            tbl[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

    verdict = "채택 (Sharpe 개선)" if ds > 0 else "기각 (Sharpe 악화)"
    vcolor  = "#1B5E20" if ds > 0 else "#B71C1C"
    ax4.text(0.5, 0.26, f"Sharpe 기준 판정: {verdict}",
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=vcolor, transform=ax4.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _h2_page(pdf, all_results, hyp_results_full=None):
    """H2: Decay 단축 백테스트 결과 + 산점도 (원본→단축 화살표)"""
    decay_data = []
    for label, res in all_results.items():
        if label not in STRATEGIES:
            continue
        d_n = STRATEGIES[label][1]
        ls  = res.ls_returns.dropna()
        if len(ls) == 0:
            continue
        decay_data.append({
            "label":   label,
            "decay":   d_n,
            "sharpe":  sharpe(ls),
            "fitness": fitness(ls, res),
        })
    if not decay_data:
        return
    decay_data.sort(key=lambda x: x["decay"])
    df_d = pd.DataFrame(decay_data)

    # H2 백테스트 결과 수집
    h2_data = {}
    if hyp_results_full:
        for lbl, info in hyp_results_full.items():
            if not lbl.startswith("H2"):
                continue
            bl = info["base_label"]
            h2_data[bl] = {
                "new_decay":   HYPOTHESIS_STRATEGIES[lbl][1],
                "new_sharpe":  info["metrics"]["sharpe"],
                "new_fitness": info["metrics"]["fitness"],
            }

    fig, axes = plt.subplots(1, 2, figsize=(11.69, 7))
    fig.suptitle("H2: 개인투자자 단기 노이즈 — Decay 단축 효과 검증",
                 fontsize=13, fontweight="bold", y=0.98)

    ax = axes[0]
    sc = ax.scatter(df_d["decay"], df_d["sharpe"],
                    c=df_d["sharpe"], cmap="RdYlGn",
                    vmin=-0.7, vmax=1.4, s=130, zorder=5)
    for _, row in df_d.iterrows():
        ax.annotate(row["label"][:6], (row["decay"], row["sharpe"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.5, alpha=0.85)

    # 화살표: 원본 Decay → 단축 Decay
    improve_cnt = 0
    for entry in decay_data:
        bl = entry["label"]
        if bl not in h2_data:
            continue
        h2 = h2_data[bl]
        improved = h2["new_sharpe"] > entry["sharpe"]
        if improved:
            improve_cnt += 1
        ax.annotate("", xy=(h2["new_decay"], h2["new_sharpe"]),
                    xytext=(entry["decay"], entry["sharpe"]),
                    arrowprops=dict(arrowstyle="->", lw=1.5,
                                   color="#1B5E20" if improved else "#B71C1C"))
        ax.scatter([h2["new_decay"]], [h2["new_sharpe"]], marker="D", s=70, zorder=6,
                   color="#1B5E20" if improved else "#B71C1C")

    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Decay 창 (일)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("원본(●)→단축(◆)  녹=개선 / 빨=악화")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Sharpe (원본)")

    ax2 = axes[1]
    ax2.axis("off")
    cols = ["전략", "원본\nDecay", "단축\nDecay", "ΔSharpe", "ΔFitness"]
    rows_tbl = []
    for entry in decay_data:
        bl = entry["label"]
        if bl not in h2_data:
            continue
        h2 = h2_data[bl]
        ds  = h2["new_sharpe"]  - entry["sharpe"]
        ef  = entry.get("fitness", np.nan)
        nf  = h2.get("new_fitness", np.nan)
        df_ = nf - ef if (pd.notna(nf) and pd.notna(ef)) else np.nan
        rows_tbl.append([bl[:12], str(int(entry["decay"])), str(int(h2["new_decay"])),
                         f"{ds:+.2f}", f"{df_:+.2f}" if pd.notna(df_) else "N/A"])

    if rows_tbl:
        tbl = ax2.table(cellText=rows_tbl, colLabels=cols,
                        bbox=[0.0, 0.28, 1.0, 0.68], cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for j in range(5):
            tbl[0, j].set_facecolor(_C_HEAD)
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, entry in enumerate(
                [e for e in decay_data if e["label"] in h2_data], 1):
            h2 = h2_data[entry["label"]]
            ds  = h2["new_sharpe"]  - entry["sharpe"]
            ef  = entry.get("fitness", np.nan)
            nf  = h2.get("new_fitness", np.nan)
            df_ = nf - ef if (pd.notna(nf) and pd.notna(ef)) else np.nan
            tbl[i, 3].set_facecolor(_C_POS if ds > 0 else _C_NEG)
            if pd.notna(df_):
                tbl[i, 4].set_facecolor(_C_POS if df_ > 0 else _C_NEG)
            for j in [0, 1, 2]:
                tbl[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

    total = len(rows_tbl)
    verdict = (f"가설 {'채택' if improve_cnt > total / 2 else '기각'}: "
               f"{improve_cnt}/{total}개 전략에서 Sharpe 개선")
    ax2.text(0.5, 0.24, f"[ H2 결론 ]\n{verdict}",
             ha="center", va="top", fontsize=9, linespacing=1.7,
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _hyp_summary_page(pdf, hyp_rows, hyp_tags, title_suffix):
    """가설 검증 전체 결과 요약 테이블 (지정 가설 그룹)"""
    n = len(hyp_tags)
    fig, axes = plt.subplots(n, 1, figsize=(11.69, 4.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"가설 검증 전체 결과 요약 — {title_suffix}", fontsize=14,
                 fontweight="bold", y=0.99)

    all_titles = {
        "H1": "H1: 재벌 계열사 동조화 (SubIndustry 중립화)",
        "H2": "H2: 개인투자자 단기 노이즈 (Decay 단축 검증)",
        "H3": "H3: 외국인 패시브 시총 편향 (Cap Bucket 중립화)",
        "H4": "H4: 신호 희석 (롱숏 상하위 20% 농축)",
        "H5": "H5: 반기 리밸런싱 (분기 → 6개월)",
        "H6": "H6: 소형주 유동성 노이즈 (대형주 유니버스)",
        "H7": "H7: 전략별 최적 Truncation 교정 (그리드서치)",
    }
    for hyp_tag, ax in zip(hyp_tags, axes):
        ax.axis("off")
        ax.set_title(all_titles.get(hyp_tag, hyp_tag), fontsize=10, fontweight="bold", pad=5)

        sub = [r for r in hyp_rows if r["가설"] == hyp_tag]
        if not sub:
            continue

        cols = ["전략", "Sharpe 원본", "Sharpe 가설", "ΔSharpe",
                "Fitness 원본", "Fitness 가설", "ΔFitness",
                "Margin 원본", "Margin 가설", "ΔMargin"]
        data = []
        for r in sub:
            data.append([
                r["전략"].replace("H1_", "").replace("H2_", "").replace("H3_", ""),
                r["Sharpe_base"], r["Sharpe_hyp"], r["ΔSharpe"],
                r["Fitness_base"], r["Fitness_hyp"], r["ΔFitness"],
                r["Margin_base"], r["Margin_hyp"], r["ΔMargin"],
            ])

        tbl = ax.table(cellText=data, colLabels=cols,
                       bbox=[0.0, 0.0, 1.0, 0.92], cellLoc="center")
        tbl.auto_set_font_size(False)
        n_rows = len(data)
        tbl.set_fontsize(6.5 if n_rows > 8 else 7.5)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor(_C_HEAD)
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, row in enumerate(data, 1):
            for col_idx in [3, 6, 9]:  # ΔSharpe, ΔFitness, ΔMargin — 높을수록 좋음
                try:
                    v = float(row[col_idx].replace("%", "").replace("+", ""))
                    tbl[i, col_idx].set_facecolor(_C_POS if v > 0 else _C_NEG)
                except Exception:
                    pass
            for j in [0, 1, 2, 4, 5, 7, 8]:
                tbl[i, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_hypothesis_pdf(pdf_path, all_results, hyp_results_full, hyp_rows):
    """가설 검증 리포트 PDF 생성"""
    print(f"\n  PDF 생성: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        _hyp_title_page(pdf)
        print("  표지 완료")

        # H1: SubIndustry 중립화
        for label in [k for k in hyp_results_full if k.startswith("H1")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            _comparison_page(pdf, label, "SubIndustry 중립화", bl,
                             info["res"], br, info["metrics"], bm)
        _hyp_summary_page(pdf, hyp_rows, ["H1"], "H1: 재벌 계열사 동조화")
        print("  H1 페이지 + 요약 완료")

        # H2: Decay 단축
        for label in [k for k in hyp_results_full if k.startswith("H2")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            orig_d = STRATEGIES[bl][1] if bl in STRATEGIES else "?"
            new_d  = HYPOTHESIS_STRATEGIES[label][1]
            _comparison_page(pdf, label, f"Decay 단축 ({orig_d}→{new_d})", bl,
                             info["res"], br, info["metrics"], bm)
        _h2_page(pdf, all_results, hyp_results_full)
        _hyp_summary_page(pdf, hyp_rows, ["H2"], "H2: 개인투자자 단기 노이즈")
        print("  H2 페이지 + 요약 완료")

        # H3: Cap Bucket 중립화
        for label in [k for k in hyp_results_full if k.startswith("H3")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            _comparison_page(pdf, label, "Cap Bucket 중립화", bl,
                             info["res"], br, info["metrics"], bm)
        _hyp_summary_page(pdf, hyp_rows, ["H3"], "H3: 외국인 패시브 시총 편향")
        print("  H3 페이지 + 요약 완료")

        # H4: 롱숏 집중도
        for label in [k for k in hyp_results_full if k.startswith("H4")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            _comparison_page(pdf, label, "롱숏 집중도 (50%→20%)", bl,
                             info["res"], br, info["metrics"], bm)
        _hyp_summary_page(pdf, hyp_rows, ["H4"], "H4: 신호 희석 (롱숏 농축)")
        print("  H4 페이지 + 요약 완료")

        # H5: 반기 리밸런싱
        for label in [k for k in hyp_results_full if k.startswith("H5")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            _comparison_page(pdf, label, "반기 리밸런싱 (분기→6개월)", bl,
                             info["res"], br, info["metrics"], bm)
        _hyp_summary_page(pdf, hyp_rows, ["H5"], "H5: 분기 리밸런싱 불일치")
        print("  H5 페이지 + 요약 완료")

        # H6: 대형주 유니버스
        for label in [k for k in hyp_results_full if k.startswith("H6")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            _comparison_page(pdf, label, "대형주 유니버스 (시총 상위 50%)", bl,
                             info["res"], br, info["metrics"], bm)
        _hyp_summary_page(pdf, hyp_rows, ["H6"], "H6: 소형주 유동성 노이즈")
        print("  H6 페이지 + 요약 완료")

        # H7: 전략별 최적 Truncation 교정
        for label in [k for k in hyp_results_full if k.startswith("H7")]:
            info = hyp_results_full[label]
            bl   = info["base_label"]
            if bl not in all_results:
                continue
            br = all_results[bl]
            orig_trunc  = STRATEGIES[bl][3] if bl in STRATEGIES else "?"
            opt_trunc   = info.get("trunc", "?")
            bm = {
                "returns":  returns_total(br.ls_returns),
                "sharpe":   sharpe(br.ls_returns),
                "drawdown": drawdown_wq(br.ls_returns),
                "turnover": br.avg_turnover,
                "fitness":  fitness(br.ls_returns, br),
                "margin":   margin(br.ls_returns, br),
            }
            desc = (f"Truncation ({orig_trunc:.2f}→{opt_trunc:.2f})"
                    if isinstance(opt_trunc, float) else f"Truncation ({orig_trunc:.2f}→?)")
            _comparison_page(pdf, label, desc, bl,
                             info["res"], br, info["metrics"], bm)
        _hyp_summary_page(pdf, hyp_rows, ["H7"], "H7: Truncation 불균등")
        print("  H7 페이지 + 요약 완료")

    print(f"  저장 완료: {pdf_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(strategy_filter=None):
    print("KR 데이터 로드 중...")
    price_pivot, panel, universe, sector_s, industry_s, subindustry_s = load_kr()

    # 백테스트 기간 필터
    price_pivot = price_pivot.loc[BT_START:BT_END]
    panel = panel[(panel["date"] >= BT_START) & (panel["date"] <= BT_END)].copy().reset_index(drop=True)

    print(f"  종목 수: {price_pivot.shape[1]}, 기간: "
          f"{price_pivot.index[0].date()} ~ {price_pivot.index[-1].date()}")
    print(f"  sector 있는 종목: {len(sector_s)}, industry 있는 종목: {len(industry_s)}")
    print(f"  panel 컬럼: {list(panel.columns)}")

    all_results = {}
    summary_rows = []

    for label, (fn, decay_n, neut_level, trunc) in STRATEGIES.items():
        if strategy_filter and label != strategy_filter:
            continue
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

        ls  = res.ls_returns
        c   = cagr(ls)
        s   = sharpe(ls)
        m   = mdd(ls)
        rt  = returns_total(ls)
        dq  = drawdown_wq(ls)
        t   = res.avg_turnover
        fit = fitness(ls, res)
        mg  = margin(ls, res)
        fit_str = f"{fit:.2f}" if pd.notna(fit) else "N/A"
        mg_str  = f"{mg:.1f}" if pd.notna(mg) else "N/A"
        print(f"  Returns {rt:.2%}  Sharpe {s:.2f}  Turnover {t:.2%}  "
              f"Fitness {fit_str}  Margin {mg_str}bps")

        all_results[label] = res
        summary_rows.append({
            "전략": label, "전략명": name,
            "Returns": rt,
            "CAGR": c, "Sharpe": s, "MDD": m, "Drawdown": dq,
            "Turnover": t, "Fitness": fit, "Margin": mg,
        })
        plot_single(res, name, label)

    if not all_results:
        print("\n실행된 전략 없음.")
        return

    print("\n비교 차트 생성 중...")
    plot_cumulative_all(all_results)
    plot_summary(summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    for col, fmt in [
        ("Returns", "{:.2%}"), ("CAGR", "{:.2%}"), ("Sharpe", "{:.2f}"),
        ("MDD", "{:.2%}"), ("Drawdown", "{:.2%}"), ("Turnover", "{:.2%}"),
        ("Fitness", "{:.2f}"), ("Margin", "{:.1f}"),
    ]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map(
                lambda x, f=fmt: f.format(x) if pd.notna(x) else "N/A"
            )
    csv_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n요약 CSV 저장: {csv_path}")
    print(f"\n[완료] 결과: {RESULTS_DIR}")
    print(summary_df.to_string(index=False))

    # ── 가설 검증 (H1: SubIndustry / H3: CapBucket) ──────────────────────────
    print("\n\n" + "=" * 65)
    print("  가설 검증: H1=SubIndustry 중립화 / H3=CapBucket 중립화")
    print("=" * 65)

    # H3에 필요한 시총 DataFrame 계산
    shares_p = panel.pivot_table(index="date", columns="ticker", values="shares").sort_index()
    cap_df = (price_pivot * shares_p.reindex(columns=price_pivot.columns).ffill(limit=252)
              ).replace(0, np.nan)
    print(f"  시총 DataFrame: {cap_df.shape}")

    # H7: 전략별 최적 truncation 그리드서치 후 반영
    h7_optimal = _find_h7_truncations(
        all_results, panel, price_pivot, universe,
        sector_s, industry_s, subindustry_s, cap_df,
    )
    effective_hyp = {}
    for lbl, cfg_entry in HYPOTHESIS_STRATEGIES.items():
        if lbl.startswith("H7") and lbl in h7_optimal:
            entry = list(cfg_entry)
            entry[3] = h7_optimal[lbl]
            effective_hyp[lbl] = tuple(entry)
        else:
            effective_hyp[lbl] = cfg_entry

    # 기준선: H1_SXX → SXX, H3_SXX → SXX 자동 매핑
    BASE_MAP = {hyp: hyp[3:] for hyp in effective_hyp}  # "H1_" 또는 "H3_" 제거

    hyp_rows = []
    hyp_results_full = {}
    for label, config_entry in effective_hyp.items():
        fn, decay_n, neut_level, trunc = config_entry[:4]
        extra = config_entry[4] if len(config_entry) > 4 else {}

        if strategy_filter and label != strategy_filter and not label.startswith(strategy_filter):
            continue
        hyp_tag = label[:2]  # "H1" ~ "H7"
        long_pct   = extra.get("long_pct",  0.5)
        short_pct  = extra.get("short_pct", 0.5)
        rebalance  = extra.get("rebalance", "QE")
        uni_filter = extra.get("universe",  None)
        print(f"\n[{label}]  (Decay={decay_n}, Neut={neut_level}, Trunc={trunc}"
              + (f", L/S={int(long_pct*100)}%" if long_pct != 0.5 else "")
              + (f", Reb={rebalance}" if rebalance != "QE" else "")
              + (f", Uni={uni_filter}" if uni_filter else "")
              + ")")
        try:
            raw_f = fn(panel, price_pivot)
        except Exception as e:
            print(f"  → 예외: {e}")
            continue
        if raw_f is None:
            print(f"  → 건너뜀 (데이터 없음)")
            continue

        factor = build_factor(raw_f, decay_n, neut_level, trunc,
                              sector_s, industry_s, panel,
                              subindustry_s=subindustry_s, cap_df=cap_df)
        if factor.notna().sum().sum() == 0:
            print(f"  → 유효 팩터값 없음, 건너뜀")
            continue

        # H6: 대형주 유니버스 필터 (시총 중위수 이상 종목만 유지)
        if uni_filter == "largecap":
            cap_aligned = cap_df.reindex(index=factor.index, columns=factor.columns)
            median_cap  = cap_aligned.median(axis=1)
            cap_mask    = cap_aligned.ge(median_cap, axis=0)
            factor      = factor.where(cap_mask, np.nan)

        cfg = Config(long_pct=long_pct, short_pct=short_pct,
                     rebalance=rebalance, truncation=trunc, score_weight=True)
        bt  = LongShortBacktester(price=price_pivot, factor=factor,
                                  universe=universe, config=cfg)
        res = bt.run()
        if len(res.ls_returns) == 0:
            print(f"  → 수익률 없음")
            continue

        ls  = res.ls_returns
        s   = sharpe(ls)
        rt  = returns_total(ls)
        dq  = drawdown_wq(ls)
        to  = res.avg_turnover
        fit = fitness(ls, res)
        mg  = margin(ls, res)
        fit_str = f"{fit:.2f}" if pd.notna(fit) else "N/A"
        mg_str  = f"{mg:.1f}"  if pd.notna(mg)  else "N/A"
        base_label = BASE_MAP.get(label, "")
        base_res   = all_results.get(base_label)
        if base_res:
            bs   = sharpe(base_res.ls_returns)
            bfit = fitness(base_res.ls_returns, base_res)
            bmg  = margin(base_res.ls_returns, base_res)
            ds   = s - bs
            dfit = fit - bfit if (pd.notna(fit) and pd.notna(bfit)) else None
            dmg  = mg  - bmg  if (pd.notna(mg)  and pd.notna(bmg))  else None
            print(f"  Sharpe {s:.2f} (Δ{ds:+.2f})  Fitness {fit_str}  Margin {mg_str}bps")
        else:
            bs, bfit, bmg, ds, dfit, dmg = None, None, None, None, None, None
            print(f"  Sharpe {s:.2f}  Fitness {fit_str}  Margin {mg_str}bps")

        hyp_rows.append({
            "가설": hyp_tag, "전략": label, "기준": base_label,
            "Sharpe_base":  f"{bs:.2f}"   if bs   is not None else "-",
            "Sharpe_hyp":   f"{s:.2f}",
            "ΔSharpe":      f"{ds:+.2f}"  if ds   is not None else "-",
            "Fitness_base": f"{bfit:.2f}" if bfit  is not None else "-",
            "Fitness_hyp":  fit_str,
            "ΔFitness":     f"{dfit:+.2f}" if dfit is not None else "-",
            "Margin_base":  f"{bmg:.1f}"  if bmg   is not None else "-",
            "Margin_hyp":   mg_str,
            "ΔMargin":      f"{dmg:+.1f}" if dmg   is not None else "-",
        })
        hyp_results_full[label] = {
            "res":        res,
            "metrics":    {
                "returns":  rt,
                "sharpe":   s,
                "drawdown": dq,
                "turnover": to,
                "fitness":  fit,
                "margin":   mg,
            },
            "base_label": base_label,
            "neut_level": neut_level,
            "trunc":      trunc,
        }

    if hyp_rows:
        print("\n\n[가설 검증 결과 요약]")
        hyp_df = pd.DataFrame(hyp_rows)
        print(hyp_df.to_string(index=False))
        hyp_path = RESULTS_DIR / "hypothesis_results.csv"
        hyp_df.to_csv(hyp_path, index=False, encoding="utf-8-sig")
        print(f"\n가설 검증 CSV 저장: {hyp_path}")

    if hyp_results_full:
        hyp_pdf_path = RESULTS_DIR / "hypothesis_report.pdf"
        make_hypothesis_pdf(hyp_pdf_path, all_results, hyp_results_full, hyp_rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KR 시장 전략 백테스트 (WQ Brain 스타일)")
    parser.add_argument(
        "--strategy", "-s", default=None,
        help="특정 전략 하나만 실행 (예: -s S02_OEY  또는  -s H1_S02_OEY)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="전략 목록 출력 후 종료",
    )
    args = parser.parse_args()

    if args.list:
        print("=== 기본 전략 (15개) ===")
        for k, (_, d, n, t) in STRATEGIES.items():
            print(f"  {k:35s}  Decay={d:2d}  Neut={n}")
        print("\n=== 가설 검증 전략 (9개) ===")
        for k, (_, d, n, t) in HYPOTHESIS_STRATEGIES.items():
            print(f"  {k:35s}  Decay={d:2d}  Neut={n}")
    else:
        main(strategy_filter=args.strategy)
