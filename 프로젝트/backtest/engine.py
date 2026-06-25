"""
Phase 3  백테스팅 루프
- 전략: 롱숏, 상위 20% 롱 / 하위 20% 숏, 동일가중
- 리밸런싱: 분기말
- 거래비용: 없음
- 상장폐지 처리: 상폐 직전까지 수익률 반영 후 잔여 비중 균등 재배분
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import LONG_PCT, SHORT_PCT, REBALANCE


@dataclass
class Config:
    long_pct:  float = LONG_PCT
    short_pct: float = SHORT_PCT
    rebalance: str   = REBALANCE   # "QE" = 분기말


class LongShortBacktester:
    """
    팩터 기반 롱숏 백테스터

    Parameters
    ----------
    price  : (date × ticker) 수정종가 — datetime 인덱스
    factor : (date × ticker) z-score 팩터 — 클수록 좋은 종목
             NaN = 해당 날짜에 유니버스 제외
    universe : DataFrame(date, ticker) — 분기별 유효 유니버스
               None이면 factor 비-NaN 종목 전체 사용
    config : Config
    """

    def __init__(self,
                 price: pd.DataFrame,
                 factor: pd.DataFrame,
                 universe: pd.DataFrame | None = None,
                 config: Config | None = None):
        self.price     = price.sort_index()
        self.factor    = factor.sort_index()
        self.universe  = universe
        self.cfg       = config or Config()
        self._trade_dates = pd.DatetimeIndex(self.price.index)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _rebal_dates(self) -> pd.DatetimeIndex:
        """분기말 날짜를 실제 거래일에 맞춤"""
        qe = pd.date_range(
            self._trade_dates[0], self._trade_dates[-1], freq=self.cfg.rebalance
        )
        aligned = []
        for d in qe:
            later = self._trade_dates[self._trade_dates >= d]
            if len(later):
                aligned.append(later[0])
        return pd.DatetimeIndex(sorted(set(aligned)))

    def _get_universe_at(self, date: pd.Timestamp) -> list[str]:
        """특정 날짜 유니버스 종목 목록"""
        if self.universe is not None:
            uni = self.universe[self.universe["date"] <= date]
            if not uni.empty:
                return uni[uni["date"] == uni["date"].max()]["ticker"].tolist()
        # 팩터 비-NaN 종목 전체
        f = self.factor.loc[self.factor.index <= date]
        if f.empty:
            return []
        return f.iloc[-1].dropna().index.tolist()

    def _get_factor_at(self, date: pd.Timestamp,
                       universe: list[str]) -> pd.Series:
        """특정 날짜 팩터 값 (유니버스 내 + 가격 존재 종목)"""
        f_hist = self.factor.loc[self.factor.index <= date]
        if f_hist.empty:
            return pd.Series(dtype=float)
        f = f_hist.iloc[-1]

        # 가격 존재 확인
        p_hist = self.price.loc[self.price.index <= date]
        if p_hist.empty:
            return pd.Series(dtype=float)
        p = p_hist.iloc[-1].dropna()

        valid = [t for t in universe if t in f.index and t in p.index and pd.notna(f[t])]
        return f[valid]

    # ── 메인 실행 ────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        rebal_dates = self._rebal_dates()
        all_daily   = []
        all_turnover = []
        prev_long  = pd.Index([])
        prev_short = pd.Index([])

        for i, rb in enumerate(rebal_dates):
            uni = self._get_universe_at(rb)
            f   = self._get_factor_at(rb, uni)
            if len(f) < 10:
                continue

            n = len(f)
            n_long  = max(1, round(n * self.cfg.long_pct))
            n_short = max(1, round(n * self.cfg.short_pct))

            ranked    = f.sort_values(ascending=False)
            long_idx  = ranked.iloc[:n_long].index
            short_idx = ranked.iloc[-n_short:].index

            # 턴오버: 변경된 종목 비율
            if len(prev_long):
                changed = len(set(long_idx) - set(prev_long)) / n_long
                all_turnover.append({"date": rb, "turnover": changed})
            prev_long, prev_short = long_idx, short_idx

            # 보유 기간 결정
            next_rb = (
                rebal_dates[i + 1]
                if i + 1 < len(rebal_dates)
                else self._trade_dates[-1]
            )
            period = self.price[
                (self.price.index > rb) & (self.price.index <= next_rb)
            ]
            if len(period) < 2:
                continue

            # 일별 수익률 계산
            rets = period.pct_change().fillna(0).iloc[1:]

            for date, row in rets.iterrows():
                # 상장폐지: 가격이 있는 종목만 유지, 나머지 균등 재배분
                l_avail = [t for t in long_idx  if t in row.index and not pd.isna(period.loc[date, t])]
                s_avail = [t for t in short_idx if t in row.index and not pd.isna(period.loc[date, t])]

                l_ret = row[l_avail].mean() if l_avail else 0.0
                s_ret = row[s_avail].mean() if s_avail else 0.0

                all_daily.append({
                    "date":      date,
                    "long_ret":  l_ret,
                    "short_ret": s_ret,
                    "ls_ret":    l_ret - s_ret,
                })

        ret_df  = (pd.DataFrame(all_daily).set_index("date").sort_index()
                   if all_daily else pd.DataFrame())
        turn_df = (pd.DataFrame(all_turnover).set_index("date")
                   if all_turnover else pd.DataFrame())

        return BacktestResult(ret_df, turn_df)


# ── 결과 컨테이너 ────────────────────────────────────────────────────────────

class BacktestResult:
    def __init__(self, ret_df: pd.DataFrame, turn_df: pd.DataFrame):
        self.ret_df  = ret_df
        self.turn_df = turn_df

    @property
    def ls_returns(self) -> pd.Series:
        return self.ret_df["ls_ret"] if "ls_ret" in self.ret_df else pd.Series(dtype=float)

    @property
    def long_returns(self) -> pd.Series:
        return self.ret_df["long_ret"] if "long_ret" in self.ret_df else pd.Series(dtype=float)

    @property
    def short_returns(self) -> pd.Series:
        return self.ret_df["short_ret"] if "short_ret" in self.ret_df else pd.Series(dtype=float)

    @property
    def avg_turnover(self) -> float:
        return self.turn_df["turnover"].mean() if not self.turn_df.empty else np.nan
