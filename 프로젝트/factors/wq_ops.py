"""
WQ Brain 연산자 라이브러리
입력/출력: (date × ticker) DataFrame (피벗 형태)
groups: ticker → group name Series
"""
import numpy as np
import pandas as pd


# ── 횡단면 연산 ──────────────────────────────────────────────────────────────

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """횡단면 rank → 0~1"""
    return df.rank(axis=1, pct=True, na_option="keep")


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """횡단면 z-score"""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1)
    return df.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)


def bucket(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """횡단면 분위 → 0~(n-1) 정수"""
    def _row(row):
        valid = row.dropna()
        if len(valid) < n:
            return row * np.nan
        try:
            q = pd.qcut(valid, n, labels=False, duplicates="drop")
        except ValueError:
            return row * np.nan
        out = row.copy() * np.nan
        out[valid.index] = q
        return out
    return df.apply(_row, axis=1)


def group_neutralize(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """각 날짜에서 그룹 평균 제거"""
    result = df.copy()
    for g in groups.unique():
        cols = groups[groups == g].index.intersection(df.columns)
        if len(cols) == 0:
            continue
        gm = df[cols].mean(axis=1)
        result[cols] = df[cols].sub(gm, axis=0)
    return result


def group_rank(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """각 날짜에서 그룹 내 rank → 0~1"""
    result = df.copy() * np.nan
    for g in groups.unique():
        cols = groups[groups == g].index.intersection(df.columns)
        if len(cols) < 2:
            continue
        result[cols] = df[cols].rank(axis=1, pct=True, na_option="keep")
    return result


def group_neutralize_dynamic(df: pd.DataFrame,
                              group_df: pd.DataFrame) -> pd.DataFrame:
    """날짜마다 그룹이 달라지는 동적 중립화 (날짜별 loop)"""
    result = df.copy() * np.nan
    common_dates = df.index.intersection(group_df.index)
    for date in common_dates:
        row = df.loc[date]
        groups = group_df.loc[date].dropna()
        for g in groups.unique():
            cols = groups[groups == g].index.intersection(row.index)
            vals = row[cols].dropna()
            if len(vals) == 0:
                continue
            result.loc[date, vals.index] = vals - vals.mean()
    return result


# ── 시계열 연산 ──────────────────────────────────────────────────────────────

def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(1, n // 2)).mean()


def ts_std(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(1, n // 2)).std()


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """시계열 rank over n 기간 → 0~1"""
    return df.rolling(n, min_periods=max(1, n // 2)).rank(pct=True)


def ts_zscore(df: pd.DataFrame, n: int) -> pd.DataFrame:
    mu = df.rolling(n, min_periods=max(1, n // 2)).mean()
    sigma = df.rolling(n, min_periods=max(1, n // 2)).std()
    return (df - mu) / sigma.replace(0, np.nan)


def ts_delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df - df.shift(n)


def ts_delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.shift(n)


# ── 수학 연산 ────────────────────────────────────────────────────────────────

def signed_power(df: pd.DataFrame, exp: float) -> pd.DataFrame:
    return df.abs() ** exp * np.sign(df)


def df_max(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.maximum(a.values, b.values),
        index=a.index, columns=a.columns
    )


def group_zscore(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """각 날짜에서 그룹 내 z-score"""
    result = df.copy() * np.nan
    for g in groups.unique():
        cols = groups[groups == g].index.intersection(df.columns)
        if len(cols) < 2:
            continue
        gm  = df[cols].mean(axis=1)
        gsd = df[cols].std(axis=1).replace(0, np.nan)
        result[cols] = df[cols].sub(gm, axis=0).div(gsd, axis=0)
    return result


def ts_corr(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    """시계열 rolling 상관계수"""
    result = a.copy() * np.nan
    for col in a.columns.intersection(b.columns):
        result[col] = a[col].rolling(n, min_periods=n // 2).corr(b[col])
    return result


def ts_backfill(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """NaN을 최근 유효값으로 채움 (최대 n 기간)"""
    return df.fillna(method="ffill", limit=n)


def trade_when(condition: pd.DataFrame, signal: pd.DataFrame,
               otherwise) -> pd.DataFrame:
    """condition True일 때 signal, 아닐 때 otherwise 반환"""
    if isinstance(otherwise, (int, float)):
        alt = pd.DataFrame(otherwise, index=signal.index, columns=signal.columns)
    else:
        alt = otherwise
    return signal.where(condition, other=alt)
