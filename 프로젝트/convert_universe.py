import pandas as pd
from pathlib import Path

src = Path(__file__).parent / "data" / "raw" / "sp500_historical.csv"
dst = Path(__file__).parent / "data" / "raw" / "us_universe.parquet"
dst.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
df["ticker"] = df["ticker"].str.split(",")
universe = df.explode("ticker")
universe["ticker"] = universe["ticker"].str.strip()
universe["date"] = pd.to_datetime(universe["date"])
universe = universe.reset_index(drop=True)

universe.to_parquet(dst, index=False)
print(f"저장 완료: {dst}")
print(f"행 수: {len(universe):,}  |  기간: {universe['date'].min().date()} ~ {universe['date'].max().date()}")
print(universe.head())
