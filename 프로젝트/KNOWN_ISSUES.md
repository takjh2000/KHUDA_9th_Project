# 알려진 문제 — KR 백테스트 데이터 커버리지

## 문제 요약

`run_kr_backtest.py`의 15개 전략 중 다수가 재무 데이터(DART 기반 `kr_extra_finance.parquet`)의
필드 결측 때문에, 실제로는 최근 1~2년치 데이터로만 성과가 계산되고 있음.
`results/kr_custom/summary.csv`의 CAGR/Sharpe 수치가 다년간 백테스트처럼 보이지만,
전략에 따라 실질 표본 기간이 크게 다름.

## 확인 방법

`build_factor()` 적용 후(decay/neutralize/truncate까지 마친) 팩터의 연도별 평균 유효 종목 수를 집계.
`LongShortBacktester`는 유효 종목 10개 미만인 리밸런싱일을 건너뛰므로(`engine.py:131` `if len(f) < 10: continue`),
연평균 유효 종목 수가 10 미만인 해는 사실상 백테스트에 기여하지 않음.

```python
valid_count = f.notna().sum(axis=1)
yearly = valid_count.groupby(valid_count.index.year).mean()
```

## 결과 (2025-07-01 기준 확인)

| 전략 | 2017년 평균 유효종목수 | 2020년 | 2024년 | 실질 사용 가능 시작연도 |
|---|---|---|---|---|
| S03_CashCFDivergence | 161 | 227 | 212 | 2017 (정상) |
| S06_DebtSpikeReversal | 93 | 91 | 92 | 2017 (정상) |
| S10_ProfitableBuyback | 56 | 61 | 67 | 2017 (정상) |
| S11_SGADriven | 46 | 55 | 58 | 2017 (정상) |
| S14_VolRegimeDebt | 27 | 6 | 23 | 2017 (정상, 2020년 표본 얇음 주의) |
| S01_LowAccrual | 0 | 0 | 233 | **2024부터만 (사실상 1년)** |
| S02_OEY | 0 | 0 | 52 | **2024부터만** |
| S05_IndustryNeutralCFY | 0 | 0 | 209 | **2024부터만** |
| S07_AggressiveDualValue | 0 | 0 | 294 | **2024부터만** |
| S12_GoodwillOverval | 0 | 0 | 336 | **2024부터만** |
| S15_BookToCapMom | 0 | 0 | 248 | **2024부터만** |
| S04_GrowthWeightedLT | 0 | 0 | 0 | **전 기간 계산 안 됨 (버그 의심)** |

## 원인 추정 (미확인)

S01/S02/S05/S07/S12/S15는 공통적으로 `equity`, `bps`(또는 `cashflow_op`, `total_assets` 등)
2개 이상의 필드를 동시에 조합해서 "실제 시가총액"을 근사하거나 비율을 계산함.
`kr_extra_finance.parquet`의 필드별 non-null 비율은 개별적으로는 55~78%로 낮지 않은데,
**여러 필드의 교집합**이 최근 1~2년에만 형성되는 것으로 보임 — DART 데이터 수집 시점이나
`merge_asof` 병합 로직(`run_kr_backtest.py:56-61`)에서 과거 연도 데이터가 제대로 안 붙는
파이프라인 문제일 가능성이 있음. 실제 데이터 부족인지 병합 버그인지는 추가 확인 필요.

S04는 완전히 0 — `sps`/`total_assets` 필드 자체 결측이거나 `ts_delta`/`shift(252)` 조합에서
NaN이 전파되는 구조적 문제로 추정.

## 대응

- 이번 E-1/E-2/E-3 가설 검증은 **표본이 안정적인 5개 전략(S03, S06, S10, S11, S14)에 한정**해서 진행.
- S01/S02/S05/S07/S12/S15/S04는 재무 데이터 병합 로직을 별도로 재점검하기 전까지 결과 신뢰 보류.
