# KHUDA 9기 프로젝트 — 퀀트 팩터 백테스팅

한국(KOSPI 200) 및 미국(S&P 500) 주식시장 대상 펀더멘털 팩터 롱숏 백테스팅.
추가로 WorldQuant Brain 스타일 전략 11개를 한국 시장에 적용하고, 성과 저하 원인에 대한 가설 11개를 백테스트로 검증합니다.

---

## 프로젝트 구조

```
프로젝트/
├── config.py                   # 경로, 기간, 유니버스 설정
├── run_pipeline.py             # Step 1: 데이터 전처리 파이프라인
├── run_backtest.py             # Step 2: 5대 팩터 롱숏 백테스트
├── run_wq_all.py               # Step 3: WQ Brain 11개 전략 백테스트
├── run_hypothesis_test.py      # Step 4: 11개 가설 검증
│
├── data/
│   ├── build_panel.py          # 가격 + 재무 데이터 병합 → 패널 parquet
│   └── pipeline/
│       ├── kr_price.py         # KOSPI 200 과거 구성종목 + 가격 (FDR/pykrx)
│       ├── kr_finance.py       # KR 재무 (DART → pykrx → 네이버 폴백)
│       ├── us_price.py         # S&P 500 과거 구성종목 + 가격 (yfinance)
│       └── us_finance.py       # US 재무 (yfinance)
│
├── factors/
│   ├── compute.py              # 5대 팩터: BP, EP, SP, ROE, GPA
│   └── wq_ops.py               # WQ Brain 연산: rank, zscore, ts_*, group_* 등
│
├── backtest/
│   ├── engine.py               # LongShortBacktester (상/하위 20%, 분기 리밸런싱)
│   └── metrics.py              # CAGR, Sharpe, MDD
│
├── analysis/
│   └── compare.py              # KR vs US 비교 차트
│
└── sp500_historical.csv        # S&P 500 과거 구성종목 (date, tickers)
```

---

## 실행 방법

```bash
# 1. 패키지 설치
pip install FinanceDataReader yfinance pykrx pandas numpy matplotlib

# 2. 데이터 파이프라인 실행
python run_pipeline.py          # KR + US 전체
python run_pipeline.py --kr     # KR만
python run_pipeline.py --us     # US만

# 3-A. 5대 팩터 백테스트
python run_backtest.py --market KR --factor BP
python run_backtest.py --market US --factor EP

# 3-B. WQ Brain 11개 전략 백테스트 (한국 시장)
python run_wq_all.py
# → results/WQ_Brain_KR_종합분석_리포트.pdf

# 3-C. 가설 검증 백테스트 (한국 시장)
python run_hypothesis_test.py
# → results/WQ_가설검증_리포트.pdf
```

> **참고:** KR 재무 데이터 전체 수집을 위해서는 DART API 키가 필요합니다.
> 실행 전 환경변수 `DART_API_KEY`를 설정하세요.
> 키가 없으면 pykrx → 네이버 금융 순으로 자동 폴백됩니다.

---

## Part 1 — 5대 팩터 롱숏 백테스트

상위 20% 롱 / 하위 20% 숏, 분기 리밸런싱 (2010–2024).

| 팩터 | 설명 |
|------|------|
| **BP** | Book-to-Price — 저평가 가치주 |
| **EP** | Earnings-to-Price — 이익 수익률 |
| **SP** | Sales-to-Price — 매출 수익률 |
| **ROE** | Return on Equity — 자본 효율성 |
| **GPA** | Gross Profit-to-Assets — Novy-Marx 수익성 |

전처리: 상하위 1% 윈저라이징 → 횡단면 z-score 정규화

---

## Part 2 — WQ Brain 11개 전략 분석 (한국 시장)

WorldQuant Brain 스타일의 알파 전략 11개를 KOSPI 200 대상으로 백테스트 (2016–2024).

| # | 전략명 | 신호 로직 |
|---|--------|----------|
| 1 | LowAccrual | `-(eps - ops) / bps`, 시총 중립화 |
| 2 | OpIncEY | `ops / close`, 업종 내 상대 순위 |
| 5 | CFYield | `group_rank(ts_zscore(ops/close, 63), 업종)` |
| 6 | DebtSpike | `signed_power(-ts_zscore(debt_ps, 252), 4)` |
| 7 | DualValue | `max(EBITDA_rank, PBR_rank)` + 모멘텀 중립화 |
| 10 | Buyback | `-shares / delay(shares, 252)` × ROA 품질 필터 |
| 11 | SGAEfficiency | SGA 비율 변화율 × 매출 성장 조건 |
| 12 | Goodwill | `-ts_zscore(무형자산 프리미엄 / sps, 63)` |
| 14 | DebtDecay | `signed_power(-ts_zscore(debt_ps, 63), 1.8)`, 저변동성 구간 한정 |
| 15 | BookCapMom | 자본/시총 모멘텀 (21일 + 42일 래그 합산) |

---

## Part 3 — 가설 검증

전략이 한국 시장에서 성과가 저하되는 원인에 대한 11개 가설을 각각 백테스트로 검증합니다.

| 그룹 | 가설 | 주요 결과 |
|------|------|----------|
| A — 시장 국면 | A-1: P/B 체제 필터 | MDD 소폭 개선 |
| | A-2: 소형주 스트레스 필터 | CAGR 유지, MDD 확대 |
| | A-3: IC 적응형 필터 | MDD −45% → −30% (전략 2) |
| B — 데이터 품질 | B-1: Buyback 수익성 조건 강화 | 음수 알파 심화 |
| | B-2: Goodwill — IT·헬스케어 제외 | 효과 미미 (소규모 유니버스) |
| | B-3: Accrual — 업종 내 상대 순위 | **CAGR +3.85%** (전략 1: −2% → +2%) |
| C — 신호 설계 | C-1: AND 멀티팩터 (전략 2 ∩ 5) | CAGR 하락, MDD 감소 |
| | C-2: 부채 전략 — 고레버리지 제외 | **CAGR +6.45%** (전략 14: +1% → +7%) |
| | C-3: 영업이익 계절성 제거 (252일 MA) | CAGR +0.32% |
| D — 유니버스 | D-1: 저유동성 종목 제거 | 전략 7 알파 파괴 (−10.6%) |
| | D-2: 시총 분위별 분리 | 전략 7 알파의 100%가 소형주에 집중 (CAGR +19.8%) |

---

## 데이터 소스

| 데이터 | 주 소스 | 폴백 |
|--------|---------|------|
| KR 가격 | FinanceDataReader | — |
| KR 유니버스 (KOSPI 200 이력) | pykrx | FDR 시총 상위 200 |
| KR 재무 | DART (OpenDartReader) | pykrx → 네이버 금융 |
| US 가격 | yfinance | — |
| US 유니버스 (S&P 500 이력) | `sp500_historical.csv` | Wikipedia 현재 목록 |
| US 재무 | yfinance | — |

---

## 주요 설정 (`config.py`)

```python
START_DATE         = "2010-01-01"
END_DATE           = "2024-12-31"
LONG_PCT           = 0.20   # 상위 20% 롱
SHORT_PCT          = 0.20   # 하위 20% 숏
REBALANCE          = "QE"   # 분기 리밸런싱
WINSORIZE_PCT      = 0.01
US_FILING_LAG_DAYS = 60
KR_FILING_LAG_DAYS = 0
```
