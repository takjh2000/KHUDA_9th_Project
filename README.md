# KHUDA 9기 금융트랙 — 투자에 활용가능한 미국시장 대비 한국시장의 특징

> 한국(KOSPI 200) · 미국(S&P 500) 주식시장 대상 펀더멘털 팩터 롱숏 백테스팅
> WorldQuant Brain 스타일 알파 전략 11개를 한국 시장에 적용하고, 성과 저하 원인을 **Value 팩터 / Quality 팩터** 관점의 가설로 나누어 백테스트로 검증합니다.


---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [환경 설정 및 실행](#환경-설정-및-실행)
4. [Part 1 — 5대 팩터 롱숏 백테스트](#part-1--5대-팩터-롱숏-백테스트)
5. [Part 2 — WQ Brain 11개 전략 (한국 시장)](#part-2--wq-brain-11개-전략-한국-시장)
6. [Part 3 — 가설 설정 및 검증](#part-3--가설-설정-및-검증)
7. [결론](#결론)
8. [활용 방안](#활용-방안)
9. [데이터 소스](#데이터-소스)
10. [주요 설정](#주요-설정-configpy)

---

## 프로젝트 개요

### 배경 및 목적

WorldQuant Brain에 공개된 알파 전략들은 주로 미국 시장 데이터를 기반으로 개발되었습니다. 반면 한국 시장은 다음과 같은 구조적 차이를 가집니다.

- **개인 투자자의 폭발적 유입**: 코로나 이후 동학개미 열풍으로 개인 거래 비중이 60%를 넘어섰고, 개인이 시장을 주도하면서 미국과는 다른 가격 형성 구조가 나타남
- **한국형 팩터의 필요성**: 개인 수급, 데이터 환경, 시장 구조를 고려한 팩터 재설계가 필요

본 프로젝트는 다음 세 가지 질문에 답합니다.

1. **한국 vs 미국**: 동일한 펀더멘털 팩터(BP, EP, SP, ROE, GPA)의 롱숏 성과가 두 시장에서 얼마나 다른가?
2. **WQ 전략의 한국 적용**: WorldQuant Brain 스타일 알파 전략 11개를 KOSPI 200에 그대로 적용하면 어떤 결과가 나오는가?
3. **성과 저하 원인 분석 및 개선**: 전략이 한국 시장에서 잘 안 되는 이유는 무엇이며, 미국 전략을 한국에 맞게 변환하는 규칙을 어떻게 찾을 수 있는가?

### 분석 범위

| 구분               | 기간          | 유니버스                |
| ---------------- | ----------- | ------------------- |
| Part 1 (5대 팩터)   | 2010 – 2024 | KOSPI 200 / S&P 500 |
| Part 2–3 (WQ 전략) | 2019 – 2023 | KOSPI 200           |

---

## 프로젝트 구조

```
프로젝트/
├── config.py                   # 경로, 기간, 유니버스 설정
├── run_pipeline.py             # Step 1: 데이터 전처리 파이프라인
├── run_backtest.py             # Step 2: 5대 팩터 롱숏 백테스트
├── run_wq_all.py               # Step 3: WQ Brain 11개 전략 백테스트
├── run_wq_backtest.py          # 단일 WQ 전략 개별 실행
├── run_hypothesis_test.py      # Step 4: 가설 검증 (Value/Quality)
├── generate_report.py          # PDF 리포트 생성
│
├── data/
│   ├── build_panel.py          # 가격 + 재무 데이터 병합 → 패널 parquet
│   └── pipeline/
│       ├── kr_price.py         # KOSPI 200 과거 구성종목 + 가격 (FDR/pykrx)
│       ├── kr_finance.py       # KR 재무 (DART → pykrx → 네이버 금융 폴백)
│       ├── us_price.py         # S&P 500 과거 구성종목 + 가격 (yfinance)
│       └── us_finance.py       # US 재무 (yfinance)
│
├── factors/
│   ├── compute.py              # 5대 팩터: BP, EP, SP, ROE, GPA
│   └── wq_ops.py               # WQ Brain 연산: rank, zscore, ts_*, group_* 등
│
├── backtest/
│   ├── engine.py               # LongShortBacktester (상/하위 20~50%, 분기 리밸런싱)
│   └── metrics.py              # Sharpe, CAGR, MDD, Turnover, Margin, Fitness
│
├── analysis/
│   └── compare.py              # KR vs US 비교 차트
│
├── results/                    # 백테스트 결과 이미지 + PDF 리포트
└── sp500_historical.csv        # S&P 500 과거 구성종목 (date, tickers)
```

---

## 환경 설정 및 실행

### 패키지 설치

```
pip install FinanceDataReader yfinance pykrx pandas numpy matplotlib OpenDartReader
```

### DART API 키 설정 (KR 재무 수집 시 필요)

```
# Windows
set DART_API_KEY=your_api_key_here

# macOS / Linux
export DART_API_KEY=your_api_key_here
```
> 키가 없으면 pykrx → 네이버 금융 순으로 자동 폴백됩니다.

### 실행 순서

```
# Step 1. 데이터 파이프라인 실행
python run_pipeline.py          # KR + US 전체
python run_pipeline.py --kr     # KR만
python run_pipeline.py --us     # US만

# Step 2-A. 5대 팩터 롱숏 백테스트
python run_backtest.py --market KR --factor BP
python run_backtest.py --market US --factor EP
# → results/pnl_KR_BP.png, heatmap_KR.png 등

# Step 2-B. WQ Brain 11개 전략 백테스트 (한국 시장)
python run_wq_all.py
# → results/WQ_Brain_KR_종합분석_리포트.pdf

# Step 2-C. 가설 검증 백테스트 (한국 시장)
python run_hypothesis_test.py
# → results/WQ_가설검증_리포트.pdf
```

---

## Part 1 — 5대 팩터 롱숏 백테스트

### 전략 개요

상위 20% 롱 / 하위 20% 숏, 분기(QE) 리밸런싱, 2010–2024.

**데이터 구성**
- 한국 상장주식 유니버스 + 일별 시세, 분기 재무데이터를 `merge_asof`로 병합 (공시 지연을 반영해 시점별로 유효한 재무 데이터만 매칭)
- Flow 항목(영업이익 등)은 최근 4개 분기 합(TTM)으로 보정, 재무상태표(Stock) 항목(자본/부채 등)은 최신값 그대로 사용

**전처리 파이프라인**
```
원시 재무데이터 → 상하위 1% 윈저라이징 → 횡단면 z-score 정규화 → 팩터 신호 생성
```

### 팩터 정의

| 팩터      | 공식                                | 투자 논리        |
| ------- | --------------------------------- | ------------ |
| **BP**  | Book-to-Price (장부가 / 시가총액)        | 저평가 가치주 포착   |
| **EP**  | Earnings-to-Price (순이익 / 시가총액)    | 이익 수익률 기반 가치 |
| **SP**  | Sales-to-Price (매출 / 시가총액)        | 매출 기반 가치     |
| **ROE** | Return on Equity (순이익 / 자기자본)     | 자본 효율성       |
| **GPA** | Gross Profit / Assets (Novy-Marx) | 총이익 기반 수익성   |

> 실제 결과는 `results/pnl_KR_*.png`, `results/heatmap_KR.png` 참조

---

## Part 2 — WQ Brain 9개 전략 (한국 시장)

WorldQuant Brain 스타일의 알파 전략을 KOSPI 200 대상으로 백테스트 (2019.01–2023.12, 상위 50% 롱 / 하위 50% 숏, 스코어 가중 방식으로 종목별 비중 산출, 팩터별 트렁케이션 적용).

### 전략 목록 (Value / Quality 팩터 분류)

가설 설정을 위해 전략을 두 가지 팩터로 분류했습니다.

| 팩터 | 정의 |
|---|---|
| **Value 팩터** | 기업의 펀더멘털(이익, 자산, 매출) 대비 싸게 거래되는 주식이 장기적으로 초과수익을 낸다 |
| **Quality 팩터** | 재무 건전성과 수익성이 우수한 기업이 장기적으로 더 안정적이고 우수한 수익을 낸다 |

| 구분 | 전략명 | 신호 로직 | 핵심 아이디어 |
| --- | --- | --- | --- |
| Quality#1 | Low Accrual Clean | `-(eps - ops) / bps`, 시총·모멘텀 중립화 | 발생주의 낮은 기업 = 이익의 질이 높음 |
| Value#1 | Operating Income Earnings Yield | `group_rank(ts_rank(ops/close, 126), 업종)` | 업종 내 영업이익 수익률 상위 종목 |
| Quality#2 | Cash & Cash Flow Divergence Information | `-ts_corr(ts_mean(bps,5), ts_mean(eps,5), 252)` | BPS 변화와 EPS 간 괴리 = 정보 비효율 |
| Value#2 | Industry-Neutral Cash Flow Yield | `group_rank(ts_zscore(ops/close, 63), 업종)` | 업종 내 현금흐름 수익률 상위 종목 |
| Quality#3 | Profitable Buyback & Cash Flow Distortion | `-shares / delay(shares, 252)` × 업종 내 ROA 가중 | 자사주 매입 = 주식수 감소 × 수익성 필터 |
| Value#3 | Aggressive Dual Value Blend with Momentum Neutralization | `max(업종중립 영업이익 순위, 업종내 BPS/P 순위)` + 모멘텀 중립화 | 가치·수익성 두 신호가 동시에 지지하는 종목 |
| Quality#4 | SG&A-Driven Marketing Efficiency Dynamic | `trade_when(매출성장, zscore(SGA비율변화), -신호)` | 매출 성장 기업의 판관비 효율 변화 |
| Quality#5 | Goodwill Overvaluation & Financial Accrued Risk | `-ts_zscore(무형자산 프리미엄 / sps, 63)` | 무형자산 과대 계상 기업 숏 |
| Value#4 | Book to Cap Momentum | `zscore(Δ(bps/close), 21) + zscore(Δ(bps/close 21일 래그), 42)` | 자본/시총 비율의 단기 모멘텀 |

---

## Part 3 — 가설 설정 및 검증

전략을 Value/Quality 두 팩터로 분류한 뒤, **"가설이 한 팩터에 속하는 대부분의 전략을 개선시키면 타당하다"**는 기준으로 각 팩터별 가설을 세우고 검증했습니다.

### Value 팩터 가설

1. **시가총액 차지 비율 구조 차이를 보정해야 한다.** 소수의 초대형주가 KOSPI200 전체 시가총액의 상당 부분을 차지 → 시가총액을 그대로 시그널의 분모로 사용하면 실제 시그널의 영향이 약화됨
2. **대주주 지분이 커서 저평가가 오래간다.** 오너가 저평가를 해소할 유인이 약함
3. **개인투자자 비중이 높아 저평가 상태가 방치된다.** 재무제표보다 테마·모멘텀·뉴스에 더 반응
4. **한국은 저평가 신호의 주가 반영이 느리다.** → 한국 적용 시 더 긴 보유기간 필요

**적용 방법**: 시가총액에 대한 **로그 보정법** 사용 (`시그널 / cap` → `log(시그널) - log(cap)`)

### Quality 팩터 가설

1. **분기 데이터는 Flow와 Stock을 구분해 다뤄야 한다.** Flow(영업이익 등)는 계절성 왜곡 방지를 위해 4개 분기 합(TTM), Stock(자기자본·부채)은 최신값 그대로 사용
2. **신호를 구간(등급)으로 나누면 성과가 떨어진다.** 연속값을 몇 단계로 끊으면 경계 근처의 미세한 차이가 사라짐
3. **비슷한 기업끼리 묶어 상대평가하면 오히려 성과가 나빠진다.** 특정 그룹에 몰려있는 진짜 신호까지 그룹 평균 대비로 깎이게 됨

**적용 방법**: 현금흐름·영업이익 등 유량 항목에 TTM 보정 + 수익성·효율성이 좋은 기업일수록 신호를 더 신뢰하도록 조건부 가중 결합

### 가설 검증 결과 (기준 로직 vs 가설 적용, 2019.01–2023.12)

| 전략 | 구분 | Returns | Sharpe | Margin | Turnover | Drawdown | Fitness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Quality#2** (Cash & Cash Flow Divergence) | 기준 | 0.90% | 0.13 | 9.8bp | 1.83% | 11.72% | 0.10 |
| | 가설 | **7.25%** | **0.84** | 191.2bp | 0.75% | 10.47% | 1.80 |
| **Quality#3** (Profitable Buyback) | 기준 | −4.91% | −0.60 | −82.0bp | 1.19% | 34.37% | −1.06 |
| | 가설 | **7.29%** | **1.09** | 183.4bp | 0.79% | 6.62% | 2.35 |
| **Quality#5** (Goodwill Overvaluation) | 기준 | −2.76% | −0.32 | −42.0bp | 1.30% | 27.32% | −0.43 |
| | 가설 | **3.47%** | **0.42** | 56.6bp | 1.22% | 17.13% | 0.62 |
| **Value#2** (Industry-Neutral Cash Flow Yield) | 기준 | 4.75% | 0.67 | 8.0bp | 11.81% | 16.31% | 1.17 |
| | 가설 | **7.30%** | **1.01** | 11.2bp | 12.92% | 14.81% | 2.11 |
| **Value#3** (Aggressive Dual Value Blend) | 기준 | 1.40% | 0.23 | 2.2bp | 12.38% | 10.73% | 0.22 |
| | 가설 | **2.36%** | **0.38** | 3.6bp | 12.91% | 10.36% | 0.45 |
| **Value#4** (Book to Cap Momentum) | 기준 | 6.25% | 0.60 | 9.0bp | 13.78% | 21.53% | 1.10 |
| | 가설 | **9.38%** | **0.73** | 17.1bp | 10.88% | 16.85% | 1.79 |

> 원본(기준) 로직과 개선안(가설)을 동일 기간·동일 파이프라인에서 나란히 백테스트해 상대 비교했습니다.

---

## 결론

### 1. Quality

- **근거**: 세 전략 모두 재무제표상의 단순 비율만으로는 기업의 진짜 상태가 제대로 드러나지 않고, 여기에 수익성·효율성이라는 조건을 더해야 신호의 신뢰도가 높아진다는 공통된 가설에서 출발
- **적용 방법**: 현금흐름·영업이익처럼 분기마다 값이 들쭉날쭉한 유량 항목을 최신 분기값 대신 최근 4개 분기 합(TTM)으로 보정하고, 수익성·효율성이 좋은 기업일수록 신호를 더 신뢰하도록 조건부 가중을 결합
- **결과**: 세 전략 모두 Sharpe 비율이 큰 폭으로 개선 — Quality#2 0.13 → 0.84, Quality#3 −0.60 → 1.09, Quality#5 −0.32 → 0.42

### 2. Value

- **근거**: 세 전략 모두 저평가 종목을 찾는 과정에서 시가총액을 아무런 보정 없이 사용했고, 한국시장의 특성상 종목들의 시가총액이 양극화되어 있다는 사실에서 가설을 도출
- **적용 방법**: 로그 보정법 사용 (`log(시그널) - log(cap)`)
- **결과**: 세 전략 모두 Sharpe 비율 개선 — Value#2 0.67 → 1.01, Value#3 0.23 → 0.38, Value#4 0.60 → 0.73

---

## 활용 방안

1. **미국 전략을 한국 시장에 자동 변환하는 "변환 함수"로 활용**: 퀀트 투자에서 일종의 함수를 만들어서, 미국에서 통하는 전략을 함수를 통해 손쉽게 한국 시장에서 사용할 수 있는 전략으로 변환
2. **개인투자자를 위한 "한국 시장 투자 시 유의사항" 가이드로 활용**: 퀀트 모델이 없는 일반 개인투자자에게도 미국시장과 한국시장을 어떻게 다르게 투자해야 하는지, 유의해야 할 점이 무엇인지 안내하는 가이드로 활용 가능

---

## 데이터 소스

| 데이터                    | 주 소스                   | 폴백              |
| ---------------------- | ---------------------- | --------------- |
| KR 가격                  | FinanceDataReader      | —               |
| KR 유니버스 (KOSPI 200 이력) | pykrx                  | FDR 시총 상위 200   |
| KR 재무                  | DART (OpenDartReader)  | pykrx → 네이버 금융  |
| US 가격                  | yfinance               | —               |
| US 유니버스 (S&P 500 이력)   | `sp500_historical.csv` | Wikipedia 현재 목록 |
| US 재무                  | yfinance               | —               |

---

## 주요 설정 (`config.py`)

```
START_DATE         = "2010-01-01"
END_DATE           = "2024-12-31"

FACTORS            = ["BP", "EP", "SP", "ROE", "GPA"]
MARKETS            = ["KR", "US"]

LONG_PCT           = 0.20       # 상위 20% 롱 (Part 1 기준)
SHORT_PCT          = 0.20       # 하위 20% 숏 (Part 1 기준)
REBALANCE          = "QE"       # 분기말 리밸런싱

WINSORIZE_PCT      = 0.01       # 상하위 1% 윈저라이징
US_FILING_LAG_DAYS = 60         # 분기말 + 60일 후 재무 데이터 사용
KR_FILING_LAG_DAYS = 0          # DART 사용 시 45로 복원 예정
```

> Part 2–3(WQ 전략, 2019–2023)은 상위 50% 롱 / 하위 50% 숏, 스코어 가중 방식(score_weight)과 팩터별 트렁케이션을 적용합니다.

---

## 참고 사항

- **WQ Brain 전략 번호**: 원본 WorldQuant Brain 알파 ID를 그대로 사용합니다.
- **데이터 제약**: 거래량·VWAP 데이터 미확보로 일부 전략은 재무비율 프록시로 근사합니다.
- **섹터 분류**: 외부 업종 데이터 미확보 시, 재무비율(매출총이익률·자산회전율·부채비율)로 HighLev / IT_Health / Heavy / Other 4개 그룹을 자동 분류합니다.
- **생성 파일**: `results/` 디렉토리에 PDF 리포트와 개별 전략 PnL 차트가 생성됩니다.
