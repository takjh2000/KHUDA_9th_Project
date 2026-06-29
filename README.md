# KHUDA 9기 금융트랙 — 퀀트 팩터 백테스팅

> 한국(KOSPI 200) · 미국(S&P 500) 주식시장 대상 펀더멘털 팩터 롱숏 백테스팅  
> WorldQuant Brain 스타일 알파 전략 11개를 한국 시장에 적용하고, 성과 저하 원인 11개 가설을 백테스트로 검증합니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 구조](#프로젝트-구조)
3. [환경 설정 및 실행](#환경-설정-및-실행)
4. [Part 1 — 5대 팩터 롱숏 백테스트](#part-1--5대-팩터-롱숏-백테스트)
5. [Part 2 — WQ Brain 11개 전략 (한국 시장)](#part-2--wq-brain-11개-전략-한국-시장)
6. [Part 3 — 가설 검증](#part-3--가설-검증)
7. [데이터 소스](#데이터-소스)
8. [주요 설정](#주요-설정)

---

## 프로젝트 개요

### 배경 및 목적

WorldQuant Brain에 공개된 알파 전략들은 주로 미국 시장 데이터를 기반으로 개발되었습니다.  
본 프로젝트는 다음 세 가지 질문에 답합니다.

1. **한국 vs 미국**: 동일한 펀더멘털 팩터(BP, EP, SP, ROE, GPA)의 롱숏 성과가 두 시장에서 얼마나 다른가?
2. **WQ 전략의 한국 적용**: WorldQuant Brain 스타일 알파 전략 11개를 KOSPI 200에 그대로 적용하면 어떤 결과가 나오는가?
3. **성과 저하 원인 분석**: 전략이 한국 시장에서 잘 안 되는 이유는 무엇이며, 어떻게 개선할 수 있는가?

### 분석 범위

| 구분 | 기간 | 유니버스 |
|------|------|---------|
| Part 1 (5대 팩터) | 2010 – 2024 | KOSPI 200 / S&P 500 |
| Part 2–3 (WQ 전략) | 2016 – 2024 | KOSPI 200 |

---

## 프로젝트 구조

```
프로젝트/
├── config.py                   # 경로, 기간, 유니버스 설정
├── run_pipeline.py             # Step 1: 데이터 전처리 파이프라인
├── run_backtest.py             # Step 2: 5대 팩터 롱숏 백테스트
├── run_wq_all.py               # Step 3: WQ Brain 11개 전략 백테스트
├── run_wq_backtest.py          # 단일 WQ 전략 개별 실행
├── run_hypothesis_test.py      # Step 4: 11개 가설 검증
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
│   ├── engine.py               # LongShortBacktester (상/하위 20%, 분기 리밸런싱)
│   └── metrics.py              # CAGR, Sharpe, MDD
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

```bash
pip install FinanceDataReader yfinance pykrx pandas numpy matplotlib OpenDartReader
```

### DART API 키 설정 (KR 재무 수집 시 필요)

```bash
# Windows
set DART_API_KEY=your_api_key_here

# macOS / Linux
export DART_API_KEY=your_api_key_here
```

> 키가 없으면 pykrx → 네이버 금융 순으로 자동 폴백됩니다.

### 실행 순서

```bash
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

**전처리 파이프라인**:
```
원시 재무데이터 → 상하위 1% 윈저라이징 → 횡단면 z-score 정규화 → 팩터 신호 생성
```

### 팩터 정의

| 팩터 | 공식 | 투자 논리 |
|------|------|---------|
| **BP** | Book-to-Price (장부가 / 시가총액) | 저평가 가치주 포착 |
| **EP** | Earnings-to-Price (순이익 / 시가총액) | 이익 수익률 기반 가치 |
| **SP** | Sales-to-Price (매출 / 시가총액) | 매출 기반 가치 |
| **ROE** | Return on Equity (순이익 / 자기자본) | 자본 효율성 |
| **GPA** | Gross Profit / Assets (Novy-Marx) | 총이익 기반 수익성 |

### 결과 요약

| 팩터 | KR CAGR | US CAGR | KR Sharpe | US Sharpe |
|------|---------|---------|-----------|-----------|
| BP | — | — | — | — |
| EP | — | — | — | — |
| SP | — | — | — | — |
| ROE | — | — | — | — |
| GPA | — | — | — | — |

> 실제 결과는 `results/pnl_KR_*.png`, `results/heatmap_KR.png` 참조

---

## Part 2 — WQ Brain 11개 전략 (한국 시장)

WorldQuant Brain 스타일의 알파 전략 11개를 KOSPI 200 대상으로 백테스트 (2016–2024).

> **미구현 전략** (데이터 미확보): #4 (fnd6_newqv1300), #8 (fnd6_drc), #9 (fn_profit_loss_q), #13 (fnd6_acdo)

### 전략 목록

| # | 전략명 | 신호 로직 | 핵심 아이디어 |
|---|--------|----------|------------|
| 1 | **LowAccrual** | `-(eps - ops) / bps`, 시총 분위 중립화, 모멘텀 중립화 | 발생주의 낮은 기업 = 이익의 질이 높음 |
| 2 | **OpIncEY** | `group_rank(ts_rank(ops/close, 126), 업종)` | 업종 내 영업이익 수익률 상위 종목 |
| 3 | **CashCFDivergence** | `-ts_corr(ts_mean(bps,5), ts_mean(eps,5), 252)` | BPS 변화와 EPS 간 괴리 = 정보 비효율 |
| 5 | **CFYield** | `group_rank(ts_zscore(ops/close, 63), 업종)` | 업종 내 현금흐름 수익률 상위 종목 |
| 6 | **DebtSpike** | `signed_power(-ts_zscore(debt_ps, 252), 4)` | 부채 급증 후 역반전 모멘텀 |
| 7 | **DualValue** | `max(업종중립 영업이익 순위, 업종내 BPS/P 순위)` + 모멘텀 중립화 | 가치·수익성 두 신호가 동시에 지지하는 종목 |
| 10 | **Buyback** | `-shares / delay(shares, 252)` × 업종 내 수익성(ROA) 가중 | 자사주 매입 = 주식수 감소 × 수익성 필터 |
| 11 | **SGAEfficiency** | `trade_when(매출성장, zscore(SGA비율변화), -신호)` | 매출 성장 기업의 판관비 효율 변화 |
| 12 | **Goodwill** | `-ts_zscore(무형자산 프리미엄 / sps, 63)` | 무형자산 과대 계상 기업 숏 |
| 14 | **DebtDecay** | `signed_power(-ts_zscore(debt_ps, 63), 1.8)`, 저변동성 구간 한정 | 저변동성 국면에서 부채 감소 기업 매수 |
| 15 | **BookCapMom** | `zscore(Δ(bps/close), 21) + zscore(Δ(bps/close 21일 래그), 42)` | 자본/시총 비율의 단기 모멘텀 |

---

## Part 3 — 가설 검증

WQ 전략이 한국 시장에서 성과가 저하되는 원인에 대한 11개 가설을 각각 백테스트로 검증합니다.

### 가설 그룹 A — 시장 국면 (Market Regime)

| 가설 | 대상 전략 | 설명 | 주요 결과 |
|------|----------|------|---------|
| **A-1** KOSPI P/B 체제 필터 | 전략7 DualValue | 유니버스 중앙값 P/B가 역사적 70th pct 이상 → 포지션 50% 축소, 90th pct 이상 → 완전 중립 | MDD 소폭 개선 |
| **A-2** 소형주 스트레스 필터 | 전략7 DualValue | 시총 Q30/Q70 비율이 역사적 하위 20th pct 이하일 때 시총 하위 30% 종목 포지션 제거 | CAGR 유지, MDD 확대 |
| **A-3** IC 적응형 필터 | 전략2 OpIncEY | Rolling 63일 Spearman IC < 0.02 구간 → 포지션 제거 | **MDD −45% → −30% 개선** |

> **A-3 핵심 인사이트**: 2020–2021년 테마·이벤트 장세처럼 펀더멘털과 주가의 연결고리가 끊어진 구간에서는 포지션 자체를 중립화하는 것이 효과적입니다.

### 가설 그룹 B — 데이터 품질 (Data Quality)

| 가설 | 대상 전략 | 설명 | 주요 결과 |
|------|----------|------|---------|
| **B-1** Buyback 수익성 조건 강화 | 전략10 Buyback | ROA > 0 AND 매출 전년비 성장 > 0 기업만 신호 활성화 | 음수 알파 심화 |
| **B-2** Goodwill — IT·헬스케어 제외 | 전략12 Goodwill | 매출총이익률 > 40% AND 자산회전율 < 1.0 (고마진·저회전) 섹터 제외 | 효과 미미 (소규모 유니버스) |
| **B-3** Accrual 업종 내 상대 순위 | 전략1 LowAccrual | 발생주의 신호를 절대값 대신 업종 내 상대 순위로 변환 | **CAGR +3.85%** (−2% → +2%) |

> **B-3 핵심 인사이트**: 한국 GAAP → K-IFRS 전환 후 발생주의 측정 방식 변화로 절대값에 노이즈가 증가했습니다. 업종 내 상대 비교로 전환하면 회계 기준 차이가 상쇄됩니다.
>
> **B-1 핵심 인사이트**: 미국에서 자사주 매입은 잉여현금 활용 신호(상승 선행)이지만, 한국에서는 주가 방어 목적 매입(하락 중 매입)이 많아 역방향 신호가 됩니다.

### 가설 그룹 C — 신호 설계 (Signal Design)

| 가설 | 대상 전략 | 설명 | 주요 결과 |
|------|----------|------|---------|
| **C-1** AND 멀티팩터 | 전략2 ∩ 전략5 | 두 신호가 같은 방향(둘 다 롱 또는 둘 다 숏)일 때만 진입 | CAGR 하락, MDD 감소 |
| **C-2** 부채 전략 — 고레버리지 제외 | 전략14 DebtDecay | 부채비율 > 70% (항공·조선·건설 등 구조적 고부채) 종목 제외 | **CAGR +6.45%** (+1% → +7%) |
| **C-3** 영업이익 계절성 제거 | 전략2 OpIncEY | `ops`를 `ts_mean(ops, 252)`(rolling 1년 평균)으로 대체 | CAGR +0.32% |

> **C-2 핵심 인사이트**: 항공·조선·건설 등은 구조적으로 고부채 상태이므로 부채 감소 신호가 실제 재무 개선이 아닌 노이즈입니다. 제외 시 '실제 부채 감소' 신호의 순도가 높아집니다.
>
> **C-3 핵심 인사이트**: 한국 반도체·제조업의 영업이익 계절성은 매우 크며, Q4 결산 이후 급등/급락이 전략 신호를 왜곡합니다.

### 가설 그룹 D — 유니버스 구성 (Universe Design)

| 가설 | 대상 전략 | 설명 | 주요 결과 |
|------|----------|------|---------|
| **D-1** 저유동성 종목 제거 | 전략7 DualValue | 시총(close×shares) 하위 20% 종목 제외 | 전략7 알파 파괴 (CAGR −10.6%) |
| **D-2** 시총 분위별 분리 | 전략7 DualValue | 상위 50%(대형주) vs 하위 50%(소형주) 분리 백테스트 | **소형주 CAGR +19.8%** (알파의 100%가 소형주 집중) |

> **D-2 핵심 인사이트**: 전략7의 알파가 전적으로 소형주에서 발생합니다. 대형주에서는 유의미한 알파가 없으므로, 실제 구현 시 소형주 특화 전략으로 설계해야 합니다.
>
> **D-1 핵심 인사이트**: 저유동성 종목 제거는 오히려 알파를 파괴합니다. 전략7의 알파 소스가 유동성이 낮은 소형주이기 때문입니다.

### 가설 검증 핵심 결과 요약

| 그룹 | 가장 효과적인 수정 | ΔCAGR |
|------|-----------------|-------|
| A (시장 국면) | A-3 IC 적응형 필터 | MDD −15%p 개선 |
| B (데이터 품질) | B-3 업종 내 Accrual 순위 | **+3.85%p** |
| C (신호 설계) | C-2 고레버리지 제외 | **+6.45%p** |
| D (유니버스) | D-2 소형주 특화 | **+19.8%p** (소형주 분리 시) |

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

FACTORS            = ["BP", "EP", "SP", "ROE", "GPA"]
MARKETS            = ["KR", "US"]

LONG_PCT           = 0.20       # 상위 20% 롱
SHORT_PCT          = 0.20       # 하위 20% 숏
REBALANCE          = "QE"       # 분기말 리밸런싱

WINSORIZE_PCT      = 0.01       # 상하위 1% 윈저라이징
US_FILING_LAG_DAYS = 60         # 분기말 + 60일 후 재무 데이터 사용
KR_FILING_LAG_DAYS = 0          # DART 사용 시 45로 복원
```

---

## 참고 사항

- **WQ Brain 전략 번호**: 원본 WorldQuant Brain 알파 ID를 그대로 사용합니다 (1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15).
- **데이터 제약**: 거래량·VWAP 데이터 미확보로 일부 전략은 재무비율 프록시로 근사합니다.
- **섹터 분류**: 외부 업종 데이터 미확보 시, 재무비율(매출총이익률·자산회전율·부채비율)로 HighLev / IT_Health / Heavy / Other 4개 그룹을 자동 분류합니다.
- **생성 파일**: `results/` 디렉토리에 PDF 리포트 4개와 개별 전략 PnL 차트가 생성됩니다.
