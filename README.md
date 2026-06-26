# KHUDA_9th_Project

---
# 펀더멘털 팩터 기반 롱숏 전략 백테스팅 (KR / US)

KHUDA 금융트랙 프로젝트 — 한국·미국 주식시장에서 5개 펀더멘털 팩터의 롱숏 전략 성과를 비교 분석합니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 분석 기간 | 2010-01-01 ~ 2024-12-31 |
| 대상 시장 | 한국 (KR), 미국 (US) |
| 전략 | 팩터 상위 20% 롱 / 하위 20% 숏 (동일가중) |
| 리밸런싱 | 분기말 (QE) |
| 팩터 전처리 | 상하위 1% 윈소라이즈 → 횡단면 z-score |

---

## 분석 팩터

| 팩터 | 정의 | 의미 |
|------|------|------|
| **BP** | BPS / 주가 | 저평가 가치주 |
| **EP** | EPS / 주가 | 수익성 대비 가격 |
| **SP** | SPS / 주가 | 매출 대비 가격 |
| **ROE** | 자기자본이익률 | 자본 효율성 |
| **GPA** | 매출총이익 / 총자산 | 수익성 (Novy-Marx) |

---

## 프로젝트 구조

```
├── config.py              # 전역 설정 (기간, 팩터, 전략 파라미터)
├── run_pipeline.py        # Phase 1: 데이터 수집 및 전처리
├── run_backtest.py        # Phase 2~5: 팩터 계산 → 백테스트 → 시각화
│
├── data/
│   ├── pipeline/
│   │   ├── kr_price.py    # 한국 가격 데이터
│   │   ├── kr_finance.py  # 한국 재무 데이터
│   │   ├── us_price.py    # 미국 가격 데이터
│   │   └── us_finance.py  # 미국 재무 데이터
│   ├── build_panel.py     # 재무+가격 패널 구성 (분기 → 일별 ffill)
│   ├── raw/               # 원시 데이터 저장 경로
│   └── processed/         # 전처리 완료 데이터 저장 경로
│
├── factors/
│   └── compute.py         # 팩터 계산 및 정규화
│
├── backtest/
│   ├── engine.py          # 롱숏 백테스터
│   └── metrics.py         # CAGR, Sharpe Ratio, MDD 산출
│
├── analysis/
│   └── compare.py         # KR vs US 비교 시각화, 연도별 히트맵, IC 분석
│
└── results/               # 차트 저장 경로
```
---

## 실행 방법

### 1. 패키지 설치

```bash
pip install pandas numpy scipy matplotlib pyarrow

2. Phase 1 — 데이터 파이프라인

# KR + US 전체
python run_pipeline.py

# 시장별 개별 실행
python run_pipeline.py --kr
python run_pipeline.py --us

3. Phase 2~5 — 백테스팅 및 분석

# KR + US 전체 (5개 팩터)
python run_backtest.py

# 시장 또는 팩터 지정
python run_backtest.py --market KR
python run_backtest.py --market US
python run_backtest.py --factor BP

---
출력 결과

┌───────────────────────────────────┬────────────────────────────────┐
│               파일                │              설명              │
├───────────────────────────────────┼────────────────────────────────┤
│ results/pnl_{MARKET}_{FACTOR}.png │ 팩터별 누적 수익률 + 낙폭 차트 │
├───────────────────────────────────┼────────────────────────────────┤
│ results/pnl_comparison.png        │ KR vs US 팩터별 PnL 비교       │
├───────────────────────────────────┼────────────────────────────────┤
│ results/heatmap_KR.png            │ KR 팩터 × 연도별 수익률 히트맵 │
├───────────────────────────────────┼────────────────────────────────┤
│ results/heatmap_US.png            │ US 팩터 × 연도별 수익률 히트맵 │
└───────────────────────────────────┴────────────────────────────────┘

콘솔에는 팩터 × 시장 10개 조합의 CAGR / Sharpe / MDD / Turnover 요약 테이블이 출력됩니다.

---
주요 설계

- 공시 지연 반영 — 미국 +60일, 한국 +45일 적용하여 look-ahead bias 방지
- 상장폐지 처리 — 상폐 직전까지 수익률 반영 후 잔여 비중 균등 재배분
- 유니버스 필터 — 재무 결측 3분기(≈63 거래일) 초과 종목 제거
- IC 분석 — 팩터 IC(Spearman)와 개인투자자 순매수 비중 상관 분석 포함

---
