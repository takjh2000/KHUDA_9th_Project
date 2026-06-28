# KHUDA 9th Project — Quantitative Factor Backtesting

Fundamental factor long-short backtesting across Korean (KOSPI 200) and US (S&P 500) equity markets, with an extended analysis of 11 WorldQuant Brain-style strategies on the Korean market and hypothesis-driven strategy improvement.

---

## Project Structure

```
프로젝트/
├── config.py                   # paths, dates, universe params
├── run_pipeline.py             # Step 1: build all data
├── run_backtest.py             # Step 2: 5-factor long-short backtest
├── run_wq_all.py               # Step 3: 11 WQ Brain strategy backtest
├── run_hypothesis_test.py      # Step 4: 11 hypothesis validation
│
├── data/
│   ├── build_panel.py          # merge price + financials → panel parquet
│   └── pipeline/
│       ├── kr_price.py         # KOSPI 200 historical universe + prices (FDR/pykrx)
│       ├── kr_finance.py       # KR financials (DART → pykrx → Naver fallback)
│       ├── us_price.py         # S&P 500 historical universe + prices (yfinance)
│       └── us_finance.py       # US financials (yfinance)
│
├── factors/
│   ├── compute.py              # 5 classic factors: BP, EP, SP, ROE, GPA
│   └── wq_ops.py               # WQ Brain ops: rank, zscore, ts_*, group_*, etc.
│
├── backtest/
│   ├── engine.py               # LongShortBacktester (top/bottom 20%, QE rebalance)
│   └── metrics.py              # CAGR, Sharpe, MDD
│
├── analysis/
│   └── compare.py              # KR vs US comparative charts
│
└── sp500_historical.csv        # S&P 500 historical constituents (date, tickers)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install FinanceDataReader yfinance pykrx pandas numpy matplotlib

# 2. Build data pipeline
python run_pipeline.py          # KR + US
python run_pipeline.py --kr     # KR only
python run_pipeline.py --us     # US only

# 3-A. Classic 5-factor backtest
python run_backtest.py --market KR --factor BP
python run_backtest.py --market US --factor EP

# 3-B. WQ Brain 11-strategy backtest (KR)
python run_wq_all.py
# → results/WQ_Brain_KR_종합분석_리포트.pdf

# 3-C. Hypothesis validation (KR)
python run_hypothesis_test.py
# → results/WQ_가설검증_리포트.pdf
```

> **Note:** KR financial data requires a DART API key for full coverage.
> Set `DART_API_KEY` in your environment before running the pipeline.
> Without it, the pipeline falls back to pykrx → Naver Finance automatically.

---

## Part 1 — Classic 5-Factor Backtest

Long top 20% / Short bottom 20% with quarterly rebalancing (2010–2024).

| Factor | Description |
|--------|-------------|
| **BP** | Book-to-Price — value |
| **EP** | Earnings-to-Price — earnings yield |
| **SP** | Sales-to-Price — revenue yield |
| **ROE** | Return on Equity — capital efficiency |
| **GPA** | Gross Profit-to-Assets — Novy-Marx profitability |

Preprocessing: winsorize at 1% extremes → cross-sectional z-score normalization.

---

## Part 2 — WQ Brain 11-Strategy Analysis (Korean Market)

11 WorldQuant Brain-inspired alpha strategies backtested on KOSPI 200 (2016–2024).

| # | Strategy | Signal Logic |
|---|----------|-------------|
| 1 | LowAccrual | `-(eps - ops) / bps`, size-neutralized |
| 2 | OpIncEY | `ops / close`, industry-relative rank |
| 5 | CFYield | `group_rank(ts_zscore(ops/close, 63), industry)` |
| 6 | DebtSpike | `signed_power(-ts_zscore(debt_ps, 252), 4)` |
| 7 | DualValue | `max(EBITDA_rank, PBR_rank)` + momentum neutralization |
| 10 | Buyback | `-shares / delay(shares, 252)` × ROA quality filter |
| 11 | SGAEfficiency | SGA ratio change × revenue growth condition |
| 12 | Goodwill | `-ts_zscore(intangible_premium / sps, 63)` |
| 14 | DebtDecay | `signed_power(-ts_zscore(debt_ps, 63), 1.8)`, low-vol regime only |
| 15 | BookCapMom | equity/cap momentum (21d + 42d lag) |

---

## Part 3 — Hypothesis Testing

11 hypotheses on why strategies underperform in the Korean market, each backtested against a baseline.

| Group | Hypothesis | Key Finding |
|-------|-----------|-------------|
| A — Market Regime | A-1: P/B regime filter | Marginal MDD improvement |
| | A-2: Small-cap stress filter | Stable CAGR, wider drawdown |
| | A-3: IC adaptive filter | MDD −45% → −30% (Strat 2) |
| B — Data Quality | B-1: Buyback profitability screen | Negative alpha deepens |
| | B-2: Goodwill — exclude IT/Health proxy | Minimal effect (small universe) |
| | B-3: Accrual — industry-relative rank | **+3.85% CAGR** (Strat 1: −2% → +2%) |
| C — Signal Design | C-1: AND multi-factor (Strat 2 ∩ 5) | Lower CAGR, lower MDD |
| | C-2: Debt strategy — exclude high-leverage | **+6.45% CAGR** (Strat 14: +1% → +7%) |
| | C-3: OI seasonal smoothing (252d MA) | +0.32% CAGR |
| D — Universe | D-1: Remove low-liquidity stocks | Destroys Strat 7 alpha (−10.6%) |
| | D-2: Size split | Strat 7 alpha 100% in small-cap (+19.8%) |

---

## Data Sources

| Data | Source | Fallback |
|------|--------|----------|
| KR prices | FinanceDataReader | — |
| KR universe (KOSPI 200 history) | pykrx | FDR market-cap top 200 |
| KR financials | DART (OpenDartReader) | pykrx → Naver Finance |
| US prices | yfinance | — |
| US universe (S&P 500 history) | `sp500_historical.csv` | Wikipedia current list |
| US financials | yfinance | — |

---

## Key Configuration (`config.py`)

```python
START_DATE      = "2010-01-01"
END_DATE        = "2024-12-31"
LONG_PCT        = 0.20          # top 20% long
SHORT_PCT       = 0.20          # bottom 20% short
REBALANCE       = "QE"          # quarterly
WINSORIZE_PCT   = 0.01
US_FILING_LAG_DAYS = 60
KR_FILING_LAG_DAYS = 0
```
