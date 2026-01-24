# Portfolio Time Series Forecasting - Task 1: EDA & Preprocessing

## Project Overview
This project aims to forecast the trends of three diverse financial assets to optimize portfolio management. 
- **TSLA**: High-growth/High-risk Tesla stock.
- **BND**: Vanguard Total Bond Market ETF (Stability).
- **SPY**: S&P 500 ETF (Market exposure).

## Task 1 Results: Intensive Review

### 1. Data Quality & Cleaning
- **Period**: Jan 2, 2015 - Jan 15, 2026.
- **Integrity**: Handled "triple-header" CSV artifacts, enforced numeric data types, and applied linear interpolation for minor missing gaps. Duplicate records were removed to ensure time-series continuity.

### 2. Key Insights from Visualizations
- **Price Trends**: TSLA shows exponential growth compared to the linear/stable trajectories of SPY and BND. However, TSLA exhibits significant price corrections throughout the period.
- **Volatility**: TSLA's daily returns show frequent spikes exceeding 5%, highlighting its high-risk nature. BND is extremely muted, serving as a reliable hedge.
- **Rolling Statistics**: 20-day standard deviations spikes correlate strongly with market-wide events (e.g., Early 2020), with TSLA showing the highest dispersion from its mean.

### 3. Statistical Analysis Summary
| Metric | TSLA | BND | SPY |
| :--- | :--- | :--- | :--- |
| **Avg Close Price** | $138.47 | $67.33 | $339.07 |
| **ADF P-Value (Price)** | 0.9991 (Non-Stat) | 0.4496 (Non-Stat) | 0.9990 (Non-Stat) |
| **ADF P-Value (Returns)** | 0.0000 (Stat) | 0.0000 (Stat) | 0.0000 (Stat) |
| **Value at Risk (95%)** | -5.56% | -0.47% | -1.67% |
| **Sharpe Ratio** | 0.8306 | -0.1751 | 0.8040 |

### 4. Technical Implications for Forecasting
- **Stationarity**: Since the raw prices for all assets are non-stationary (ADF p-values >> 0.05), differencing ($d=1$) or transformation into returns is **mandatory** for ARIMA/SARIMA models.
- **Risk Assessment**: TSLA's high VaR suggests needing a robust outlier-handling mechanism in the forecasting models.

## Repository Structure
- `data/processed/`: Contains the final cleaned CSVs.
- `notebooks/EDA.ipynb`: Full step-by-step analysis and visualizations.
- `scripts/`: Implementation details for data extraction and metrics.
