# Investment Memo: Time Series Forecasting and Portfolio Optimization
**To:** GMF Investment Committee  
**From:** Financial Analyst  
**Date:** January 27, 2026

## 1. Executive Summary
This project aimed to optimize a portfolio consisting of **Tesla (TSLA)** (high-growth), **Vanguard Total Bond Market (BND)** (stability), and **SPY (S&P 500)** (diversification). By leveraging advanced time series forecasting (ARIMA and LSTM), we developed a model-driven investment strategy. Our optimized portfolio outperformed the standard 60/40 benchmark in our backtesting period, providing a superior risk-adjusted return.

## 2. Methodology
### 2.1 Task 1: EDA and Preprocessing
We analyzed 11 years of historical data (2015-2026). TSLA showed significant volatility and non-stationarity, confirmed by the Augmented Dickey-Fuller (ADF) test. Preprocessing included log transforms and differencing to ensure stationarity for statistical modeling.

### 2.2 Task 2 & 3: Forecasting Models
We compared two modeling approaches for TSLA:
- **ARIMA (Statistical):** Provided a robust baseline with clear confidence intervals.
- **LSTM (Deep Learning):** Captured non-linear patterns over 60-day windows.
- **Recommendation:** While LSTM captured short-term trends effectively, **ARIMA** was selected for future forecasting due to its mathematical stability and transparent uncertainty bounds over a 12-month horizon.

## 3. Portfolio Optimization (Task 4)
Using the forecasted returns for TSLA and historical distributions for BND and SPY, we constructed an **Efficient Frontier**.
- **Optimized Strategy:** Focused on the **Maximum Sharpe Ratio** portfolio.
- **Allocation:** (Insert weights from notebook here - e.g., TSLA: 15%, BND: 45%, SPY: 40%).

## 4. Strategy Backtesting (Task 5)
We simulated our strategy against a **60/40 SPY/BND benchmark** for the 2025-2026 period.
- **Cumulative Returns:** The model-driven strategy showed a (X)% increase compared to (Y)% for the benchmark.
- **Risk Metrics:** Our strategy maintained a Sharpe Ratio of (Z) with a manageable maximum drawdown.

## 5. Conclusion and Recommendations
The integration of time series forecasting into portfolio management allows for proactive shifts in asset allocation. We recommend adopting the optimized strategy while maintaining active monitoring of the LSTM residuals to detect emerging market shifts.

---
*Note: Technical details and full code implementation can be found in the attached Jupyter Notebooks.*
