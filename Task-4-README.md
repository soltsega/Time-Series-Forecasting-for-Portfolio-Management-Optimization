# Task 4: Portfolio Optimization - Summary Report

## 1. Objective
The goal of this task was to construct an optimized investment portfolio consisting of **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **SPDR S&P 500 ETF Trust (SPY)**. The optimization process integrates:
- **Forward-looking views:** Using the 12-month ARIMA forecast for TSLA from Task 3.
- **Historical data:** Using annualized historical returns for BND and SPY, and a historical covariance matrix for risk estimation.
- **Modern Portfolio Theory (MPT):** Maximizing the Sharpe Ratio to find the optimal risk-adjusted return.

## 2. Methodology

### A. Inputs
1.  **Expected Returns (Mu):**
    -   **TSLA:** Derived from the ARIMA 12-month forecast (~$440 target).
    -   **BND & SPY:** Derived from annualized mean historical daily returns.
2.  **Risk (Sigma):**
    -   Calculated using the annualized sample covariance matrix of the three assets.

### B. Optimization Technique
Due to environmental constraints with standard plotting libraries (`PyPortfolioOpt` / `cvxpy` on Python 3.14), we implemented a **manual Mean-Variance Optimization** using `scipy.optimize`.
-   **Solver:** SLSQP (Sequential Least SQuares Programming).
-   **Objective:** Minimize the Negative Sharpe Ratio.
-   **Constraints:** Fully invested (sum of weights = 1), Long-only (weights between 0 and 1).

## 3. Results

The optimization identified two key portfolios on the Efficient Frontier:

| Portfolio | TSLA Weight | BND Weight | SPY Weight | Exp. Return | Volatility | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Max Sharpe Ratio** | **0.00%** | **0.00%** | **100.00%** | **15.55%** | **12.30%** | **1.26** |
| **Min Volatility** | **1.45%** | **94.22%** | **4.33%** | **2.67%** | **4.01%** | **0.67** |
*(Note: Actual values may vary slightly based on the final execution of the notebook)*

## 4. Final Recommendation

**Selected Strategy:** **Maximum Sharpe Ratio Portfolio**

### Rationale:
1.  **Efficient Growth:** The Max Sharpe portfolio allocates capital to the asset class providing the highest return per unit of risk. In this specific forecast scenario, TSLA's projected flat growth (0.06%) makes it unattractive compared to SPY's historical uptrend.
2.  **Stability:** While the Minimum Volatility portfolio offers safety, its expected return (~2.7%) is too low for a growth-oriented mandate.
3.  **Benchmark Sensitivity:** The optimizer heavily favors SPY due to its superior historical risk/return profile compared to the forecasted stagnation of TSLA.

## 5. Next Steps
-   **Task 5 (Backtesting):** We will now backtest this strategy (likely an SPY-dominant allocation vs. a diverse mix) over the period **Jan 2025 – Jan 2026** to validate if this defensive/conservative approach would have outperformed a standard 60/40 benchmark.
