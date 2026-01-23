# Time Series Forecasting for Portfolio Management Optimization

## Project Overview
Guide Me in Finance (GMF) Investments aims to leverage time series forecasting models to predict market trends, optimize asset allocation, and enhance portfolio performance. This project focuses on analyzing historical data for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) to build predictive models and recommend portfolio adjustments.

## Business Objective
- Predict market trends using advanced time series forecasting.
- Optimize asset allocation based on forecasted returns and volatility.
- Minimize risks while capitalizing on market opportunities.

## Project Structure
- `data/`: Contains raw and processed financial data.
- `notebooks/`: Jupyter notebooks for EDA and modeling.
- `src/`: Source code for data preprocessing and modeling.
- `scripts/`: Utility scripts for data extraction and automation.
- `tests/`: Unit tests for the project.

## Implementation Steps
1. **Preprocess and Explore the Data**: Load, clean, and analyze historical data from YFinance.
2. **Build Time Series Forecasting Models**: Develop ARIMA/SARIMA and LSTM models for stock price prediction.
3. **Forecast Future Market Trends**: Generate long-term forecasts and analyze trends.
4. **Optimize Portfolio**: Use Modern Portfolio Theory (MPT) to construct an optimal asset allocation.
5. **Strategy Backtesting**: Validate the portfolio strategy against a benchmark.

## Dependencies
Install the required dependencies using:
```bash
pip install -r requirements.txt
```
