import yfinance as yf
import pandas as pd
import os

def extract_data(tickers, start_date, end_date, save_path):
    """
    Extracts historical financial data for multiple tickers using yfinance.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if not data.empty:
            file_name = f"{ticker}_historical_data.csv"
            data.to_csv(os.path.join(save_path, file_name))
            print(f"Data for {ticker} saved to {os.path.join(save_path, file_name)}")
        else:
            print(f"No data found for {ticker}")

if __name__ == "__main__":
    assets = ["TSLA", "BND", "SPY"]
    start = "2015-01-01"
    end = "2026-01-15"
    processed_path = "data/processed"
    
    extract_data(assets, start, end, processed_path)
