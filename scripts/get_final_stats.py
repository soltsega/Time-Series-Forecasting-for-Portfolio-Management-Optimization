import pandas as pd
import numpy as np
import os
import logging
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use relative or specific path
data_path = 'data/processed' if os.path.exists('data/processed') else '../data/processed'
assets = ['TSLA', 'BND', 'SPY']

def load_and_summarize(ticker):
    fname = f'{ticker}_historical_data.csv'
    fpath = os.path.join(data_path, fname)
    try:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Historical data file not found: {fpath}")
            
        logger.info(f"Summarizing data for {ticker} from {fpath}")
        df = pd.read_csv(fpath)
        
        # Cleaning Logic
        if 'Price' in df.columns:
            df = df[~df['Price'].isin(['Ticker', 'Date'])].copy()
            df.rename(columns={'Price': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').interpolate().dropna().set_index('Date')
        
        # Basic Stats
        stats = df['Close'].describe()
        ret = df['Close'].pct_change().dropna()
        
        # Statistical tests
        p_close = adfuller(df['Close'])[1]
        p_ret = adfuller(ret)[1]
        var = np.percentile(ret, 5)
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
        
        print(f"Asset: {ticker}")
        print(f"  Range: {df.index.min()} to {df.index.max()}")
        print(f"  ADF P-Close: {p_close:.4f} | ADF P-Ret: {p_ret:.4f}")
        print(f"  VaR (95%): {var:.4f} | Sharpe: {sharpe:.4f}")
        print(f"  Avg Close: {stats['mean']:.2f}")
        
    except Exception as e:
        logger.error(f"Error processing stats for {ticker}: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists(data_path):
        logger.error(f"Data path not found index: {data_path}")
    else:
        for a in assets:
            load_and_summarize(a)
