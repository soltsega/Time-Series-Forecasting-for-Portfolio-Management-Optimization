import yfinance as yf
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_data(tickers, start_date, end_date, save_path):
    """
    Extracts historical financial data for multiple tickers using yfinance.
    """
    try:
        if not os.path.exists(save_path):
            logger.info(f"Creating directory: {save_path}")
            os.makedirs(save_path)
        
        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if not data.empty:
                    file_name = f"{ticker}_historical_data.csv"
                    target_file = os.path.join(save_path, file_name)
                    data.to_csv(target_file)
                    logger.info(f"Data for {ticker} successfully saved to {target_file}")
                else:
                    logger.warning(f"No data found for {ticker} within the range {start_date} to {end_date}")
            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {str(e)}")
                
    except Exception as e:
        logger.critical(f"Critical failure in extraction process: {str(e)}")

if __name__ == "__main__":
    assets = ["TSLA", "BND", "SPY"]
    start = "2015-01-01"
    end = "2026-01-15"
    processed_path = "data/processed"
    
    extract_data(assets, start, end, processed_path)
