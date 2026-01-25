import nbformat as nbf
import os
import logging

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

path = 'notebooks/Task-2-TimeSeriesForecasting.ipynb'

cells_to_update = {
    "import pandas as pd\nimport os": """import pandas as pd
import os
import logging

# Configure logging for the notebook
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
data_path = "../data/processed"
assets = ["TSLA", "BND", "SPY"]
splits = {}

def prepare_data(ticker):
    try:
        # Load the processed data from Task 1
        file_path = os.path.join(data_path, f"{ticker}_final_processed.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info(f"Loading and splitting data for {ticker} from {file_path}")
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # Sort chronologically
        df = df.sort_index()
        
        # Split Chronologically
        train = df.loc['2015-01-01':'2024-12-31']
        test = df.loc['2025-01-01':'2026-01-15']
        
        print(f"--- {ticker} Data Split ---")
        print(f"Training set: {train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)")
        print(f"Testing set:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)\\n")
        
        return train, test
    except Exception as e:
        logger.error(f"Failed to prepare data for {ticker}: {str(e)}")
        return None, None

# Store splits for each asset
for asset in assets:
    train, test = prepare_data(asset)
    if train is not None:
        splits[asset] = (train, test)""",

    "def find_best_params": """import pmdarima as pm

def find_best_params(ticker, train_data):
    try:
        logger.info(f"Finding optimal parameters for {ticker}")
        print(f"--- Finding optimal parameters for {ticker} ---")
        
        # auto_arima will iterate through combinations to minimize AIC
        model = pm.auto_arima(train_data['Close'], 
                              seasonal=True, m=5, # m=5 for weekly trading patterns
                              stepwise=True, 
                              suppress_warnings=True, 
                              error_action="ignore")
        
        print(f"Best model for {ticker}: {model.order} x {model.seasonal_order}")
        return model
    except Exception as e:
        logger.error(f"Failed to find parameters for {ticker}: {str(e)}")
        return None

# Store the discovered model configurations
asset_models = {}
for asset in assets:
    if asset in splits:
        train, _ = splits[asset]
        model = find_best_params(asset, train)
        if model is not None:
            asset_models[asset] = model""",

    "fitted_models = {}": """# --- Model Training & Parameter Documentation ---
import logging
logger = logging.getLogger(__name__)

# Dictionary to hold the fitted results
fitted_models = {}

# Documentation Table (Hyperparameters)
print(f"{'Asset':<10} | {'ARIMA Order (p,d,q)':<20} | {'Seasonal Order (P,D,Q,s)':<25}")
print("-" * 65)

for asset in assets:
    try:
        if asset not in splits or asset not in asset_models:
            logger.warning(f"Skipping {asset} due to missing split or model configuration")
            continue
            
        train_data, _ = splits[asset]
        model_config = asset_models[asset]
        
        logger.info(f"Fitting model for {asset}")
        # Train the selected model on the training data
        fitted_models[asset] = model_config.fit(train_data['Close'])
        
        # Document parameters
        print(f"{asset:<10} | {str(model_config.order):<20} | {str(model_config.seasonal_order):<25}")
    except Exception as e:
        logger.error(f"Failed to fit model for {asset}: {str(e)}")

print("\\nModel Training attempts complete.")""",

    "test_forecasts = {}": """import pandas as pd
import warnings
import logging
logger = logging.getLogger(__name__)

# --- This specific block eliminates the red warning text ---
warnings.filterwarnings('ignore', message='No supported index is available')
warnings.filterwarnings('ignore', category=FutureWarning)

test_forecasts = {}

for asset in assets:
    try:
        if asset not in fitted_models or asset not in splits:
            logger.warning(f"Skipping forecast for {asset} - model not fitted")
            continue
            
        print(f"Generating forecast for {asset}...")
        _, test_data = splits[asset]
        fitted_row = fitted_models[asset]
        
        # Predict for the length of the test set
        n_periods = len(test_data)
        
        # Grabbing .values here is key
        raw_predictions = fitted_row.predict(n_periods=n_periods)
        
        # Manually re-indexing ensures your time series is aligned for the evaluation step
        test_forecasts[asset] = pd.Series(raw_predictions.values, index=test_data.index)
        logger.info(f"Forecast generated for {asset}")
    except Exception as e:
        logger.error(f"Failed to generate forecast for {asset}: {str(e)}")

print("\\nTask 2 Forecast generation complete.")"""
}

def improve_notebook():
    try:
        if not os.path.exists(path):
            logger.error(f"Notebook not found: {path}")
            return
            
        logger.info(f"Opening notebook: {path}")
        nb = nbf.read(path, as_version=4)
        
        updated_count = 0
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                for marker, new_content in cells_to_update.items():
                    if marker in source:
                        cell.source = new_content
                        updated_count += 1
                        logger.info(f"Updated cell matching marker: {marker[:30]}...")
                        # Remove marker to avoid double updating
                        # (though in this case it doesn't matter much)
        
        logger.info(f"Saving updated notebook to {path}")
        with open(path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        logger.info(f"Improved {updated_count} cells in {path}")
            
    except Exception as e:
        logger.error(f"Failed to improve notebook: {str(e)}")

if __name__ == "__main__":
    improve_notebook()
