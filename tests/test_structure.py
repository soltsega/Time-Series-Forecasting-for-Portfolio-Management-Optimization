import os
import pytest

def test_data_files_exist():
    data_path = "data/processed"
    assets = ["TSLA", "BND", "SPY"]
    for asset in assets:
        file_path = f"{data_path}/{asset}_final_processed.csv"
        assert os.path.exists(file_path), f"Processed data for {asset} missing at {file_path}"

def test_notebooks_exist():
    expected_notebooks = [
        "notebooks/EDA.ipynb",
        "notebooks/Task-2-TimeSeriesForecasting.ipynb",
        "notebooks/Task-4-5-PortfolioAndBacktesting.ipynb"
    ]
    for nb in expected_notebooks:
        assert os.path.exists(nb), f"Required notebook {nb} missing"
