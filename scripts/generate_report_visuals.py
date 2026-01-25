import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Setup paths
data_path = 'data/processed' if os.path.exists('data/processed') else '../data/processed'
output_path = 'deliverables/visuals'
if not os.path.exists(output_path):
    os.makedirs(output_path)

assets = ['TSLA', 'BND', 'SPY']
dfs = {}

def load_data(ticker):
    fpath = os.path.join(data_path, f'{ticker}_final_processed.csv')
    df = pd.read_csv(fpath, index_col='Date', parse_dates=True)
    return df

for a in assets:
    dfs[a] = load_data(a)

# 1. Price Trends Line Chart
plt.figure(figsize=(15, 7))
for a in assets:
    plt.plot(dfs[a].index, dfs[a]['Close'], label=f'{a} (Close)')
plt.title('Asset Closing Price Trends (2015 - 2026)', fontsize=15)
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_path, 'price_trends.png'), dpi=300)
plt.close()

# 2. Daily Percentage Change (Volatility)
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
colors = ['#e74c3c', '#2980b9', '#27ae60']
for i, a in enumerate(assets):
    ret = dfs[a]['Close'].pct_change()
    axes[i].plot(ret.index, ret, color=colors[i], alpha=0.7)
    axes[i].set_title(f'{a} Daily Returns (Volatility Spectrum)', fontsize=12)
    axes[i].set_ylabel('% Change')
    axes[i].grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'volatility_spectrum.png'), dpi=300)
plt.close()

# 3. Rolling Mean & Std Dev Plots
fig, axes = plt.subplots(3, 1, figsize=(15, 20))
for i, a in enumerate(assets):
    df = dfs[a]
    rolling_mean = df['Close'].rolling(window=50).mean()
    rolling_std = df['Close'].rolling(window=50).std()
    
    ax1 = axes[i]
    ax1.plot(df.index, df['Close'], label=f'{a} Price', color='blue', alpha=0.3)
    ax1.plot(rolling_mean.index, rolling_mean, label='50-Day Rolling Mean', color='red', linewidth=1.5)
    ax1.set_ylabel('Price (USD)')
    
    ax2 = ax1.twinx()
    ax2.plot(rolling_std.index, rolling_std, label='50-Day Rolling Volatility', color='green', linestyle=':', alpha=0.7)
    ax2.set_ylabel('Volatility (Std Dev)', color='green')
    
    ax1.set_title(f'{a} Price Support vs. Rolling Volatility Analysis', fontsize=12)
    ax1.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'rolling_stats_spectrum.png'), dpi=300)
plt.close()

print(f"Visuals generated in {output_path}/")
