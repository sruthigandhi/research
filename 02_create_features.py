import pandas as pd
import numpy as np

# Load the data
close_prices = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

print(f"Data shape: {close_prices.shape}")
print(f"Date range: {close_prices.index.min()} to {close_prices.index.max()}\n")

def create_target_and_features(prices):
    """Create features WITHOUT leaking future info (CLEAN version)"""
    df = pd.DataFrame()
    
    for stock in prices.columns:
        stock_prices = prices[stock].copy()
        
        # 1. Returns (% change from yesterday to today)
        df[f'{stock}_return'] = stock_prices.pct_change()
        
        # 2. Rolling volatility (20-day rolling std)
        df[f'{stock}_volatility'] = stock_prices.pct_change().rolling(window=20).std()
        
        # 3. Rolling mean (average price over last 20 days)
        df[f'{stock}_price_ma20'] = stock_prices.rolling(window=20).mean()
        
        # 4. Momentum (average return over last 5 days)
        df[f'{stock}_momentum'] = stock_prices.pct_change().rolling(window=5).mean()
    
    # TARGET: Next day's direction (1 = up, 0 = down)
    all_returns = prices.pct_change()
    target = (all_returns.mean(axis=1).shift(-1) > 0).astype(int)
    
    df['target'] = target
    df = df.dropna()
    
    return df

# Create clean dataset
dataset = create_target_and_features(close_prices)

print(f"✓ Clean dataset shape: {dataset.shape}")
print(f"  Target distribution: {dataset['target'].value_counts().to_dict()}")
print(f"  Baseline (always guess up): {dataset['target'].mean():.1%}\n")

# Save it
dataset.to_csv('features_clean.csv')
print("✓ Saved features_clean.csv")
