import pandas as pd
import numpy as np

# Load original prices
close_prices = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)


def create_leaky_features(prices):
    """Create features WITH information leakage (CHEATING version)"""
    df = pd.DataFrame()
    
    for stock in prices.columns:
        stock_prices = prices[stock].copy()
        returns = stock_prices.pct_change()
        
        # LEAKY: Use full-sample statistics (computed with ALL data including future)
        full_sample_mean = returns.mean()
        full_sample_std = returns.std()
        
        # Centered features normalized to global mean/std (not rolling!)
        df[f'{stock}_return_centered'] = (returns - full_sample_mean) / full_sample_std
        
        # LEAKY: Use expanding window (sees future data)
        df[f'{stock}_momentum_leaky'] = returns.expanding().mean()
        
        # LEAKY: Volatility scaled by full-sample stats
        df[f'{stock}_volatility_leaky'] = returns / full_sample_std
    
    # Same target
    target = (prices.pct_change().mean(axis=1).shift(-1) > 0).astype(int)
    df['target'] = target
    df = df.dropna()
    
    return df


dataset_leaky = create_leaky_features(close_prices)

print(f"✓ Leaky dataset shape: {dataset_leaky.shape}")
print(f"  Target distribution: {dataset_leaky['target'].value_counts().to_dict()}\n")

dataset_leaky.to_csv('features_leaky.csv')
print("✓ Saved features_leaky.csv")
