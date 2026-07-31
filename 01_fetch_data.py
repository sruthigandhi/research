import yfinance as yf
import pandas as pd

# List of stocks to download
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'XOM', 'JNJ', 'MA', 'V', 'WMT']

# Download 5 years of daily data
print("Fetching data...")
data = yf.download(tickers, start='2019-01-01', end='2024-12-31', progress=False)

# Keep only Close price
close_prices = data['Close']

# Save to CSV
close_prices.to_csv('stock_data.csv')

print(f"✓ Downloaded data for {len(tickers)} stocks")
print(f"  Shape: {close_prices.shape} (rows, columns)")
print(f"  Date range: {close_prices.index.min()} to {close_prices.index.max()}")
print(f"\n✓ Saved to stock_data.csv")
