import pandas as pd

# Load both datasets
dataset_clean = pd.read_csv('features_clean.csv', index_col='Date', parse_dates=True)
dataset_leaky = pd.read_csv('features_leaky.csv', index_col='Date', parse_dates=True)

# TIME SERIES SPLIT (80% train, 20% test)
n = len(dataset_clean)
split_idx = int(0.8 * n)

# CLEAN DATASET
train_clean = dataset_clean.iloc[:split_idx]
test_clean = dataset_clean.iloc[split_idx:]

# LEAKY DATASET
train_leaky = dataset_leaky.iloc[:split_idx]
test_leaky = dataset_leaky.iloc[split_idx:]

print("="*60)
print("TRAIN/TEST SPLIT (Time Series)")
print("="*60)

print(f"\nCLEAN Dataset:")
print(f"  Train: {train_clean.shape[0]} rows ({train_clean.index.min().date()} to {train_clean.index.max().date()})")
print(f"  Test:  {test_clean.shape[0]} rows ({test_clean.index.min().date()} to {test_clean.index.max().date()})")

print(f"\nLEAKY Dataset:")
print(f"  Train: {train_leaky.shape[0]} rows ({train_leaky.index.min().date()} to {train_leaky.index.max().date()})")
print(f"  Test:  {test_leaky.shape[0]} rows ({test_leaky.index.min().date()} to {test_leaky.index.max().date()})")

# Save all splits
train_clean.to_csv('train_data_clean.csv')
test_clean.to_csv('test_data_clean.csv')
train_leaky.to_csv('train_data_leaky.csv')
test_leaky.to_csv('test_data_leaky.csv')

print("\n✓ Saved all train/test splits")
print("  - train_data_clean.csv")
print("  - test_data_clean.csv")
print("  - train_data_leaky.csv")
print("  - test_data_leaky.csv")
