import pandas as pd
import json
from apply_categorical_thresholds import categorize_prediction, load_quartile_thresholds

# Load the predictions CSV
print("Loading predictions from better_LSTM_monthly_hotspot_forecasts_2025.csv...")
predictions_df = pd.read_csv('improved_monthly_hotspot_forecasts_2025.csv')
# predictions_df = pd.read_csv('monthly_hotspot_forecasts_2025.csv')
predictions_df['year_month'] = pd.to_datetime(predictions_df['year_month'])
predictions_df = predictions_df.set_index('year_month')

print(f"Loaded {len(predictions_df)} months of predictions")
print(f"Date range: {predictions_df.index.min().strftime('%Y-%m')} to {predictions_df.index.max().strftime('%Y-%m')}")

# Load quartile thresholds
print("\nLoading quartile thresholds from 2023-2024 data...")
thresholds = load_quartile_thresholds()

# Apply categorical classification
print("\n" + "=" * 70)
print("APPLYING CATEGORICAL THRESHOLDS TO PREDICTIONS")
print("=" * 70)

categorical_df = predictions_df.copy()

# Get tile columns
tile_columns = [col for col in predictions_df.columns if col.startswith('tile_')]

# Apply categorization to each tile
for tile in tile_columns:
    categorical_df[tile] = predictions_df[tile].apply(
        lambda x: categorize_prediction(x, thresholds[tile])
    )

# Save categorical predictions
output_file = 'categorical_forecasts_2025.csv'
categorical_df.to_csv(output_file)

print(f"\nCategorical predictions saved to: {output_file}")

# Display results
print("\n" + "=" * 70)
print("CATEGORICAL FORECAST SUMMARY")
print("=" * 70)

print("\nPredictions by tile and month:")
print("-" * 70)

for tile in tile_columns:
    # Check if tile has any activity
    tile_values = predictions_df[tile]
    if tile_values.max() >= 0.5:
        print(f"\n{tile}:")
        print(f"  Thresholds: Q50={thresholds[tile]['q50']:.2f}, Q75={thresholds[tile]['q75']:.2f}")
        
        for idx, month in enumerate(predictions_df.index):
            pred_value = predictions_df.loc[month, tile]
            category = categorical_df.loc[month, tile]
            month_str = month.strftime('%Y-%m')
            print(f"    {month_str}: {pred_value:.2f} → {category}")

# Category distribution per tile
print("\n" + "=" * 70)
print("CATEGORY DISTRIBUTION PER TILE (2025)")
print("=" * 70)

for tile in tile_columns:
    counts = categorical_df[tile].value_counts()
    if 'No Activity' not in counts or counts['No Activity'] < 12:
        print(f"\n{tile}:")
        for category in ['High', 'Medium', 'Low', 'No Activity']:
            if category in counts:
                print(f"  {category}: {counts[category]} months")

# High-risk months
print("\n" + "=" * 70)
print("HIGH-RISK MONTHS (Tiles with 'High' or 'Medium' predictions)")
print("=" * 70)

for month in categorical_df.index:
    month_str = month.strftime('%Y-%m')
    high_tiles = categorical_df.loc[month][categorical_df.loc[month] == 'High']
    medium_tiles = categorical_df.loc[month][categorical_df.loc[month] == 'Medium']
    
    if len(high_tiles) > 0 or len(medium_tiles) > 0:
        print(f"\n{month_str}:")
        
        if len(high_tiles) > 0:
            print(f"  HIGH ({len(high_tiles)} tiles):")
            for tile in high_tiles.index:
                value = predictions_df.loc[month, tile]
                print(f"    {tile}: {value:.2f}")
        
        if len(medium_tiles) > 0:
            print(f"  MEDIUM ({len(medium_tiles)} tiles):")
            for tile in medium_tiles.index:
                value = predictions_df.loc[month, tile]
                print(f"    {tile}: {value:.2f}")

# Summary statistics
print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)

# Count total by category
all_categories = []
for tile in tile_columns:
    all_categories.extend(categorical_df[tile].tolist())

category_counts = pd.Series(all_categories).value_counts()
print("\nTotal predictions across all tiles and months:")
for category in ['High', 'Medium', 'Low', 'No Activity']:
    if category in category_counts:
        percentage = (category_counts[category] / len(all_categories)) * 100
        print(f"  {category}: {category_counts[category]} ({percentage:.1f}%)")

print("\n" + "=" * 70)
print("Categorization complete!")
