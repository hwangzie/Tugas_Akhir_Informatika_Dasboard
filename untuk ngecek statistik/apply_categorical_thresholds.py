import pandas as pd
import numpy as np
import json

def load_quartile_thresholds():
    """
    Load quartile thresholds from 2024 data
    """
    # Read the monthly hotspot data
    df = pd.read_csv('improved_monthly_hotspot_forecasts_2025.csv')
    df['year_month'] = pd.to_datetime(df['year_month'])
    df = df.set_index('year_month')
    
    # Filter data for 2024 to calculate quartiles
    train_data = df[(df.index.year == 2025)]
    
    # Get tile columns
    tile_columns = [col for col in df.columns if col.startswith('tile_')]
    
    # Calculate quartiles for each tile
    quartile_thresholds = {}
    
    for tile in tile_columns:
        # Get non-zero values for quartile calculation
        non_zero_values = train_data[tile][train_data[tile] > 0]
        
        if len(non_zero_values) > 0:
            q25 = non_zero_values.quantile(0.25)
            q50 = non_zero_values.quantile(0.50)
            q75 = non_zero_values.quantile(0.75)
            
            quartile_thresholds[tile] = {
                'q25': float(q25),
                'q50': float(q50),
                'q75': float(q75)
            }
        else:
            # If no activity, set thresholds to 0
            quartile_thresholds[tile] = {
                'q25': 0.0,
                'q50': 0.0,
                'q75': 0.0
            }
    
    return quartile_thresholds


def categorize_prediction(value, thresholds):
    """
    Categorize prediction value based on quartile thresholds:
    - No Activity: value ≈ 0 (< 0.5)
    - Low: 0.5 ≤ value ≤ Q50
    - Medium: Q50 < value ≤ Q75
    - High: value > Q75
    """
    if value <= thresholds['q25']:
        return 'Low'
    elif value <= thresholds['q50']:
        return 'Medium'
    elif value <= thresholds['q75']:
        return 'High'
    else:
        return 'High'

def apply_thresholds_to_predictions(predictions_df, thresholds):
    """
    Apply categorical thresholds to prediction values
    
    Args:
        predictions_df: DataFrame with predictions (rows=months, cols=tiles)
        thresholds: Dictionary of quartile thresholds per tile
    
    Returns:
        DataFrame with categorical labels
    """
    categorical_predictions = predictions_df.copy()
    
    for tile in predictions_df.columns:
        if tile in thresholds:
            categorical_predictions[tile] = predictions_df[tile].apply(
                lambda x: categorize_prediction(x, thresholds[tile])
            )
    
    return categorical_predictions

def main():
    print("Loading quartile thresholds from 2023-2024 data...")
    print("=" * 70)
    
    # Load thresholds
    thresholds = load_quartile_thresholds()
    
    # Save thresholds to JSON for future use
    with open('quartile_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print("Quartile thresholds saved to: quartile_thresholds.json")
    print("\nThresholds per tile:")
    print("-" * 70)
    
    for tile, vals in thresholds.items():
        if vals['q50'] > 0:
            print(f"{tile}: Q25={vals['q25']:.2f}, Q50={vals['q50']:.2f}, Q75={vals['q75']:.2f}")
    
    print("\n" + "=" * 70)
    print("How to use these thresholds with your predictions:")
    print("-" * 70)

if __name__ == "__main__":
    main()
