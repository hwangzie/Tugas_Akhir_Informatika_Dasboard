"""
Inference script for the Monthly LSTM Hotspot Forecasting models.

This loads the already-trained per-tile .h5 models (from
Monthly_LSTM_Hotspot_Forecasting_hyperparameter_mlflow.ipynb) and
generates rolling monthly forecasts, WITHOUT retraining anything.

Folder layout expected:

    lstm_inference/
    ├── data/
    │   ├── monthly_hotspot_sum_train.csv  <- FROZEN copy of the csv the
    │   │                                      models were originally
    │   │                                      trained on. Never edit
    │   │                                      this file again.
    │   └── monthly_hotspot_sum.csv        <- LIVE file. Starts as a copy
    │                                          of the train csv; append
    │                                          new real monthly rows to
    │                                          the bottom as they arrive
    │                                          (e.g. 2025 actuals). Never
    │                                          edit existing rows.
    ├── models/
    │   └── best_model_tile_0.h5, best_model_tile_1.h5, ...
    ├── output/                          <- forecasts get written here
    └── predict.py                       <- this file

Why two CSVs instead of one:
    The notebook fit a MinMaxScaler for the target and another for the
    temporal features (month, year, time_trend, month_sin, month_cos)
    PER TILE, in memory, and never saved them to disk. MinMaxScaler has
    no randomness, so re-fitting it on the ORIGINAL training csv
    reproduces the exact scaler used in training.

    But 'year' and 'time_trend' both grow every time you add a new
    month of real data. If you re-fit the scaler on an ever-growing csv,
    the max value used for normalization keeps changing -- which
    silently changes the scaled value of EVERY historical row, not just
    the new ones. The model's learned weights assume the ORIGINAL
    scaling, so feeding it a different scaling produces wrong
    predictions with no error raised.

    The fix: fit scalers once, only on the frozen original training csv.
    Use the live/growing csv only to TRANSFORM (never re-fit) the actual
    last-12-months of real data, which becomes the input context for the
    rolling forecast. This is exactly what forecast_next_year() in the
    notebook already did for *predicted* future months -- extending
    time_trend/year past the range the scaler was fit on is expected and
    matches the notebook's own behavior.

Usage:
    python predict.py                     # forecast all tiles, 12 months ahead
    python predict.py --tile tile_5        # forecast a single tile
    python predict.py --months 6           # forecast a different horizon
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config - match the training notebook
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
TRAIN_CSV_PATH = BASE_DIR / "data" / "monthly_hotspot_sum_train.csv"  # frozen
LIVE_CSV_PATH = BASE_DIR / "data" / "monthly_hotspot_sum.csv"         # can grow
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

LOOKBACK_MONTHS = 12
N_FEATURES = 6  # value, month, year, time_trend, month_sin, month_cos


# ---------------------------------------------------------------------------
# Same preprocessing helpers as the training notebook
# (kept in sync with cell "4. Data Preprocessing Functions")
# ---------------------------------------------------------------------------
def load_monthly_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df.set_index("date", inplace=True)
    df.drop("year_month", axis=1, inplace=True)
    df.sort_index(inplace=True)
    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": df.index.month,
            "year": df.index.year - df.index.year.min(),
            "time_trend": range(len(df)),
            "month_sin": np.sin(2 * np.pi * df.index.month / 12),
            "month_cos": np.cos(2 * np.pi * df.index.month / 12),
        },
        index=df.index,
    )


def fit_tile_scalers(df_train: pd.DataFrame, tile_name: str):
    """
    Fit scalers ONLY on the frozen original training data -- exactly what
    prepare_tile_data() did during training. These must NEVER be re-fit
    on an extended/live csv, or scaling will silently drift (see module
    docstring).
    """
    tile_values = df_train[tile_name].values.reshape(-1, 1)
    features = build_temporal_features(df_train)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(tile_values)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features)

    return scaler, feature_scaler


def transform_with_frozen_scalers(df_live: pd.DataFrame, tile_name: str, scaler, feature_scaler):
    """
    Transforms (never fits) the live/current data using scalers that were
    fit once on the frozen training csv. Rows beyond the original training
    range (e.g. newly appended 2025 actuals) are legitimately extrapolated
    past [0, 1] for year/time_trend -- same as the notebook's own
    forecast_next_year() does for predicted future months.
    """
    tile_values = df_live[tile_name].values.reshape(-1, 1)
    scaled_data = scaler.transform(tile_values).flatten()

    features = build_temporal_features(df_live)
    scaled_features = feature_scaler.transform(features)

    return scaled_data, scaled_features


def sanity_check_prefix_unchanged(df_train: pd.DataFrame, df_live: pd.DataFrame, tile_columns):
    """
    Warn (don't crash) if the historical rows shared by both files don't
    match -- editing old rows instead of only appending new ones would
    silently break the frozen-scaler assumption above.
    """
    n = len(df_train)
    if len(df_live) < n:
        print(f"WARNING: {LIVE_CSV_PATH.name} has fewer rows than {TRAIN_CSV_PATH.name}. "
              "It should be the training csv plus any newly appended months.")
        return
    live_prefix = df_live.iloc[:n]
    train_prefix = df_train
    if not live_prefix.index.equals(train_prefix.index):
        print("WARNING: the first rows' dates in the live csv don't match the train csv. "
              "Only append new rows at the bottom -- don't reorder or edit existing ones.")
        return
    mismatch = ~np.isclose(live_prefix[tile_columns].values, train_prefix[tile_columns].values)
    if mismatch.any():
        print("WARNING: some historical values differ between the train csv and the live csv. "
              "This breaks the frozen-scaler assumption -- only append new rows, never edit old ones.")


def warn_if_extrapolating_far(df_train: pd.DataFrame, df_live: pd.DataFrame, threshold_months=24):
    """
    The model's weights never change after training -- only the input
    window does. The further the live data grows past the original
    training period, the further year/time_trend get extrapolated beyond
    the range the model actually learned on. This doesn't error, but
    forecast quality can degrade. Nudge toward periodic retraining.
    """
    months_beyond_train = len(df_live) - len(df_train)
    if months_beyond_train > threshold_months:
        years_beyond = months_beyond_train / 12
        print(
            f"NOTE: the live csv now has {months_beyond_train} months ({years_beyond:.1f} years) "
            "of data beyond what these models were trained on. The models' weights haven't "
            "changed since training -- only the input window has -- so forecasts this far out "
            "are an increasingly large extrapolation. Consider re-running the training notebook "
            "on the full updated dataset to refresh the models and scalers."
        )


# ---------------------------------------------------------------------------
# Rolling forecast (same logic as forecast_next_year() in the notebook)
# ---------------------------------------------------------------------------
def forecast_next_months(
    model,
    scaler,
    feature_scaler,
    last_12_months_data,
    last_12_months_features,
    start_date,
    year_min,
    total_history_len,
    months_ahead=12,
):
    forecasts = []
    current_sequence = last_12_months_data.copy()
    current_features = last_12_months_features.copy()

    future_dates = pd.date_range(start=start_date, periods=months_ahead, freq="MS")

    for i, date in enumerate(future_dates):
        future_feature = pd.DataFrame(
            {
                "month": [date.month],
                "year": [date.year - year_min],
                "time_trend": [total_history_len + i],
                "month_sin": [np.sin(2 * np.pi * date.month / 12)],
                "month_cos": [np.cos(2 * np.pi * date.month / 12)],
            }
        )
        scaled_future_feature = feature_scaler.transform(future_feature)

        input_sequence = []
        for j in range(len(current_sequence)):
            input_sequence.append(
                [
                    current_sequence[j],
                    current_features[j, 0],
                    current_features[j, 1],
                    current_features[j, 2],
                    current_features[j, 3],
                    current_features[j, 4],
                ]
            )
        input_sequence = np.array(input_sequence).reshape(1, len(current_sequence), N_FEATURES)

        next_pred = model.predict(input_sequence, verbose=0)[0, 0]
        forecasts.append(next_pred)

        current_sequence = np.append(current_sequence[1:], next_pred)
        current_features = np.vstack([current_features[1:], scaled_future_feature])

    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_unscaled = scaler.inverse_transform(forecasts).flatten()
    forecasts_unscaled = np.maximum(0, forecasts_unscaled)  # hotspots can't be negative

    return forecasts_unscaled, future_dates


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def find_model_path(tile_name: str) -> Path:
    candidate = MODELS_DIR / f"best_model_{tile_name}.h5"
    if candidate.exists():
        return candidate
    matches = list(MODELS_DIR.glob(f"*{tile_name}*.h5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No .h5 model found for {tile_name} in {MODELS_DIR}")


def load_model_for_inference(model_path: Path):
    # compile=False: we only need forward passes (model.predict), so there's
    # no need to register the custom r_squared metric used during training.
    return tf.keras.models.load_model(model_path, compile=False)


def tile_names_from_models_dir():
    names = []
    for path in sorted(MODELS_DIR.glob("*.h5")):
        m = re.match(r"best_model_(tile_.+)\.h5", path.name)
        names.append(m.group(1) if m else path.stem)
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Forecast future months using trained per-tile LSTM models.")
    parser.add_argument("--tile", type=str, default=None, help="Forecast a single tile, e.g. tile_5. Default: all tiles found in models/")
    parser.add_argument("--months", type=int, default=12, help="Number of months ahead to forecast (default: 12)")
    args = parser.parse_args()

    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected the FROZEN original training csv at {TRAIN_CSV_PATH}. "
            "This must be an untouched copy of whatever csv the models were trained on."
        )
    if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.h5")):
        raise FileNotFoundError(f"No .h5 model files found in {MODELS_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    df_train = load_monthly_data(TRAIN_CSV_PATH)
    if LIVE_CSV_PATH.exists():
        df_live = load_monthly_data(LIVE_CSV_PATH)
    else:
        print(f"No {LIVE_CSV_PATH.name} found -- using the train csv as the live data too.")
        df_live = df_train

    tile_columns = [c for c in df_train.columns if c.startswith("tile_")]
    sanity_check_prefix_unchanged(df_train, df_live, tile_columns)
    warn_if_extrapolating_far(df_train, df_live)

    year_min = df_train.index.year.min()
    start_date = (df_live.index.max() + pd.DateOffset(months=1)).replace(day=1)
    print(f"Forecasting starting from {start_date.strftime('%Y-%m')}")

    tiles = [args.tile] if args.tile else tile_names_from_models_dir()

    all_forecasts = {}
    for tile_name in tiles:
        if tile_name not in df_train.columns:
            print(f"Skipping {tile_name}: not found in {TRAIN_CSV_PATH.name}")
            continue

        print(f"Forecasting {tile_name}...")
        model_path = find_model_path(tile_name)
        model = load_model_for_inference(model_path)

        # Scalers: fit ONCE on the frozen training csv.
        scaler, feature_scaler = fit_tile_scalers(df_train, tile_name)

        # Context window: transform (don't fit) the live/current data.
        scaled_data, scaled_features = transform_with_frozen_scalers(df_live, tile_name, scaler, feature_scaler)
        last_12_data = scaled_data[-LOOKBACK_MONTHS:]
        last_12_features = scaled_features[-LOOKBACK_MONTHS:]

        forecast_values, forecast_dates = forecast_next_months(
            model,
            scaler,
            feature_scaler,
            last_12_data,
            last_12_features,
            start_date,
            year_min,
            total_history_len=len(df_live),
            months_ahead=args.months,
        )

        all_forecasts[tile_name] = forecast_values
        print(f"  {tile_name}: {forecast_values.round(2).tolist()}")

    if not all_forecasts:
        print("No forecasts generated.")
        return

    result_df = pd.DataFrame(all_forecasts, index=[d.strftime("%Y-%m") for d in forecast_dates])
    result_df.index.name = "year_month"
    out_path = OUTPUT_DIR / "forecasts.csv"
    result_df.to_csv(out_path)
    print(f"\nSaved forecasts for {len(all_forecasts)} tile(s) to {out_path}")


if __name__ == "__main__":
    main()