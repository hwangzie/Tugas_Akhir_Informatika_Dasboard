# LSTM Hotspot Forecasting — Inference Only

Runs your already-trained per-tile LSTM models to generate forecasts,
without retraining anything.

## 1. Folder setup

```
lstm_inference/
├── data/
│   └── monthly_hotspot_sum.csv      <- copy the SAME csv used to train
├── models/
│   └── best_model_tile_0.h5         <- copy all your trained .h5 files here
│   └── best_model_tile_1.h5
│   └── ...
├── output/                          <- forecasts.csv appears here after running
├── predict.py
└── requirements.txt
```

Put every `best_model_tile_X.h5` file the notebook produced into `models/`,
and put `monthly_hotspot_sum.csv` into `data/`.

## 2. Why the CSV is needed

The notebook scales each tile's data with a `MinMaxScaler` fit in memory,
but never saves that scaler to disk. Since `MinMaxScaler` has no
randomness — it's just the min/max of the series — re-running the same
fitting code on the same CSV reproduces the *exact* scaler used in
training. That's what `predict.py` does internally before calling
`model.predict()`. If the CSV doesn't match what a model was trained on,
forecasts will be wrong.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run

Forecast all tiles that have a model in `models/`, 12 months ahead:

```bash
python predict.py
```

Forecast a single tile:

```bash
python predict.py --tile tile_5
```

Forecast a different horizon (e.g. 6 months):

```bash
python predict.py --months 6
```

Results are written to `output/forecasts.csv`, one row per future month,
one column per tile — same shape as `monthly_hotspot_forecasts_2025.csv`
in the original notebook.
