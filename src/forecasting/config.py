"""Experiment settings: date splits, chosen model orders, and trading thresholds.

Model orders were chosen on the 2018 validation year (see scripts/select_models.py).
"""

DATA_START = "2015-01-01"
DATA_END = "2019-12-31"

SPLITS = {
    "train": ("2015-01-01", "2017-12-31"),
    "val": ("2018-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2019-12-31"),
}

TICKERS = ["MSFT", "TSLA"]

# Final hyperparameters used for the 2019 test-year backtest.
MODELS = {
    "AR": {"kind": "sarimax", "order": (1, 0, 0)},
    "ARMA": {"kind": "sarimax", "order": (1, 0, 1)},
    "SARMA": {"kind": "sarimax", "order": (1, 0, 1), "seasonal_order": (1, 0, 1, 62)},
    "RandomForest": {"kind": "random_forest", "n_lags": 5, "n_estimators": 100},
}

# A model "buys" for the next day when its predicted log return exceeds delta.
DELTAS = [0.0, 0.001]

RANDOM_SEED = 42
