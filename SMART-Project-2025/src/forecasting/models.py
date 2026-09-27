"""One-step-ahead forecasters evaluated with an expanding (walk-forward) window.

For every day t in the evaluation period, the model is refit on all returns before t
and asked to predict the return on day t, so no future information leaks into a forecast.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import RANDOM_SEED
from .data import lag_features


def _sarimax_forecast(history: pd.Series, order, seasonal_order=(0, 0, 0, 0)) -> float:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        # statsmodels warns about convergence and frequency on many refits; the fits are still used.
        warnings.simplefilter("ignore")
        fitted = SARIMAX(history.to_numpy(), order=order, seasonal_order=seasonal_order).fit(disp=False)
    return float(fitted.forecast(1)[0])


def walk_forward(returns: pd.Series, start: str, end: str, spec: dict) -> pd.DataFrame:
    """Walk-forward predictions for every date in [start, end].

    returns: log returns covering the training history and the evaluation period.
    spec:    one entry of config.MODELS.
    Returns a frame indexed by date with 'prediction' and 'actual' columns.
    """
    eval_dates = returns.loc[start:end].index
    kind = spec["kind"]

    if kind == "random_forest":
        frame = lag_features(returns, spec["n_lags"])
        features = [c for c in frame.columns if c.startswith("lag_")]

    predictions = []
    for date in eval_dates:
        if kind == "sarimax":
            history = returns.loc[returns.index < date]
            pred = _sarimax_forecast(history, spec["order"], spec.get("seasonal_order", (0, 0, 0, 0)))
        elif kind == "random_forest":
            train = frame.loc[frame.index < date]
            model = RandomForestRegressor(n_estimators=spec["n_estimators"], random_state=RANDOM_SEED)
            model.fit(train[features], train["target"])
            pred = float(model.predict(frame.loc[[date], features])[0])
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        predictions.append(pred)

    return pd.DataFrame(
        {"prediction": predictions, "actual": returns.loc[eval_dates].to_numpy()},
        index=eval_dates,
    )


def mse(results: pd.DataFrame) -> float:
    """Mean squared error of a walk_forward result."""
    return float(np.mean((results["prediction"] - results["actual"]) ** 2))
