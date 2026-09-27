"""Downloading prices, splitting by date, and building return features."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_END, DATA_START, SPLITS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(ticker: str, start: str = DATA_START, end: str = DATA_END) -> pd.Series:
    """Daily closing prices for one ticker, cached to data/<TICKER>.csv after the first download."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{ticker}.csv"
    if cache.exists():
        prices = pd.read_csv(cache, index_col="Date", parse_dates=True)["Close"]
    else:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"]
        prices = close[ticker] if isinstance(close, pd.DataFrame) else close
        prices = prices.rename("Close")
        prices.index.name = "Date"
        prices.to_frame().to_csv(cache)
    return prices.sort_index().rename(ticker)


def split(series: pd.Series, splits: dict = SPLITS) -> dict:
    """Split a date-indexed series into {'train', 'val', 'test'} by calendar ranges."""
    return {name: series.loc[start:end] for name, (start, end) in splits.items()}


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: log(P_t / P_{t-1})."""
    return np.log(prices).diff().dropna()


def lag_features(returns: pd.Series, n_lags: int) -> pd.DataFrame:
    """Frame with the target return plus its previous n_lags values as features."""
    frame = pd.DataFrame({"target": returns})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = returns.shift(lag)
    return frame.dropna()
