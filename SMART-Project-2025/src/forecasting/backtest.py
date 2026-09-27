"""Turning forecasts into daily long-or-flat trades and scoring them.

Trading rule: at the close of day t-1, buy one share if the signal for day t is on,
and sell it at the close of day t. Profit is measured in dollars per share, with no
transaction costs.
"""

import numpy as np
import pandas as pd


def price_changes(prices: pd.Series, start: str, end: str) -> pd.Series:
    """Dollar change from the previous close for every day in [start, end]."""
    return prices.diff().loc[start:end]


def threshold_signal(predictions: pd.Series, delta: float) -> pd.Series:
    """Buy when the predicted log return is above delta."""
    return predictions > delta


def momentum_signal(returns: pd.Series, dates: pd.Index) -> pd.Series:
    """Buy when the previous day's actual return was positive."""
    return (returns.shift(1) > 0).reindex(dates, fill_value=False)


def random_signal(dates: pd.Index, p: float = 0.5, seed: int = 0) -> pd.Series:
    """Buy at random with probability p (a no-skill benchmark)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(dates)) < p, index=dates)


def oracle_signal(changes: pd.Series) -> pd.Series:
    """Buy only on days the price actually rises (the maximum achievable profit)."""
    return changes > 0


def buy_and_hold_signal(dates: pd.Index) -> pd.Series:
    """Hold every day."""
    return pd.Series(True, index=dates)


def daily_pnl(signal: pd.Series, changes: pd.Series) -> pd.Series:
    """Profit or loss per day: the price change on days the strategy holds, else 0."""
    signal = signal.reindex(changes.index, fill_value=False).astype(bool)
    return changes.where(signal, 0.0)


def summarize(signal: pd.Series, changes: pd.Series) -> dict:
    """Total profit, number of trades, and share of trades that made money."""
    pnl = daily_pnl(signal, changes)
    traded = pnl[signal.reindex(changes.index, fill_value=False).astype(bool)]
    return {
        "profit": float(pnl.sum()),
        "trades": int(len(traded)),
        "win_rate": float((traded > 0).mean()) if len(traded) else float("nan"),
    }
