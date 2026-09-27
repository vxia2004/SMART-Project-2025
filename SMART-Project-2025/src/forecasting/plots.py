"""Figures for the README and poster."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import daily_pnl

BLUE = "#2057e7"


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prices_and_returns(prices: pd.Series, returns: pd.Series, ticker: str, path) -> None:
    """Closing prices next to log returns, showing why returns are modeled instead of prices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(prices, color=BLUE)
    ax1.set(title=f"{ticker} daily close", ylabel="Price ($)")
    ax2.plot(returns, color=BLUE, linewidth=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set(title=f"{ticker} daily log returns", ylabel="Log return")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    _finish(fig, path)


def plot_trades(prices: pd.Series, signal: pd.Series, changes: pd.Series, title: str, path) -> None:
    """Price path with winning (green) and losing (red) trades marked."""
    signal = signal.reindex(changes.index, fill_value=False).astype(bool)
    wins = signal & (changes > 0)
    losses = signal & (changes < 0)
    window = prices.loc[changes.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window, color=BLUE, label="Close")
    ax.scatter(window.index[wins], window[wins], color="green", marker="^", s=18, label="Winning trade")
    ax.scatter(window.index[losses], window[losses], color="red", marker="v", s=18, label="Losing trade")
    ax.set(title=title, ylabel="Price ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, path)


def plot_cumulative_profit(strategies: dict, changes: pd.Series, title: str, path) -> None:
    """Cumulative dollar profit over the test year for several strategies."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, signal in strategies.items():
        ax.plot(daily_pnl(signal, changes).cumsum(), label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title=title, ylabel="Cumulative profit ($ per share)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)
