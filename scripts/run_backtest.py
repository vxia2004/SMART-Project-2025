"""Backtest the chosen models on the 2019 test year and write tables and figures to results/.

Usage:
    python scripts/run_backtest.py                  # MSFT and TSLA
    python scripts/run_backtest.py --tickers AAPL   # any other ticker

Model predictions are cached in results/predictions/, so rerunning only redoes
the trading simulation. Delete that folder to refit the models.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting import backtest, plots  # noqa: E402
from forecasting.config import DELTAS, MODELS, SPLITS, TICKERS  # noqa: E402
from forecasting.data import load_prices, log_returns  # noqa: E402
from forecasting.models import mse, walk_forward  # noqa: E402

RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
FIG_DIR = RESULTS / "figures"


def test_predictions(ticker: str, name: str, returns: pd.Series) -> pd.DataFrame:
    """Walk-forward predictions for the test year, loaded from cache when available."""
    path = PRED_DIR / f"{ticker}_{name}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"  fitting {name} ...", flush=True)
    start, end = SPLITS["test"]
    results = walk_forward(returns, start, end, MODELS[name])
    results.to_csv(path)
    return results


def run_ticker(ticker: str) -> list[dict]:
    print(f"{ticker}:")
    prices = load_prices(ticker)
    returns = log_returns(prices)
    start, end = SPLITS["test"]
    changes = backtest.price_changes(prices, start, end)
    dates = changes.index

    rows, strategies = [], {}
    for name in MODELS:
        results = test_predictions(ticker, name, returns)
        for delta in DELTAS:
            signal = backtest.threshold_signal(results["prediction"], delta)
            label = f"{name} (δ = {delta:g})"
            rows.append({"ticker": ticker, "strategy": label, "test_mse": mse(results),
                         **backtest.summarize(signal, changes)})
            if delta == 0:
                strategies[name] = signal

    baselines = {
        "Momentum": backtest.momentum_signal(returns, dates),
        "Random (p = 0.5)": backtest.random_signal(dates, seed=0),
        "Buy and hold": backtest.buy_and_hold_signal(dates),
        "Oracle (perfect foresight)": backtest.oracle_signal(changes),
    }
    for label, signal in baselines.items():
        rows.append({"ticker": ticker, "strategy": label, "test_mse": float("nan"),
                     **backtest.summarize(signal, changes)})

    train_start, train_end = SPLITS["train"]
    plots.plot_prices_and_returns(prices.loc[train_start:train_end], returns.loc[train_start:train_end],
                                  ticker, FIG_DIR / f"{ticker}_prices_returns.png")
    plots.plot_trades(prices, strategies["ARMA"], changes, f"{ticker}: ARMA(1,1) trades, 2019",
                      FIG_DIR / f"{ticker}_arma_trades.png")
    plots.plot_cumulative_profit(
        {**strategies, "Momentum": baselines["Momentum"], "Buy and hold": baselines["Buy and hold"]},
        changes, f"{ticker}: cumulative profit, 2019 (δ = 0)", FIG_DIR / f"{ticker}_cumulative_profit.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    args = parser.parse_args()

    for folder in (PRED_DIR, FIG_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    rows = [row for ticker in args.tickers for row in run_ticker(ticker)]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "backtest_summary.csv", index=False)

    profit = table.pivot(index="strategy", columns="ticker", values="profit")
    profit = profit.reindex(table["strategy"].drop_duplicates())
    profit.columns = [f"{c} profit, 2019 ($/share)" for c in profit.columns]
    (RESULTS / "backtest_summary.md").write_text(profit.round(2).to_markdown() + "\n")
    print("\n" + profit.round(2).to_string())
    print(f"\nWrote {RESULTS / 'backtest_summary.csv'} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
