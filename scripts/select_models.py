"""Model selection on the 2018 validation year.

Produces ACF/PACF plots of the 2015-2017 training returns, AIC for candidate AR and
ARMA orders, and walk-forward validation MSE for each candidate. The orders chosen
from this step are recorded in src/forecasting/config.py.

Usage:
    python scripts/select_models.py --tickers MSFT TSLA
    python scripts/select_models.py --skip-sarma      # SARMA refits are slow
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecasting.config import SPLITS, TICKERS  # noqa: E402
from forecasting.data import load_prices, log_returns  # noqa: E402
from forecasting.models import mse, walk_forward  # noqa: E402

RESULTS = ROOT / "results"

CANDIDATES = {
    **{f"AR({p})": {"kind": "sarimax", "order": (p, 0, 0)} for p in (1, 2, 7)},
    **{f"ARMA({p},{q})": {"kind": "sarimax", "order": (p, 0, q)} for p in (1, 2) for q in (1, 2)},
    **{f"SARMA(1,1)x(1,1,{s})": {"kind": "sarimax", "order": (1, 0, 1), "seasonal_order": (1, 0, 1, s)}
       for s in (18, 27, 50, 62)},
    **{f"RF lags 1-{k}": {"kind": "random_forest", "n_lags": k, "n_estimators": 100} for k in (3, 4, 5, 6, 7)},
}


def acf_pacf_plot(train_returns: pd.Series, ticker: str, path: Path, lags: int = 40) -> None:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train_returns, lags=lags, ax=ax1, title=f"{ticker} ACF (train)")
    plot_pacf(train_returns, lags=lags, ax=ax2, title=f"{ticker} PACF (train)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aic(train_returns: pd.Series, spec: dict) -> float:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(train_returns.to_numpy(), order=spec["order"],
                         seasonal_order=spec.get("seasonal_order", (0, 0, 0, 0))).fit(disp=False)
    return float(fitted.aic)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--skip-sarma", action="store_true", help="skip the slow seasonal candidates")
    args = parser.parse_args()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in args.tickers:
        returns = log_returns(load_prices(ticker)).loc[: SPLITS["val"][1]]
        train = returns.loc[SPLITS["train"][0]: SPLITS["train"][1]]
        acf_pacf_plot(train, ticker, RESULTS / "figures" / f"{ticker}_acf_pacf.png")

        for name, spec in CANDIDATES.items():
            if args.skip_sarma and name.startswith("SARMA"):
                continue
            print(f"{ticker}: {name}", flush=True)
            val = walk_forward(returns, *SPLITS["val"], spec)
            rows.append({
                "ticker": ticker,
                "model": name,
                "train_aic": aic(train, spec) if spec["kind"] == "sarimax" else float("nan"),
                "val_mse": mse(val),
            })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "model_selection.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
