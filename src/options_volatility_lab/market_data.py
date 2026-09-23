from pathlib import Path

import pandas as pd

from options_volatility_lab.dividends import add_trailing_dividend_yield
from options_volatility_lab.rates import (
    add_risk_free_rate,
    load_treasury_data,
)


def prepare_market_data(options, underlying, treasury):
    options = options.copy()
    underlying = underlying.copy()

    options["date"] = pd.to_datetime(options["date"])
    options["expiration"] = pd.to_datetime(options["expiration"])
    underlying["date"] = pd.to_datetime(underlying["date"])

    underlying = add_trailing_dividend_yield(underlying)

    underlying_prices = underlying[["date", "close", "trailing_dividend_yield"]]

    data = options.merge(
        underlying_prices,
        on="date",
        how="left",
    )

    data["T"] = (
        data["expiration"] - data["date"]
    ).dt.days / 365.0

    data = data[data["T"] > 0].copy()
    data = add_risk_free_rate(data, treasury)

    return data


def load_spy_data(data_dir, treasury_path, year):
    data_dir = Path(data_dir)

    options = pd.read_parquet(
        data_dir / f"options_{year}.parquet"
    )

    underlying = pd.read_parquet(
        data_dir / "underlying_prices.parquet"
    )

    treasury = load_treasury_data(treasury_path)

    return prepare_market_data(options, underlying, treasury)