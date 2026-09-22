from pathlib import Path

import pandas as pd


def prepare_market_data(options, underlying):
    options = options.copy()
    underlying = underlying.copy()

    options["date"] = pd.to_datetime(options["date"])
    options["expiration"] = pd.to_datetime(options["expiration"])
    underlying["date"] = pd.to_datetime(underlying["date"])

    underlying_prices = underlying[["date", "close"]]

    data = options.merge(
        underlying_prices,
        on="date",
        how="left",
    )

    data["T"] = (
        data["expiration"] - data["date"]
    ).dt.days / 365.0

    data = data[data["T"] > 0].copy()

    return data


def load_spy_data(data_dir, year):
    data_dir = Path(data_dir)

    options = pd.read_parquet(
        data_dir / f"options_{year}.parquet"
    )

    underlying = pd.read_parquet(
        data_dir / "underlying_prices.parquet"
    )

    return prepare_market_data(options, underlying)