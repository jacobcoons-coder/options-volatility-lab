import pandas as pd


def add_trailing_dividend_yield(underlying):
    underlying = underlying.copy()

    underlying["date"] = pd.to_datetime(underlying["date"])
    underlying = underlying.sort_values("date")

    rolling_dividends = (
        underlying.set_index("date")["dividend_amount"]
        .rolling("365D")
        .sum()
    )

    underlying["trailing_dividends"] = rolling_dividends.to_numpy()

    underlying["trailing_dividend_yield"] = (
        underlying["trailing_dividends"] / underlying["close"]
    )

    return underlying