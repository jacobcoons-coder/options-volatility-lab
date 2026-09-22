import numpy as np
import pandas as pd

TREASURY_COLUMNS = [
    "1 Mo",
    "1.5 Month",
    "2 Mo",
    "3 Mo",
    "4 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
]

TREASURY_MATURITIES = np.array([
    1 / 12,
    1.5 / 12,
    2 / 12,
    3 / 12,
    4 / 12,
    6 / 12,
    1.0,
    2.0,
    3.0,
])


def interpolate_rate(T, maturities, rates):
    return np.interp(T, maturities, rates)


def prepare_treasury_data(treasury):
    treasury = treasury.copy()

    treasury["Date"] = pd.to_datetime(treasury["Date"])

    treasury = treasury.sort_values("Date").reset_index(drop=True)

    return treasury


def get_treasury_curve(row):
    rates = (
        row[TREASURY_COLUMNS]
        .to_numpy(dtype=float)
        / 100.0
    )

    valid = ~np.isnan(rates)


    return TREASURY_MATURITIES[valid], rates[valid]


def to_continuous_rate(rate):
    return np.log1p(rate)


def get_rate(treasury, date, T):
    date = pd.to_datetime(date)

    available_data = treasury[
        treasury["Date"] <= date
    ]

    if available_data.empty:
        raise ValueError(
            f"No Treasury data available on or before {date.date()}"
        )

    row = available_data.iloc[-1]

    maturities, rates = get_treasury_curve(row)

    interpolated_rate = interpolate_rate(
    T,
    maturities,
    rates,
    )

    return to_continuous_rate(interpolated_rate)


def load_treasury_data(path):
    treasury = pd.read_csv(path)

    return prepare_treasury_data(treasury)

