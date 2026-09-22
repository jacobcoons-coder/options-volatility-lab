import numpy as np
import pandas as pd

from options_volatility_lab.rates import (
    get_rate,
    get_treasury_curve,
    interpolate_rate,
    prepare_treasury_data,
    to_continuous_rate,
)


def test_interpolate_rate():
    maturities = np.array([
        1 / 12,
        3 / 12,
        6 / 12,
        1.0,
        2.0,
    ])

    rates = np.array([
        0.0445,
        0.0436,
        0.0425,
        0.0417,
        0.0425,
    ])

    rate = interpolate_rate(
        0.75,
        maturities,
        rates,
    )

    assert np.isclose(rate, 0.0421)

def test_prepare_treasury_data():
    treasury = pd.DataFrame(
        {
            "Date": ["01/03/2025", "01/02/2025"],
            "1 Mo": [4.44, 4.45],
        }
    )

    result = prepare_treasury_data(treasury)

    assert result.iloc[0]["Date"] == pd.Timestamp("2025-01-02")
    assert result.iloc[1]["Date"] == pd.Timestamp("2025-01-03")


def test_get_treasury_curve_removes_missing_rates():
    row = pd.Series(
        {
            "1 Mo": 4.45,
            "1.5 Month": np.nan,
            "2 Mo": 4.36,
            "3 Mo": 4.36,
            "4 Mo": 4.31,
            "6 Mo": 4.25,
            "1 Yr": 4.17,
            "2 Yr": 4.25,
            "3 Yr": 4.29,
        }
    )

    maturities, rates = get_treasury_curve(row)

    assert len(maturities) == 8
    assert len(rates) == 8
    assert not np.isnan(rates).any()


def test_to_continuous_rate():
    rate = to_continuous_rate(0.0421)

    assert np.isclose(rate, np.log(1.0421))


def test_get_rate():
    treasury = pd.DataFrame(
        {
            "Date": ["2025-01-02"],
            "1 Mo": [4.45],
            "1.5 Month": [np.nan],
            "2 Mo": [4.36],
            "3 Mo": [4.36],
            "4 Mo": [4.31],
            "6 Mo": [4.25],
            "1 Yr": [4.17],
            "2 Yr": [4.25],
            "3 Yr": [4.29],
        }
    )

    treasury = prepare_treasury_data(treasury)

    rate = get_rate(
        treasury,
        "2025-01-02",
        0.75,
    )

    expected_rate = np.log1p(0.0421)

    assert np.isclose(rate, expected_rate)