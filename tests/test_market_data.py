import numpy as np
import pandas as pd

from options_volatility_lab.market_data import prepare_market_data


def test_prepare_market_data():
    options = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "expiration": ["2025-01-03", "2025-01-02"],
            "strike": [100, 100],
            "type": ["call", "call"],
            "bid": [5.0, 4.0],
            "ask": [5.2, 4.2],
        }
    )

    underlying = pd.DataFrame(
    {
        "date": [
            "2024-03-15",
            "2024-06-21",
            "2024-09-20",
            "2024-12-20",
            "2025-01-02",
        ],
        "close": [
            90.0,
            92.0,
            95.0,
            98.0,
            100.0,
        ],
        "dividend_amount": [
            0.25,
            0.25,
            0.25,
            0.25,
            0.0,
        ],
    }
    )

    treasury = pd.DataFrame(
    {
        "Date": pd.to_datetime(["2025-01-02"]),
        "1 Mo": [4.45],
        "1.5 Month": [4.40],
        "2 Mo": [4.36],
        "3 Mo": [4.36],
        "4 Mo": [4.31],
        "6 Mo": [4.25],
        "1 Yr": [4.17],
        "2 Yr": [4.25],
        "3 Yr": [4.29],
    }
    )

    result = prepare_market_data(options, underlying, treasury)

    assert len(result) == 1
    assert result.iloc[0]["close"] == 100.0
    assert np.isclose(result.iloc[0]["T"], 1 / 365)
    assert np.isclose(result.iloc[0]["trailing_dividend_yield"], 0.01)
    assert np.isclose(result.iloc[0]["risk_free_rate"], np.log1p(0.0445))