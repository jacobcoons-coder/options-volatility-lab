import numpy as np
import pandas as pd

from options_volatility_lab.dividends import (
    add_trailing_dividend_yield,
)


def test_add_trailing_dividend_yield():
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
                509.83,
                544.51,
                568.25,
                591.15,
                584.64,
            ],
            "dividend_amount": [
                1.5949,
                1.7590,
                1.7455,
                1.9655,
                0.0,
            ],
        }
    )

    result = add_trailing_dividend_yield(underlying)

    expected_dividends = (
        1.5949 + 1.7590 + 1.7455 + 1.9655
    )

    expected_yield = expected_dividends / 584.64

    assert np.isclose(
        result.iloc[-1]["trailing_dividends"],
        expected_dividends,
    )

    assert np.isclose(
        result.iloc[-1]["trailing_dividend_yield"],
        expected_yield,
    )