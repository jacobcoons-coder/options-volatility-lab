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
            "date": ["2025-01-02"],
            "close": [100.0],
        }
    )

    result = prepare_market_data(options, underlying)

    assert len(result) == 1
    assert result.iloc[0]["close"] == 100.0
    assert np.isclose(result.iloc[0]["T"], 1 / 365)