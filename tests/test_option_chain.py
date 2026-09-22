import numpy as np
import pandas as pd
import pytest

from options_volatility_lab.black_scholes import call_price
from options_volatility_lab.option_chain import (
    add_mid_price,
    clean_option_chain,
    process_option_chain,
)


def test_add_mid_price():
    chain = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "bid": [8.00, 5.00, 2.50],
            "ask": [8.40, 5.40, 2.90],
        }
    )

    result = add_mid_price(chain)

    expected_mid = np.array([8.20, 5.20, 2.70])

    assert np.allclose(result["mid"], expected_mid)


@pytest.mark.parametrize(
    "true_sigma",
    [0.10, 0.20, 0.40, 0.80],
)
def test_process_option_chain_recovers_volatility(true_sigma):
    S = 100
    r = 0.05
    T = 0.5

    strikes = [90, 100, 110]

    market_prices = [
        call_price(S, K, T, r, true_sigma)
        for K in strikes
    ]

    chain = pd.DataFrame(
        {
            "strike": strikes,
            "bid": market_prices,
            "ask": market_prices,
            "T": [T, T, T],
            "option_type": ["call", "call", "call"],
        }
    )

    result = process_option_chain(chain, S, r)

    expected_volatility = np.array(
        [true_sigma, true_sigma, true_sigma]
    )

    assert np.allclose(
        result["implied_volatility"],
        expected_volatility,
        rtol=1e-6,
    )

def test_clean_option_chain():
    chain = pd.DataFrame(
        {
            "strike": [100, 105, 110, 115, 120, 125],
            "bid": [5.00, -1.00, 2.00, 4.00, 0.00, np.nan],
            "ask": [5.20, 3.00, 0.00, 3.50, 1.00, 2.00],
            "T": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "option_type": [
                "call",
                "call",
                "call",
                "call",
                "call",
                "call",
            ],
        }
    )

    result = clean_option_chain(chain)

    assert list(result["strike"]) == [100, 120]

def test_invalid_iv_becomes_nan():
    S = 100
    r = 0.05
    T = 1.0
    true_sigma = 0.20

    valid_price = call_price(S, 100, T, r, true_sigma)

    chain = pd.DataFrame(
        {
            "strike": [100, 90, 100],
            "bid": [valid_price, 10.00, valid_price],
            "ask": [valid_price, 10.00, valid_price],
            "T": [T, T, T],
            "option_type": ["call", "call", "call"],
        }
    )

    result = process_option_chain(chain, S, r)

    assert np.isclose(result.iloc[0]["implied_volatility"], true_sigma)
    assert np.isnan(result.iloc[1]["implied_volatility"])
    assert np.isclose(result.iloc[2]["implied_volatility"], true_sigma)