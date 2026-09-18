import numpy as np
import pytest

from options_volatility_lab.black_scholes import call_price, put_price
from options_volatility_lab.implied_volatility import (
    implied_volatility,
    implied_volatility_brent,
)


@pytest.mark.parametrize(
    "true_sigma",
    [0.10, 0.20, 0.40, 0.80],
)
def test_newton_recovers_known_volatility(true_sigma):
    S = 100
    K = 100
    T = 1
    r = 0.05

    market_price = call_price(S, K, T, r, true_sigma)

    calculated_sigma = implied_volatility(
        market_price, S, K, T, r
    )

    assert np.isclose(calculated_sigma, true_sigma, rtol=1e-6)


@pytest.mark.parametrize(
    "true_sigma",
    [0.10, 0.20, 0.40, 0.80],
)
def test_brent_recovers_known_volatility(true_sigma):
    S = 100
    K = 100
    T = 1
    r = 0.05

    market_price = call_price(S, K, T, r, true_sigma)

    calculated_sigma = implied_volatility_brent(
        market_price, S, K, T, r
    )

    assert np.isclose(calculated_sigma, true_sigma, rtol=1e-6)

@pytest.mark.parametrize(
    "true_sigma",
    [0.10, 0.20, 0.40, 0.80],
)
def test_brent_recovers_put_volatility(true_sigma):
    S = 100
    K = 110
    T = 0.5
    r = 0.04
    
    market_price = put_price(S, K, T, r, true_sigma)

    calculated_sigma = implied_volatility_brent(
        market_price,
        S,
        K,
        T,
        r,
        option_type="put",
    )

    assert np.isclose(calculated_sigma, true_sigma, rtol=1e-6)


def test_invalid_option_type_raises_error():
    with pytest.raises(ValueError):
        implied_volatility(
            10,
            100,
            100,
            1,
            0.05,
            option_type="banana",
        )