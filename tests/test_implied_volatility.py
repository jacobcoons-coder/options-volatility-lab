import numpy as np
import pytest

from options_volatility_lab.black_scholes import call_price, put_price
from options_volatility_lab.implied_volatility import implied_volatility


@pytest.mark.parametrize(
    "true_sigma",
    [0.10, 0.20, 0.40, 0.80],
)
def test_implied_volatility_recovers_call_volatility(true_sigma):
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
def test_implied_volatility_recovers_put_volatility(true_sigma):
    S = 100
    K = 110
    T = 0.5
    r = 0.04
    
    market_price = put_price(S, K, T, r, true_sigma)

    calculated_sigma = implied_volatility(
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


def test_falls_back_to_brent_when_newton_fails(monkeypatch):
    import options_volatility_lab.implied_volatility as iv_module

    S = 100
    K = 100
    T = 1
    r = 0.05
    true_sigma = 0.20

    market_price = call_price(S, K, T, r, true_sigma)

    def failing_newton(*args, **kwargs):
        raise ValueError("Forced Newton failure")

    monkeypatch.setattr(
        iv_module,
        "_implied_volatility_newton",
        failing_newton,
    )

    calculated_sigma = iv_module.implied_volatility(
        market_price, S, K, T, r
    )

    assert np.isclose(calculated_sigma, true_sigma, rtol=1e-6)

def test_call_price_below_arbitrage_bound_raises_error():
    S = 100
    K = 90
    T = 1
    r = 0.05

    impossible_price = 10.00

    with pytest.raises(ValueError):
        implied_volatility(
            impossible_price,
            S,
            K,
            T,
            r,
            option_type="call",
        )

def test_put_price_above_arbitrage_bound_raises_error():
    S = 100
    K = 100
    T = 1
    r = 0.05

    impossible_price = 100.00

    with pytest.raises(ValueError):
        implied_volatility(
            impossible_price,
            S,
            K,
            T,
            r,
            option_type="put",
        )    