import numpy as np

from options_volatility_lab.black_scholes import call_price, put_price


def test_known_call_price():
    price = call_price(100, 100, 1, 0.05, 0.20)
    assert np.isclose(price, 10.450583572185565)


def test_known_put_price():
    price = put_price(100, 100, 1, 0.05, 0.20)
    assert np.isclose(price, 5.573526022256971)


def test_put_call_parity():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    q = 0.0

    call = call_price(S, K, T, r, sigma, q)
    put = put_price(S, K, T, r, sigma, q)

    left_side = call - put
    right_side = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert np.isclose(left_side, right_side)