import numpy as np

from options_volatility_lab.black_scholes import call_price
from options_volatility_lab.greeks import call_delta, call_rho, call_theta, gamma, vega


def test_call_delta_against_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    h = 0.01

    numerical_delta = (
        call_price(S + h, K, T, r, sigma)
        - call_price(S - h, K, T, r, sigma)
    ) / (2 * h)

    analytical_delta = call_delta(S, K, T, r, sigma)

    assert np.isclose(numerical_delta, analytical_delta, rtol=1e-5)

def test_gamma_against_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    h = 0.01

    numerical_gamma = (
        call_price(S + h, K, T, r, sigma)
        - 2 * call_price(S, K, T, r, sigma)
        + call_price(S - h, K, T, r, sigma)
    ) / h**2

    analytical_gamma = gamma(S, K, T, r, sigma)

    assert np.isclose(numerical_gamma, analytical_gamma, rtol=1e-4)

def test_vega_against_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    h = 0.0001

    numerical_vega = (
        call_price(S, K, T, r, sigma + h)
        - call_price(S, K, T, r, sigma - h)
    ) / (2 * h)

    analytical_vega = vega(S, K, T, r, sigma)

    assert np.isclose(numerical_vega, analytical_vega, rtol=1e-5)

def test_call_theta_against_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    h = 0.0001

    numerical_theta = -(
        call_price(S, K, T + h, r, sigma)
        - call_price(S, K, T - h, r, sigma)
    ) / (2 * h)

    analytical_theta = call_theta(S, K, T, r, sigma)

    assert np.isclose(numerical_theta, analytical_theta, rtol=1e-5)


def test_call_rho_against_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    h = 0.0001

    numerical_rho = (
        call_price(S, K, T, r + h, sigma)
        - call_price(S, K, T, r - h, sigma)
    ) / (2 * h)

    analytical_rho = call_rho(S, K, T, r, sigma)

    assert np.isclose(numerical_rho, analytical_rho, rtol=1e-5)