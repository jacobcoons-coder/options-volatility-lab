import numpy as np
from scipy.optimize import brentq

from options_volatility_lab.black_scholes import call_price, put_price
from options_volatility_lab.greeks import vega


def _option_price(S, K, T, r, sigma, option_type, q):
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)

    if option_type == "put":
        return put_price(S, K, T, r, sigma, q)

    raise ValueError("option_type must be 'call' or 'put'")


def _implied_volatility_newton(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    q=0.0,
    initial_guess=0.20,
    tolerance=1e-8,
    max_iterations=100,
):
    sigma = initial_guess

    for _ in range(max_iterations):
        model_price = _option_price(S, K, T, r, sigma, option_type, q)

        difference = model_price - market_price

        if abs(difference) < tolerance:
            return sigma

        option_vega = vega(S, K, T, r, sigma, q)

        if abs(option_vega) < 1e-12:
            raise ValueError("Vega is too small for Newton-Raphson")

        sigma = sigma - difference / option_vega

        if sigma <= 0:
            raise ValueError("Newton-Raphson produced non-positive volatility")

    raise ValueError("Implied volatility did not converge")

def _implied_volatility_brent(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    q=0.0,
    lower_bound=1e-6,
    upper_bound=5.0,
):
    def objective(sigma):
        return (
            _option_price(S, K, T, r, sigma, option_type, q)
            - market_price
        )

    return brentq(objective, lower_bound, upper_bound)

def _validate_market_price(
    market_price,
    S,
    K,
    T,
    r,
    option_type,
    q,
):
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        lower_bound = max(
            0.0,
            discounted_spot - discounted_strike,
        )
        upper_bound = discounted_spot

    elif option_type == "put":
        lower_bound = max(
            0.0,
            discounted_strike - discounted_spot,
        )
        upper_bound = discounted_strike

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if not lower_bound <= market_price <= upper_bound:
        raise ValueError(
            "Market price violates no-arbitrage bounds"
        )

def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    q=0.0,
    initial_guess=0.20,
):
    _validate_market_price(
        market_price,
        S,
        K,
        T,
        r,
        option_type,
        q,
    )
    try:
        return _implied_volatility_newton(
            market_price,
            S,
            K,
            T,
            r,
            option_type,
            q,
            initial_guess,
        )
    except ValueError:
        return _implied_volatility_brent(
            market_price,
            S,
            K,
            T,
            r,
            option_type,
            q,
        )