import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma, q=0.0):
    return (
        np.log(S / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

def call_price(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)
    d_2 = d2(S, K, T, r, sigma, q)

    return (
        S * np.exp(-q * T) * norm.cdf(d_1)
        - K * np.exp(-r * T) * norm.cdf(d_2)
    )

def put_price(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)
    d_2 = d2(S, K, T, r, sigma, q)

    return (
        K * np.exp(-r * T) * norm.cdf(-d_2)
        - S * np.exp(-q * T) * norm.cdf(-d_1)
    )

    