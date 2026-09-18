import numpy as np
from scipy.stats import norm

from options_volatility_lab.black_scholes import d1, d2


def call_delta(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d_1)


def put_delta(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * (norm.cdf(d_1) - 1)

def gamma(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)

    return (
        np.exp(-q * T) * norm.pdf(d_1)
        / (S * sigma * np.sqrt(T))
    )

def vega(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)

    return (
        S * np.exp(-q * T) * norm.pdf(d_1) * np.sqrt(T)
    )

def call_theta(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)
    d_2 = d2(S, K, T, r, sigma, q)

    first_term = (
        -S * np.exp(-q * T) * norm.pdf(d_1) * sigma
        / (2 * np.sqrt(T))
    )

    second_term = -r * K * np.exp(-r * T) * norm.cdf(d_2)

    third_term = q * S * np.exp(-q * T) * norm.cdf(d_1)

    return first_term + second_term + third_term


def put_theta(S, K, T, r, sigma, q=0.0):
    d_1 = d1(S, K, T, r, sigma, q)
    d_2 = d2(S, K, T, r, sigma, q)

    first_term = (
        -S * np.exp(-q * T) * norm.pdf(d_1) * sigma
        / (2 * np.sqrt(T))
    )

    second_term = r * K * np.exp(-r * T) * norm.cdf(-d_2)

    third_term = -q * S * np.exp(-q * T) * norm.cdf(-d_1)

    return first_term + second_term + third_term

def call_rho(S, K, T, r, sigma, q=0.0):
    d_2 = d2(S, K, T, r, sigma, q)

    return (
        K * T * np.exp(-r * T) * norm.cdf(d_2)
    )


def put_rho(S, K, T, r, sigma, q=0.0):
    d_2 = d2(S, K, T, r, sigma, q)

    return (
        -K * T * np.exp(-r * T) * norm.cdf(-d_2)
    )
