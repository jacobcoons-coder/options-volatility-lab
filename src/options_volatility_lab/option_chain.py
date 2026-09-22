import numpy as np

from options_volatility_lab.implied_volatility import implied_volatility


def add_mid_price(chain):
    chain = chain.copy()

    chain["mid"] = (chain["bid"] + chain["ask"]) / 2

    return chain

def _safe_implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type,
    q,
):
    try:
        return implied_volatility(
            market_price,
            S,
            K,
            T,
            r,
            option_type=option_type,
            q=q,
        )
    except ValueError:
        return np.nan

def add_implied_volatility(chain, S, r, q=0.0):
    chain = chain.copy()

    chain["implied_volatility"] = chain.apply(
        lambda row: _safe_implied_volatility(
            row["mid"],
            S,
            row["strike"],
            row["T"],
            r,
            option_type=row["option_type"],
            q=q,
        ),
        axis=1,
    )

    return chain

def clean_option_chain(chain):
    chain = chain.copy()

    chain = chain.dropna(
        subset=["strike", "bid", "ask", "T", "option_type"]
    )

    chain = chain[
        (chain["strike"] > 0)
        & (chain["bid"] >= 0)
        & (chain["ask"] > 0)
        & (chain["ask"] >= chain["bid"])
        & (chain["T"] > 0)
    ]

    return chain

def process_option_chain(chain, S, r, q=0.0):
    chain = clean_option_chain(chain)
    chain = add_mid_price(chain)
    chain = add_implied_volatility(chain, S, r, q)

    return chain
