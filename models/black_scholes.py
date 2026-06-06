import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Closed-form Black-Scholes price for a European option.

    Parameters
    ----------
    S           : spot price
    K           : strike
    T           : time to expiry in years
    r           : risk-free rate
    sigma       : volatility
    option_type : 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes delta — sensitivity of option price to spot moves.
    This is the primary hedging ratio used in practice.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1


def implied_vol(market_price, S, K, T, r, option_type="call"):
    """
    Recover implied volatility from a market price by inverting Black-Scholes.

    Uses Brent's method — more robust than Newton when the pricing function
    is nearly flat (deep ITM or OTM options). Returns NaN on failure.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0) + 1e-10
    if market_price <= intrinsic:
        return np.nan

    objective = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=1e-8)
    except ValueError:
        return np.nan


def compute_smile(strikes, prices, S, T, r):
    """
    Compute the implied vol smile across a range of strikes.

    Returns strikes and implied vols with NaN entries removed.
    """
    ivols = np.array([implied_vol(p, S, K, T, r) for p, K in zip(prices, strikes)])
    mask = ~np.isnan(ivols)
    return strikes[mask], ivols[mask]
