import numpy as np
import yfinance as yf
from datetime import datetime


def load_spx_options(expiry_index=4, moneyness_band=0.15, min_price=0.5, min_oi=500):
    """
    Fetch live SPX call options from Yahoo Finance.

    Parameters
    ----------
    expiry_index    : which expiry to pick from the options chain (0 = nearest)
    moneyness_band  : keep strikes within this fraction of spot (e.g. 0.15 = ±15%)
    min_price       : minimum option price to avoid stale quotes
    min_oi          : minimum open interest to ensure liquidity

    Returns
    -------
    spot   : float, current SPX spot price
    expiry : str, expiry date string (YYYY-MM-DD)
    T      : float, time to expiry in years
    strikes: np.ndarray of strike prices
    prices : np.ndarray of mid option prices
    """
    spx = yf.Ticker("^SPX")
    spot = spx.history(period="1d")["Close"].iloc[-1]

    expiry = spx.options[expiry_index]
    chain = spx.option_chain(expiry)
    calls = chain.calls

    # Filter for liquid, near-the-money strikes
    calls = calls[(calls["strike"] > spot * (1 - moneyness_band)) &
                  (calls["strike"] < spot * (1 + moneyness_band))]
    calls = calls[calls["lastPrice"] > min_price]
    calls = calls[calls["openInterest"] > min_oi]
    calls = calls.reset_index(drop=True)

    T = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365

    strikes = calls["strike"].values
    prices = calls["lastPrice"].values

    print(f"Loaded {len(strikes)} strikes  |  Spot: {spot:.1f}  |  Expiry: {expiry}  |  T: {T:.4f}y")
    return spot, expiry, T, strikes, prices
