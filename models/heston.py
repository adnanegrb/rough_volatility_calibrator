import numpy as np
from models.black_scholes import implied_vol


def heston_mc(S, K, T, r, v0, kappa, theta, xi, rho, n_paths=50000, n_steps=200):
    """
    Monte Carlo price of a European call under Heston stochastic volatility.

    Dynamics:
        dS = r S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2
        corr(dW1, dW2) = rho

    Parameters
    ----------
    v0    : initial variance (vol^2, e.g. 0.04 for 20% vol)
    kappa : mean-reversion speed
    theta : long-run variance
    xi    : volatility of variance (vol of vol)
    rho   : spot-vol correlation (typically negative, the leverage effect)
    """
    dt = T / n_steps
    S_t = np.full(n_paths, float(S))
    v_t = np.full(n_paths, float(v0))

    for _ in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.standard_normal(n_paths)

        # Full truncation scheme — prevents negative variance
        v_t = np.maximum(v_t, 0)
        v_t = v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t * dt) * Z2

        S_t = S_t * np.exp((r - 0.5 * v_t) * dt + np.sqrt(np.maximum(v_t, 0) * dt) * Z1)

    payoff = np.maximum(S_t - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


def heston_smile(S, strikes, T, r, v0, kappa, theta, xi, rho, n_paths=50000):
    """
    Compute the implied vol smile produced by Heston across a range of strikes.
    Returns an array of implied vols aligned with the input strikes.
    """
    ivols = []
    for K in strikes:
        price = heston_mc(S, K, T, r, v0, kappa, theta, xi, rho, n_paths=n_paths)
        iv = implied_vol(price, S, K, T, r)
        ivols.append(iv)
    return np.array(ivols)
