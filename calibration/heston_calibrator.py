import numpy as np
from scipy.optimize import differential_evolution
from models.black_scholes import implied_vol
from models.heston import heston_mc


def calibrate_heston(S, strikes, market_ivols, T, r, n_paths=10000, n_steps=100):
    """
    Calibrate Heston parameters to a market implied vol smile.

    Minimizes RMSE between model-implied vols and market-implied vols using
    differential evolution — a global optimizer that handles the non-convex
    Heston loss surface much better than local methods like Nelder-Mead.

    Parameters
    ----------
    S            : spot price
    strikes      : array of strike prices
    market_ivols : array of market implied vols (same length as strikes)
    T            : time to expiry in years
    r            : risk-free rate
    n_paths      : MC paths used during calibration (lower = faster)
    n_steps      : MC time steps used during calibration

    Returns
    -------
    params : dict with calibrated v0, kappa, theta, xi, rho
    rmse   : root mean squared error on implied vols (in vol points)
    """

    def objective(x):
        v0, kappa, theta, xi, rho = x
        model_ivols = []
        for K in strikes:
            price = heston_mc(S, K, T, r, v0, kappa, theta, xi, rho,
                              n_paths=n_paths, n_steps=n_steps)
            iv = implied_vol(price, S, K, T, r)
            model_ivols.append(iv)
        model_ivols = np.array(model_ivols)
        mask = ~np.isnan(model_ivols)
        if mask.sum() < 3:
            return 1.0
        return np.sqrt(np.mean((model_ivols[mask] - market_ivols[mask]) ** 2))

    bounds = [
        (1e-4, 0.5),    # v0
        (0.1,  15.0),   # kappa
        (1e-4, 0.5),    # theta
        (0.01, 2.0),    # xi
        (-0.99, -0.01), # rho
    ]

    print("Calibrating Heston with differential evolution...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=80, tol=1e-4,
        popsize=8, mutation=(0.5, 1.0), recombination=0.7,
        workers=1, updating="deferred"
    )

    v0, kappa, theta, xi, rho = result.x
    params = dict(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    rmse = result.fun

    print(f"Heston calibrated  |  RMSE: {rmse*100:.4f} vol pts")
    print(f"  v0={v0:.4f}  kappa={kappa:.4f}  theta={theta:.4f}  xi={xi:.4f}  rho={rho:.4f}")

    return params, rmse
