import numpy as np
from scipy.optimize import differential_evolution
from models.rough_bergomi import rbergomi_smile


def calibrate_rbergomi(strikes, market_ivols, S, T, r, n_paths=50000):
    """
    Calibrate Rough Bergomi parameters (H, eta, rho) to a market smile.

    The rBergomi loss surface is highly non-convex — local optimizers routinely
    get stuck. Differential evolution explores the full parameter space globally
    before converging, which makes calibration reliable in practice.

    xi0 is fixed to the ATM variance (a standard industry choice that removes
    one degree of freedom and keeps the model anchored to the market level).

    Parameters
    ----------
    strikes      : array of strike prices
    market_ivols : array of market implied vols
    S            : spot price
    T            : time to expiry in years
    r            : risk-free rate
    n_paths      : MC paths (100k gives ~0.1 vol pt MC error)

    Returns
    -------
    params : dict with calibrated H, eta, rho, xi0
    rmse   : root mean squared error on implied vols (in vol points)
    """

    # ATM variance anchor
    atm_idx = np.argmin(np.abs(strikes - S))
    xi0 = market_ivols[atm_idx] ** 2

    # Normalize strikes to moneyness for the rBergomi pricer
    strikes_norm = strikes / S

    def objective(x):
        H, eta, rho = x
        model_ivols = rbergomi_smile(strikes_norm, T, r, H, eta, rho, xi0,
                                     n_paths=n_paths)
        mask = model_ivols > 0
        if mask.sum() < 3:
            return 1.0
        return np.sqrt(np.mean((model_ivols[mask] - market_ivols[mask]) ** 2))

    bounds = [
        (0.01, 0.49),   # H  — roughness parameter (empirically 0.05-0.15 for SPX)
        (0.5,  4.0),    # eta — vol of vol
        (-0.99, -0.1),  # rho — leverage correlation
    ]

    print("Calibrating Rough Bergomi with differential evolution...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=60, tol=5e-4,
        popsize=10, mutation=(0.5, 1.2), recombination=0.8,
        workers=1, updating="deferred"
    )

    H, eta, rho = result.x
    params = dict(H=H, eta=eta, rho=rho, xi0=xi0)
    rmse = result.fun

    print(f"Rough Bergomi calibrated  |  RMSE: {rmse*100:.4f} vol pts")
    print(f"  H={H:.4f}  eta={eta:.4f}  rho={rho:.4f}  xi0={xi0:.6f}")

    return params, rmse
