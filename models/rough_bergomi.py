import numpy as np
from models.black_scholes import implied_vol


def simulate_rough_bergomi(T, r, H, eta, rho, xi0, n_paths=100000, n_steps=200):
    """
    Simulate Rough Bergomi paths using the exact kernel representation.

    The variance process follows:
        v(t) = xi0 * exp(eta * W^H(t) - 0.5 * eta^2 * t^(2H))

    where W^H is a Riemann-Liouville fractional Brownian motion with
    Hurst exponent H. The kernel is computed exactly — no Cholesky trick.

    Variance reduction: antithetic variates halve MC error at no extra cost.
    Martingale correction: enforces E[S_T] = S_0 * e^(rT) exactly.

    Parameters
    ----------
    H    : Hurst exponent (empirically ~0.1 for SPX realized vol)
    eta  : vol of vol
    rho  : spot-vol correlation
    xi0  : initial forward variance (ATM vol squared is a good proxy)

    Returns
    -------
    S_T : terminal spot values, normalized to S0 = 1
    v   : variance paths, shape (n_paths, n_steps)
    """
    assert n_paths % 2 == 0, "n_paths must be even for antithetic variates"

    half = n_paths // 2
    dt = T / n_steps
    grid = np.arange(1, n_steps + 1) * dt
    alpha = H - 0.5

    # Draw half the paths; antithetic half is just the negation
    Z1_h = np.random.standard_normal((half, n_steps))
    Z2_h = rho * Z1_h + np.sqrt(1 - rho ** 2) * np.random.standard_normal((half, n_steps))

    Z1 = np.concatenate([Z1_h, -Z1_h], axis=0)
    Z2 = np.concatenate([Z2_h, -Z2_h], axis=0)

    dW2 = Z2 * np.sqrt(dt)

    # Fractional kernel: W^H(t_i) = sum_{j<=i} (t_i - t_j + dt)^alpha * dW2_j
    V_H = np.zeros((n_paths, n_steps))
    for i in range(n_steps):
        j = np.arange(i + 1)
        kernel = (grid[i] - grid[j] + dt) ** alpha
        V_H[:, i] = (dW2[:, : i + 1] * kernel[np.newaxis, :]).sum(axis=1)

    # Variance: Dolean-Dade exponential with Ito correction
    var_VH = grid ** (2 * H) / (2 * H)
    v = xi0 * np.exp(eta * V_H - 0.5 * eta ** 2 * var_VH[np.newaxis, :])

    # Log-Euler scheme for spot
    log_S = np.zeros(n_paths)
    for i in range(n_steps):
        vi = np.maximum(v[:, i], 0)
        log_S += (r - 0.5 * vi) * dt + np.sqrt(vi * dt) * Z1[:, i]

    S_T = np.exp(log_S)

    # Martingale correction: enforce E[S_T] = e^(rT)
    S_T *= np.exp(r * T) / S_T.mean()

    return S_T, v


def rbergomi_smile(strikes_norm, T, r, H, eta, rho, xi0, n_paths=100000):
    """
    Compute the implied vol smile for Rough Bergomi across normalized strikes.

    Parameters
    ----------
    strikes_norm : array of K/S (moneyness ratios)

    Returns
    -------
    ivols : array of implied vols, NaN replaced with 0 for optimizer stability
    """
    S_T, _ = simulate_rough_bergomi(T, r, H, eta, rho, xi0, n_paths=n_paths)

    ivols = []
    for K in strikes_norm:
        price = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
        iv = implied_vol(price, 1.0, K, T, r)
        ivols.append(iv if not np.isnan(iv) else 0.0)

    return np.array(ivols)
