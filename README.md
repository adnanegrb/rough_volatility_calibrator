# Rough Volatility Calibrator

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Models](https://img.shields.io/badge/Models-BS%20%7C%20Heston%20%7C%20rBergomi-8A2BE2?style=flat-square)
![Data](https://img.shields.io/badge/Data-SPX%20Live-green?style=flat-square)
![Optimizer](https://img.shields.io/badge/Optimizer-Differential%20Evolution-orange?style=flat-square)

Volatility models calibrated on live SPX options data, from Black-Scholes to Rough Bergomi.

Black-Scholes assumes volatility is constant. Markets don't. This project builds from the BS
baseline through Heston to the Rough Bergomi model (Bayer, Friz, Gatheral 2016), which captures
the empirical roughness of realized volatility with H ≈ 0.1 and reproduces the short-term
implied vol smile that classical models fail to fit.


## Why Rough Bergomi

The volatility smile — the fact that implied vol varies across strikes — is a well-known failure
of Black-Scholes. Heston partially fixes this but breaks down at short maturities because its
variance process is driven by a standard Brownian motion, which is too smooth.

Gatheral, Jaisson and Rosenbaum (2018) showed that realized volatility on equity indexes behaves
like a fractional Brownian motion with Hurst exponent H ≈ 0.1, far rougher than standard BM
(H = 0.5). The Rough Bergomi model encodes this directly in the variance kernel and fits the
SPX smile with only three parameters: H, eta, and rho.


## Project Structure

```
rough_volatility_calibrator/
├── data/
│   └── loader.py                  fetch and clean live SPX options from Yahoo Finance
├── models/
│   ├── black_scholes.py           BS pricing, delta, implied vol inversion
│   ├── heston.py                  Monte Carlo Heston with full truncation scheme
│   └── rough_bergomi.py           rBergomi with antithetic variates + martingale correction
├── calibration/
│   ├── heston_calibrator.py       global calibration via differential evolution
│   └── rbergomi_calibrator.py     global calibration via differential evolution
├── notebooks/
│   ├── 01_bs_smile.ipynb          extract and visualize the SPX implied vol smile
│   ├── 02_heston_calibration.ipynb calibrate Heston and compare to market
│   └── 03_rbergomi_calibration.ipynb calibrate rBergomi and run the full model comparison
├── results/                       plots saved here automatically
└── requirements.txt
```


## Results

The table below shows typical RMSE values on the SPX smile (one near-term expiry, ±15% moneyness).

| Model | RMSE (vol pts) | Parameters |
|---|---|---|
| Black-Scholes | ~2.5 | 0 |
| Heston | ~0.8 | 5 |
| Rough Bergomi | ~0.3 | 3 |

Rough Bergomi fits the steep short-term skew that Heston systematically underestimates.
The calibrated H ≈ 0.1 is consistent with the empirical findings of Gatheral et al. (2018).


## Calibration Design

Both Heston and rBergomi are calibrated with **differential evolution** — a global optimizer
that explores the full parameter space before converging. Local methods like Nelder-Mead or
gradient descent routinely get stuck in flat regions of the loss surface for these models.

The Rough Bergomi calibration fixes xi0 at the ATM variance (a standard industry choice) and
optimizes over (H, eta, rho) only, which keeps the problem tractable.


## Setup

```bash
pip install -r requirements.txt
```

Then run the notebooks in order from the `notebooks/` folder.
No GPU required. The full calibration pipeline runs in under 10 minutes on a standard laptop.


## References

- Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*.
- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility. *Review of Financial Studies*.
- Bayer, C., Friz, P. and Gatheral, J. (2016). Pricing under Rough Volatility. *Quantitative Finance*.
- Gatheral, J., Jaisson, T. and Rosenbaum, M. (2018). Volatility is Rough. *Quantitative Finance*.
