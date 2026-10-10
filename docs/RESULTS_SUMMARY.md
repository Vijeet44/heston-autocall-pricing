# Everything tried, with metrics

Vol RMSE in vol points (unweighted, implied vols inverted with the same r and q for market and model). Price RMSE in index points on call-equivalent prices. "Short" = 1M and 3M expiries, "6M+" = the rest. "Product strikes" = mean |vol error| at 1Y–4Y, 60–100%. Quote sets differ between rows where stated; compare within a block.

## A. Heston calibrations by snapshot and set-up

| Run | Data | Quotes | Fit set | Vol RMSE all / short / 6M+ | Price RMSE all / 6M+ | Product strikes | Note price |
|---|---|---|---|---|---|---|---|
| v1 original (Dec 2025 submission) | 3 Dec 2025, 8 expiries, unfiltered | 906 | all, price RMSE objective | (1.53 on the filtered set) | 13.29 | 0.37 | 987.5 (98.75%) |
| v1 params re-scored after the quote filter | 3 Dec 2025, 8 expiries | 883 | – | 1.53 / 2.9* / 0.55* | 11.51 / – | 0.37 | – |
| v2, vol objective on all 8 expiries (first attempt) | 3 Dec 2025 | 883 | all | 1.10 / 1.6* / 0.7* | 14.87 / – | – | – |
| v2, maturity weights 0.25/0.5/1 | 3 Dec 2025 | 883 | all, weighted | 1.12 | 13.41 | – | – |
| v2, maturity weights 0.1/0.5/1 | 3 Dec 2025 | 883 | all, weighted | 1.23 | 12.09 | – | – |
| v2, strike-range weights only | 3 Dec 2025 | 883 | all, weighted | 1.09 | 16.93 | – | – |
| v2, maturity + strike weights | 3 Dec 2025 | 883 | all, weighted | 1.10 | 14.01 | – | – |
| **v2 final Dec: fit 6M+ only** | 3 Dec 2025, 9 expiries after filter | 1049 (682 in fit) | 6M+ | 1.63 / – / 0.42 (fit set) | 12.79 / 10.48 (fit set) | 0.24 | 986.1 (98.61%) |
| v2 live, 9 expiries (extra 1.75y pillar) | 30 Sep 2026 | 1435 (847 in fit) | 6M+ | 0.97 / – / 0.56 (fit set) | 10.28 / 12.35 (fit set) | 0.16 | 975.6 |
| v1 method on the live data (price objective, all expiries) | 30 Sep 2026 | 1435 | all | 1.49 / – / 0.75 (fit set) | 9.49 / 11.70 (fit set) | 0.20 | 974.9 |
| **v2 live final: v1 expiry rule, fit 6M+** (`main.ipynb`) | 30 Sep 2026, 8 expiries | 1273 (685 in fit) | 6M+ | 1.08 / 1.44 / 0.615 | 10.73 / 13.47 | 0.156 | 975.4 (97.54%) |
| v1 method, v1 expiry rule, live | 30 Sep 2026 | 1273 | all | 1.61 / 2.18 / 0.85 | 10.01 / 12.91 | 0.22 | – |

\* per-expiry numbers only; short/6M+ splits reconstructed from the fit tables.

## B. Models, all fitted to all 8 expiries with the same objective (live, 1273 quotes)

| Model | Params | Vol RMSE all / short / 6M+ | Price RMSE all / short / 6M+ | Max vol miss | Product strikes | Calib time |
|---|---|---|---|---|---|---|
| Heston | 5 | 0.90 / 1.15 / 0.61 | 12.2 / 8.2 / 14.9 | 9.4 | 0.20 | 35 s |
| Heston, fit 6M+ only (scored on all) | 5 | 1.08 / 1.44 / 0.615 | 10.7 / 6.2 / 13.5 | 13.2 | 0.16 | 30 s |
| Bates (λ ≤ 5) | 8 | 0.42 / 0.47 / 0.36 | 9.9 / 2.0 / 13.4 | 5.7 | 0.21 | 78 s |
| Bates (λ ≤ 20) | 8 | 0.41 / 0.44 / 0.38 | 9.9 / 2.2 / 13.3 | 5.4 | 0.23 | 116 s |
| Double Heston, DE seed 42 | 10 | 0.30 / 0.27 / 0.32 | 9.1 / 0.8 / 12.4 | 4.2 | 0.21 | 131 s |
| Double Heston, DE seed 7 (best) | 10 | **0.22 / 0.17 / 0.25** | **8.9** / 0.9 / 12.1 | 3.5 | 0.13 | 132 s |
| Double Heston, least_squares 8 starts | 10 | (objective 0.362, same as seed 7) | – | – | – | 102 s |
| Rough Heston (H = 0.33) | 6 | 0.80 / 0.99 / 0.58 | 11.8 / 6.7 / 14.8 | 6.7 | 0.13 | 263 s |

Per-expiry vol RMSE (live): 1M Heston 2.02 / Bates 1.21 / DH 0.42 / rough 1.59; 3M 1.03 / 0.33 / 0.13 / 0.92; 4.2Y 0.95 / 0.83 / 0.81 / 0.89 (quote-width floor).

## C. Calibration objectives (Heston, live, all 8 expiries, DE + L-BFGS-B)

| Objective | Vol RMSE all / short / 6M+ | Price RMSE all / 6M+ | Max vol miss | Product strikes | σ, ρ |
|---|---|---|---|---|---|
| Price RMSE (v1) | 1.61 / 2.18 / 0.85 | **10.0** / 12.9 | 18.6 | 0.22 | 1.07, −0.62 |
| Relative price RMSE | 1.99 / 1.77 / 2.16 | 26.9 / 36.4 | 13.2 | 2.66 | 1.70, −0.67 |
| Vega-weighted price | 0.99 / 1.30 / 0.61 | 12.2 / 15.0 | 12.2 | 0.17 | 1.70, −0.62 |
| Vol RMSE, uniform | 0.94 / 1.18 / 0.67 | 13.1 / 15.4 | 6.4 | 0.22 | 2.54, −0.57 |
| **Vol RMSE, 1/spread² (current)** | **0.90 / 1.15 / 0.61** | 12.2 / 14.9 | 9.4 | 0.20 | 2.07, −0.59 |
| Huber vol loss | 1.02 / 1.34 / 0.615 | 11.7 / 14.8 | 12.8 | 0.15 | 1.66, −0.59 |

## D. Optimisers (current objective, Heston 5 params; objective in vol points, best / worst of two seeds)

| Optimiser | Objective | Evaluations | Time |
|---|---|---|---|
| DE best1bin + L-BFGS-B (current) | 0.7729 / 0.7729 | 3,888 | 65 s |
| DE best1bin, no local step | 0.7729 / 0.7731 | 3,825 | 63 s |
| DE rand1bin + L-BFGS-B | 0.8553 / 0.8874 | 3,825 | 63 s |
| DE currenttobest1bin | 0.7767 / 0.7851 | 3,825 | 64 s |
| DE small population | 0.7729 / 0.7729 | 1,713 | 28 s |
| Dual annealing (maxiter 60) | 0.9163 / 1.0258 | 901 | 16 s |
| Basin hopping (12 hops) | 0.7729 / 0.7729 | 3,138 | 63 s |
| SHGO (sobol, n = 32) | 0.7729 | 903 | 18 s |
| Nelder–Mead from heuristic start | 0.7729 | 575 | 10 s |
| L-BFGS-B, 12 LHS starts | 0.7729 / 0.7729 | 5,106 | 101 s |
| **least_squares (trf) from heuristic start** | **0.7729** | 8 (+ Jacobians) | **0.9 s** |
| least_squares, 6 starts | 0.7729 / 0.7729 | 92 | 9 s |

Double Heston (10 params): DE seed 42 0.408 (230 s), DE seed 7 0.364 (246 s), least_squares 8 LHS starts 0.362 (102 s), least_squares polish of DE 0.362 (5 s).

## E. Expiry selection (Heston fit 6M+, live)

| Rule | Expiries (y) | Quotes | Vol RMSE 6M+ | Product strikes |
|---|---|---|---|---|
| Nearest listed (v1) | 0.09, 0.25, 0.5, 1.0, 1.31, 2.21, 3.22, 4.22 | 1273 | 0.615 | 0.156 |
| Most liquid within ±15% | 0.14, 0.25, 0.5, 0.96, 1.71, 2.21, 3.22, 4.22 | 1838 | 0.571 | 0.147 |

## F. Vanilla pricing engines (Heston, 1273 quotes, vs adaptive quadrature)

| Engine | Max err (pts) | Max err (vol pts) | RMS err (vol pts) | ms / surface |
|---|---|---|---|---|
| Lewis FFT + cubic spline (current) | 0.419 | 0.047 | 0.0036 | 9 |
| Lewis, Gauss–Legendre 1500 nodes | 0.003 | 0.009 | 0.0003 | 196 |
| COS, 256 terms | 0.019 | 0.001 | 0.0002 | 15 |
| COS, 512 terms | 0.019 | 0.001 | 0.0002 | 17 |
| Carr–Madan FFT, α = 1.5 | **0.003** | 0.003 | 0.0001 | 11 |
| Carr–Madan FFT, α = 0.75 | 0.208 | 46.4 | 1.75 | 11 |
| Heston 1993 two integrals | 0.268 | 46.4 | 1.74 | 3,234 |

## G. Implied-vol inversion (1273 options)

| Solver | Max error (vol pts) | Time |
|---|---|---|
| Brent per option (reference) | 0 | 1,271 ms |
| Bisection, 50 iterations (current) | 2e-11 | 7.4 ms |
| **Bisection, 30 iterations** | 2e-7 | **4.3 ms** |
| Newton safeguarded, 25 it | 5e-4 | 6.2 ms |
| Newton safeguarded, 10 it | 15 (fails) | 2.4 ms |

## H. Monte Carlo for the note (Heston, daily steps, 16,384 paths; efficiency relative to plain QE)

| Method | Price | s.e. per 100k paths | Relative efficiency |
|---|---|---|---|
| Euler full truncation | 975.3 | 0.49 | 1.29 (cheaper per path; biased at coarse steps) |
| QE pseudo-random (current) | 976.9 | 0.48 | 1.00 |
| QE + 4 control variates | 975.6 | 0.34 | **2.03** |
| QE antithetic | 978.9 | 0.46 | 1.24 |
| QE antithetic + control variates | 976.7 | 0.34 | **2.30** |
| QE Sobol + Brownian bridge | 976.1 | 0.42 | 0.58 |
| QE Sobol, no bridge | 976.5 | 0.32 | 1.13 |
| Step size: 12/yr Euler / QE | 970.0 / 977.1 | – | Euler biased −5.4, QE +1.7 (vs 975.4 at daily) |

## I. Vol surfaces for local vol (live, 1273 quotes)

| Surface | Fit vol RMSE | Max miss | 1M / 3M / 6M / 1Y / 1.3Y / 2.2Y / 3.2Y / 4.2Y | Arbitrage checks |
|---|---|---|---|---|
| SSVI (global ρ, power-law φ) | 1.27 | 11.4 | 2.31 / 1.54 / 1.11 / 0.29 / 0.30 / 0.44 / 0.60 / 0.95 | η(1+\|ρ\|) = 2.0 (binding), θ increasing |
| **SVI slices** | **0.25** | 2.4 | 0.24 / 0.21 / 0.27 / 0.09 / 0.14 / 0.15 / 0.14 / 0.75 | g(k) ≥ 0 all slices; calendar gap ≥ −6e-5 |

## J. Dynamics for the note (live, same 50,000 paths)

| Dynamics | Vanilla repricing RMS vs SVI surface (vol pts) | Note price (s.e.) | KI put | P(KI) | E[level \| KI] | Fair coupon at par |
|---|---|---|---|---|---|---|
| Local vol (Dupire on SVI) | 0.37 | 960.1 (0.86) | −37.3 | 6.7% | 32% | 4.76% |
| LSV (Heston + leverage) | 0.53 | 972.8 (0.73) | −27.0 | 5.2% | 37% | 4.12% |
| Heston (6M+ fit) | 2.17 | 975.4 (0.70) | −24.9 | 4.9% | 38% | 4.03% |
| Bates (all-expiry fit) | – | 975.9 (0.70) | −24.7 | 4.8% | 37% | 3.98% |
| Double Heston (all-expiry fit) | – | 976.4 (0.69) | −24.4 | 4.7% | 37% | 3.97% |

LV vs LSV, same vanillas: 12.7 per 1,000 of pure dynamics (forward skew) risk.
