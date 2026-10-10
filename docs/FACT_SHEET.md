# Fact sheet: Athena autocall on SPX, Heston v2, live snapshot of 30 September 2026

All figures per 1,000 notional unless stated. Monte Carlo: 50,000 paths, daily steps, QE scheme, seed 42, standard error 0.70. December 2025 snapshot figures in brackets where they exist.

## Product (re-struck at the snapshot spot)
| Item | Value |
|---|---|
| Underlying, spot | SPX, 7,696.50 at 15:22 EDT (Dec: 6,829.37) |
| Maturity | 4 years, final valuation 3 Oct 2030 |
| Observation dates | 7, semi-annual from year 1: 1 Oct 2027, 1 Apr 2028, 1 Oct 2028, 2 Apr 2029, 2 Oct 2029, 3 Apr 2030, 3 Oct 2030 |
| Autocall level | 100% of initial, snowball coupon 3% per semester (n × 3%) |
| Final redemption | 124% if S_T ≥ 70%; 100% if 60% ≤ S_T < 70%; S_T / S_0 if S_T < 60% (European knock-in) |
| Payment lag | 7 calendar days after each valuation (term sheet) |
| Decomposition | zero-coupon bond + autocall digitals at 100% + 24% digital at 70% − (60% put + 40% digital put at 60%) |

## Market data
| Item | Value |
|---|---|
| Option quotes pulled | 5,923 on 30 expiries, OTM only, 45–125% of spot, volume ≥ 1 |
| Expiry rule | 1M, 3M, 6M and the semi-annual product dates, nearest listed expiry: 0.09, 0.25, 0.50, 1.00, 1.31, 2.21, 3.22, 4.22 years |
| Quotes used | 1,360 selected, 87 removed by the call-spread no-arbitrage filter (all last traded 12+ days earlier), 1,273 kept; 685 in the 6M+ fit set |
| Quote width | median 0.04–0.09 vol points to 1Y; 0.65 at 3.2Y; 4.5 at 4.2Y |
| Discount curve | FRED Treasury constant-maturity par curve as of 28 Sep 2026 (SOFR OIS proxy); zeros 0.25y 4.26%, 1y 4.49%, 2y 4.88%, 4y 4.99% (Dec: Bloomberg SOFR OIS, 1y 3.46%, 4y 3.20%) |
| Implied dividend (parity, ATM strike) | 0.34% at 1M, 0.38–0.61% from 3M to 4.2Y, all positive (Dec: −2.48% at 1M from the overnight snapshot) |

## Calibration (Heston, fit to 6M–4.2Y, spread-weighted vol objective, DE + L-BFGS-B)
| Item | Value |
|---|---|
| κ, θ, σ, ρ, v0 | 2.87, 0.0527, 1.53, −0.61, 0.0247 (Dec: 2.18, 0.0562, 1.24, −0.69, 0.0377) |
| Spot vol, long-run vol | 15.7%, 22.9% (Dec: 19.4%, 23.7%) |
| Half-life of a vol shock | 2.9 months |
| Feller 2κθ vs σ² | 0.30 vs 2.33, violated; 42% of simulated days at zero variance (Dec: 0.25 vs 1.55, 33%) |
| Objective (weighted vol RMSE on fit set) | 0.541 vol points |
| Vol RMSE all / 1M–3M / 6M+ | 1.08 / 1.44 / 0.615 vol points (1M–3M out of sample) |
| Price RMSE all / 6M+ | 10.7 / 13.5 index points; 0.14% of spot; 30% of OTM option price (median 6.4%) |
| 1Y–3.2Y only | 0.29 vol points; 7.6 points; 0.10% of spot; 3.6% of option price (median 1.8%) |
| Product strikes (1Y–4Y, 60–100%), mean |vol error| | 0.16 vol points (v1 price objective: 0.22) |
| Calibration time | 30 s (least_squares reaches the same optimum in 0.9 s) |

## Price and legs
| Item | Live | Dec 2025 |
|---|---|---|
| Fair value | 975.40 = 97.54% (s.e. 0.70) | 986.13 = 98.61% (s.e. 0.79) |
| Principal leg | 928.69 | 947.03 |
| Coupon leg | 71.62 | 73.80 |
| Knock-in put leg (client short) | −24.91 | −34.70 |
| Fair coupon at par / at 98.75% | 4.03% / 3.51% per semester | 3.56% / 3.06% |
| Margin at the 3% coupon | 2.46% of notional | 1.39% |

## Outcome statistics (50,000 paths)
| Item | Live | Dec 2025 |
|---|---|---|
| Autocall at year 1 | 70.0% | 68.3% |
| Autocall by date | 70.0, 7.8, 3.9, 2.4, 1.7, 1.3% | 68.3, 7.4, 3.5, 2.4, 1.7, 1.3% |
| Autocall at any date | 87.0% | 84.6% |
| Reach maturity | 13.0% | 15.4% |
| of which bonus (≥ 70%) / par (60–70%) / knock-in (< 60%) | 6.5% / 1.6% / 4.9% | 7.3% / 1.9% / 6.2% |
| Probability of loss | 4.9% | 6.2% |
| Mean final level given knock-in | 38.0% (loss 62%) | 36.5% |
| Expected loss, unconditional, undiscounted | 3.05% of notional | 3.9% |
| Effective maturity (expected life) | 1.57 years | 1.64 years |

## Greeks at inception, client seat (desk is the mirror image)
PDE = finite-difference solution (delta and gamma read off the grid; vega, vol-of-vol and correlation sensitivities by re-solving with bumped parameters on the same grid, 28 solves). MC = bump-and-reprice on 100,000 unit paths with common random numbers, spot bumps of ±1% and ±3%, parameter bumps of ±0.002. Per 1,000 notional.

| Leg | Value PDE / MC | Index hedge % of notional, PDE / MC ±1% / ±3% | Cash gamma per 1%, PDE / MC ±1% / ±3% | Vega per vol pt, PDE / MC | d/dσ per +0.1, PDE / MC | d/dρ per +0.1, PDE / MC |
|---|---|---|---|---|---|---|
| Total | 976.2 / 975.4 | 15.7 / 16.4 / 16.3 | −12.1 / −10.4 / −11.3 | −0.57 / −0.51 | +0.93 / +0.60 | −0.28 / −0.29 |
| Principal | 929.3 / 928.7 | 15.8 / 16.1 / 16.4 | −15.6 / −18.7 / −15.7 | −0.27 / −0.29 | +0.68 / +0.58 | −1.40 / −1.67 |
| Coupons | 71.3 / 71.6 | −10.6 / −10.6 / −11.0 | +11.6 / +13.8 / +11.3 | +0.09 / +0.13 | −0.29 / −0.43 | +1.11 / +1.03 |
| Knock-in put | −24.5 / −24.9 | 10.5 / 11.0 / 10.8 | −8.0 / −5.4 / −6.9 | −0.39 / −0.36 | +0.54 / +0.45 | +0.01 / +0.36 |
| Rates, discounting only (forward fixed) | | −0.146 per bp | | | | |
| Rates, full parallel shift (forward moves too) | | −1.13 per 10 bp | | | | |
| Theta | | about +0.7 per week (inside MC noise) | | | | |

Reading: the client is long delta (the desk buys about 16% of notional in index), short gamma at the money, short vega (−0.57 per vol point, mostly the put and the principal's discounting through earlier calls). The vol-of-vol and correlation sensitivities of the whole note are small and dominated by autocall timing rather than by the put: a flatter skew (ρ less negative) delays year-1 calls, which lowers the principal leg (−1.4) and raises the coupon leg (+1.1), net −0.3 per 0.1, while the put leg itself barely moves. Earlier bump-and-reprice figures on 50,000 paths (gamma −3.1, d/dσ −0.16, d/dρ +0.35) were noise; the PDE values are the reference and the 100k-path MC with wider bumps agrees with them.

## Hedging costs and structure sensitivities (v2 live, 100,000 QE paths, common random numbers; client seat, desk = mirror image)
Base 976.34 (s.e. 0.49). Differences are paired, s.e. of the difference in brackets.

| Change | vs base (per 1,000) |
|---|---|
| Autocall level 100% → 99% / 101% | +0.92 / −0.99 (0.08) |
| Bonus barrier 70% → 69.3% / 70.7% | +0.23 / −0.24 (0.02) |
| Knock-in 60% → 59.4% / 60.6% | +0.34 / −0.29 (0.03) |
| All three barriers 1% against the desk | +1.47 (0.09); fair coupon at par 3.99% → 3.94% |
| Coupon ±10 bp per semester (bonus 8×) | ∓2.39; +0.239 per bp; with the bonus held at 24%: +1.97 per +10 bp |
| Autocall digitals as call spreads, width 1% / 2% / 5% | +0.48 / +0.93 / +2.29 |
| 70% bonus digital as a call spread, 1% / 2% / 5% | +0.11 / +0.22 / +0.55 |
| 60% knock-in digital as a put spread, 1% / 2% / 5% | +0.17 / +0.32 / +0.74 |
| All digitals as spreads, 1% / 2% / 5% | +0.76 / +1.46 / +3.47; fair coupon with 2% spreads 3.94% |
| Payment lag 0 / 14 days (base 7) | +0.93 / −0.93 |
| Knock-in observed daily instead of at maturity | −8.50 (0.14) |
| Dividend yield ±10 bp | ∓0.42 (0.04) |
| All observation dates +5 / −5 days | −1.23 / +1.09 (0.24, re-simulated) |
| Rates +10 bp, discounting only | −1.46 |
| Vega-hedge crossing cost (put-leg vega 0.39/vol pt × half the 4.2Y 60% quote width, 2.8 vol pts) | about 1.1; with 1Y–2Y puts about 0.04 |
| Model-risk reserve (local vol vs LSV, same vanillas) | 12.7 |

## Model comparison on the same note (same paths, same payoff)
| Dynamics | Price | vs Heston | Put leg | P(KI) | E[level | KI] | Fair coupon | Vanilla fit vs market (7 expiries ≤ 4y): vol RMSE / price RMSE / % of option price |
|---|---|---|---|---|---|---|---|
| Local vol (Dupire on SVI slices) | 960.1 (0.86) | −15.3 | −37.3 | 6.7% | 32% | 4.76% | 0.27 vol pts / 2.6 pts / 5.9% |
| LSV (Heston + particle leverage) | 972.8 (0.73) | −2.6 | −27.0 | 5.2% | 37% | 4.12% | 0.34 / 2.9 / 10.2% |
| Heston (6M+ fit) | 975.4 (0.70) | – | −24.9 | 4.9% | 38% | 4.03% | 1.08 / 6.7 / 30% |
| Heston, finite differences (no MC noise) | 976.2 | +0.8 | −24.5 | | | | same model |
| Bates (all-expiry fit) | 975.9 (0.70) | +0.5 | −24.7 | 4.8% | 37% | 3.98% | 0.42 / 9.9 (8 expiries) / 9.4% |
| Double Heston (all-expiry fit) | 976.4 (0.69) | +1.0 | −24.4 | 4.7% | 37% | 3.97% | 0.22 / 8.9 (8 expiries) / 4.2% |

LV − LSV = −12.7 with the same vanillas: pure dynamics (forward skew) model risk, 1.3% of notional. LSV − Heston = −2.6: the value of fitting every vanilla. Stochastic-vol models among themselves: within 1 per 1,000, inside noise.

## Price attribution, v1 method to v2 (live, same random numbers)
| Step | Price | Change |
|---|---|---|
| v1 method (price objective, all expiries), Euler, no payment lag | 974.9 | |
| + QE scheme | 975.4 | +0.5 |
| + v2 calibration (vol objective, filtered quotes, 6M+) | 976.5 | +1.1 |
| + 7-day payment lag | 975.6 | −0.9 |
| Discounting 25 bp lower (SOFR OIS instead of Treasury) | 979.3 | +3.7 |

## Validation (all in the notebook)
| Check | Result |
|---|---|
| Characteristic function, φ(−i) = e^{(r−q)T} | exact to 1e-15 at every expiry |
| FFT + spline vs direct quadrature | ≤ 0.004 index points at 1Y and beyond (0.04 at 1M); Carr–Madan / COS alternatives 100× tighter |
| Martingale, E[S_t]/F(t) − 1 | within about 1 s.e. at 0.25, 1, 2, 3, 4 years |
| Vanilla repricing, MC vs FFT, 60–110% at 1Y and 4Y | largest gap 1.64 s.e. |
| Step-size convergence, 12 / 52 / 252 steps a year | Euler off by −5.4 at monthly steps (6 s.e.), QE +1.7; both agree at daily steps |
| Path convergence, 25k / 50k / 100k / 200k | 976.4 / 975.4 / 976.3 / 975.8; s.e. × √n constant at 154–156 |
| Limit case: no autocall, KI = 0, vs closed-form bond + 24% digital at 70% | 986.23 (s.e. 0.31) vs 986.78, −1.8 s.e. |
| Pure bond limit | exact (817.94) |
| Independent implementation: 2-D Heston PDE (Craig–Sneyd ADI, 509 × 80 grid, autocall as jump conditions) | note 976.19 vs MC 975.40 (s.e. 0.70); legs 929.3 / 71.3 / −24.5 vs 928.7 / 71.6 / −24.9; European calls vs FFT within 0.03–0.8 points (≤ 0.03 vol); limit case 986.91 vs closed form 986.78; grid refinement 976.5 → 976.2 → 976.0; time-step doubling no change |
| Monte Carlo efficiency | control variates 2.0×, antithetic + control variates 2.3×; Sobol no gain |

## Is the Feller violation a bug? (checked 2 October 2026)
| Check | Result |
|---|---|
| Model prices below the intrinsic bound (would floor the vol inversion and inflate σ) | none, in either fit |
| Same calibration with the COS engine (100× more accurate than the FFT + spline) | identical optimum: σ 1.51 vs 1.53, objective 0.5411 in both |
| κ–σ ridge (σ²/κ held at 0.81, θ, ρ, v0 re-optimised) | objective 0.69 at κ = 1 / σ = 0.9, 0.54 at κ = 2.9 / σ = 1.5, 0.65 at κ = 6 / σ = 2.2: a real minimum |
| Heston with Feller enforced (σ² ≤ 2κθ), 6M+ fit | σ 0.38, ρ −0.97; vol RMSE 6M+ 0.61 → 2.58, price RMSE 10.7 → 25.8, product strikes 0.16 → 1.11; note 970.9 (QE), 970.8 (Euler), 971.1 (PDE) |
| Heston with Feller enforced, all-expiry fit | σ 0.53, ρ −0.93; vol RMSE 0.90 → 3.45 |
| Euler vs QE vs PDE, current parameters, daily steps | Euler 974.4, QE 975.4, PDE 976.2 (s.e. 0.7): Euler 2.5 s.e. below the PDE, QE 1.1; with the Feller-constrained parameters all three agree (970.8 / 970.9 / 971.1) |
| Share of simulated days at zero variance | 42% (QE) / 38% (Euler) with the fitted parameters; 0% with Feller enforced |

Reading: the growth of σ across runs follows the objective, not a defect. Price-space objectives see mostly ATM and long-dated options and settle at σ ≈ 1.1–1.2; vol-space objectives see the wings, and the only tool Heston has for wing curvature is σ. Enforcing Feller costs 2 vol points of fit everywhere and 4.5 per 1,000 on the note. Euler is not better: it is slightly more biased than QE at daily steps when Feller is badly violated and identical when it is not.

## v1 benchmark (December parameters, v1 conventions, `v1_benchmark.ipynb`)
| Item | Value |
|---|---|
| Parameters (reproduced exactly) | κ 2.4095, θ 0.0548, σ 1.1947, ρ −0.7226, v0 0.0310; price RMSE 13.2876 on 906 quotes |
| Same fit in vol terms | 1.52 / 2.27 / 0.62 vol points (all / 1M–3M / 6M+); largest miss 24.5 vol points at the 1M 50% put |
| v1 Euler MC, seed 42 | 986.83 (s.e. 0.79) = 947.74 + 73.68 − 34.59; submission 987.53 (no seed) |
| PDE, v1 conventions (flat drift, no lag) | 988.23 = 948.06 + 73.53 − 33.36; +1.40 vs MC = 1.8 s.e. |
| PDE Greeks at spot | delta 13.0% of notional as index hedge (script: "about 13%"); cash gamma per 1% −7.8 |
| Limit case | PDE 1055.71, closed form 1055.55, MC 1055.18 |
| Euler step convergence | 983.6 / 987.1 / 986.8 at 12 / 52 / 252 steps a year; path convergence 987.9 / 986.8 / 988.6 / 987.9 at 25k–200k |
| Same parameters, QE scheme (reference only) | 987.22 (s.e. 0.79); Euler 986.81; PDE 988.23 |
| Filter + QE on v1's set-up (section 7 of `v1_benchmark.ipynb`) | filter removes 23 of 906 quotes; v1 objective on the 883 left: κ 2.53, σ 1.20, ρ −0.72, Feller 0.189 (from 0.185); price RMSE 13.29 → 11.44, product strikes 0.35 → 0.36; MC–PDE gap: Euler −1.8 s.e. → QE −1.3 s.e. (filtered params: −1.75 → −1.23); s.e. unchanged at 0.79 |

## Where the errors are
| Region | Vol RMSE | Cause |
|---|---|---|
| 1M, 3M | 2.0, 1.0 (Heston) | a diffusion cannot build the short-dated skew; Bates 1.2 / 0.3, double Heston 0.4 / 0.1 |
| 1Y–3.2Y | 0.19–0.39 | inside quote width |
| 4.2Y | 0.85–0.95 for every model | quotes 4.5 vol points wide; data floor, not model |
| Price RMSE growth with maturity | | vega: 26 points per vol point at 1Y, 39 at 4.2Y |
