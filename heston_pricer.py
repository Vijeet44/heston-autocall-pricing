import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import copy
import os
import sys
from contextlib import contextmanager

from scipy.fft import ifft
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution, OptimizeResult, brentq
from scipy.stats import norm

from numba import njit


#---------------------Bootstrapping & Curves---------------------
class OISBootstrapper:
    # Bloomberg SOFR OIS par swap rates, 3 December 2025 (I forgot but the command might be USOSFRA BGN Curncy <GO>)
    DEFAULT_MARKET_DATA = [
        (1/52, 4.0433), (2/52, 4.0353), (3/52, 3.9704),
        (1/12, 3.9175), (2/12, 3.8650), (3/12, 3.8182),
        (4/12, 3.7930), (5/12, 3.7605), (6/12, 3.7283),
        (9/12, 3.6226), (1,  3.5245),
        (1.5,  3.3602), (2,  3.2860), (3,  3.2410),
        (4,  3.2557), (5,  3.2940), (7,  3.4145),
        (10, 3.6005), (20, 3.9565), (30, 3.9435),
    ]

    def __init__(self, market_data=None, coupon_freq=1):
        """
        Args:
            market_data: list of (tenor in years, par rate in %). Defaults to the December 2025 SOFR OIS quotes.
            coupon_freq: fixed-leg payments per year for tenors >= 2y (1 for OIS swaps, 2 for Treasury par yields).
        """
        self.market_data = list(market_data) if market_data is not None else list(self.DEFAULT_MARKET_DATA)
        self.market_data.sort(key=lambda x: x[0]) # Ensure sorted by T
        self.coupon_freq = coupon_freq


    def bootstrap(self):
        known_ts = [] # maturities successfully calculated
        known_rs = [] # annualized, continuously compounded zero-coupon rates for known_ts

        # Helper for linear interpolation of zero rates
        def get_df(t):
            if t <= 0:
                return 1
            # If t is beyond our current knowledge, use the last known zero rate (flat extrapolation)
            if not known_ts:
                return 1

            # Interpolate R_zero, NOT DF
            r = np.interp(t, known_ts, known_rs)
            return np.exp(-r * t)

        for T, rate_pct in self.market_data:
            R_par = rate_pct / 100
            if T < 2: # 18M max
                # Short tenor: simple discounting
                df_T = 1 / (1 + R_par * T)
                r_zero = -np.log(df_T) / T

            else:
                # Long tenor: par instrument with coupon_freq fixed payments a year
                step = 1 / self.coupon_freq
                coupon = R_par * step
                # First calculate value of the fixed coupons before maturity
                prev_coupons_val = 0
                payment_dates = np.arange(step, T - 0.01, step)
                for t_pay in payment_dates:
                    prev_coupons_val += coupon * get_df(t_pay)
                # Boorstap from discounted principal_fixed coupon value at maturity
                df_T = (1 - prev_coupons_val) / (1 + coupon)
                r_zero = -np.log(df_T) / T

            known_ts.append(T)
            known_rs.append(r_zero)

        return interp1d(known_ts, known_rs, kind='linear', fill_value="extrapolate")

# Instantiate global curve for module-level access
_bootstrapper = OISBootstrapper()
sofr_zero_curve = _bootstrapper.bootstrap()


def set_rate_curve(market_data, coupon_freq=1):
    """
    Rebuilds the module-level discount curve from (tenor_years, rate_pct) pairs.
    rate_curve_func and DF read the module global, so every later call uses the new curve.

    Returns:
        the new zero-curve interpolator
    """
    global _bootstrapper, sofr_zero_curve
    _bootstrapper = OISBootstrapper(market_data, coupon_freq)
    sofr_zero_curve = _bootstrapper.bootstrap()
    return sofr_zero_curve


# Define rate curve function (wrapper)
def rate_curve_func(T):
    """Returns annualized zero rate for T"""
    if isinstance(T, (np.ndarray, list, pd.Series)):
        T = np.maximum(T, 1e-4) # Avoid T=0 division
    else:
        if T <= 0:
            T = 1e-4
    return sofr_zero_curve(T)

# Define Discount Factor Function
def DF(T):
    """Calculates discount factor e^(-rT)"""
    r = rate_curve_func(T)
    return np.exp(-r * T)



#---------------------Black-Scholes helpers (vectorised)---------------------
def bs_call_price(S, K, T, r, q, vol):
    """Vectorised Black-Scholes call. All inputs broadcast."""
    vol = np.maximum(vol, 1e-12)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqT)
    d2 = d1 - vol * sqT
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, q, vol):
    """Vectorised Black-Scholes vega (per unit of vol)."""
    vol = np.maximum(vol, 1e-12)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqT)
    return S * np.exp(-q * T) * norm.pdf(d1) * sqT


def implied_vol_call_vec(C, S, K, T, r, q, vol_lo=1e-4, vol_hi=5.0, n_iter=50):
    """
    Vectorised implied vol by bisection (Black-Scholes call price is monotone in vol).

    A price below the no-arbitrage lower bound converges to vol_lo, a price above the
    upper bound converges to vol_hi. That is deliberate: inside a calibration objective an
    impossible model price should produce a large residual, not an exception.

    Args:
        C: call prices (array)
        S, K, T, r, q: broadcastable inputs
    Returns:
        implied vols, same shape as C
    """
    C = np.asarray(C, dtype=float)
    lo = np.full(C.shape, vol_lo, dtype=float)
    hi = np.full(C.shape, vol_hi, dtype=float)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        above = bs_call_price(S, K, T, r, q, mid) > C
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)



#---------------------Quote hygiene---------------------
def attach_quote_details(df_clean, xlsx_path, r_curve_func):
    """
    Merges bid, ask, lastTradeDate, volume and openInterest from the 'Original Data' sheet
    onto the cleaned calibration set, and converts the put bid/ask to call-equivalent bid/ask
    with the same parity relation used for the mid: C = P + S e^{-qT} - K e^{-rT}.

    Args:
        df_clean: DataFrame from the 'Cleaned Data' sheet (possibly already resampled)
        xlsx_path: path to the calibration workbook
        r_curve_func: zero-rate function
    Returns:
        DataFrame with extra columns: bid, ask, bid_call_eq, ask_call_eq, lastTradeDate, volume,
        openInterest, quote_age_days
    """
    orig = pd.read_excel(xlsx_path, sheet_name="Original Data")
    spot_sheet = pd.read_excel(xlsx_path, sheet_name="Spot Price")
    S0 = float(spot_sheet['S0'].iloc[0])
    snapshot = pd.to_datetime(spot_sheet['Date'].iloc[0])

    cols = ['strike', 'T', 'type', 'bid', 'ask', 'lastTradeDate', 'volume', 'openInterest']
    merged = df_clean.merge(
        orig[cols],
        left_on=['strike', 'T', 'type_before_conversion'],
        right_on=['strike', 'T', 'type'],
        how='left', validate='1:1'
    ).drop(columns=['type'])

    if merged['bid'].isna().any():
        raise ValueError("Some cleaned quotes could not be matched to the original snapshot.")

    T = merged['T'].values
    r = np.asarray(r_curve_func(T), dtype=float)
    q = merged['implied_q'].values
    parity_shift = S0 * np.exp(-q * T) - merged['strike'].values * np.exp(-r * T)
    is_put = (merged['type_before_conversion'] == 'put').values

    merged['bid_call_eq'] = np.where(is_put, merged['bid'].values + parity_shift, merged['bid'].values)
    merged['ask_call_eq'] = np.where(is_put, merged['ask'].values + parity_shift, merged['ask'].values)
    merged['lastTradeDate'] = pd.to_datetime(merged['lastTradeDate'])
    merged['quote_age_days'] = (snapshot.normalize() - merged['lastTradeDate'].dt.normalize()).dt.days
    return merged


def filter_call_spread_arbitrage(df, r_curve_func, tol=0.02):
    """
    Static no-arbitrage filter on call-equivalent mid prices, one maturity at a time.

    Between adjacent strikes a call price must satisfy  -e^{-rT} <= dC/dK <= 0
    (a call spread cannot be worth more than its discounted width, and calls cannot get
    dearer as the strike rises). A quote pair that breaks the bound is inconsistent; we
    drop the quote that traded less recently (needs 'quote_age_days'), which is the one
    more likely to be stale. Repeats until every slice is clean.

    Args:
        df: DataFrame with 'strike', 'T', 'market_price_call_equivalent' and, ideally,
            'quote_age_days' and bid/ask columns
        r_curve_func: zero-rate function
        tol: relative slack on the slope bound (0.02 = 2%)
    Returns:
        (kept DataFrame, dropped DataFrame with a 'reason' column)
    """
    kept = []
    dropped = []

    for T, grp in df.groupby('T'):
        g = grp.sort_values('strike').reset_index(drop=True)
        df_T = float(np.exp(-r_curve_func(T) * T))
        while True:
            K = g['strike'].values
            C = g['market_price_call_equivalent'].values
            if len(g) < 2:
                break
            slope = np.diff(C) / np.diff(K)
            bad = np.where((slope < -df_T * (1 + tol)) | (slope > tol * df_T))[0]
            if len(bad) == 0:
                break
            i = bad[0]
            a, b = g.iloc[i], g.iloc[i + 1]
            if 'quote_age_days' in g.columns and a['quote_age_days'] != b['quote_age_days']:
                victim = i if a['quote_age_days'] > b['quote_age_days'] else i + 1
            elif 'bid' in g.columns:
                rel_a = (a['ask'] - a['bid']) / max(a['mid_price'], 1e-8)
                rel_b = (b['ask'] - b['bid']) / max(b['mid_price'], 1e-8)
                victim = i if rel_a > rel_b else i + 1
            else:
                victim = i + 1
            row = g.iloc[victim].copy()
            row['reason'] = f"call-spread slope {slope[i]:.3f} vs bound {-df_T:.3f} (pair {K[i]:.0f}/{K[i+1]:.0f})"
            dropped.append(row)
            g = g.drop(index=victim).reset_index(drop=True)
        kept.append(g)

    kept_df = pd.concat(kept, ignore_index=True)
    dropped_df = pd.DataFrame(dropped).reset_index(drop=True) if dropped else pd.DataFrame(columns=list(df.columns) + ['reason'])
    return kept_df, dropped_df



#---------------------Heston FFT Calibrator---------------------
class HestonFFTCalibrator:
    def __init__(self, market_data, S0, r_curve_func,
                 objective='iv', weight_mode='spread', maturity_weights=None,
                 min_vol_spread=0.005):
        """
        Lewis (2001) FFT Calibrator with Hybrid Optimization (Global + Local).

        Args:
            market_data: DF with ['strike', 'T', 'market_price_call_equivalent', 'implied_q'];
                         optional ['bid_call_eq', 'ask_call_eq'] enable bid-ask weighting
            S0: Spot price
            r_curve_func: zero-rate function
            objective: 'iv'    -> weighted RMSE of implied vols (default)
                       'price' -> plain RMSE of call prices (the original objective)
            weight_mode: 'spread'  -> weight = 1 / (bid-ask spread in vol)^2, needs bid/ask columns
                         'uniform' -> every quote weighs the same
            maturity_weights: optional dict {T: weight}. Each maturity slice is first normalised
                              to total weight 1 ("one vote per expiry"), then scaled by this.
            min_vol_spread: floor on the vol spread used for weights (0.005 = half a vol point),
                            so a handful of tight ATM quotes cannot dominate the fit
        """
        self.market_data = market_data.copy()
        self.S0 = S0
        self.r_curve = r_curve_func  # Store the function
        self.objective = objective
        self.weight_mode = weight_mode
        self.min_vol_spread = min_vol_spread


        # Optimization Bounds [kappa, theta, sigma, rho, v0]
        self.bounds = [
            (0.01, 15),   # kappa: Mean reversion speed
            (0.001, 1),   # theta: Long-run variance
            (0.01, 10),    # sigma: Vol of vol
            (-0.999, 0.999),# rho: Correlation (avoid exact -1/1)
            (0.001, 1)    # v0: Initial variance
        ]

        # Grid Parameters
        self.N = 4096
        self.B = 300

        self.du = self.B / self.N
        self.u = np.arange(self.N) * self.du

        self.dk = 2 * np.pi / self.B
        self.b = self.N * self.dk / 2
        self.k_fft = -self.b + self.dk * np.arange(self.N)

        # Simpson's Weights
        self.weights = 3 + (-1) ** (np.arange(self.N) + 1)
        self.weights[0] = 1
        self.weights[self.N - 1] = 1
        self.weights = self.weights * self.du / 3

        # Pre-Group Data
        self.unique_T = self.market_data['T'].unique()
        self.data_batches = []

        for T in self.unique_T:
            mask = self.market_data['T'] == T
            batch = {
                'T': T,
                'strikes': self.market_data.loc[mask, 'strike'].values,
                'market_prices': self.market_data.loc[mask, 'market_price_call_equivalent'].values,
                'k_market': np.log(S0 / self.market_data.loc[mask, 'strike'].values),
                'q': self.market_data.loc[mask, 'implied_q'].iloc[0] # Extract the unique implied q for this maturity
            }
            if 'bid_call_eq' in self.market_data.columns:
                batch['bid'] = self.market_data.loc[mask, 'bid_call_eq'].values
                batch['ask'] = self.market_data.loc[mask, 'ask_call_eq'].values
            self.data_batches.append(batch)

        # Flat arrays in batch order (same order as price_surface output)
        self.flat_T = np.concatenate([np.full(len(b['strikes']), b['T']) for b in self.data_batches])
        self.flat_K = np.concatenate([b['strikes'] for b in self.data_batches])
        self.flat_q = np.concatenate([np.full(len(b['strikes']), b['q']) for b in self.data_batches])
        self.flat_r = np.asarray(self.r_curve(self.flat_T), dtype=float)
        self.flat_market = np.concatenate([b['market_prices'] for b in self.data_batches])

        # Market implied vols with OUR r and q (so market and model vols are comparable)
        self.market_iv = implied_vol_call_vec(self.flat_market, S0, self.flat_K, self.flat_T,
                                              self.flat_r, self.flat_q)

        self._build_weights(maturity_weights)

    def _build_weights(self, maturity_weights):
        """
        Quote weights for the 'iv' objective.
        spread mode: w_i = 1 / max(iv_ask - iv_bid, floor)^2, i.e. the market's own uncertainty
        about each quote, then normalised so that each maturity slice sums to one.
        """
        n = len(self.flat_K)
        has_ba = all('bid' in b for b in self.data_batches)

        if self.weight_mode == 'spread' and has_ba:
            bid = np.concatenate([b['bid'] for b in self.data_batches])
            ask = np.concatenate([b['ask'] for b in self.data_batches])
            iv_bid = implied_vol_call_vec(bid, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
            iv_ask = implied_vol_call_vec(ask, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
            self.vol_spread = np.maximum(iv_ask - iv_bid, self.min_vol_spread)
            w = 1.0 / self.vol_spread**2
        else:
            self.vol_spread = np.full(n, np.nan)
            w = np.ones(n)

        # Optional per-quote multiplier supplied with the data (e.g. product strike range)
        if 'weight_mult' in self.market_data.columns:
            w = w * np.concatenate([self.market_data.loc[self.market_data['T'] == b['T'], 'weight_mult'].values
                                    for b in self.data_batches])

        # One vote per expiry, optionally rescaled
        for b in self.data_batches:
            m = self.flat_T == b['T']
            w[m] = w[m] / w[m].sum()
            if maturity_weights is not None:
                w[m] *= maturity_weights.get(b['T'], 1.0)
        self.quote_weights = w / w.sum()

    def _heston_char_func(self, u, T, r_t, q_t, params):
        """
        Stable Albrecher/Schoutens form Heston characteristic function.

        Args:
        u: Frequency parameter of the Heston characteristic function
        T: Time to maturity
        r_t: Drift
        q_t: Dividend Yield
        params: [kappa, theta, sigma, rho, v0]

        Returns:
            Model prices for each strike and maturity, flattened; dimensions: [n_strikes * n_maturities]
        """
        kappa, theta, sigma, rho, v0 = params

        # Drift uses the specific rate for this T
        mu = r_t - q_t

        xi = kappa - sigma * rho * u * 1j
        d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
        g1 = (xi + d) / (xi - d)
        g2 = 1 / g1

        cf = np.exp(
            1j * u * mu * T
            + (kappa * theta) / (sigma**2) * ((xi - d) * T - 2 * np.log((1 - g2 * np.exp(-d * T)) / (1 - g2)))
            + (v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * T)) / (1 - g2 * np.exp(-d * T))
        )
        return cf

    def price_calls(self, params, T, strikes, q_t):
        """
        Lewis (2001) call prices for one maturity via FFT + cubic spline in log-moneyness.

        Args:
            params: [kappa, theta, sigma, rho, v0]
            T: maturity (years)
            strikes: array of strikes
            q_t: dividend yield for this maturity
        Returns:
            call prices, same shape as strikes
        """
        strikes = np.asarray(strikes, dtype=float)
        r_t = float(self.r_curve(T))

        # Characteristic Function
        cf_vals = self._heston_char_func(self.u - 0.5j, T, r_t, q_t, params)

        # Input Array
        input_array = np.exp(-1j * self.b * self.u) * cf_vals * self.weights / (self.u**2 + 0.25)

        # Inverse FFT
        integral_val = np.real(ifft(input_array) * self.N)

        # Interpolation
        spline = interp1d(self.k_fft, integral_val, kind='cubic', fill_value="extrapolate")
        integral_interp = spline(np.log(self.S0 / strikes))

        # Lewis Formula
        discount = np.exp(-r_t * T)
        div_factor = np.exp(-q_t * T)

        return (self.S0 * div_factor) - (np.sqrt(self.S0 * strikes) * discount / np.pi) * integral_interp

    def price_calls_quadrature(self, params, T, strikes, q_t, u_max=300.0, n_u=20001):
        """
        Same Lewis integral evaluated by direct trapezoidal quadrature on a fine grid.
        Independent of the FFT grid and of the spline: used as an engine check.
        """
        strikes = np.asarray(strikes, dtype=float)
        r_t = float(self.r_curve(T))
        u = np.linspace(1e-8, u_max, n_u)
        cf_vals = self._heston_char_func(u - 0.5j, T, r_t, q_t, params)
        k = np.log(self.S0 / strikes)[:, None]
        integrand = np.real(np.exp(1j * u[None, :] * k) * cf_vals[None, :]) / (u[None, :]**2 + 0.25)
        integral = np.trapezoid(integrand, u, axis=1)
        return self.S0 * np.exp(-q_t * T) - np.sqrt(self.S0 * strikes) * np.exp(-r_t * T) / np.pi * integral

    def price_surface(self, params):
        """
        Heston model prices for every calibration quote, in batch order.

        Args:
            params: [kappa, theta, sigma, rho, v0]

        Returns:
            Model prices, flattened; dimensions: [n_strikes * n_maturities]
        """
        model_prices = [self.price_calls(params, b['T'], b['strikes'], b['q']) for b in self.data_batches]
        return np.concatenate(model_prices)

    def model_iv(self, params):
        """Model implied vols for every calibration quote (same r, q as the market vols)."""
        prices = self.price_surface(params)
        return implied_vol_call_vec(prices, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)

    def objective_function(self, params):
        """
        Calibration objective.
            'price': RMSE of call prices (index points)
            'iv':    weighted RMSE of implied vols, weights = self.quote_weights

        Args:
            params: [kappa, theta, sigma, rho, v0]

        Returns:
            objective value
        """
        try:
            model_vals = self.price_surface(params)
        except Exception:
            return 1e10 # Catch numerical overflows

        if not np.all(np.isfinite(model_vals)):
            return 1e10

        if self.objective == 'price':
            return np.sqrt(np.mean((model_vals - self.flat_market)**2))

        iv_model = implied_vol_call_vec(model_vals, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
        resid = iv_model - self.market_iv
        return np.sqrt(np.sum(self.quote_weights * resid**2))

    def price_rmse(self, params):
        return float(np.sqrt(np.mean((self.price_surface(params) - self.flat_market)**2)))

    def iv_rmse(self, params, weighted=False):
        resid = self.model_iv(params) - self.market_iv
        if weighted:
            return float(np.sqrt(np.sum(self.quote_weights * resid**2)))
        return float(np.sqrt(np.mean(resid**2)))

    def fit_report(self, params):
        """
        Per-maturity fit table: price RMSE, IV RMSE, weighted IV RMSE, worst IV miss.
        The last row is the whole surface.
        """
        prices = self.price_surface(params)
        iv_model = implied_vol_call_vec(prices, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
        resid_iv = iv_model - self.market_iv
        resid_px = prices - self.flat_market
        rows = []
        for b in self.data_batches:
            m = self.flat_T == b['T']
            w = self.quote_weights[m] / self.quote_weights[m].sum()
            rows.append({
                'T': b['T'], 'n': int(m.sum()),
                'price RMSE': np.sqrt(np.mean(resid_px[m]**2)),
                'IV RMSE (vol pts)': 100 * np.sqrt(np.mean(resid_iv[m]**2)),
                'weighted IV RMSE (vol pts)': 100 * np.sqrt(np.sum(w * resid_iv[m]**2)),
                'max |IV err| (vol pts)': 100 * np.max(np.abs(resid_iv[m])),
                'strike of max': self.flat_K[m][np.argmax(np.abs(resid_iv[m]))],
            })
        rows.append({
            'T': 'ALL', 'n': len(self.flat_K),
            'price RMSE': np.sqrt(np.mean(resid_px**2)),
            'IV RMSE (vol pts)': 100 * np.sqrt(np.mean(resid_iv**2)),
            'weighted IV RMSE (vol pts)': 100 * np.sqrt(np.sum(self.quote_weights * resid_iv**2)),
            'max |IV err| (vol pts)': 100 * np.max(np.abs(resid_iv)),
            'strike of max': self.flat_K[np.argmax(np.abs(resid_iv))],
        })
        return pd.DataFrame(rows)

    def calibrate(self, verbose=True, fixed_params=None, maxiter=50, popsize=15, seed=42, workers=1):
        """
        Global optimization with local refinement:
        1. Global optimizer (Differential Evolution) explores; may not find global optimum if popsize is small.
        2. Local optimizer (L-BFGS-B) refines using global results as starting point.
           (DE's built-in polish is switched off so the local step is explicit.)

        Args:
            verbose: Print progress.
            fixed_params: [kappa, theta, sigma, rho, v0], hardcoded for debugging.
            maxiter, popsize, seed, workers: Differential Evolution settings.

        Returns:
            scipy OptimizeResult; .x = [kappa, theta, sigma, rho, v0]
        """

        if fixed_params is not None:
            if verbose:
                print(f"Skipping optimization. Using fixed parameters: {fixed_params}")

            # Create a fake result object that looks exactly like scipy's output
            fixed_array = np.array(fixed_params)

            # Optional: Calculate the objective for these parameters so you see the error metric
            fun = self.objective_function(fixed_array)

            return OptimizeResult(x=fixed_array, fun=fun, message="Fixed Parameters Used")

        if verbose: print(f"Starting global optimization (Differential Evolution), objective = '{self.objective}'.")

        # Global Optimization
        result_global = differential_evolution(
            self.objective_function,
            bounds=self.bounds,
            strategy='best1bin',
            maxiter=maxiter,      # Cap iterations to save time
            popsize=popsize,      # Population size
            tol=1e-4,
            polish=False,
            workers=workers,
            seed=seed
        )

        if verbose:
            print(f"Global Best: {result_global.x}, objective: {result_global.fun:.6f} ({result_global.nfev} evaluations)")
            print("Starting Local Optimization (L-BFGS-B).")

        # Local Refinement: use the global result as the starting point (x0)
        result_local = minimize(
            self.objective_function,
            x0=result_global.x,
            method='L-BFGS-B',
            bounds=self.bounds,
            tol=1e-10
        )

        if verbose:
            print(f"Local Best: {result_local.x}, objective: {result_local.fun:.6f}")

        return result_local




#---------------------Heston Paths Simulation---------------------
# Define static numba functions outside of the class to speed up simulation
@njit(fastmath=True)
def heston_engine_euler_numba(S0, v0, kappa, theta, sigma, rho, dts, drifts, Z1, Z2):
    """
    Euler-Maruyama with full truncation (Lord, Koekkoek, van Dijk 2010), log-Euler for the spot.
    Kept as the benchmark scheme.

    Args:
        S0, v0: starting spot and variance
        kappa, theta, sigma, rho: Heston parameters
        dts: step lengths (N_steps,)
        drifts: log-forward increment per step, log F(t_{k+1}) - log F(t_k)  (N_steps,)
        Z1, Z2: independent standard normals (N_steps, n_paths); Z1 drives the variance

    Returns:
        S_paths: (N_steps + 1, n_paths)
        zero_frac: fraction of (step, path) cells where the variance was at or below zero
    """
    N_steps, n_paths = Z1.shape
    S_paths = np.empty((N_steps + 1, n_paths))
    v = np.empty(n_paths)
    for p in range(n_paths):
        S_paths[0, p] = S0
        v[p] = v0

    sq1mr2 = math.sqrt(1.0 - rho * rho)
    n_zero = 0
    for t in range(N_steps):
        dt = dts[t]
        sqrt_dt = math.sqrt(dt)
        mu = drifts[t]
        for p in range(n_paths):
            v_curr = v[p]
            v_pos = v_curr if v_curr > 0.0 else 0.0
            if v_curr <= 0.0:
                n_zero += 1
            zv = Z1[t, p]
            zs = rho * zv + sq1mr2 * Z2[t, p]
            S_paths[t + 1, p] = S_paths[t, p] * math.exp(mu - 0.5 * v_pos * dt + math.sqrt(v_pos) * sqrt_dt * zs)
            v[p] = v_curr + kappa * (theta - v_pos) * dt + sigma * math.sqrt(v_pos) * sqrt_dt * zv

    return S_paths, n_zero / (N_steps * n_paths)


@njit(fastmath=True)
def heston_engine_qe_numba(S0, v0, kappa, theta, sigma, rho, dts, drifts, Z1, Z2,
                           psi_c=1.5, gamma1=0.5, gamma2=0.5):
    """
    Andersen (2008) Quadratic-Exponential scheme with martingale correction.

    Variance: exact conditional mean m and variance s^2 of the CIR step; psi = s^2/m^2.
        psi <= psi_c: v' = a (b + Z)^2           (moment-matched squared Gaussian)
        psi >  psi_c: v' = 0 with prob p, else exponential tail  (moment-matched, mass at zero)
    Log-spot: Andersen eq. (33) with K0 replaced by the martingale-corrected K0* (eqs. 37-38),
    so that E[S_{t+dt} / S_t] = exp(drift) holds exactly step by step.

    Args:
        as heston_engine_euler_numba. Z1 drives the variance (its normal cdf gives the uniform
        for the exponential branch), Z2 drives the spot.
    Returns:
        S_paths: (N_steps + 1, n_paths)
        zero_frac: fraction of (step, path) cells where the variance was sampled at zero
    """
    N_steps, n_paths = Z1.shape
    S_paths = np.empty((N_steps + 1, n_paths))
    v = np.empty(n_paths)
    for p in range(n_paths):
        S_paths[0, p] = S0
        v[p] = v0

    rho_sig = rho / sigma
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    n_zero = 0

    for t in range(N_steps):
        dt = dts[t]
        mu = drifts[t]
        E = math.exp(-kappa * dt)
        K0 = -rho * kappa * theta * dt / sigma
        K1 = gamma1 * dt * (kappa * rho_sig - 0.5) - rho_sig
        K2 = gamma2 * dt * (kappa * rho_sig - 0.5) + rho_sig
        K3 = gamma1 * dt * (1.0 - rho * rho)
        K4 = gamma2 * dt * (1.0 - rho * rho)
        A = K2 + 0.5 * K4
        c1 = sigma * sigma * E * (1.0 - E) / kappa
        c2 = theta * sigma * sigma * (1.0 - E) * (1.0 - E) / (2.0 * kappa)

        for p in range(n_paths):
            vt = v[p]
            m = theta + (vt - theta) * E
            s2 = vt * c1 + c2
            psi = s2 / (m * m)

            if psi <= psi_c:
                inv = 2.0 / psi
                b2 = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
                a = m / (1.0 + b2)
                b = math.sqrt(b2)
                zv = Z1[t, p]
                vn = a * (b + zv) * (b + zv)
                den = 1.0 - 2.0 * A * a
                if den > 0.0:
                    K0s = -A * b2 * a / den + 0.5 * math.log(den) - (K1 + 0.5 * K3) * vt
                else:
                    K0s = K0
            else:
                pp = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - pp) / m
                u = 0.5 * (1.0 + math.erf(Z1[t, p] * inv_sqrt2))
                if u <= pp:
                    vn = 0.0
                else:
                    vn = math.log((1.0 - pp) / (1.0 - u)) / beta
                if beta > A:
                    K0s = -math.log(pp + beta * (1.0 - pp) / (beta - A)) - (K1 + 0.5 * K3) * vt
                else:
                    K0s = K0

            if vn <= 0.0:
                n_zero += 1

            x = mu + K0s + K1 * vt + K2 * vn + math.sqrt(K3 * vt + K4 * vn) * Z2[t, p]
            S_paths[t + 1, p] = S_paths[t, p] * math.exp(x)
            v[p] = vn

    return S_paths, n_zero / (N_steps * n_paths)


#---------------------Heston Pricing Engine (Including Sensitivities)---------------------
class HestonAutocallPricer:
    def __init__(self, S0, S_ref, r_curve_func, q_curve_func, params, scheme='qe'):
        """
        Heston Autocall Pricer.

        Args:
            S0: Current spot price (live)
            S_ref: initial price (reference level)
            r_curve_func: Function accepting T (years) returning zero rate
            q_curve_func: Function accepting T (years) returning dividend yield
            params: [kappa, theta, sigma, rho, v0]
            scheme: 'qe' (Andersen quadratic-exponential, default) or 'euler' (full truncation)

        Attributes:
            S0: Current spot price (live)
            S_ref: initial price (reference level)
            r_curve_func: Function accepting T (years) returning Zero Rate
            kappa, theta, sigma, rho, v0: Heston parameters
            time_grid: Time grid for simulation; dimensions (N_time_steps + 1)
            S_paths: Simulated stock paths; dimensions (N_time_steps + 1, N_paths)
            is_simulated: Flag indicating if simulation has been run
            expected_life: Expected life of the note (in years)
            zero_variance_frac: share of simulated (day, path) cells with zero variance
            std_error: Monte Carlo standard error of the last price_structure call
            outcomes: dict of path statistics from the last price_structure call
        """
        self.S0 = S0
        self.S_ref = S_ref
        self.r_curve_func = r_curve_func
        self.q_curve_func = q_curve_func
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = params
        self.scheme = scheme

        # State variables to hold simulation data
        self.time_grid = None
        self.S_paths = None
        self.is_simulated = False
        self.expected_life = 0
        self.zero_variance_frac = np.nan
        self.std_error = np.nan
        self.outcomes = {}

        # Sensitivities: delta, gamma, and sensitivities to v0/sigma/rho
        self.sensitivities = {}

    def _build_time_grid(self, T_max, steps_per_year, obs_times):
        """Uniform 'daily' grid, with the observation dates inserted exactly."""
        N_steps = int(np.ceil(T_max * steps_per_year))
        grid = np.linspace(0, T_max, N_steps + 1)
        if obs_times is not None:
            extra = []
            for t in np.atleast_1d(obs_times):
                if 0 < t <= T_max and np.min(np.abs(grid - t)) > 1e-6:
                    extra.append(t)
            if extra:
                grid = np.sort(np.concatenate([grid, np.array(extra)]))
        return grid

    def _log_forward(self, t):
        """log F(t)/S0 = (r(t) - q(t)) t from the zero curves."""
        t = np.asarray(t, dtype=float)
        r = np.asarray(self.r_curve_func(np.maximum(t, 1e-8)), dtype=float)
        q = np.asarray(self.q_curve_func(np.maximum(t, 1e-8)), dtype=float)
        return (r - q) * t

    def run_simulation(self, T_max, steps_per_year=252, n_paths=50000, force_unit_start=False,
                       obs_times=None, seed=None, scheme=None, verbose=True):
        """
        Simulates Heston paths and stores time_grid and S_paths internally.

        The drift of each step is the increment of the log forward, log F(t_{k+1}) - log F(t_k),
        read off the zero-rate and dividend curves, so the simulated forward matches the curve
        (and the calibration) at every date instead of using one flat r - q.

        Args:
            T_max: Time to maturity in years
            steps_per_year: Number of time steps per year
            n_paths: Number of paths to simulate
            force_unit_start: start paths at 1.0 (for Greeks by rescaling)
            obs_times: observation times (years) inserted exactly into the grid
            seed: optional numpy seed for the normals
            scheme: 'qe' or 'euler'; defaults to self.scheme
            verbose: print a summary line

        Returns:
            None
        """
        scheme = self.scheme if scheme is None else scheme

        # Setup grid
        self.time_grid = self._build_time_grid(T_max, steps_per_year, obs_times)
        dts = np.diff(self.time_grid)
        N_steps = len(dts)

        # Term-structure drift: log-forward increments
        drifts = np.diff(self._log_forward(self.time_grid))

        # Independent normals (correlation is applied inside the engines)
        if seed is not None:
            np.random.seed(seed)
        Z1 = np.random.normal(size=(N_steps, n_paths)) # Variance noise
        Z2 = np.random.normal(size=(N_steps, n_paths)) # Stock noise (uncorrelated)

        # If forcing unit start, we simulate starting at 1.0 regardless of self.S0
        start_val = 1.0 if force_unit_start else self.S0

        if scheme == 'euler':
            engine = heston_engine_euler_numba
        elif scheme == 'qe':
            engine = heston_engine_qe_numba
        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Use 'qe' or 'euler'.")

        self.S_paths, self.zero_variance_frac = engine(
            start_val, self.v0,
            self.kappa, self.theta, self.sigma, self.rho,
            dts, drifts,
            Z1, Z2
        )

        self.is_simulated = True
        if verbose:
            print(f"Simulation complete ({scheme}): {n_paths} paths over {T_max:.2f} years ({N_steps} steps), "
                  f"zero-variance share {self.zero_variance_frac:.1%}.")


    def _get_time_index(self, T_target):
        """
        Finds the closest simulation index for a requested time T.

        Args:
            T_target: Time in years

        Returns:
            idx: Closest simulation index
        """
        if not self.is_simulated:
            raise RuntimeError("Run simulation first!")
        if self.S_paths is not None and self.S_paths.shape[0] != len(self.time_grid):
            raise RuntimeError("time_grid and S_paths are out of sync: paths were swapped in without their grid.")
        idx = int(np.argmin(np.abs(self.time_grid - T_target)))
        if T_target > self.time_grid[-1] + 1e-9:
            print(f"Observation date {T_target:.2f}y is beyond simulation. Using index {idx} (Time {self.time_grid[-1]:.2f}y).")
        return idx


    @staticmethod
    def schedule_to_times(full_schedule, pricing_date):
        """Observation times in years (ACT/365) for the dates after pricing_date."""
        times = []
        for date_str in full_schedule:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            if d > pricing_date:
                times.append((d - pricing_date).days / 365)
        return np.array(times)


    def price_structure(self, full_schedule,
                        coupon_step=0.0452,
                        barrier_redemption=0.70,
                        barrier_knock_in=0.60,
                        exit_bonus=0.3616,
                        nominal=1000,
                        payment_lag_days=0,
                        pricing_date=None,
                        verbose=True):
        """
        Live Pricer with flexible product parameters. All rates in decimal!
        Decomposes the Fair Value into:
        1. Principal (Bond)
        2. Coupons (Digital Calls)
        3. Risk (Short Put)

        WARNING:
        This method contains hardcoded logic specific to the ADAPTED BNP Athena Note.
        The period index calculation (i + 2) assumes 'full_schedule' begins at the
        Year 1 observation. Do not use this method for other products without
        modifying the 'future_period_indices' logic.

        Args:
            full_schedule: list of all valuation dates "YYYY-MM-DD"
            coupon_step: coupon rate per observation period
            barrier_redemption: redemption barrier (final valuation)
            barrier_knock_in: knock-in barrier (final valuation)
            exit_bonus: bonus rate if redemption condition met at maturity
            nominal: nominal value of the note
            payment_lag_days: calendar days between a valuation date and its payment
                              (the BNP term sheet pays 7 days after each valuation date)
            pricing_date: if not passed, use today by default; otherwise use a future date (for sensitivity analysis)
            verbose: print the decomposition
        Returns:
            a tuple of four elements:
            total_price: Total price of the note
            avg_principal: Average principal component
            avg_coupons: Average coupon component
            avg_risk: Average risk component
        """

        if not self.is_simulated:
            raise RuntimeError("Please run .run_simulation() before pricing.")
        # Note: Frist autocall observation date is one year from the pricing date

        # Filter future dates
        if pricing_date is not None:
            today = pricing_date
        else:
            today = datetime.now()
        future_times = []
        future_period_indices = []

        for i, date_str in enumerate(full_schedule):
            d = datetime.strptime(date_str, "%Y-%m-%d")
            if d > today:
                T_rem = (d - today).days / 365
                future_times.append(T_rem)
                # i=0 is pricing_date. If it's TODAY,
                # First Exit Rate is 2 * coupon_step in accordance with Page 19.
                future_period_indices.append(i + 2)

        if not future_times:
            return 0

        lag = payment_lag_days / 365

        # Setup barriers (based on S_ref)
        B_autocall = 1 * self.S_ref
        B_redemption = barrier_redemption * self.S_ref
        B_knock_in = barrier_knock_in * self.S_ref

        n_paths = self.S_paths.shape[1]
        active_mask = np.ones(n_paths, dtype=bool)

        # Setup termination tracking: initialize all paths to terminate at maturity
        # If they autocall early, update this array
        final_maturity_T = future_times[-1]
        termination_times = np.full(n_paths, final_maturity_T)

        # Setup path pv's for three components; add up to note pv
        pv_principal = np.zeros(n_paths)
        pv_coupons   = np.zeros(n_paths)
        pv_risk      = np.zeros(n_paths)

        call_probs = []

        # Loop through all future observation dates except maturity
        for t_obs, n_period in zip(future_times[:-1], future_period_indices[:-1]):

            idx = self._get_time_index(t_obs)
            S_current = self.S_paths[idx, :]

            # Check autocall
            did_autocall = active_mask & (S_current >= B_autocall)

            # Record the current time if autocalled
            termination_times[did_autocall] = t_obs

            # Discount to the payment date
            t_pay = t_obs + lag
            r_t = self.r_curve_func(t_pay)
            df = np.exp(-r_t * t_pay)

            # Payoff split: principal, nominal; coupons, nominal * coupon; add up to nominal * (1 + coupon)
            pv_principal[did_autocall] = nominal * df
            pv_coupons[did_autocall]   = (nominal * n_period * coupon_step) * df

            active_mask[did_autocall] = False
            call_probs.append(did_autocall.mean())

        # Observation at maturity
        idx_final = self._get_time_index(future_times[-1])
        S_final = self.S_paths[idx_final, :]
        T_final = future_times[-1]

        t_pay_final = T_final + lag
        r_mat = self.r_curve_func(t_pay_final)
        df_final = np.exp(-r_mat * t_pay_final)

        survivors = active_mask

        # Scenario A: redemption condition met
        # Payoff is 100% nominal (principal) + Exit coupon
        mask_A = survivors & (S_final >= B_redemption)
        pv_principal[mask_A] = nominal * df_final
        pv_coupons[mask_A]   = (nominal * exit_bonus) * df_final

        # Scenario B: safe zone
        # Payoff is 100% nominal only
        mask_B = survivors & (S_final >= B_knock_in) & (S_final < B_redemption)
        pv_principal[mask_B] = nominal * df_final

        # Scenario C: knock-in loss
        # Payoff is the underlying price, which can be decomposed into:
        # 100% nominal + nominal * (S_final/S_ref - 100%); since S < S_ref, this term is negative
        mask_C = survivors & (S_final < B_knock_in)
        performance = S_final / self.S_ref
        pv_principal[mask_C] = nominal * df_final
        pv_risk[mask_C]      = nominal * (performance[mask_C] - 1) * df_final

        # Calculate expected life
        self.expected_life = np.mean(termination_times)

        # Aggregate
        avg_principal = np.mean(pv_principal)
        avg_coupons   = np.mean(pv_coupons)
        avg_risk      = np.mean(pv_risk)

        total_price = avg_principal + avg_coupons + avg_risk

        # Monte Carlo diagnostics
        pv_total = pv_principal + pv_coupons + pv_risk
        self.last_pv_paths = pv_total          # per-path PV, for control variates and convergence studies
        self.std_error = float(np.std(pv_total, ddof=1) / np.sqrt(n_paths))
        self.outcomes = {
            'call_prob_by_date': np.array(call_probs),
            'prob_reach_maturity': float(survivors.mean()),
            'prob_bonus_at_maturity': float(mask_A.mean()),
            'prob_safe_zone': float(mask_B.mean()),
            'prob_knock_in': float(mask_C.mean()),
            'mean_perf_given_ki': float(performance[mask_C].mean()) if mask_C.any() else np.nan,
            'expected_life': float(self.expected_life),
            'std_error': self.std_error,
        }

        if verbose:
            print(f"Autocallable note price: {total_price:.2f} (MC s.e. {self.std_error:.2f}), \
            \nPrincipal component: {avg_principal:.2f}, \
            \nCoupon component: {avg_coupons:.2f}, \
            \nPut risk component (short down-and-in put): {avg_risk:.2f}")

        return (total_price, avg_principal, avg_coupons, avg_risk)

    def solve_fair_coupon(self, product_params, target_price=1000.0, bonus_periods=8,
                          pricing_date=None, lo=0.0, hi=0.20):
        """
        Solves for the coupon per period that makes the note worth target_price, on the
        SAME simulated paths (common random numbers), with the exit bonus kept at
        coupon x bonus_periods as in the term sheet.

        Args:
            product_params: dict as passed to price_structure (coupon_step/exit_bonus are overridden)
            target_price: e.g. 1000 for par, 987.5 for a 1.25% margin
            bonus_periods: number of coupon periods in the final bonus
        Returns:
            fair coupon per period (decimal)
        """
        base = {k: v for k, v in product_params.items() if k not in ('coupon_step', 'exit_bonus')}

        def gap(c):
            px = self.price_structure(coupon_step=c, exit_bonus=c * bonus_periods,
                                      pricing_date=pricing_date, verbose=False, **base)[0]
            return px - target_price

        return brentq(gap, lo, hi, xtol=1e-7)

    def mc_vanilla_calls(self, T, strikes):
        """
        Prices European calls from the stored paths at the grid time closest to T.
        Used to check the simulation scheme against the FFT pricer.

        Returns:
            (prices, standard errors)
        """
        idx = self._get_time_index(T)
        S_T = self.S_paths[idx, :]
        t = self.time_grid[idx]
        df = np.exp(-self.r_curve_func(t) * t)
        strikes = np.atleast_1d(strikes)
        payoffs = np.maximum(S_T[None, :] - strikes[:, None], 0.0) * df
        return payoffs.mean(axis=1), payoffs.std(axis=1, ddof=1) / np.sqrt(S_T.shape[0])

    def martingale_check(self, times):
        """
        E[S_t] / F(t) - 1 at the requested times, with standard errors.
        Should be zero within noise for a correctly drifted risk-neutral simulation.
        """
        rows = []
        for T in np.atleast_1d(times):
            idx = self._get_time_index(T)
            t = self.time_grid[idx]
            F = self.S_paths[0, 0] * np.exp(self._log_forward(t))
            ratio = self.S_paths[idx, :] / F
            rows.append({'T': t, 'E[S_t]/F(t) - 1': ratio.mean() - 1,
                         's.e.': ratio.std(ddof=1) / np.sqrt(ratio.shape[0])})
        return pd.DataFrame(rows)

    def generate_greek_paths(self, T_max=4, n_paths=10000, eps_vol=2e-3, obs_times=None):
        """
        Explicitly generates unit paths for sensitivity analysis.
        Call this once before scanning.
        """
        path_set = {}

        def gen_run(key, p_variant):
            np.random.seed(42)
            p_variant.run_simulation(T_max, n_paths=n_paths, force_unit_start=True, obs_times=obs_times)
            path_set[key] = p_variant.S_paths
            path_set['_time_grid'] = p_variant.time_grid   # the grid these paths live on (same for every key)

        # 1. Base (on a copy, so the caller's pricing paths are left untouched)
        gen_run('Base', copy.copy(self))

        # 2. Vega
        p = copy.copy(self); p.v0 += eps_vol; gen_run('Vega_Up', p)
        p.v0 -= 2*eps_vol; gen_run('Vega_Dn', p)

        # 3. Volga
        p = copy.copy(self); p.sigma += eps_vol; gen_run('Volga_Up', p)
        p.sigma -= 2*eps_vol; gen_run('Volga_Dn', p)

        # 4. Rho
        p = copy.copy(self); p.rho += eps_vol; gen_run('Rho_Up', p)
        p.rho -= 2*eps_vol; gen_run('Rho_Dn', p)

        return path_set

    def compute_sensitivities(self, product_params, path_set=None, eps_spot=0.01, eps_vol=2e-3, pricing_date=None):
        """
        Calculates Delta, Gamma, Vega, Volga, and Rho.
        Stores the result in self.sensitivities.

        Args:
            path_set: Dictionary of pre-computed unit paths. If None, run slow re-simulations.
        """
        p = copy.copy(self)
        orig_S0 = p.S0

        def get_pv(curr_pricer, path_key=None):
            # Use pre-computed paths (and their own time grid)
            if path_set and path_key in path_set:
                curr_pricer.S_paths = path_set[path_key] * curr_pricer.S0
                if '_time_grid' in path_set:
                    curr_pricer.time_grid = path_set['_time_grid']
                curr_pricer.is_simulated = True

            # Re-simulate
            else:
                np.random.seed(42)
                if curr_pricer.time_grid is not None:
                    T_max = curr_pricer.time_grid[-1]
                else:
                    T_max = 4
                curr_pricer.run_simulation(T_max, n_paths=20000, force_unit_start=False)

            # Calculate and return array (Total, Principal, Coupon, Put)
            return np.array(curr_pricer.price_structure(**product_params, pricing_date=pricing_date))

        # Delta, Gamma (perturb S0)
        vals_base = get_pv(p, 'Base')

        p.S0 = orig_S0 * (1 + eps_spot)
        vals_up = get_pv(p, 'Base')
        p.S0 = orig_S0 * (1 - eps_spot)
        vals_dn = get_pv(p, 'Base')
        p.S0 = orig_S0 # Reset

        deltas = (vals_up - vals_dn) / (2 * eps_spot * orig_S0)
        gammas = (vals_up - 2*vals_base + vals_dn) / ((eps_spot * orig_S0)**2)
        gammas_cash = gammas * (orig_S0**2) / 100

        # Vega (perturb v0)
        if path_set:
            vals_v_up = get_pv(p, 'Vega_Up')
            vals_v_dn = get_pv(p, 'Vega_Dn')
        else:
            orig_v0 = p.v0
            p.v0 = orig_v0 + eps_vol
            vals_v_up = get_pv(p)
            p.v0 = orig_v0 - eps_vol
            vals_v_dn = get_pv(p)
            p.v0 = orig_v0 # Reset

        vegas = (vals_v_up - vals_v_dn) / (2 * eps_vol)

        # Volga (perturb sigma)
        if path_set:
            vals_s_up = get_pv(p, 'Volga_Up')
            vals_s_dn = get_pv(p, 'Volga_Dn')
        else:
            orig_sig = p.sigma
            p.sigma = orig_sig + eps_vol
            vals_s_up = get_pv(p)
            p.sigma = orig_sig - eps_vol
            vals_s_dn = get_pv(p)
            p.sigma = orig_sig

        volgas = (vals_s_up - vals_s_dn) / (2 * eps_vol)

        # Rho (perturb rho)
        if path_set:
            vals_r_up = get_pv(p, 'Rho_Up')
            vals_r_dn = get_pv(p, 'Rho_Dn')
        else:
            orig_rho = p.rho
            p.rho = orig_rho + eps_vol
            vals_r_up = get_pv(p)
            p.rho = orig_rho - eps_vol
            vals_r_dn = get_pv(p)
            p.rho = orig_rho

        rhos = (vals_r_up - vals_r_dn) / (2 * eps_vol)

        # Package Results
        components = ['Total', 'Principal', 'Coupon', 'Put']
        self.sensitivities = {}

        for i, name in enumerate(components):
            self.sensitivities[name] = {
                'Delta': deltas[i], 'Gamma': gammas_cash[i],
                'Vega': vegas[i], 'Volga': volgas[i], 'Rho': rhos[i]
            }

        return self.sensitivities



#---------------------Helper Function to Mute Printouts---------------------
@contextmanager
def suppress_stdout():
    """A context manager that redirects stdout to devnull."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
