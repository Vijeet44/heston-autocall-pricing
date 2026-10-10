import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import copy
import os
import sys
from contextlib import contextmanager

from scipy.fft import ifft
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution, OptimizeResult

from numba import njit


#---------------------Bootstrapping & Curves---------------------
class OISBootstrapper:
    def __init__(self):
        # Bloomberg SOFR OIS par swap rates (I forgot but the command might be USOSFRA BGN Curncy <GO>)
        self.market_data = [
            (1/52, 4.0433), (2/52, 4.0353), (3/52, 3.9704),
            (1/12, 3.9175), (2/12, 3.8650), (3/12, 3.8182),
            (4/12, 3.7930), (5/12, 3.7605), (6/12, 3.7283),
            (9/12, 3.6226), (1,  3.5245),
            (1.5,  3.3602), (2,  3.2860), (3,  3.2410),
            (4,  3.2557), (5,  3.2940), (7,  3.4145),
            (10, 3.6005), (20, 3.9565), (30, 3.9435),
        ]
        self.market_data.sort(key=lambda x: x[0]) # Ensure sorted by T
    

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
                # Long tenor
                # First calculate value of the fixed coupons before maturity
                prev_coupons_val = 0
                payment_dates = np.arange(1, int(T) + 0.01)
                payment_dates = payment_dates[payment_dates < T - 0.01]
                for t_pay in payment_dates:
                    prev_coupons_val += R_par * get_df(t_pay)
                # Boorstap from discounted principal_fixed coupon value at maturity
                df_T = (1 - prev_coupons_val) / (1 + R_par)
                r_zero = -np.log(df_T) / T

            known_ts.append(T)
            known_rs.append(r_zero)

        return interp1d(known_ts, known_rs, kind='linear', fill_value="extrapolate")

# Instantiate global curve for module-level access
_bootstrapper = OISBootstrapper()
sofr_zero_curve = _bootstrapper.bootstrap()


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



#---------------------Heston FFT Calibrator---------------------
class HestonFFTCalibrator:
    def __init__(self, market_data, S0, r_curve_func):
        """
        Lewis (2001) FFT Calibrator with Hybrid Optimization (Global + Local).

        Args:
            market_data: DF with ['strike', 'T', 'market_price_call_equivalent']
            S0: Spot price
            r_curve_func: DF function
        """
        self.market_data = market_data.copy()
        self.S0 = S0
        self.r_curve = r_curve_func  # Store the function


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
            self.data_batches.append(batch)

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

    def price_surface(self, params):
        """
        FFT Discretized Formula. Calculates Heston model prices for each strike and maturity.

        Args:
            params: [kappa, theta, sigma, rho, v0], calibrated parameters

        Returns:
            Model prices for each strike and maturity, flattened; dimensions: [n_strikes * n_maturities]
        """
        model_prices = []

        for batch in self.data_batches:
            T = batch['T']
            r_t = self.r_curve(T)
            q_t = batch['q']

            # Characteristic Function
            cf_vals = self._heston_char_func(self.u - 0.5j, T, r_t, q_t, params)

            # Input Array
            input_array = np.exp(-1j * self.b * self.u) * cf_vals * self.weights / (self.u**2 + 0.25)

            # Inverse FFT
            integral_val = np.real(ifft(input_array) * self.N)

            # Interpolation
            spline = interp1d(self.k_fft, integral_val, kind='cubic', fill_value="extrapolate")
            integral_interp = spline(batch['k_market'])

            # Lewis Formula
            discount = np.exp(-r_t * T)
            div_factor = np.exp(-q_t * T)

            prices = (self.S0 * div_factor) - (np.sqrt(self.S0 * batch['strikes']) * discount / np.pi) * integral_interp
            model_prices.extend(prices)

        return np.array(model_prices)

    def objective_function(self, params):
        """
        RMSE objective.

        Args:
            params: [kappa, theta, sigma, rho, v0]

        Returns:
            RMSE
        """
        try:
            model_vals = self.price_surface(params)
        except:
            return 1e10 # Catch numerical overflows

        market_vals = np.concatenate([b['market_prices'] for b in self.data_batches])
        #weights = 1 / (market_vals + 1) # Add a small epsilon to avoid division by zero
        mse = np.mean((model_vals - market_vals)**2)
        #mse = np.mean(((model_vals - market_vals) * weights) ** 2)
        return np.sqrt(mse)

    def calibrate(self, verbose=True, fixed_params=None):
        """
        Global optimization with local refinement:
        1. Global optimizer (Differential Evolution) explores; may not find global optimum if popsize is small.
        2. Local optimizer (L-BFGS-B) refines using global results as starting point.

        Args:
            verbose: Print progress.
            fixed_params: [kappa, theta, sigma, rho, v0], hardcoded for debugging.

        Returns:
            Calibrated parameters [kappa, theta, sigma, rho, v0]; returns hardcoded parameters if provided for debugging.
        """

        if fixed_params is not None:
            if verbose:
                print(f"Skipping optimization. Using fixed parameters: {fixed_params}")

            # Create a fake result object that looks exactly like scipy's output
            fixed_array = np.array(fixed_params)

            # Optional: Calculate the RMSE for these parameters so you see the error metric
            rmse = self.objective_function(fixed_array)

            return OptimizeResult(x=fixed_array, fun=rmse, message="Fixed Parameters Used")

        if verbose: print("Starting global optimization (Differential Evolution).")

        # Global Optimization
        result_global = differential_evolution(
            self.objective_function,
            bounds=self.bounds,
            strategy='best1bin',
            maxiter=50,      # Cap iterations to save time
            popsize=15,      # Population size
            tol=1e-4,
            workers=-1,       # Parallelize
            seed=42
        )

        if verbose:
            print(f"Global Best: {result_global.x}, RMSE: {result_global.fun:.4f}")
            print("Starting Local Optimization (L-BFGS-B).")

        # Local Refinement: use the global result as the starting point (x0)
        result_local = minimize(
            self.objective_function,
            x0=result_global.x,
            method='L-BFGS-B',
            bounds=self.bounds,
            tol=1e-6
        )

        if verbose:
            print(f"Local Best: {result_local.x}, RMSE: {result_local.fun:.4f}")

        return result_local
    



#---------------------Heston Paths Simulation---------------------
# Define static numba function outside of the class to speed up simulation
@njit(fastmath=True)
def heston_engine_numba(S0, v0, kappa, theta, sigma, rho,
                        dt, sqrt_dt, Z1, Z2, r_drift_scalar):
    """
    Standalone Heston simulation engine compiled by Numba.
    Passes raw arrays/floats only (no 'self').

    Args:
    See HestonAutocallPricer class

    Returns:
    S_paths: Simulated stock paths; dimensions (N_steps + 1, n_paths)
    v_paths: Simulated variance paths; dimensions (N_steps + 1, n_paths)
    """
    N_steps, n_paths = Z1.shape

    # Pre-allocate
    S_paths = np.zeros((N_steps + 1, n_paths))
    v = np.zeros((N_steps + 1, n_paths))

    S_paths[0, :] = S0
    v[0, :] = v0

    # Simulation Loop (C-Speed)
    for t in range(N_steps):
        v_curr = v[t, :]
        v_pos = np.maximum(0.0, v_curr)

        # Variance Step (Full Truncation)
        v[t+1, :] = v_curr + kappa * (theta - v_pos) * dt + \
                    sigma * np.sqrt(v_pos) * sqrt_dt * Z1[t, :]

        # Stock Step
        # Pre-calculating the correlated noise term
        W_S = rho * Z1[t, :] + np.sqrt(1 - rho**2) * Z2[t, :]

        S_paths[t+1, :] = S_paths[t, :] * np.exp(
            (r_drift_scalar - 0.5 * v_pos) * dt +
            np.sqrt(v_pos) * sqrt_dt * W_S
        )

    return S_paths, v # Return both if you need volatility paths later


#---------------------Heston Pricing Engine (Including Sensitivities)---------------------
class HestonAutocallPricer:
    def __init__(self, S0, S_ref, r_curve_func, q_curve_func, params):
        """
        Heston Autocall Pricer.

        Args:
            S0: Current spot price (live)
            S_ref: initial price (reference level)
            r_curve_func: Function accepting T (years) returning zero rate
            q_curve_func: Function accepting T (years) returning dividend yield
            params: [kappa, theta, sigma, rho, v0]

        Attributes:
            S0: Current spot price (live)
            S_ref: initial price (reference level)
            r_curve_func: Function accepting T (years) returning Zero Rate
            kappa, theta, sigma, rho, v0: Heston parameters
            time_grid: Time grid for simulation; dimensions (N_time_steps + 1)
            S_paths: Simulated stock paths; dimensions (N_time_steps + 1, N_paths)
            is_simulated: Flag indicating if simulation has been run
            expected_life: Expected life of the note (in years)
        """
        self.S0 = S0
        self.S_ref = S_ref
        self.r_curve_func = r_curve_func
        self.q_curve_func = q_curve_func
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = params

        # State variables to hold simulation data
        self.time_grid = None
        self.S_paths = None
        self.is_simulated = False
        self.expected_life = 0

        # Sensitivities: delta, gamma, and sensitivities to v0/sigma/rho
        self.sensitivities = {}


    def run_simulation(self, T_max, steps_per_year=252, n_paths=50000, force_unit_start=False):
        """
        Runs the Heston Euler-Maruyama simulation with Full Truncation.
        Stores the time_grid and paths internally.

        Args:
            T_max: Time to maturity in years
            steps_per_year: Number of time steps per year
            n_paths: Number of paths to simulate

        Returns:
            None
        """
        # Get flat rate for the simulation horizon drift
        r_val = self.r_curve_func(T_max)
        q_val = self.q_curve_func(T_max)
        r_drift = r_val - q_val

        # Setup grid: calculate the "daily" steps needed to fit into the remaining time
        N_steps = int(np.ceil(T_max * steps_per_year))
        self.time_grid = np.linspace(0, T_max, N_steps + 1)
        dt = T_max / N_steps
        sqrt_dt = np.sqrt(dt)


        # Correlated Brownian motions
        Z1 = np.random.normal(size=(N_steps, n_paths)) # Variance noise
        Z2 = np.random.normal(size=(N_steps, n_paths)) # Stock noise (uncorrelated)
        W_v = Z1
        W_S = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        # Initialize arrays
        self.S_paths = np.zeros((N_steps + 1, n_paths))
        v = np.zeros((N_steps + 1, n_paths))

        # If forcing unit start, we simulate starting at 1.0 regardless of self.S0
        if force_unit_start:
            start_val = 1.0
        else:
            start_val = self.S0

        self.S_paths[0] = start_val
        v[0] = self.v0

        # Time stepping (Euler-Maruyama)
        # for t in range(N_steps):
        #     v_curr = v[t]
        #     # Full truncation: Use max(0, v) for drift and diffusion calculations
        #     v_pos = np.maximum(0, v_curr)

        #     # Variance step
        #     # Note: Add the step to the un-truncated v_curr, allowing it to go negative.
        #     # Formula: v_{t+1} = v_t + kappa*(theta - v^+)*dt + sigma*sqrt(v^+)*dW_v
        #     v[t+1] = v_curr + self.kappa * (self.theta - v_pos) * dt + \
        #              self.sigma * np.sqrt(v_pos) * sqrt_dt * W_v[t]

        #     # Stock step (Log-Euler)
        #     # Uses v_pos for both the drift correction and the diffusion; drift uses r_drift=r-q (risk-neutral)
        #     self.S_paths[t+1] = self.S_paths[t] * np.exp(
        #         (r_drift - 0.5 * v_pos) * dt +
        #         np.sqrt(v_pos) * sqrt_dt * W_S[t]
        #     )

        # Time stepping (Euler-Maruyama)
        # Unwrap 'self' variables and pass them as simple arguments
        self.S_paths, _ = heston_engine_numba(
            start_val, self.v0,
            self.kappa, self.theta, self.sigma, self.rho,
            dt, sqrt_dt,
            Z1, Z2,
            r_drift
        )

        self.is_simulated = True
        print(f"Simulation complete: {n_paths} paths over {T_max:.2f} years ({N_steps} steps).")


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
        idx = np.searchsorted(self.time_grid, T_target)
        if idx >= len(self.time_grid):
            idx = len(self.time_grid) - 1
            print(f"Observation date {T_target:.2f}y is beyond simulation. Using index {idx} (Time {self.time_grid[-1]:.2f}y).")
        return idx


    def price_structure(self, full_schedule,
                        coupon_step=0.0452,
                        barrier_redemption=0.70,
                        barrier_knock_in=0.60,
                        exit_bonus=0.3616,
                        nominal=1000,
                        pricing_date=None):
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
            pricing_date: if not passed, use today by default; otherwise use a future date (for sensitivity analysis)
        Returns:
            a tuple of four elements:
            total_price: Total price of the note
            avg_principal: Average principal component
            avg_coupons: Average coupon component
            avg_risk: Average risk component
        """

        if not self.is_simulated:
            raise RuntimeError("Please run .run_simulation() before pricing.")
        # Note: Frist autocall observation date is one year from TODAY

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

        # Loop through all future observation dates except maturity
        for t_obs, n_period in zip(future_times[:-1], future_period_indices[:-1]):

            idx = self._get_time_index(t_obs)
            S_current = self.S_paths[idx, :]

            # Check autocall
            did_autocall = active_mask & (S_current >= B_autocall)

            # Record the current time if autocalled
            termination_times[did_autocall] = t_obs

            # Discount
            r_t = self.r_curve_func(t_obs)
            df = np.exp(-r_t * t_obs)

            # Payoff split: principal, nominal; coupons, nominal * coupon; add up to nominal * (1 + coupon)
            pv_principal[did_autocall] = nominal * df
            pv_coupons[did_autocall]   = (nominal * n_period * coupon_step) * df

            active_mask[did_autocall] = False

        # Observation at maturity
        idx_final = self._get_time_index(future_times[-1])
        S_final = self.S_paths[idx_final, :]
        T_final = future_times[-1]

        r_mat = self.r_curve_func(T_final)
        df_final = np.exp(-r_mat * T_final)

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
        print(f"Autocallable note price: {total_price:.2f}, \
        \nPrincipal component: {avg_principal:.2f}, \
        \nCoupon component: {avg_coupons:.2f}, \
        \nPut risk component (short down-and-in put): {avg_risk:.2f}")

        return (total_price, avg_principal, avg_coupons, avg_risk)

    def generate_greek_paths(self, T_max=4, n_paths=10000, eps_vol=2e-3):
        """
        Explicitly generates unit paths for sensitivity analysis.
        Call this once before scanning.
        """
        path_set = {}

        def gen_run(key, p_variant):
            np.random.seed(42)
            p_variant.run_simulation(T_max, n_paths=n_paths, force_unit_start=True)
            path_set[key] = p_variant.S_paths

        # 1. Base
        gen_run('Base', self)

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
            # Use pre-computed paths
            if path_set and path_key in path_set:
                curr_pricer.S_paths = path_set[path_key] * curr_pricer.S0
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