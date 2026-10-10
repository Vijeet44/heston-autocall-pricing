"""
Alternative numerical methods for every stage of the pipeline, for methods_comparison.ipynb.

  Vanilla pricing engines   : Lewis FFT+spline (the calibrator's), Lewis Gauss-Legendre, COS (Fang-Oosterlee 2008),
                              Carr-Madan damped FFT (1999), Heston 1993 two-integral form, adaptive-quadrature reference
  Implied vol solvers       : bisection (heston_pricer), safeguarded Newton, Brent per option
  Calibration objectives    : price / relative price / vega-weighted price / vol uniform / vol spread-weighted / Huber
  Optimisers                : DE variants, dual annealing, basin hopping, SHGO, Nelder-Mead, multistart L-BFGS-B,
                              least_squares (Levenberg-Marquardt / trust-region on the residual vector)
  Monte Carlo               : antithetic variates, control variates, scrambled Sobol + Brownian bridge
  Local vol / LSV           : SSVI surface (Gatheral-Jacquier 2014), Dupire local vol, particle-method leverage
                              (Guyon, Henry-Labordere 2011), LV and LSV path engines
"""
import math
import time
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
from scipy.interpolate import interp1d
from numba import njit
from functools import lru_cache

from heston_pricer import bs_call_price, bs_vega, implied_vol_call_vec, HestonAutocallPricer
from vol_models import HestonModel, FFTCalibrator


# ====================================================================== 1. vanilla pricing engines
def lewis_integrand(u, k, cf_shifted):
    """Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) with k = log(S0/K)."""
    return np.real(np.exp(1j * u * k) * cf_shifted) / (u**2 + 0.25)


def price_lewis_reference(model, params, S0, K, T, r, q, epsabs=1e-10, limit=800):
    """Adaptive quadrature (QUADPACK) of the Lewis integral, one strike at a time. Slow; the reference."""
    K = np.atleast_1d(K).astype(float)
    out = np.empty_like(K)
    for j, Kj in enumerate(K):
        k = math.log(S0 / Kj)
        f = lambda u: float(np.real(np.exp(1j * u * k) * model.char_func(np.array([u - 0.5j]), T, r, q, params)[0]) / (u * u + 0.25))
        val, _ = integrate.quad(f, 0.0, np.inf, epsabs=epsabs, epsrel=1e-10, limit=limit)
        out[j] = S0 * math.exp(-q * T) - math.sqrt(S0 * Kj) * math.exp(-r * T) / math.pi * val
    return out


@lru_cache(maxsize=8)
def _leggauss(n):
    return np.polynomial.legendre.leggauss(n)


def price_lewis_gauss_legendre(model, params, S0, K, T, r, q, u_max=300.0, n=1500):
    """Lewis integral on [0, u_max] with Gauss-Legendre nodes, all strikes at once."""
    x, w = _leggauss(n)
    u = 0.5 * u_max * (x + 1); w = 0.5 * u_max * w
    cf = model.char_func(u - 0.5j, T, r, q, params)
    K = np.atleast_1d(K).astype(float)
    k = np.log(S0 / K)[:, None]
    integ = (np.real(np.exp(1j * u[None, :] * k) * cf[None, :]) / (u[None, :]**2 + 0.25)) @ w
    return S0 * np.exp(-q * T) - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * integ


def cumulants_numeric(model, params, T, r, q, h=1e-3):
    """c1, c2 of log(S_T/S0) from the characteristic function (central differences of log phi at 0)."""
    lp = np.log(model.char_func(np.array([h, -h], dtype=complex), T, r, q, params))
    c1 = float(np.imag(lp[0] - lp[1]) / (2 * h))
    c2 = float(-np.real(lp[0] + lp[1]) / h**2)
    return c1, max(c2, 1e-12)


def price_cos(model, params, S0, K, T, r, q, N=512, L=10.0):
    """COS method (Fang & Oosterlee 2008) for calls; one Fourier-cosine expansion serves all strikes."""
    K = np.atleast_1d(K).astype(float)
    x = np.log(S0 / K)                       # initial log-moneyness per strike
    c1, c2 = cumulants_numeric(model, params, T, r, q)
    a = x.min() + c1 - L * math.sqrt(c2)
    b = x.max() + c1 + L * math.sqrt(c2)
    kk = np.arange(N)
    u = kk * np.pi / (b - a)
    cf = model.char_func(u.astype(complex), T, r, q, params)
    # payoff coefficients for the call on y in [0, b]
    c, d = 0.0, b
    arg_d = kk * np.pi * (d - a) / (b - a); arg_c = kk * np.pi * (c - a) / (b - a)
    chi = (np.cos(arg_d) * math.exp(d) - np.cos(arg_c) * math.exp(c) + u * (np.sin(arg_d) * math.exp(d) - np.sin(arg_c) * math.exp(c))) / (1 + u**2)
    psi = np.empty(N); psi[0] = d - c
    psi[1:] = (np.sin(arg_d[1:]) - np.sin(arg_c[1:])) * (b - a) / (kk[1:] * np.pi)
    V = 2.0 / (b - a) * (chi - psi)          # times K below
    V[0] *= 0.5                                # the prime on the sum: half weight on k = 0
    terms = np.real(np.exp(1j * u[None, :] * (x[:, None] - a)) * cf[None, :]) * V[None, :]
    return K * np.exp(-r * T) * terms.sum(axis=1)


def price_carr_madan_fft(model, params, S0, K, T, r, q, alpha=1.5, N=4096, eta=0.25):
    """Carr-Madan (1999) damped call transform with Simpson weights and an FFT on a log-strike grid."""
    K = np.atleast_1d(K).astype(float)
    lam = 2 * np.pi / (N * eta); b = 0.5 * N * lam
    v = np.arange(N) * eta
    k_grid = -b + lam * np.arange(N)
    cf = model.char_func(v - (alpha + 1) * 1j, T, r, q, params) * np.exp(1j * (v - (alpha + 1) * 1j) * math.log(S0))
    psi = np.exp(-r * T) * cf / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    w = 3 + (-1) ** (np.arange(N) + 1); w[0] = 1; w = w / 3.0
    xin = np.exp(1j * b * v) * psi * eta * w
    C_grid = np.exp(-alpha * k_grid) / np.pi * np.real(np.fft.fft(xin))
    return interp1d(k_grid, C_grid, kind='cubic')(np.log(K))


def price_heston_two_integral(model, params, S0, K, T, r, q, u_max=200.0, n=4000):
    """Heston (1993) / Gil-Pelaez form C = S e^{-qT} P1 - K e^{-rT} P2 with 1/u integrands, Gauss-Legendre on [0, u_max]."""
    K = np.atleast_1d(K).astype(float)
    x = np.log(S0 / K)[:, None]
    xg, wg = _leggauss(n)
    u = 0.5 * u_max * (xg + 1) + 1e-10; w = 0.5 * u_max * wg
    phi_u = model.char_func(u.astype(complex), T, r, q, params)
    phi_umi = model.char_func(u - 1j, T, r, q, params)
    phi_mi = model.char_func(np.array([-1j + 1e-9]), T, r, q, params)[0]
    P2 = 0.5 + (np.real(np.exp(1j * u[None, :] * x) * phi_u[None, :] / (1j * u[None, :])) @ w) / np.pi
    P1 = 0.5 + (np.real(np.exp(1j * u[None, :] * x) * phi_umi[None, :] / (1j * u[None, :] * phi_mi)) @ w) / np.pi
    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


# ====================================================================== 2. implied vol solvers
def implied_vol_newton(C, S, K, T, r, q, n_iter=25, vol_lo=1e-4, vol_hi=5.0):
    """Vectorised Newton on vol with a bisection safeguard (falls back to the bracket midpoint when Newton leaves it)."""
    C = np.asarray(C, float)
    lo = np.full(C.shape, vol_lo); hi = np.full(C.shape, vol_hi)
    # Brenner-Subrahmanyam start, clipped
    sig = np.clip(np.sqrt(2 * np.pi / T) * C / S, 0.05, 1.5)
    for _ in range(n_iter):
        px = bs_call_price(S, K, T, r, q, sig)
        above = px > C
        hi = np.where(above, sig, hi); lo = np.where(above, lo, sig)
        vega = bs_vega(S, K, T, r, q, sig)
        step = np.where(vega > 1e-12, (px - C) / np.maximum(vega, 1e-12), 0.0)
        new = sig - step
        bad = (new <= lo) | (new >= hi) | ~np.isfinite(new)
        sig = np.where(bad, 0.5 * (lo + hi), new)
    return sig


def implied_vol_brent_loop(C, S, K, T, r, q, xtol=1e-12):
    """scipy brentq per option (the notebook's original scalar solver)."""
    out = np.full(len(C), np.nan)
    for i in range(len(C)):
        f = lambda s: bs_call_price(S, K[i], T[i], r[i], q[i], s) - C[i]
        try:
            out[i] = optimize.brentq(f, 1e-4, 5.0, xtol=xtol)
        except ValueError:
            out[i] = 1e-4 if f(1e-4) > 0 else 5.0
    return out


# ====================================================================== 3. calibration objectives
class ObjectiveCalibrator(FFTCalibrator):
    """
    FFTCalibrator with selectable objective:
      'price'      RMSE of call prices (index points)                      [v1]
      'relprice'   RMSE of (model - market) / market
      'vega'       RMSE of (model - market) / vega_mkt  (linearised vol error)
      'iv_uniform' RMSE of implied vols, every quote equal, one vote per expiry
      'iv_spread'  weighted RMSE of implied vols, weights 1/spread^2      [v2, the default elsewhere]
      'iv_huber'   Huber loss on spread-weighted vol residuals (delta = 1 vol pt), robust to outliers
    """

    def __init__(self, market_data, S0, r_curve_func, mode='iv_spread', huber_delta=0.01, **kw):
        wm = 'uniform' if mode == 'iv_uniform' else 'spread'
        super().__init__(market_data, S0, r_curve_func, model=HestonModel(), objective='iv', weight_mode=wm, **kw)
        self.mode = mode
        self.huber_delta = huber_delta
        self.vega_mkt = bs_vega(self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q, self.market_iv)
        self.uniform_w = np.ones(len(self.flat_K))
        for b in self.data_batches:                      # one vote per expiry
            m = self.flat_T == b['T']; self.uniform_w[m] /= m.sum()
        self.uniform_w /= self.uniform_w.sum()

    def objective_function(self, params):
        try:
            model_vals = self.price_surface(params)
        except Exception:
            return 1e10
        if not np.all(np.isfinite(model_vals)):
            return 1e10
        px_res = model_vals - self.flat_market
        if self.mode == 'price':
            return float(np.sqrt(np.mean(px_res**2)))
        if self.mode == 'relprice':
            return float(np.sqrt(np.mean((px_res / self.flat_market)**2)))
        if self.mode == 'vega':
            return float(np.sqrt(np.sum(self.uniform_w * (px_res / np.maximum(self.vega_mkt, 1.0))**2)))
        iv = implied_vol_call_vec(model_vals, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
        res = iv - self.market_iv
        if self.mode == 'iv_uniform':
            return float(np.sqrt(np.sum(self.uniform_w * res**2)))
        if self.mode == 'iv_spread':
            return float(np.sqrt(np.sum(self.quote_weights * res**2)))
        if self.mode == 'iv_huber':
            d = self.huber_delta; a = np.abs(res)
            hub = np.where(a <= d, 0.5 * res**2, d * (a - 0.5 * d))
            return float(np.sqrt(2 * np.sum(self.quote_weights * hub)))
        raise ValueError(self.mode)

    def residual_vector(self, params):
        """sqrt(w) * (iv_model - iv_mkt) for least-squares optimisers (spread weights)."""
        try:
            model_vals = self.price_surface(params)
            if not np.all(np.isfinite(model_vals)):
                raise ValueError
            iv = implied_vol_call_vec(model_vals, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
            return np.sqrt(self.quote_weights) * (iv - self.market_iv)
        except Exception:
            return np.full(len(self.flat_K), 1.0)


# ====================================================================== 4. optimisers
def heuristic_start(cal):
    """Cheap starting point: v0 from the shortest ATM vol, theta from the longest, kappa 2, sigma 1, rho -0.7."""
    Ts = sorted(b['T'] for b in cal.data_batches)
    def atm_iv(T):
        m = cal.flat_T == T; j = np.argmin(np.abs(cal.flat_K[m] - cal.S0)); return cal.market_iv[m][j]
    return np.array([2.0, atm_iv(Ts[-1])**2, 1.0, -0.7, atm_iv(Ts[0])**2])


def run_optimizer(cal, method, seed=42, **kw):
    """
    Returns dict(method, x, fun, nfev, seconds). fun is always the calibrator's own objective at x.
    Methods: 'de_lbfgs' (current), 'de', 'de_rand1', 'de_c2b', 'de_small_pop', 'dual_annealing', 'basinhopping', 'shgo',
             'nelder_mead', 'lbfgs_multistart', 'least_squares', 'least_squares_multistart'
    """
    f = cal.objective_function; bounds = cal.bounds; lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    t0 = time.perf_counter(); nfev = 0
    if method.startswith('de'):
        strat = {'de_rand1': 'rand1bin', 'de_c2b': 'currenttobest1bin'}.get(method, 'best1bin')
        pop = 5 if method == 'de_small_pop' else 15
        it = 150 if method == 'de_small_pop' else 50
        res = optimize.differential_evolution(f, bounds, strategy=strat, maxiter=it, popsize=pop, tol=1e-4, polish=False, seed=seed)
        x, nfev = res.x, res.nfev
        if method == 'de_lbfgs':
            r2 = optimize.minimize(f, x, method='L-BFGS-B', bounds=bounds, tol=1e-10); x = r2.x; nfev += r2.nfev
    elif method == 'dual_annealing':
        res = optimize.dual_annealing(f, bounds, maxiter=60, maxfun=12000, seed=seed, minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 40}})
        x, nfev = res.x, res.nfev
    elif method == 'basinhopping':
        x0 = heuristic_start(cal)
        res = optimize.basinhopping(f, x0, niter=12, T=0.002, stepsize=0.5, seed=seed,
                                    minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds})
        x, nfev = res.x, res.nfev
    elif method == 'shgo':
        res = optimize.shgo(f, bounds, n=32, iters=1, sampling_method='sobol', minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 40}})
        x, nfev = res.x, res.nfev
    elif method == 'nelder_mead':
        x0 = heuristic_start(cal)
        res = optimize.minimize(f, x0, method='Nelder-Mead', bounds=bounds, options={'maxfev': 4000, 'xatol': 1e-6, 'fatol': 1e-9, 'adaptive': True})
        x, nfev = res.x, res.nfev
    elif method == 'lbfgs_multistart':
        rng = np.random.default_rng(seed)
        starts = lo + stats.qmc.LatinHypercube(d=len(bounds), seed=seed).random(12) * (hi - lo)
        best = None
        for s in starts:
            r_ = optimize.minimize(f, s, method='L-BFGS-B', bounds=bounds, tol=1e-10); nfev += r_.nfev
            if best is None or r_.fun < best.fun: best = r_
        x = best.x
    elif method in ('least_squares', 'least_squares_multistart'):
        starts = [heuristic_start(cal)]
        if method == 'least_squares_multistart':
            starts += list(lo + stats.qmc.LatinHypercube(d=len(bounds), seed=seed).random(5) * (hi - lo))
        best = None
        for s in starts:
            r_ = optimize.least_squares(cal.residual_vector, np.clip(s, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method='trf', xtol=1e-10, ftol=1e-10, max_nfev=400)
            nfev += r_.nfev
            val = f(r_.x)
            if best is None or val < best[1]: best = (r_.x, val)
        x = best[0]
    else:
        raise ValueError(method)
    return {'method': method, 'seed': seed, 'x': np.asarray(x), 'fun': float(f(x)), 'nfev': int(nfev), 'seconds': time.perf_counter() - t0}


# ====================================================================== 5. Monte Carlo variance reduction
def brownian_bridge_paths(times, Z):
    """
    Brownian motion on the grid `times` (t_0 = 0 < ... < t_N) from standard normals Z of shape (N, n_paths),
    built by bisection so that Z[0] drives W(t_N), Z[1] the midpoint, etc. Returns W of shape (N + 1, n_paths).
    """
    N = len(times) - 1; n = Z.shape[1]
    W = np.zeros((N + 1, n))
    W[N] = math.sqrt(times[N]) * Z[0]
    zi = 1
    stack = [(0, N)]
    while stack:
        l, r = stack.pop(0)
        if r - l < 2:
            continue
        m = (l + r) // 2
        tl, tm, tr = times[l], times[m], times[r]
        mean = ((tr - tm) * W[l] + (tm - tl) * W[r]) / (tr - tl)
        sd = math.sqrt((tm - tl) * (tr - tm) / (tr - tl))
        W[m] = mean + sd * Z[zi]; zi += 1
        stack.append((l, m)); stack.append((m, r))
    return W


def sobol_normals(n_steps, n_paths, seed, bridge_times=None):
    """
    Scrambled Sobol normals for (variance, spot) with shape (2, n_steps, n_paths). With bridge_times, the first
    n_steps Sobol dimensions build the spot Brownian bridge and the next n_steps the variance bridge; the
    returned arrays are the increments divided by sqrt(dt), i.e. what the QE engine expects.
    """
    d = 2 * n_steps
    sob = stats.qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_paths)))
    U = sob.random_base2(m)[:n_paths]
    Z = stats.norm.ppf(np.clip(U, 1e-12, 1 - 1e-12)).T          # (d, n_paths)
    if bridge_times is None:
        return Z[n_steps:], Z[:n_steps]                          # (variance, spot) in natural order
    dt = np.diff(bridge_times)
    W_s = brownian_bridge_paths(bridge_times, Z[:n_steps]); W_v = brownian_bridge_paths(bridge_times, Z[n_steps:])
    return np.diff(W_v, axis=0) / np.sqrt(dt)[:, None], np.diff(W_s, axis=0) / np.sqrt(dt)[:, None]


def control_variate_estimate(Y, X, mu):
    """
    Regression control variates. Y: (n,) target PVs; X: (n, m) control PVs with known means mu (m,).
    Returns (estimate, standard error, beta).
    """
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Xc = X - X.mean(axis=0)
    beta = np.linalg.lstsq(Xc, Y - Y.mean(), rcond=None)[0]
    Yadj = Y - (X - mu) @ beta
    return float(Yadj.mean()), float(Yadj.std(ddof=1) / math.sqrt(len(Y))), beta


# ====================================================================== 6. SSVI surface, Dupire local vol
def ssvi_w(k, theta, rho, eta, gam):
    """SSVI total variance, power-law phi (Gatheral & Jacquier 2014)."""
    phi = eta / (theta**gam * (1 + theta)**(1 - gam))
    return 0.5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho)**2 + 1 - rho**2))


class SSVISurface:
    """
    Arbitrage-aware implied-vol surface: SSVI slices with a piecewise-linear ATM total-variance term structure
    theta(T) (theta(0) = 0). Fitted by weighted least squares in implied vol. Gives Dupire local vol on (k, T)
    with k = log(K / F_T) and total variance w.
    """

    def __init__(self, T_pillars):
        self.Tp = np.asarray(sorted(T_pillars), float)

    def theta(self, T):
        return np.interp(T, np.concatenate([[0.0], self.Tp]), np.concatenate([[0.0], self.thetas]))

    def w(self, k, T):
        th = self.theta(T)
        return ssvi_w(k, np.maximum(th, 1e-8), self.rho, self.eta, self.gam)

    def iv(self, k, T):
        return np.sqrt(self.w(k, T) / T)

    def fit(self, k, T, iv_mkt, weights=None):
        k = np.asarray(k); T = np.asarray(T); iv_mkt = np.asarray(iv_mkt)
        wts = np.ones_like(iv_mkt) if weights is None else np.sqrt(np.asarray(weights))
        th0 = []
        for Tp in self.Tp:
            m = T == Tp; j = np.argmin(np.abs(k[m])); th0.append(iv_mkt[m][j]**2 * Tp)
        x0 = np.concatenate([[-0.6, 1.0, 0.4], th0])
        lo = np.concatenate([[-0.999, 1e-3, 0.0], np.full(len(self.Tp), 1e-5)])
        hi = np.concatenate([[0.999, 10.0, 0.5], np.full(len(self.Tp), 5.0)])

        def resid(x):
            self.rho, self.eta, self.gam = x[:3]; self.thetas = x[3:]
            r = wts * (self.iv(k, T) - iv_mkt)
            # soft penalties: calendar arbitrage (theta increasing) and the butterfly-free condition eta (1 + |rho|) <= 2
            pen_cal = np.maximum(0.0, -np.diff(np.concatenate([[0.0], self.thetas]))) * 100
            pen_bf = max(0.0, self.eta * (1 + abs(self.rho)) - 2.0) * 10
            return np.concatenate([r, pen_cal, [pen_bf]])

        res = optimize.least_squares(resid, x0, bounds=(lo, hi), method='trf', xtol=1e-12, ftol=1e-12, max_nfev=2000)
        self.rho, self.eta, self.gam = res.x[:3]; self.thetas = res.x[3:]
        self.fit_rmse = float(np.sqrt(np.mean((self.iv(k, T) - iv_mkt)**2)))
        return self

    def local_var(self, k, T, hk=1e-4, hT=1e-4):
        """Dupire local variance from total variance w(k, T) (Gatheral 2006, eq. 1.10)."""
        k = np.asarray(k, float); T = np.asarray(T, float)
        w = self.w(k, T)
        wk = (self.w(k + hk, T) - self.w(k - hk, T)) / (2 * hk)
        wkk = (self.w(k + hk, T) - 2 * w + self.w(k - hk, T)) / hk**2
        wT = (self.w(k, T + hT) - self.w(k, np.maximum(T - hT, 1e-6))) / (T + hT - np.maximum(T - hT, 1e-6))
        den = 1 - k / w * wk + 0.25 * (-0.25 - 1 / w + k**2 / w**2) * wk**2 + 0.5 * wkk
        return np.clip(wT, 1e-6, None) / np.clip(den, 1e-3, None)



def svi_w(k, a, b, rho, m, sig):
    """Raw SVI total variance (Gatheral 2004)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sig**2))


def svi_g(k, a, b, rho, m, sig):
    """Butterfly-arbitrage function g(k) of Gatheral & Jacquier (2014); the slice is arbitrage-free iff g >= 0."""
    w = svi_w(k, a, b, rho, m, sig)
    R = np.sqrt((k - m)**2 + sig**2)
    wp = b * (rho + (k - m) / R)
    wpp = b * sig**2 / R**3
    w = np.maximum(w, 1e-10)
    return (1 - k * wp / (2 * w))**2 - wp**2 / 4 * (1 / w + 0.25) + wpp / 2


class SVISurface:
    """
    Slice-by-slice raw SVI in total variance, fitted in implied vol with weights, expiry by expiry from the shortest,
    with a penalty against crossing the previous slice (calendar arbitrage) on a log-moneyness grid and the
    Roger Lee bound b (1 + |rho|) <= 4 / T. Between pillars total variance is interpolated linearly in T at fixed k
    (and from zero at T = 0), which keeps calendar monotonicity if the slices do not cross. Same interface as SSVISurface.
    """

    def __init__(self, T_pillars, k_pen=np.linspace(-1.5, 0.8, 93)):
        self.Tp = np.asarray(sorted(T_pillars), float); self.k_pen = k_pen

    def fit(self, k, T, iv_mkt, weights=None):
        k = np.asarray(k); T = np.asarray(T); iv_mkt = np.asarray(iv_mkt)
        wts = np.ones_like(iv_mkt) if weights is None else np.sqrt(np.asarray(weights))
        self.params = []
        prev = None
        for Tp in self.Tp:
            m_ = T == Tp; kk = k[m_]; ivm = iv_mkt[m_]; ww = wts[m_]; ww = ww / ww.mean()
            w_mkt = ivm**2 * Tp
            j = np.argmin(np.abs(kk)); a0 = 0.8 * w_mkt[j]
            x0 = np.array([a0, 0.1 * Tp**0.5, -0.5, 0.0, 0.1])
            lo = np.array([-1.0, 1e-6, -0.999, -1.0, 0.03]); hi = np.array([5.0, 10.0, 0.999, 1.0, 5.0])

            def resid(x, Tp=Tp, kk=kk, ivm=ivm, ww=ww, prev=prev):
                w = svi_w(kk, *x)
                r = ww * (np.sqrt(np.maximum(w, 1e-10) / Tp) - ivm)
                pen = [max(0.0, x[1] * (1 + abs(x[2])) - 4.0 / Tp) * 10.0]
                if prev is not None:
                    gap = svi_w(self.k_pen, *prev) - svi_w(self.k_pen, *x)      # previous slice must lie below
                    pen += list(np.maximum(0.0, gap) * 5000.0)
                w_pen = svi_w(self.k_pen, *x)
                pen += list(np.maximum(0.0, -w_pen) * 50.0)                       # positivity
                pen += list(np.maximum(0.0, -svi_g(self.k_pen, *x)) * 2000.0)     # butterfly: Dupire denominator g >= 0
                return np.concatenate([r, pen])

            res = optimize.least_squares(resid, x0, bounds=(lo, hi), method='trf', xtol=1e-12, ftol=1e-12, max_nfev=3000)
            self.params.append(res.x); prev = res.x
        self.params = np.array(self.params)
        self.fit_rmse = float(np.sqrt(np.mean((self.iv(k, T) - iv_mkt)**2)))
        return self

    def w(self, k, T):
        k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        shape = k.shape; k = k.ravel(); T = T.ravel()
        Tg = np.concatenate([[0.0], self.Tp])
        W = np.stack([np.zeros_like(k)] + [svi_w(k, *p) for p in self.params])      # (n_pillars + 1, n)
        Tc = np.clip(T, 0.0, self.Tp[-1])
        idx = np.clip(np.searchsorted(Tg, Tc, side='right') - 1, 0, len(Tg) - 2)
        t0 = Tg[idx]; t1 = Tg[idx + 1]
        lam = (Tc - t0) / (t1 - t0)
        cols = np.arange(len(k))
        out = W[idx, cols] * (1 - lam) + W[idx + 1, cols] * lam
        out = np.where(T > self.Tp[-1], out * T / self.Tp[-1], out)      # flat implied vol beyond the last pillar
        return out.reshape(shape)

    def iv(self, k, T):
        return np.sqrt(self.w(k, T) / np.asarray(T, float))

    local_var = SSVISurface.local_var


def mc_vanilla_iv(pricer, T, strikes, r_curve_func, q_func, S0):
    """
    Implied vols from Monte Carlo paths, pricing the out-of-the-money instrument (put below the forward, call above)
    and converting with put-call parity, so that deep OTM options are not obtained as differences of large numbers.
    Returns (iv, price standard error in vol points via vega).
    """
    idx = pricer._get_time_index(T); t = pricer.time_grid[idx]; ST = pricer.S_paths[idx]
    r = float(r_curve_func(t)); q = float(q_func(t)); F = S0 * np.exp((r - q) * t); df = np.exp(-r * t)
    K = np.atleast_1d(strikes).astype(float)
    call_eq = np.empty_like(K); se = np.empty_like(K)
    for j, Kj in enumerate(K):
        if Kj < F:
            pay = np.maximum(Kj - ST, 0.0) * df
            call_eq[j] = pay.mean() + S0 * np.exp(-q * t) - Kj * df
        else:
            pay = np.maximum(ST - Kj, 0.0) * df
            call_eq[j] = pay.mean()
        se[j] = pay.std(ddof=1) / np.sqrt(len(pay))
    iv = implied_vol_call_vec(call_eq, S0, K, t, r, q)
    vega = bs_vega(S0, K, t, r, q, iv)
    return iv, se / np.maximum(vega, 1e-8)


# ====================================================================== 7. LV and LSV engines
@njit(fastmath=True)
def _qe_variance_step(vt, zv, dt, kappa, theta, sigma, psi_c=1.5):
    E = math.exp(-kappa * dt)
    m = theta + (vt - theta) * E
    s2 = vt * sigma * sigma * E * (1.0 - E) / kappa + theta * sigma * sigma * (1.0 - E) * (1.0 - E) / (2.0 * kappa)
    psi = s2 / (m * m)
    if psi <= psi_c:
        inv = 2.0 / psi
        b2 = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
        a = m / (1.0 + b2)
        return a * (math.sqrt(b2) + zv) * (math.sqrt(b2) + zv)
    pp = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - pp) / m
    u = 0.5 * (1.0 + math.erf(zv / math.sqrt(2.0)))
    if u <= pp:
        return 0.0
    return math.log((1.0 - pp) / (1.0 - u)) / beta


@njit(fastmath=True)
def engine_local_stochastic_vol(S0, v0, kappa, theta, sigma, rho, dts, drifts, Z1, Z2, k_grid, locvol_grid, use_sv, n_bins):
    """
    Local vol (use_sv = False) or local-stochastic vol (use_sv = True) paths.
      LV : dX = (mu - 1/2 sig_loc^2) dt + sig_loc dW
      LSV: dX = (mu - 1/2 L^2 V) dt + L sqrt(V) dW,  L^2(x, t) = sig_loc^2(x, t) / E[V_t | X_t = x]   (particle method)
    with V a QE variance step driven by Z1 and dW = rho Z1 + sqrt(1 - rho^2) Z2. x = log(S_t / F_t) is log-moneyness
    against the forward; sig_loc is read from locvol_grid[t, :] on k_grid by linear interpolation. E[V | x] is estimated
    at every step by a local-linear regression of V on x inside n_bins quantile bins of x (3-bin windows): the particle method with an adaptive kernel.
    Returns S paths (N + 1, n_paths) and the leverage L at the last step for inspection.
    """
    N_steps, n_paths = Z1.shape
    S = np.empty((N_steps + 1, n_paths))
    v = np.empty(n_paths); x = np.empty(n_paths)
    for p in range(n_paths):
        S[0, p] = S0; v[p] = v0; x[p] = 0.0
    sq1mr2 = math.sqrt(1.0 - rho * rho)
    n_k = len(k_grid)
    logF = 0.0
    sums = np.zeros(n_bins); cnts = np.zeros(n_bins); condv = np.zeros(n_bins)
    sx = np.zeros(n_bins); sxx = np.zeros(n_bins); sxv = np.zeros(n_bins); slope = np.zeros(n_bins); icept = np.zeros(n_bins)
    L_last = np.ones(n_paths)
    bin_of = np.zeros(n_paths, dtype=np.int64)
    for t in range(N_steps):
        dt = dts[t]; sq = math.sqrt(dt); mu = drifts[t]
        # conditional expectation E[V | x] on this step: quantile bins (equal path counts), 3-bin smoothing
        if use_sv:
            order = np.argsort(x)
            for rank in range(n_paths):
                bin_of[order[rank]] = (rank * n_bins) // n_paths
            for b in range(n_bins):
                sums[b] = 0.0; cnts[b] = 0.0; sx[b] = 0.0; sxx[b] = 0.0; sxv[b] = 0.0
            for p in range(n_paths):
                vp = v[p] if v[p] > 0.0 else 0.0
                bb = bin_of[p]
                sums[bb] += vp; cnts[bb] += 1.0; sx[bb] += x[p]; sxx[bb] += x[p] * x[p]; sxv[bb] += x[p] * vp
            for b in range(n_bins):
                s_ = sums[b]; c_ = cnts[b]; a_ = sx[b]; q_ = sxx[b]; z_ = sxv[b]
                if b > 0: s_ += sums[b - 1]; c_ += cnts[b - 1]; a_ += sx[b - 1]; q_ += sxx[b - 1]; z_ += sxv[b - 1]
                if b < n_bins - 1: s_ += sums[b + 1]; c_ += cnts[b + 1]; a_ += sx[b + 1]; q_ += sxx[b + 1]; z_ += sxv[b + 1]
                xm = a_ / c_; vm = s_ / c_
                varx = q_ / c_ - xm * xm
                slope[b] = (z_ / c_ - xm * vm) / varx if varx > 1e-12 else 0.0
                icept[b] = vm - slope[b] * xm
        for p in range(n_paths):
            # local vol by linear interpolation on the k grid (flat outside)
            xp = x[p]
            if xp <= k_grid[0]:
                sl = locvol_grid[t, 0]
            elif xp >= k_grid[n_k - 1]:
                sl = locvol_grid[t, n_k - 1]
            else:
                j = int((xp - k_grid[0]) / (k_grid[1] - k_grid[0]))
                if j >= n_k - 1: j = n_k - 2
                wgt = (xp - k_grid[j]) / (k_grid[j + 1] - k_grid[j])
                sl = locvol_grid[t, j] * (1.0 - wgt) + locvol_grid[t, j + 1] * wgt
            zv = Z1[t, p]; zs = rho * zv + sq1mr2 * Z2[t, p]
            if use_sv:
                vp = v[p] if v[p] > 0.0 else 0.0
                cv = icept[bin_of[p]] + slope[bin_of[p]] * xp        # local-linear E[V | x] in the path's bin
                if cv < 1e-5: cv = 1e-5
                L = sl / math.sqrt(cv)
                vol = L * math.sqrt(vp)
                L_last[p] = L
                v[p] = _qe_variance_step(v[p], zv, dt, kappa, theta, sigma)
            else:
                vol = sl
            dx = mu - 0.5 * vol * vol * dt + vol * sq * zs
            S[t + 1, p] = S[t, p] * math.exp(dx)
            x[p] += dx - mu          # log-moneyness against the forward: strip the drift
        logF += mu
    return S, L_last


class LocalVolPricer(HestonAutocallPricer):
    """
    HestonAutocallPricer on LV or LSV paths. surface: fitted SSVISurface; heston params for the SV part.
    Everything after run_simulation (payoff, coupon solver, checks) is inherited.
    """

    def __init__(self, S0, S_ref, r_curve_func, q_curve_func, heston_params, surface, use_sv=True,
                 k_grid=np.linspace(-1.5, 0.8, 231), n_bins=100):
        super().__init__(S0, S_ref, r_curve_func, q_curve_func, heston_params, scheme='qe')
        self.surface = surface; self.use_sv = use_sv; self.k_grid = np.asarray(k_grid); self.n_bins = n_bins

    def local_vol_grid(self, time_grid):
        tt = np.maximum(time_grid, 1e-4)
        K_, T_ = np.meshgrid(self.k_grid, tt)
        return np.sqrt(self.surface.local_var(K_.ravel(), T_.ravel())).reshape(len(tt), len(self.k_grid)).clip(0.03, 2.5)

    def run_simulation(self, T_max, steps_per_year=252, n_paths=50000, force_unit_start=False, obs_times=None, seed=None, scheme=None, verbose=True):
        self.time_grid = self._build_time_grid(T_max, steps_per_year, obs_times)
        dts = np.diff(self.time_grid); drifts = np.diff(self._log_forward(self.time_grid))
        if seed is not None:
            np.random.seed(seed)
        Z1 = np.random.normal(size=(len(dts), n_paths)); Z2 = np.random.normal(size=(len(dts), n_paths))
        lv = self.local_vol_grid(self.time_grid)
        start = 1.0 if force_unit_start else self.S0
        self.S_paths, self.leverage_last = engine_local_stochastic_vol(start, self.v0, self.kappa, self.theta, self.sigma, self.rho,
                                                                       dts, drifts, Z1, Z2, self.k_grid, lv, self.use_sv, self.n_bins)
        self.is_simulated = True; self.zero_variance_frac = np.nan
        if verbose:
            print(f"Simulation complete ({'LSV' if self.use_sv else 'LV'}): {n_paths} paths over {T_max:.2f} years ({len(dts)} steps).")
