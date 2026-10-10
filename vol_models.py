"""
Stochastic-volatility models for the model comparison, sharing the Lewis/FFT calibrator of heston_pricer.py.

Models (all give the characteristic function of log(S_T/S_0) including the drift (r-q)T):
    HestonModel        kappa, theta, sigma, rho, v0
    BatesModel         Heston + lognormal jumps (lam, muJ, sigJ)             Bates (1996)
    DoubleHestonModel  two independent CIR variance factors                  Christoffersen, Heston, Jacobs (2009)
    RoughHestonModel   fractional Riccati, Hurst H                           El Euch, Rosenbaum (2019);
                       solved with the (3,3) rational approximant of Gatheral, Radoicic (2019)

FFTCalibrator(model=...) is HestonFFTCalibrator with the model plugged in. MultiModelPricer adds QE-based Monte Carlo
for Bates and double Heston so the note can be priced under each model.
"""
import math
import numpy as np
from scipy.special import gamma as Gamma, roots_jacobi
from numba import njit

from heston_pricer import HestonFFTCalibrator, HestonAutocallPricer


# ---------------------------------------------------------------- characteristic functions
def heston_cf_core(u, T, kappa, theta, sigma, rho, v0):
    """Heston characteristic function of the driftless log return, stable 'little trap' form."""
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g2 = (xi - d) / (xi + d)
    e = np.exp(-d * T)
    return np.exp(
        (kappa * theta) / sigma**2 * ((xi - d) * T - 2 * np.log((1 - g2 * e) / (1 - g2)))
        + (v0 / sigma**2) * (xi - d) * (1 - e) / (1 - g2 * e)
    )


class HestonModel:
    name = 'Heston'
    param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
    bounds = [(0.01, 15), (0.001, 1), (0.01, 10), (-0.999, 0.999), (0.001, 1)]

    @staticmethod
    def char_func(u, T, r, q, params):
        kappa, theta, sigma, rho, v0 = params
        return np.exp(1j * u * (r - q) * T) * heston_cf_core(u, T, kappa, theta, sigma, rho, v0)


class BatesModel:
    """Heston plus compound-Poisson jumps in the log price: intensity lam, jump size ~ N(muJ, sigJ^2)."""
    name = 'Bates (Heston + jumps)'
    param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'lam', 'muJ', 'sigJ']
    bounds = HestonModel.bounds + [(0.0, 5.0), (-0.6, 0.3), (0.005, 0.6)]

    @staticmethod
    def char_func(u, T, r, q, params):
        kappa, theta, sigma, rho, v0, lam, muJ, sigJ = params
        k_bar = np.exp(muJ + 0.5 * sigJ**2) - 1            # E[e^J] - 1, the martingale compensator
        jump = np.exp(T * lam * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1) - 1j * u * T * lam * k_bar)
        return np.exp(1j * u * (r - q) * T) * heston_cf_core(u, T, kappa, theta, sigma, rho, v0) * jump


class DoubleHestonModel:
    """Two independent CIR variance factors, each with its own correlation to the spot."""
    name = 'Double Heston'
    param_names = ['kappa1', 'theta1', 'sigma1', 'rho1', 'v01', 'kappa2', 'theta2', 'sigma2', 'rho2', 'v02']
    bounds = HestonModel.bounds + HestonModel.bounds

    @staticmethod
    def char_func(u, T, r, q, params):
        p1, p2 = params[:5], params[5:]
        return np.exp(1j * u * (r - q) * T) * heston_cf_core(u, T, *p1) * heston_cf_core(u, T, *p2)


# ---------------------------------------------------------------- rough Heston
def _rough_riccati_pade(a, alpha, c0, c1, c2):
    """
    (3,3) rational approximant of the solution h(a, t) of the fractional Riccati equation
        D^alpha h = c0 + c1 h + c2 h^2,  h(0) = 0,
    in the variable x = t^alpha (Gatheral and Radoicic 2019). Matches the first three terms of the
    short-time series h = sum beta_k t^(k alpha) and the first three terms of the long-time expansion
    h = r_- + gamma_1 t^(-alpha) + gamma_2 t^(-2 alpha) + ...
    Returns p1, p2, p3, q1, q2, q3 (arrays shaped like a).
    """
    g = Gamma
    # short-time coefficients (term-by-term fractional differentiation of the series)
    b1 = c0 / g(alpha + 1)
    b2 = (c1 * b1) * g(alpha + 1) / g(2 * alpha + 1)
    b3 = (c1 * b2 + c2 * b1 * b1) * g(2 * alpha + 1) / g(3 * alpha + 1)
    # long-time: stable root and the algebraic tail (Riemann-Liouville derivative of the constant)
    d = np.sqrt(c1 * c1 - 4.0 * c2 * c0)
    rm = (-c1 - d) / (2.0 * c2)
    g1 = -rm / (d * g(1 - alpha))
    g2 = (c2 * g1 * g1 - g1 * g(1 - alpha) / g(1 - 2 * alpha)) / d
    # linear system for q (p follows): rows are the x^3 small-x match and the y^1, y^2 large-x matches
    #   b2 q1 + b1 q2 - rm q3 = -b3
    #   b1 q1 - rm q2 - g1 q3 = -b2
    #   rm q1 + g1 q2 + g2 q3 =  b1
    A11, A12, A13 = b2, b1, -rm
    A21, A22, A23 = b1, -rm, -g1
    A31, A32, A33 = rm, g1, g2
    r1, r2, r3 = -b3, -b2, b1
    det = (A11 * (A22 * A33 - A23 * A32) - A12 * (A21 * A33 - A23 * A31) + A13 * (A21 * A32 - A22 * A31))
    q1 = (r1 * (A22 * A33 - A23 * A32) - A12 * (r2 * A33 - A23 * r3) + A13 * (r2 * A32 - A22 * r3)) / det
    q2 = (A11 * (r2 * A33 - A23 * r3) - r1 * (A21 * A33 - A23 * A31) + A13 * (A21 * r3 - r2 * A31)) / det
    q3 = (A11 * (A22 * r3 - r2 * A32) - A12 * (A21 * r3 - r2 * A31) + r1 * (A21 * A32 - A22 * A31)) / det
    p1 = b1
    p2 = b2 + b1 * q1
    p3 = b3 + b2 * q1 + b1 * q2
    return p1, p2, p3, q1, q2, q3


def _pade_eval(coeffs, x):
    p1, p2, p3, q1, q2, q3 = coeffs
    return (p1 * x + p2 * x**2 + p3 * x**3) / (1.0 + q1 * x + q2 * x**2 + q3 * x**3)


_GJ_CACHE = {}


def _gauss_jacobi_01(n, a_exp, b_exp):
    """Nodes/weights for int_0^1 (1-v)^a v^b f(v) dv  =  sum w_i f(v_i)."""
    key = (n, round(a_exp, 12), round(b_exp, 12))
    if key not in _GJ_CACHE:
        x, w = roots_jacobi(n, a_exp, b_exp)
        v = 0.5 * (1 + x)
        _GJ_CACHE[key] = (v, w * 2.0**(-(a_exp + b_exp + 1)))
    return _GJ_CACHE[key]


def rough_heston_cf(u, T, r, q, params, n_nodes=40):
    """
    Characteristic function of log(S_T/S_0) in the rough Heston model
        dS/S = sqrt(V) dW,
        V_t = V_0 + 1/Gamma(alpha) int_0^t (t-s)^(alpha-1) [kappa (theta - V_s) ds + sigma sqrt(V_s) dB_s],
    alpha = H + 1/2, corr(dW, dB) = rho.  E[exp(a X_T)] = exp( kappa theta I^1 h(a,T) + V_0 I^(1-alpha) h(a,T) )
    with D^alpha h = (a^2-a)/2 + (a rho sigma - kappa) h + sigma^2/2 h^2  (El Euch, Rosenbaum 2019).
    h is the (3,3) rational approximant; the fractional integrals are Gauss-Jacobi quadratures after the
    substitution s = T v^(1/alpha), which makes the integrand a smooth function of v.

    params: [kappa, theta, sigma, rho, v0, H]  (sigma = vol of vol, so H = 1/2 is classical Heston)
    """
    kappa, theta, sigma, rho, v0, H = params
    alpha = H + 0.5
    u = np.asarray(u, dtype=complex)
    a = 1j * u
    c0 = 0.5 * (a * a - a)
    c1 = a * rho * sigma - kappa
    c2 = 0.5 * sigma**2
    coeffs = [c[:, None] for c in _rough_riccati_pade(a, alpha, c0, c1, c2)]   # (n_u, 1) against (1, n_nodes)
    Ta = T**alpha
    ia = 1.0 / alpha

    # I^1 h(T) = (T/alpha) int_0^1 v^(1/alpha - 1) h(T^alpha v) dv
    v1, w1 = _gauss_jacobi_01(n_nodes, 0.0, ia - 1.0)
    h1 = _pade_eval(coeffs, Ta * v1[None, :])
    I1 = (T * ia) * (h1 * w1[None, :]).sum(axis=1)

    # I^(1-alpha) h(T) = T^(1-alpha)/(alpha Gamma(1-alpha)) int_0^1 (1-v)^(-alpha) v^(1/alpha-1) [(1-v^(1/alpha))/(1-v)]^(-alpha) h(T^alpha v) dv
    v2, w2 = _gauss_jacobi_01(n_nodes, -alpha, ia - 1.0)
    smooth = ((1.0 - v2**ia) / (1.0 - v2))**(-alpha)
    h2 = _pade_eval(coeffs, Ta * v2[None, :])
    I1ma = T**(1 - alpha) / (alpha * Gamma(1 - alpha)) * (h2 * (w2 * smooth)[None, :]).sum(axis=1)

    return np.exp(a * (r - q) * T + kappa * theta * I1 + v0 * I1ma)


class RoughHestonModel:
    name = 'Rough Heston'
    param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'H']
    bounds = HestonModel.bounds + [(0.02, 0.45)]

    @staticmethod
    def char_func(u, T, r, q, params):
        return rough_heston_cf(u, T, r, q, params)


def rough_riccati_adams(a, alpha, c0, c1, c2, T, N=1500):
    """
    Reference solver for D^alpha h = c0 + c1 h + c2 h^2, h(0)=0 (Diethelm-Ford-Freed fractional
    Adams-Bashforth-Moulton predictor-corrector). Scalar complex a. Returns the time grid and h on it.
    O(N^2), for validation only.
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    h = np.zeros(N + 1, dtype=complex)
    F = lambda x: c0 + c1 * x + c2 * x * x
    f = np.zeros(N + 1, dtype=complex)
    f[0] = F(0.0)
    k = np.arange(N + 2, dtype=float)
    # corrector weights a_{j,n+1} and predictor weights b_{j,n+1}
    aw = (k[:-1] + 2)**(alpha + 1) - 2 * (k[:-1] + 1)**(alpha + 1) + k[:-1]**(alpha + 1)   # a_j for 1<=j<=n
    for n in range(N):
        j = np.arange(n + 1)
        bw = ((n + 1 - j)**alpha - (n - j)**alpha)                                   # predictor weights
        pred = (dt**alpha / Gamma(alpha + 1)) * np.sum(bw * f[:n + 1])
        # corrector: a_0 = n^(alpha+1) - (n-alpha)(n+1)^alpha ; a_j = (n-j+2)^(a+1) - 2(n-j+1)^(a+1) + (n-j)^(a+1)
        a0 = n**(alpha + 1) - (n - alpha) * (n + 1)**alpha
        aj = (n - j[1:] + 2)**(alpha + 1) - 2 * (n - j[1:] + 1)**(alpha + 1) + (n - j[1:])**(alpha + 1)
        corr = (dt**alpha / Gamma(alpha + 2)) * (a0 * f[0] + np.sum(aj * f[1:n + 1]) + F(pred))
        h[n + 1] = corr
        f[n + 1] = F(corr)
    return t, h


def fractional_integral_grid(t, h, T, order):
    """I^order h(T) for h given on a uniform grid, exact for piecewise-linear h (product integration)."""
    n = len(t) - 1
    dt = t[1] - t[0]
    # int_{t_j}^{t_{j+1}} (T-s)^(order-1) [h_j (t_{j+1}-s)/dt + h_{j+1} (s-t_j)/dt] ds, with T = t_n
    j = np.arange(n)
    tau0 = T - t[j]          # T - t_j
    tau1 = T - t[j + 1]      # T - t_{j+1}
    o = order
    # antiderivatives with w = T - s: s from t_j..t_{j+1} <-> w from tau0..tau1 (decreasing)
    def A(w): return -w**o / o                      # int (T-s)^(o-1) ds = -w^o/o
    def B(w): return w**(o + 1) / (o + 1) - T * w**o / o   # int s (T-s)^(o-1) ds = int (T-w) w^(o-1) (-dw)
    Ia = A(tau1) - A(tau0)
    Ib = B(tau1) - B(tau0)
    # h(s) = h_j + (h_{j+1}-h_j)(s - t_j)/dt
    val = h[j] * Ia + (h[j + 1] - h[j]) / dt * (Ib - t[j] * Ia)
    return val.sum() / Gamma(o)


def rough_heston_cf_reference(u, T, r, q, params, N=1500):
    """Reference CF via the Adams solver (scalar u), for validation of rough_heston_cf."""
    kappa, theta, sigma, rho, v0, H = params
    alpha = H + 0.5
    a = 1j * complex(u)
    c0 = 0.5 * (a * a - a); c1 = a * rho * sigma - kappa; c2 = 0.5 * sigma**2
    t, h = rough_riccati_adams(a, alpha, c0, c1, c2, T, N)
    I1 = fractional_integral_grid(t, h, T, 1.0)
    I1ma = fractional_integral_grid(t, h, T, 1.0 - alpha)
    return np.exp(a * (r - q) * T + kappa * theta * I1 + v0 * I1ma)


# ---------------------------------------------------------------- calibrator
class FFTCalibrator(HestonFFTCalibrator):
    """HestonFFTCalibrator with a pluggable model (same objective, weights, FFT grid and reports)."""

    def __init__(self, market_data, S0, r_curve_func, model=None, **kwargs):
        self.model = model if model is not None else HestonModel()
        super().__init__(market_data, S0, r_curve_func, **kwargs)
        self.bounds = list(self.model.bounds)

    def _heston_char_func(self, u, T, r_t, q_t, params):
        return self.model.char_func(u, T, r_t, q_t, params)

    def error_split(self, params, short_T=0.4):
        """Unweighted IV RMSE (vol pts) and price RMSE on the short end (T < short_T) and the rest."""
        prices = self.price_surface(params)
        from heston_pricer import implied_vol_call_vec
        iv = implied_vol_call_vec(prices, self.S0, self.flat_K, self.flat_T, self.flat_r, self.flat_q)
        res = iv - self.market_iv
        px = prices - self.flat_market
        m = self.flat_T < short_T
        out = {}
        for label, mask in [('all', np.ones_like(m)), ('short (<6M)', m), ('6M+', ~m)]:
            out[label] = {'n': int(mask.sum()), 'IV RMSE': 100 * np.sqrt(np.mean(res[mask]**2)),
                          'price RMSE': np.sqrt(np.mean(px[mask]**2)), 'max |IV err|': 100 * np.max(np.abs(res[mask]))}
        return out


# ---------------------------------------------------------------- Monte Carlo for Bates and double Heston
@njit
def _numba_seed(seed):
    np.random.seed(seed)


@njit(fastmath=True)
def _qe_step(vt, dt, kappa, theta, sigma, rho, psi_c, gamma1, gamma2):
    """One QE variance step and the log-spot coefficients for one path. Returns (v_next, A_term, B_term):
    log-spot increment = A_term + sqrt(B_term) * Z with the martingale-corrected constant included."""
    rho_sig = rho / sigma
    E = math.exp(-kappa * dt)
    K0 = -rho * kappa * theta * dt / sigma
    K1 = gamma1 * dt * (kappa * rho_sig - 0.5) - rho_sig
    K2 = gamma2 * dt * (kappa * rho_sig - 0.5) + rho_sig
    K3 = gamma1 * dt * (1.0 - rho * rho)
    K4 = gamma2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4
    m = theta + (vt - theta) * E
    s2 = vt * sigma * sigma * E * (1.0 - E) / kappa + theta * sigma * sigma * (1.0 - E) * (1.0 - E) / (2.0 * kappa)
    psi = s2 / (m * m)
    if psi <= psi_c:
        inv = 2.0 / psi
        b2 = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
        a = m / (1.0 + b2)
        b = math.sqrt(b2)
        zv = np.random.standard_normal()
        vn = a * (b + zv) * (b + zv)
        den = 1.0 - 2.0 * A * a
        K0s = (-A * b2 * a / den + 0.5 * math.log(den) - (K1 + 0.5 * K3) * vt) if den > 0.0 else K0
    else:
        pp = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - pp) / m
        uu = np.random.random()
        vn = 0.0 if uu <= pp else math.log((1.0 - pp) / (1.0 - uu)) / beta
        K0s = (-math.log(pp + beta * (1.0 - pp) / (beta - A)) - (K1 + 0.5 * K3) * vt) if beta > A else K0
    return vn, K0s + K1 * vt + K2 * vn, K3 * vt + K4 * vn


@njit(fastmath=True)
def engine_bates_qe(S0, kappa, theta, sigma, rho, v0, lam, muJ, sigJ, dts, drifts, n_paths, seed):
    """Bates: QE step for the diffusion plus compound-Poisson lognormal jumps, compensated in the drift."""
    np.random.seed(seed)
    N_steps = len(dts)
    S = np.empty((N_steps + 1, n_paths))
    v = np.empty(n_paths)
    for p in range(n_paths):
        S[0, p] = S0
        v[p] = v0
    k_bar = math.exp(muJ + 0.5 * sigJ * sigJ) - 1.0
    for t in range(N_steps):
        dt = dts[t]
        mu = drifts[t] - lam * k_bar * dt
        for p in range(n_paths):
            vn, A_term, B_term = _qe_step(v[p], dt, kappa, theta, sigma, rho, 1.5, 0.5, 0.5)
            x = mu + A_term + math.sqrt(B_term) * np.random.standard_normal()
            nj = np.random.poisson(lam * dt)
            if nj > 0:
                x += nj * muJ + sigJ * math.sqrt(nj) * np.random.standard_normal()
            S[t + 1, p] = S[t, p] * math.exp(x)
            v[p] = vn
    return S


@njit(fastmath=True)
def engine_double_heston_qe(S0, p1, p2, dts, drifts, n_paths, seed):
    """Double Heston: two independent QE variance factors; the spot step sums their drift and variance terms."""
    np.random.seed(seed)
    N_steps = len(dts)
    S = np.empty((N_steps + 1, n_paths))
    va = np.empty(n_paths); vb = np.empty(n_paths)
    for p in range(n_paths):
        S[0, p] = S0
        va[p] = p1[4]; vb[p] = p2[4]
    for t in range(N_steps):
        dt = dts[t]
        mu = drifts[t]
        for p in range(n_paths):
            vna, Aa, Ba = _qe_step(va[p], dt, p1[0], p1[1], p1[2], p1[3], 1.5, 0.5, 0.5)
            vnb, Ab, Bb = _qe_step(vb[p], dt, p2[0], p2[1], p2[2], p2[3], 1.5, 0.5, 0.5)
            x = mu + Aa + Ab + math.sqrt(Ba + Bb) * np.random.standard_normal()
            S[t + 1, p] = S[t, p] * math.exp(x)
            va[p] = vna; vb[p] = vnb
    return S


class MultiModelPricer(HestonAutocallPricer):
    """
    HestonAutocallPricer for other models. model in {'heston', 'bates', 'double_heston'}.
    The payoff logic, coupon solver and checks are inherited unchanged; only the path engine differs.
    """

    def __init__(self, S0, S_ref, r_curve_func, q_curve_func, params, model='heston'):
        params = np.asarray(params, dtype=float)
        super().__init__(S0, S_ref, r_curve_func, q_curve_func, params[:5], scheme='qe')
        self.model = model
        self.params = params

    def run_simulation(self, T_max, steps_per_year=252, n_paths=50000, force_unit_start=False,
                       obs_times=None, seed=None, scheme=None, verbose=True):
        if self.model == 'heston':
            return super().run_simulation(T_max, steps_per_year, n_paths, force_unit_start, obs_times, seed, scheme, verbose)
        self.time_grid = self._build_time_grid(T_max, steps_per_year, obs_times)
        dts = np.diff(self.time_grid)
        drifts = np.diff(self._log_forward(self.time_grid))
        start_val = 1.0 if force_unit_start else self.S0
        seed = 0 if seed is None else int(seed)
        if self.model == 'bates':
            self.S_paths = engine_bates_qe(start_val, *self.params[:8], dts, drifts, n_paths, seed)
        elif self.model == 'double_heston':
            self.S_paths = engine_double_heston_qe(start_val, self.params[:5].copy(), self.params[5:10].copy(),
                                                   dts, drifts, n_paths, seed)
        else:
            raise ValueError(f"no simulation scheme for model '{self.model}'")
        self.zero_variance_frac = np.nan
        self.is_simulated = True
        if verbose:
            print(f"Simulation complete ({self.model}, QE): {n_paths} paths over {T_max:.2f} years ({len(dts)} steps).")
