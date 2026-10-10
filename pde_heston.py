"""
Finite-difference benchmark for the autocall under Heston: 2-D PDE in (x = log S/S0, V) solved backward in time with
the Craig-Sneyd ADI scheme (in 't Hout & Foulon 2010) and the autocall as jump conditions on the observation dates.

    u_t + 1/2 V u_xx + (mu(t) - 1/2 V) u_x + rho sigma V u_xV + 1/2 sigma^2 V u_VV + kappa (theta - V) u_V - f(t) u = 0

with mu(t) the log-forward drift and f(t) the instantaneous forward rate from the zero curve, both read per time step
exactly as in the Monte Carlo engine. Boundaries: Dirichlet at x_min / x_max (asymptotic values), the degenerate PDE
itself at V = 0 (no boundary condition needed, Feller or not), u_V = 0 at V_max. Discontinuous payoffs (digitals)
are followed by two Rannacher steps (Douglas scheme with theta = 1).
"""
import math
import numpy as np
from numba import njit


# ---------------------------------------------------------------- grids
def log_spot_grid(barriers=(0.6, 0.7, 1.0), x_min=math.log(0.05), x_max=math.log(5.0), fine=0.004, coarse=0.02):
    """Union of uniform segments whose end points are the barriers (in log-moneyness), fine between 0.4 and 1.6."""
    pts = [x_min, math.log(0.4)] + [math.log(b) for b in barriers] + [math.log(1.6), x_max]
    pts = sorted(set(pts))
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        h = fine if (a >= math.log(0.4) - 1e-12 and b <= math.log(1.6) + 1e-12) else coarse
        n = max(2, int(round((b - a) / h)) + 1)
        seg = np.linspace(a, b, n)
        out.extend(seg[1:])
    return np.array(out)


def variance_grid(v_max=3.0, n=80, c=0.08):
    """sinh-spaced grid on [0, v_max], concentrated near zero (in 't Hout & Foulon)."""
    xi = np.linspace(0.0, math.asinh(v_max / c), n)
    return c * np.sinh(xi)


def _d1_d2_weights(y):
    """Three-point weights for first and second derivatives on a non-uniform grid (interior points)."""
    n = len(y); hm = np.zeros(n); hp = np.zeros(n)
    hm[1:] = y[1:] - y[:-1]; hp[:-1] = y[1:] - y[:-1]
    w1 = np.zeros((n, 3)); w2 = np.zeros((n, 3))
    i = slice(1, n - 1)
    w1[i, 0] = -hp[i] / (hm[i] * (hm[i] + hp[i])); w1[i, 1] = (hp[i] - hm[i]) / (hm[i] * hp[i]); w1[i, 2] = hm[i] / (hp[i] * (hm[i] + hp[i]))
    w2[i, 0] = 2.0 / (hm[i] * (hm[i] + hp[i])); w2[i, 1] = -2.0 / (hm[i] * hp[i]); w2[i, 2] = 2.0 / (hp[i] * (hm[i] + hp[i]))
    return w1, w2


# ---------------------------------------------------------------- batched tridiagonal solver
@njit(fastmath=True)
def _thomas_rows(a, b, c, d):
    """Solve, for every row k, the tridiagonal system with sub-diagonal a[k], diagonal b[k], super-diagonal c[k]."""
    K, n = d.shape
    x = np.empty_like(d); cp = np.empty(n); dp = np.empty(n)
    for k in range(K):
        cp[0] = c[k, 0] / b[k, 0]; dp[0] = d[k, 0] / b[k, 0]
        for i in range(1, n):
            m = b[k, i] - a[k, i] * cp[i - 1]
            cp[i] = c[k, i] / m
            dp[i] = (d[k, i] - a[k, i] * dp[i - 1]) / m
        x[k, n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            x[k, i] = dp[i] - cp[i] * x[k, i + 1]
    return x


# ---------------------------------------------------------------- the solver
class HestonADISolver:
    def __init__(self, params, x=None, v=None):
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = params
        self.x = log_spot_grid() if x is None else np.asarray(x, float)
        self.v = variance_grid() if v is None else np.asarray(v, float)
        self.nx, self.nv = len(self.x), len(self.v)
        self.w1x, self.w2x = _d1_d2_weights(self.x)
        self.w1v, self.w2v = _d1_d2_weights(self.v)
        # one-sided first derivative in V at V = 0 (upwind, V only increases there since kappa theta > 0)
        self.w1v0 = np.array([0.0, -1.0 / (self.v[1] - self.v[0]), 1.0 / (self.v[1] - self.v[0])])
        self.X, self.Vg = np.meshgrid(self.x, self.v, indexing='ij')        # (nx, nv)

    # ---- operator coefficients (nx, nv, 3) along x and along v, mixed as scalar factor
    def _coeffs(self, mu, f):
        V = self.Vg
        # A1: 1/2 V u_xx + (mu - 1/2 V) u_x - 1/2 f u
        a1 = 0.5 * V[:, :, None] * self.w2x[:, None, :] + (mu - 0.5 * V)[:, :, None] * self.w1x[:, None, :]
        a1[:, :, 1] -= 0.5 * f
        a1[0, :, :] = 0.0; a1[-1, :, :] = 0.0                                # Dirichlet rows
        # A2: 1/2 sigma^2 V u_VV + kappa (theta - V) u_V - 1/2 f u
        a2 = 0.5 * self.sigma**2 * V[:, :, None] * self.w2v[None, :, :] + (self.kappa * (self.theta - V))[:, :, None] * self.w1v[None, :, :]
        a2[:, 0, :] = (self.kappa * self.theta) * self.w1v0[None, :]          # V = 0: degenerate PDE, upwind u_V
        a2[:, :, 1] -= 0.5 * f
        a2[:, -1, :] = 0.0                                                   # V_max: Neumann handled by copy
        a2[0, :, :] = 0.0; a2[-1, :, :] = 0.0
        return a1, a2

    def _apply_x(self, a1, U):
        out = a1[:, :, 1] * U
        out[1:-1] += a1[1:-1, :, 0] * U[:-2] + a1[1:-1, :, 2] * U[2:]
        return out

    def _apply_v(self, a2, U):
        out = a2[:, :, 1] * U
        out[:, 1:-1] += a2[:, 1:-1, 0] * U[:, :-2] + a2[:, 1:-1, 2] * U[:, 2:]
        out[:, 0] = a2[:, 0, 1] * U[:, 0] + a2[:, 0, 2] * U[:, 1]
        return out

    def _apply_mixed(self, U):
        """rho sigma V u_xV with central differences on both non-uniform grids (interior only)."""
        out = np.zeros_like(U)
        # first derivative in x -> (nx, nv), then in v
        ux = np.zeros_like(U)
        ux[1:-1] = self.w1x[1:-1, 0, None] * U[:-2] + self.w1x[1:-1, 1, None] * U[1:-1] + self.w1x[1:-1, 2, None] * U[2:]
        out[:, 1:-1] = self.w1v[None, 1:-1, 0] * ux[:, :-2] + self.w1v[None, 1:-1, 1] * ux[:, 1:-1] + self.w1v[None, 1:-1, 2] * ux[:, 2:]
        out *= self.rho * self.sigma * self.Vg
        out[0, :] = 0.0; out[-1, :] = 0.0
        return out

    def _solve_x(self, a1, theta_dt, rhs, bc_lo, bc_hi):
        a = -theta_dt * a1[:, :, 0].T.copy(); b = 1.0 - theta_dt * a1[:, :, 1].T.copy(); c = -theta_dt * a1[:, :, 2].T.copy()
        d = rhs.T.copy()
        a[:, 0] = 0; b[:, 0] = 1; c[:, 0] = 0; d[:, 0] = bc_lo
        a[:, -1] = 0; b[:, -1] = 1; c[:, -1] = 0; d[:, -1] = bc_hi
        return _thomas_rows(a, b, c, d).T

    def _solve_v(self, a2, theta_dt, rhs, bc_lo, bc_hi):
        a = -theta_dt * a2[:, :, 0].copy(); b = 1.0 - theta_dt * a2[:, :, 1].copy(); c = -theta_dt * a2[:, :, 2].copy()
        d = rhs.copy()
        a[:, -1] = -1.0; b[:, -1] = 1.0; c[:, -1] = 0.0; d[:, -1] = 0.0        # u_V = 0 at V_max: u_N - u_{N-1} = 0
        a[0, :] = 0; b[0, :] = 1; c[0, :] = 0; d[0, :] = bc_lo
        a[-1, :] = 0; b[-1, :] = 1; c[-1, :] = 0; d[-1, :] = bc_hi
        return _thomas_rows(a, b, c, d)

    def step(self, U, dt, mu, f, bc_lo, bc_hi, scheme='CS', theta=0.5):
        """One backward time step of length dt. bc_lo / bc_hi: Dirichlet values (arrays over V) at the new time."""
        a1, a2 = self._coeffs(mu, f)
        A0U = self._apply_mixed(U); A1U = self._apply_x(a1, U); A2U = self._apply_v(a2, U)
        Y0 = U + dt * (A0U + A1U + A2U)
        Y1 = self._solve_x(a1, theta * dt, Y0 - theta * dt * A1U, bc_lo, bc_hi)
        Y2 = self._solve_v(a2, theta * dt, Y1 - theta * dt * A2U, bc_lo, bc_hi)
        if scheme == 'CS':
            Y0h = Y0 + 0.5 * dt * (self._apply_mixed(Y2) - A0U)
            Y1h = self._solve_x(a1, theta * dt, Y0h - theta * dt * A1U, bc_lo, bc_hi)
            Y2 = self._solve_v(a2, theta * dt, Y1h - theta * dt * A2U, bc_lo, bc_hi)
        return Y2

    def solve(self, time_grid, terminal, events, bc_lo_fn, bc_hi_fn, drift_fn, rate_fn, rannacher=2, verbose=False):
        """
        time_grid: increasing times (0 = valuation, last = maturity), containing every event time.
        terminal(x) -> (nx,) payoff at maturity (applied for all V).
        events: dict {time: (mask(x) -> bool array, payoff(x) -> (nx,))}: on that date, where mask is true the
                value becomes payoff (the autocall).
        bc_lo_fn(t), bc_hi_fn(t): Dirichlet values at x_min and x_max at time t (scalar or (nv,)).
        drift_fn(t0, t1), rate_fn(t0, t1): log-forward increment / dt and forward rate over the step.
        Returns U(x, v) at time 0.
        """
        U = np.repeat(terminal(self.x)[:, None], self.nv, axis=1)
        n_ran = rannacher
        for n in range(len(time_grid) - 1, 0, -1):
            t1, t0 = time_grid[n], time_grid[n - 1]
            if t1 in events:
                mask, pay = events[t1]
                U[mask(self.x), :] = pay(self.x)[mask(self.x)][:, None]
                n_ran = rannacher
            dt = t1 - t0
            mu = drift_fn(t0, t1); f = rate_fn(t0, t1)
            bl = np.broadcast_to(bc_lo_fn(t0), (self.nv,)).copy(); bh = np.broadcast_to(bc_hi_fn(t0), (self.nv,)).copy()
            if n_ran > 0:
                # Rannacher: two implicit (theta = 1, Douglas) half-steps
                for _ in range(2):
                    U = self.step(U, 0.5 * dt, mu, f, bl, bh, scheme='Douglas', theta=1.0)
                n_ran -= 1
            else:
                U = self.step(U, dt, mu, f, bl, bh, scheme='CS', theta=0.5)
            U[:, -1] = U[:, -2]
        if 0.0 in events:
            mask, pay = events[0.0]; U[mask(self.x), :] = pay(self.x)[mask(self.x)][:, None]
        return U

    def value_at(self, U, x0=0.0, v0=None):
        v0 = self.v0 if v0 is None else v0
        i = int(np.argmin(np.abs(self.x - x0)))
        j = int(np.searchsorted(self.v, v0)) - 1; j = min(max(j, 0), self.nv - 2)
        w = (v0 - self.v[j]) / (self.v[j + 1] - self.v[j])
        return (1 - w) * U[i, j] + w * U[i, j + 1]

    def delta_gamma_at(self, U, S0, x0=0.0, v0=None):
        """Delta and gamma in spot from the x grid at x0 (central differences, chain rule)."""
        v0 = self.v0 if v0 is None else v0
        i = int(np.argmin(np.abs(self.x - x0)))
        j = int(np.searchsorted(self.v, v0)) - 1; j = min(max(j, 0), self.nv - 2)
        w = (v0 - self.v[j]) / (self.v[j + 1] - self.v[j])
        col = (1 - w) * U[:, j] + w * U[:, j + 1]
        ux = self.w1x[i] @ col[i - 1:i + 2]; uxx = self.w2x[i] @ col[i - 1:i + 2]
        S = S0 * math.exp(self.x[i])
        delta = ux / S; gamma = (uxx - ux) / S**2
        return delta, gamma


# ---------------------------------------------------------------- the autocall on the PDE
def price_autocall_pde(params, S0, r_curve_func, q_curve_func, obs_times, coupon_step=0.03, barrier_redemption=0.70,
                       barrier_knock_in=0.60, exit_bonus=0.24, nominal=1000.0, payment_lag_days=7, steps_per_year=252,
                       leg='total', solver=None, verbose=False):
    """
    Note value at (S0, v0) by finite differences. leg in {'total', 'principal', 'coupon', 'put'} selects which cash
    flows are kept, with the same autocall logic in every case, so the three legs add to the total.
    Returns (value, U, solver).
    """
    solver = HestonADISolver(params) if solver is None else solver
    obs_times = np.asarray(obs_times, float); T = obs_times[-1]; lag = payment_lag_days / 365.0
    r = lambda t: float(r_curve_func(max(t, 1e-8))); q = lambda t: float(q_curve_func(max(t, 1e-8)))
    logF = lambda t: (r(t) - q(t)) * t
    df_lag = lambda t: math.exp(-(r(t + lag) * (t + lag) - r(t) * t))
    n = int(math.ceil(T * steps_per_year)); grid = np.linspace(0, T, n + 1)
    extra = [t for t in obs_times if np.min(np.abs(grid - t)) > 1e-9]
    grid = np.sort(np.concatenate([grid, extra])) if extra else grid
    grid = np.array([t for t in grid if t <= T + 1e-12])
    # snap event times to grid values exactly
    ev_times = [grid[int(np.argmin(np.abs(grid - t)))] for t in obs_times]
    S_ratio = lambda x: np.exp(x)
    keep_p = leg in ('total', 'principal'); keep_c = leg in ('total', 'coupon'); keep_k = leg in ('total', 'put')

    def terminal(x):
        s = S_ratio(x); d = df_lag(T); out = np.zeros_like(x)
        bonus = s >= barrier_redemption; par = (s >= barrier_knock_in) & ~bonus; ki = s < barrier_knock_in
        if keep_p: out[bonus | par | ki] += nominal * d
        if keep_c: out[bonus] += nominal * exit_bonus * d
        if keep_k: out[ki] += nominal * (s[ki] - 1.0) * d
        return out

    events = {}
    for k, t_obs in enumerate(ev_times[:-1]):
        n_period = k + 2; d = df_lag(t_obs)
        val = (nominal * d if keep_p else 0.0) + (nominal * n_period * coupon_step * d if keep_c else 0.0)
        events[t_obs] = (lambda x: S_ratio(x) >= 1.0 - 1e-12, (lambda x, val=val: np.full_like(x, val)))

    def next_call_value(t):
        """x_max boundary: certain to be called at the next observation date (or redeemed with bonus at maturity)."""
        future = [tt for tt in ev_times if tt > t + 1e-12]
        if not future: return 0.0
        tn = future[0]; k = ev_times.index(tn)
        disc = math.exp(-(r(tn) * tn - r(t) * t))
        if tn == ev_times[-1]:
            return disc * df_lag(tn) * ((nominal if keep_p else 0.0) + (nominal * exit_bonus if keep_c else 0.0))
        return disc * df_lag(tn) * ((nominal if keep_p else 0.0) + (nominal * (k + 2) * coupon_step if keep_c else 0.0))

    def deep_ki_value(t):
        """x_min boundary: never called. Below the knock-in: nominal (principal) + nominal (S_T/S0 - 1) (put);
        if the boundary sits above the knock-in (e.g. knock-in switched off): par, or par plus bonus above the bonus barrier."""
        disc = math.exp(-(r(T) * T - r(t) * t)); fwd = math.exp(logF(T) - logF(t)) * math.exp(solver.x[0])
        s_min = math.exp(solver.x[0])
        if s_min < barrier_knock_in:
            return disc * df_lag(T) * ((nominal if keep_p else 0.0) + (nominal * (fwd - 1.0) if keep_k else 0.0))
        bonus = s_min >= barrier_redemption
        return disc * df_lag(T) * ((nominal if keep_p else 0.0) + (nominal * exit_bonus if (keep_c and bonus) else 0.0))

    drift_fn = lambda t0, t1: (logF(t1) - logF(t0)) / (t1 - t0)
    rate_fn = lambda t0, t1: (r(t1) * t1 - r(t0) * t0) / (t1 - t0)
    U = solver.solve(grid, terminal, events, deep_ki_value, next_call_value, drift_fn, rate_fn, verbose=verbose)
    return solver.value_at(U), U, solver


def price_european_call_pde(params, S0, K, T, r, q, solver=None, steps_per_year=252):
    """European call by the same solver, for validation against the FFT."""
    solver = HestonADISolver(params) if solver is None else solver
    n = int(math.ceil(T * steps_per_year)); grid = np.linspace(0, T, n + 1)
    terminal = lambda x: np.maximum(S0 * np.exp(x) - K, 0.0)
    bc_hi = lambda t: S0 * math.exp(solver.x[-1]) * math.exp(-q * (T - t)) - K * math.exp(-r * (T - t))
    bc_lo = lambda t: 0.0
    U = solver.solve(grid, terminal, {}, bc_lo, bc_hi, lambda t0, t1: r - q, lambda t0, t1: r, rannacher=2)
    return solver.value_at(U), U, solver
