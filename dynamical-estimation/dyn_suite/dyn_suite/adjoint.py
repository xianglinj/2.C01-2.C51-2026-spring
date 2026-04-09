"""
dyn_suite.adjoint
-----------------
dyn_adjoint        — solve the adjoint ODE backwards in time; accumulate
                     the parameter gradient via explicit trapezoidal
                     quadrature at every RK45 internal step (Level 3),
                     fully decoupled from the data checkpoint grid (Level 1).
dyn_update_adjoint — gradient-descent update via the adjoint method.

Three nested time-discretisation levels
----------------------------------------
Level 1  Data checkpoints  t_n  (coarse, externally imposed)
         Residuals eps_n are injected as impulse kicks into a_rev.
Level 2  RK45 solver steps  (fine, adaptive, internal to solve_ivp)
         Propagates a_rev accurately; step sizes chosen solely by the
         adjoint error criterion — independent of Level-1 spacing.
Level 3  Quadrature nodes  (= Level-2 output points, seg.t)
         After each sub-interval, the integrand
             U(x(t_k), t_k, w).T @ a_rev(tau_k)
         is evaluated at every solver output point and accumulated into G
         via numpy.trapz.  Fully decoupled from Level 1: the quadrature
         density is set by RK45's adaptive grid, not by N.

         NOTE: scipy's solve_ivp (RK45) returns only the raw adaptive
         steps by default (dense_output=False).  These are the true
         integration nodes — equivalent to MATLAB ode45 with Refine=1.
         For denser quadrature nodes pass dense_output=True and supply a
         t_eval grid; the trapezoidal accumulation below works unchanged.

Notation: Barbastathis, Principles of Imaging, Ch. Dynamical Estimation.
  a_rev  <->  \\adjorev  (K,)
  G      —   gradient accumulator (P,), NOT part of the ODE state
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
def dyn_adjoint(J_fun, U_fun, w, t_sol, x_sol, t_data, resi,
                rtol=1e-8, atol=1e-10,
                h_fun=None, x0=None, x_tdata=None):
    """
    Solve the adjoint ODE backwards in time and accumulate the parameter
    gradient via explicit trapezoidal quadrature at RK45 solver steps.

    Standard mode (h_fun / x0 / x_tdata all None):
        x(T-tau) obtained from PCHIP interpolation of (t_sol, x_sol).

    Checkpointing mode (all three of h_fun, x0, x_tdata provided):
        On each backward sub-interval the forward ODE is re-solved from
        the stored checkpoint x(t_{n-1}), supplying x(t) to full RK45
        accuracy with no PCHIP bias on J or U.

    Parameters
    ----------
    J_fun  : callable (x, t, w) -> (K, K)   Jacobian  dh/dx
    U_fun  : callable (x, t, w) -> (K, P)   plant sensitivity  dh/dw
    w      : (P,)  current parameter vector
    t_sol  : (S,)  dense time grid from dyn_forward
    x_sol  : (K, S) full trajectory on t_sol
    t_data : (N,)  sampling times, t_data[-1] = T
    resi   : (K, N) residuals from dyn_residuals
    rtol, atol : solver tolerances

    Checkpointing-only
    ------------------
    h_fun  : callable (x, t, w) -> (K,)   plant operator
    x0     : (K,)  initial condition at t = 0
    x_tdata: (K, N) forward trajectory at sampling times (from dyn_forward)

    Returns
    -------
    a_tdata : (K, N)  adjoint a(t_n) just after the residual kick
    a_0     : (K,)   adjoint at t = 0  (IC gradient)
    grad_w  : (P,)   int_0^T U(x(t)).T @ a(t) dt  (parameter gradient)
    t_adj   : (S',)  adjoint time grid in forward time
    a_adj   : (K, S') adjoint trajectory, NaN-separated segments
    """
    w      = np.asarray(w,      dtype=float).ravel()
    t_data = np.asarray(t_data, dtype=float).ravel()
    resi   = np.asarray(resi,   dtype=float)
    x_sol  = np.asarray(x_sol,  dtype=float)

    K = x_sol.shape[0]
    P = len(w)
    N = len(t_data)
    T = float(t_data[-1])

    use_checkpointing = (h_fun is not None) and \
                        (x0 is not None) and \
                        (x_tdata is not None)

    # ------------------------------------------------------------------
    # Standard mode: global PCHIP interpolant over [0, T]
    # Accepts scalar or array tau; returns (K,) or (K, n_query).
    # ------------------------------------------------------------------
    if not use_checkpointing:
        _global_interp = PchipInterpolator(t_sol, x_sol.T)  # (S, K)
        def x_at_tau_global(tau):
            tau = np.asarray(tau, dtype=float)
            out = _global_interp(T - tau)       # (..., K)
            return out.T if out.ndim == 2 else out.ravel()

    # ------------------------------------------------------------------
    # Ascending-tau ordering
    # tau = T - t,  so t_N = T  =>  tau_N = 0  (first in ascending order)
    # ------------------------------------------------------------------
    tau_data          = T - t_data
    perm              = np.argsort(tau_data)
    tau_asc           = tau_data[perm]
    resi_tau          = resi[:, perm]          # (K, N) reordered

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    a_tau_asc  = np.zeros((K, N))
    tau_pieces = []      # list of (S_i,) arrays
    a_pieces   = []      # list of (K, S_i) arrays

    # ------------------------------------------------------------------
    # Main loop — Level 1: iterate over sub-intervals between checkpoints
    #
    # a_rev  (K,)  adjoint state, carried across sub-intervals
    # G      (P,)  gradient accumulator, updated by trapz (Level 3)
    #              NOT part of the ODE state
    # ------------------------------------------------------------------
    a_rev = np.zeros(K, dtype=float)
    G     = np.zeros(P, dtype=float)

    for i in range(N):

        # --- Level 1: inject residual impulse --------------------------
        a_rev          += resi_tau[:, i]
        a_tau_asc[:, i] = a_rev.copy()

        tau_start = tau_asc[i]
        tau_end   = tau_asc[i + 1] if i < N - 1 else T

        if tau_end <= tau_start + 10.0 * np.finfo(float).eps * T:
            tau_pieces.append(np.array([tau_start]))
            a_pieces.append(a_rev.reshape(K, 1))
            continue

        # --- Build x(t) supplier for this sub-interval ----------------
        if use_checkpointing:
            t_fwd_end = T - tau_start
            if i < N - 1:
                t_fwd_start = T - tau_end
                x_fwd_ic    = np.asarray(x_tdata[:, perm[i + 1]],
                                         dtype=float).ravel()
            else:
                t_fwd_start = 0.0
                x_fwd_ic    = np.asarray(x0, dtype=float).ravel()

            fwd = solve_ivp(
                lambda t, x: np.asarray(h_fun(x, t, w), dtype=float).ravel(),
                [t_fwd_start, t_fwd_end], x_fwd_ic,
                method='RK45', rtol=rtol, atol=atol, dense_output=False)
            _loc_interp = PchipInterpolator(fwd.t, fwd.y.T)  # (S_fwd, K)

            def x_at_tau_loc(tau,  # noqa: E704 – default-arg capture
                             _interp=_loc_interp, _T=T):
                tau = np.asarray(tau, dtype=float)
                out = _interp(_T - tau)
                return out.T if out.ndim == 2 else out.ravel()
        else:
            x_at_tau_loc = x_at_tau_global

        # --- Level 2: integrate adjoint ODE on [tau_start, tau_end] ---
        #
        #   d(a_rev)/dtau = J(x(T-tau), T-tau, w).T @ a_rev
        #
        #   ODE state is K-dimensional only.  RK45 step sizes are chosen
        #   solely by the a_rev error criterion — independent of Level 1.

        def _rhs_a(tau, a,
                   _xat=x_at_tau_loc, _J=J_fun, _w=w, _T=T, _K=K):
            x = np.asarray(_xat(tau), dtype=float).ravel()
            J = np.asarray(_J(x, _T - tau, _w), dtype=float).reshape(_K, _K)
            return J.T @ a

        seg = solve_ivp(_rhs_a, [tau_start, tau_end], a_rev.copy(),
                        method='RK45', rtol=rtol, atol=atol,
                        dense_output=False)
        #   seg.t: (S_i,),  seg.y: (K, S_i)
        a_rev = seg.y[:, -1].copy()

        # --- Level 3: trapezoidal quadrature for gradient integral -----
        #
        #   G += int_{tau_start}^{tau_end}
        #            U(x(T-tau), T-tau, w).T @ a_rev(tau) dtau
        #
        #   Nodes = every RK45 output point in seg.t.
        #   x_at_tau_loc called once with the full vector for efficiency.

        S_i      = seg.t.shape[0]
        tau_vec  = seg.t                           # (S_i,)
        x_nodes  = x_at_tau_loc(tau_vec)           # (K, S_i)
        a_nodes  = seg.y                           # (K, S_i)

        intg = np.zeros((P, S_i))
        for k in range(S_i):
            U_k = np.asarray(
                U_fun(x_nodes[:, k], T - tau_vec[k], w),
                dtype=float).reshape(K, P)
            intg[:, k] = U_k.T @ a_nodes[:, k]    # (P,)

        # np.trapezoid(y, x) integrates y along last axis using spacings x
        # y shape (P, S_i), x shape (S_i,)  =>  result (P,)
        if S_i >= 2:
            G += np.trapezoid(intg, tau_vec, axis=1)

        tau_pieces.append(tau_vec)                 # (S_i,)
        a_pieces.append(seg.y)                     # (K, S_i)

    # ------------------------------------------------------------------
    # Extract outputs
    # ------------------------------------------------------------------
    a_0    = a_rev.copy()
    grad_w = G.copy()

    a_tdata          = np.zeros((K, N))
    a_tdata[:, perm] = a_tau_asc

    # Assemble dense forward-time adjoint trajectory (NaN-separated)
    t_adj_segs = []
    a_adj_segs = []
    nan_col    = np.full((K, 1), np.nan)

    for i in range(N - 1, -1, -1):
        t_seg     = (T - tau_pieces[i])[::-1]      # (S_i,) ascending fwd time
        a_seg_fwd = a_pieces[i][:, ::-1]           # (K, S_i) forward order
        if i < N - 1:
            t_adj_segs.append(np.array([[np.nan]]))
            a_adj_segs.append(nan_col)
        t_adj_segs.append(t_seg.reshape(1, -1))
        a_adj_segs.append(a_seg_fwd)

    t_adj = (np.concatenate(t_adj_segs, axis=1).ravel()
             if t_adj_segs else np.array([]))
    a_adj = (np.concatenate(a_adj_segs, axis=1)
             if a_adj_segs else np.zeros((K, 0)))

    return a_tdata, a_0, grad_w, t_adj, a_adj


# ---------------------------------------------------------------------------
def dyn_update_adjoint(w, x0, alpha, beta, N, grad_w, a_0):
    """
    Gradient-descent update via the adjoint method.

    Applies:
        w_new  = w  + (alpha / N) * grad_w
        x0_new = x0 + beta * a_0

    grad_w is computed by dyn_adjoint via trapezoidal quadrature at every
    RK45 internal output node (Level-3 quadrature), NOT as a sum at the N
    data checkpoint times.  The quadrature density is governed by RK45's
    adaptive step size (Level 2), fully decoupled from N.

    Parameters
    ----------
    w      : (P,)  current parameter vector
    x0     : (K,)  current initial condition
    alpha  : (P,) or scalar  learning rates for w
    beta   : (K,) or scalar  learning rates for x0
    N      : int   number of data points (1/N normalisation)
    grad_w : (P,)  parameter gradient from dyn_adjoint output 3
    a_0    : (K,)  adjoint at t=0 from dyn_adjoint output 2

    Returns
    -------
    w_new  : (P,)  updated parameter vector
    x0_new : (K,)  updated initial condition

    Notation: Barbastathis, Principles of Imaging — eq. (dyn-update-L2-adjo)
      grad_w  =  int_0^T U(x(t)).T @ a(t) dt
      a_0     <->  \\adjom(0)
    """
    w      = np.asarray(w,      dtype=float).ravel()
    x0     = np.asarray(x0,     dtype=float).ravel()
    grad_w = np.asarray(grad_w, dtype=float).ravel()
    a_0    = np.asarray(a_0,    dtype=float).ravel()

    alpha = np.broadcast_to(np.asarray(alpha, dtype=float).ravel(), w.shape).copy()
    beta  = np.broadcast_to(np.asarray(beta,  dtype=float).ravel(), x0.shape).copy()

    w_new  = w  + (alpha / N) * grad_w
    x0_new = x0 + beta * a_0

    return w_new, x0_new
