"""
dyn_suite.sensitivity
---------------------
dyn_sensitivity        — solve the augmented (x, S, S_ic) ODE forward
dyn_update_sensitivity — gradient-descent update via sensitivity method
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
def dyn_sensitivity(h_fun, J_fun, U_fun, U_ic_fun,
                    w, x0, S0, t_data, rtol=1e-8, atol=1e-10):
    """
    Solve the sensitivity ODEs jointly with the forward ODE.

    Augmented state: z = [x (K), vec(S) (K*P), vec(S_ic) (K*K)]

        dx/dt     = h(x, t; w)
        dS/dt     = J * S + U
        dS_ic/dt  = J * S_ic + U_ic

    Parameters
    ----------
    h_fun, J_fun, U_fun : callables (x, t, w) -> (K,), (K,K), (K,P)
    U_ic_fun : callable (x, t, w) -> (K, K), or None  (None -> zero matrix)
    w        : (P,)  parameter vector
    x0       : (K,)  initial condition
    S0       : (K, P) or None  initial sensitivity dx0/dw  (None -> zeros)
    t_data   : (N,)  sampling times
    rtol, atol : ODE tolerances

    Returns
    -------
    S_tdata    : (K, P, N)  parameter sensitivity at sampling times
    S_ic_tdata : (K, K, N)  IC sensitivity at sampling times
    x_tdata    : (K, N)     state at sampling times
    t_sol      : (S,)       dense solver time grid
    x_sol      : (K, S)     full state trajectory
    S_sol      : (K, P, S)  full sensitivity trajectory  (optional)
    S_ic_sol   : (K, K, S)  full IC-sensitivity trajectory (optional)
    """
    w      = np.asarray(w,  dtype=float).ravel()
    x0     = np.asarray(x0, dtype=float).ravel()
    t_data = np.asarray(t_data, dtype=float).ravel()
    T  = float(t_data[-1])
    K  = x0.size
    P  = w.size

    if U_ic_fun is None:
        U_ic_fun = lambda x, t, w: np.zeros((K, K))
    if S0 is None:
        S0 = np.zeros((K, P))
    else:
        S0 = np.asarray(S0, dtype=float).reshape(K, P)

    Sic0 = np.eye(K)
    z0   = np.concatenate([x0, S0.ravel(), Sic0.ravel()])

    def rhs(t, z):
        x   = z[:K]
        S   = z[K:K+K*P].reshape(K, P)
        Sic = z[K+K*P:K+K*P+K*K].reshape(K, K)
        dx   = np.asarray(h_fun(x, t, w),       dtype=float).ravel()
        J    = np.asarray(J_fun(x, t, w),        dtype=float).reshape(K, K)
        U    = np.asarray(U_fun(x, t, w),        dtype=float).reshape(K, P)
        Uic  = np.asarray(U_ic_fun(x, t, w),    dtype=float).reshape(K, K)
        dS   = J @ S   + U
        dSic = J @ Sic + Uic
        return np.concatenate([dx, dS.ravel(), dSic.ravel()])

    sol = solve_ivp(rhs, [0.0, T], z0,
                    method='RK45', rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f'dyn_sensitivity: solver failed — {sol.message}')

    t_sol = sol.t    # (S_pts,)
    z_sol = sol.y    # (K*(1+P+K), S_pts)

    ix_x   = slice(0, K)
    ix_S   = slice(K, K + K*P)
    ix_Sic = slice(K + K*P, K + K*P + K*K)

    x_sol = z_sol[ix_x, :]   # (K, S_pts)

    # Extract at sampling times via PCHIP
    interp   = PchipInterpolator(t_sol, z_sol.T)    # (S_pts, dim)
    z_samp   = interp(t_data).T                      # (dim, N)

    N = t_data.size
    x_tdata    = z_samp[ix_x, :]                                     # (K, N)
    S_tdata    = z_samp[ix_S,   :].reshape(K, P, N)                  # (K, P, N)
    S_ic_tdata = z_samp[ix_Sic, :].reshape(K, K, N)                  # (K, K, N)

    # Full dense sensitivity arrays (for plotting)
    S_pts   = t_sol.size
    S_sol   = z_sol[ix_S,   :].reshape(K, P, S_pts)
    S_ic_sol = z_sol[ix_Sic, :].reshape(K, K, S_pts)

    return S_tdata, S_ic_tdata, x_tdata, t_sol, x_sol, S_sol, S_ic_sol


# ---------------------------------------------------------------------------
def dyn_update_sensitivity(w, x0, alpha, beta, S_tdata, S_ic_tdata, resi):
    """
    Gradient-descent update via the sensitivity method.

        w_new  = w  + (alpha / N) * sum_n  S_n.T @ resi_n
        x0_new = x0 + (beta  / N) * sum_n  S_ic_n.T @ resi_n

    Parameters
    ----------
    w, x0          : (P,), (K,)
    alpha, beta    : (P,), (K,)  learning rates (scalar OK)
    S_tdata        : (K, P, N)
    S_ic_tdata     : (K, K, N)
    resi           : (K, N)

    Returns
    -------
    w_new  : (P,)
    x0_new : (K,)
    """
    K, P, N = S_tdata.shape
    w   = np.asarray(w,   dtype=float).ravel()
    x0  = np.asarray(x0,  dtype=float).ravel()

    alpha = np.broadcast_to(np.atleast_1d(np.asarray(alpha, dtype=float)), (P,)).copy()
    beta  = np.broadcast_to(np.atleast_1d(np.asarray(beta,  dtype=float)), (K,)).copy()

    grad_w  = np.zeros(P)
    grad_x0 = np.zeros(K)
    for n in range(N):
        grad_w  += S_tdata[:, :, n].T    @ resi[:, n]
        grad_x0 += S_ic_tdata[:, :, n].T @ resi[:, n]

    w_new  = w  + (alpha / N) * grad_w
    x0_new = x0 + (beta  / N) * grad_x0
    return w_new, x0_new
