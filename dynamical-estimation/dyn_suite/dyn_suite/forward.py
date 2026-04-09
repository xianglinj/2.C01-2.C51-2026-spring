"""
dyn_suite.forward
-----------------
dyn_forward    — solve the forward ODE, store all internal steps
dyn_residuals  — compute residuals and L2 loss
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


def dyn_forward(h_fun, w, x0, t_data, rtol=1e-8, atol=1e-10):
    """
    Solve the forward ODE  dx/dt = h(x, t; w),  x(0) = x0.

    Parameters
    ----------
    h_fun  : callable (x, t, w) -> (K,)   plant operator
    w      : (P,)  parameter vector
    x0     : (K,)  initial condition
    t_data : (N,)  sampling times  t_1 <= ... <= t_N = T
    rtol, atol : ODE solver tolerances (default 1e-8 / 1e-10)

    Returns
    -------
    x_tdata : (K, N)  trajectory at each sampling time
    t_sol   : (S,)    dense solver time grid (all internal RK45 steps)
    x_sol   : (K, S)  full trajectory on t_sol

    Notes
    -----
    solve_ivp is called with no t_eval so that ALL internal stepper
    points are retained in t_sol / x_sol.  This dense grid is used by
    dyn_adjoint for accurate PCHIP interpolation during the backward solve.
    Notation: Barbastathis, Principles of Imaging, Ch. Dynamical Estimation.
    """
    w      = np.asarray(w,      dtype=float).ravel()
    x0     = np.asarray(x0,     dtype=float).ravel()
    t_data = np.asarray(t_data, dtype=float).ravel()
    T = float(t_data[-1])

    sol = solve_ivp(
        lambda t, x: np.asarray(h_fun(x, t, w), dtype=float).ravel(),
        [0.0, T], x0,
        method='RK45', rtol=rtol, atol=atol,
        dense_output=False   # all internal steps returned when t_eval is None
    )
    if not sol.success:
        raise RuntimeError(f'dyn_forward: solver failed — {sol.message}')

    t_sol = sol.t        # (S,)
    x_sol = sol.y        # (K, S)

    # Extract at sampling times via PCHIP interpolation on the dense grid
    interp  = PchipInterpolator(t_sol, x_sol.T)   # x_sol.T is (S, K)
    x_tdata = interp(t_data).T                     # (K, N)

    return x_tdata, t_sol, x_sol


def dyn_residuals(x_tdata, x_data):
    """
    Compute residuals  resi = x_data - x_tdata  and the L2 loss.

    Parameters
    ----------
    x_tdata : (K, N)  model predictions from dyn_forward
    x_data  : (K, N)  measurements

    Returns
    -------
    resi : (K, N)  residual matrix; column n is epsilon^[n]
    loss : float   (1/2N) * ||resi||_F^2
    """
    x_tdata = np.asarray(x_tdata, dtype=float)
    x_data  = np.asarray(x_data,  dtype=float)
    if x_tdata.shape != x_data.shape:
        raise ValueError(
            f'dyn_residuals: shape mismatch {x_tdata.shape} vs {x_data.shape}')
    resi = x_data - x_tdata
    loss = np.sum(resi**2) / (2 * x_data.shape[1])
    return resi, float(loss)
