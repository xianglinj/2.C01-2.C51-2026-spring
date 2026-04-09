"""
dyn_suite.estimation
--------------------
dyn_estimate_sensitivity — full gradient-descent loop, sensitivity method
dyn_estimate_adjoint     — full gradient-descent loop, adjoint method
"""

import numpy as np
from .forward     import dyn_forward, dyn_residuals
from .sensitivity import dyn_sensitivity, dyn_update_sensitivity
from .adjoint     import dyn_adjoint, dyn_update_adjoint


def _check_stop(loss_hist, m, E_frac_stop, W, stop_on_increase, verbose):
    """
    Windowed fractional-decrease stopping check.
    Returns True if the loop should break, False otherwise.
    """
    if E_frac_stop <= 0 or m < W:
        return False

    avg_curr = np.mean(loss_hist[m - W + 1 : m + 1])
    avg_prev = np.mean(loss_hist[m - W     : m    ])
    if avg_prev < np.finfo(float).eps:
        return True   # perfect fit

    frac = (avg_prev - avg_curr) / avg_prev
    is_increase = (frac < 0)

    if frac < E_frac_stop:
        if is_increase and not stop_on_increase:
            if verbose:
                print(f'  [Warning iter {m}: windowed loss increased '
                      f'(frac={frac:.2e}); continuing.]')
            return False
        else:
            if verbose:
                tag = 'INCREASED' if is_increase else 'converged'
                print(f'Early stop at iter {m} (windowed {tag}: frac={frac:.2e})')
            return True
    return False


# ---------------------------------------------------------------------------
def dyn_estimate_sensitivity(
        h_fun, J_fun, U_fun, U_ic_fun,
        w_init, x0_init, S0,
        t_data, x_data,
        alpha, beta,
        M, E_frac_stop=0, M_check_stop=1, stop_on_increase=False,
        rtol=1e-8, atol=1e-10, verbose=True):
    """
    Dynamical estimation via the sensitivity method.

    Parameters
    ----------
    h_fun, J_fun, U_fun : callables (x,t,w)
    U_ic_fun : callable or None
    w_init   : (P,)  initial guess
    x0_init  : (K,)  initial guess for IC
    S0       : (K,P) or None  initial sensitivity (None -> zeros)
    t_data   : (N,)  sampling times
    x_data   : (K,N) measurements
    alpha    : (P,) or scalar  learning rate for w
    beta     : (K,) or scalar  learning rate for x0
    M        : int   max iterations
    E_frac_stop  : float  fractional-decrease stopping threshold (0 = off)
    M_check_stop : int    window size for stopping check
    stop_on_increase : bool  stop if windowed loss increases
    rtol, atol : ODE tolerances
    verbose  : bool

    Returns
    -------
    w_hat     : (P,)
    x0_hat    : (K,)
    loss_hist : (m_stop+1,)  loss at each iteration (trimmed)
    """
    w  = np.asarray(w_init,  dtype=float).ravel()
    x0 = np.asarray(x0_init, dtype=float).ravel()
    N  = t_data.shape[-1] if hasattr(t_data, 'shape') else len(t_data)

    loss_hist = np.full(M + 1, np.nan)

    if verbose:
        hdr = f'--- Sensitivity method: max {M} iters'
        if E_frac_stop > 0:
            hdr += f',  E_frac_stop={E_frac_stop:.1e} (win={M_check_stop})'
        print(f'\n{hdr} ---')
        print(f'{"Iter":>6}  {"Loss":>14}')
        print('-' * 22)

    m_stop = M
    for m in range(M):
        (S_tdata, S_ic_tdata, x_tdata, *_) = dyn_sensitivity(
            h_fun, J_fun, U_fun, U_ic_fun, w, x0, S0, t_data, rtol, atol)

        resi, loss = dyn_residuals(x_tdata, x_data)
        loss_hist[m] = loss
        if verbose:
            print(f'{m:>6}  {loss:>14.6e}')

        if _check_stop(loss_hist, m, E_frac_stop, M_check_stop,
                       stop_on_increase, verbose):
            m_stop = m
            break

        w, x0 = dyn_update_sensitivity(w, x0, alpha, beta,
                                        S_tdata, S_ic_tdata, resi)
    else:
        m_stop = M

    # Final loss
    x_final, *_ = dyn_forward(h_fun, w, x0, t_data, rtol, atol)
    _, loss_final = dyn_residuals(x_final, x_data)
    loss_hist[m_stop] = loss_final

    if verbose:
        print(f'{m_stop:>6}  {loss_final:>14.6e}')
        print('-' * 22)
        print('Done.\n')

    return w, x0, loss_hist[:m_stop + 1]


# ---------------------------------------------------------------------------
def dyn_estimate_adjoint(
        h_fun, J_fun, U_fun,
        w_init, x0_init,
        t_data, x_data,
        alpha, beta,
        M, E_frac_stop=0, M_check_stop=1, stop_on_increase=False,
        rtol=1e-8, atol=1e-10, verbose=True):
    """
    Dynamical estimation via the adjoint method (with checkpointing).

    Parameters
    ----------
    h_fun, J_fun, U_fun : callables (x,t,w)
    w_init   : (P,)
    x0_init  : (K,)
    t_data   : (N,)
    x_data   : (K,N)
    alpha    : (P,) or scalar
    beta     : (K,) or scalar
    M        : int   max iterations
    E_frac_stop, M_check_stop, stop_on_increase : stopping options
    rtol, atol : ODE tolerances
    verbose  : bool

    Returns
    -------
    w_hat, x0_hat : (P,), (K,)
    loss_hist     : (m_stop+1,)
    """
    w  = np.asarray(w_init,  dtype=float).ravel()
    x0 = np.asarray(x0_init, dtype=float).ravel()
    t_data = np.asarray(t_data, dtype=float).ravel()
    N  = t_data.size

    loss_hist = np.full(M + 1, np.nan)

    if verbose:
        hdr = f'--- Adjoint method: max {M} iters'
        if E_frac_stop > 0:
            hdr += f',  E_frac_stop={E_frac_stop:.1e} (win={M_check_stop})'
        print(f'\n{hdr} ---')
        print(f'{"Iter":>6}  {"Loss":>14}')
        print('-' * 22)

    m_stop = M
    for m in range(M):
        x_tdata, t_sol, x_sol = dyn_forward(h_fun, w, x0, t_data, rtol, atol)
        resi, loss = dyn_residuals(x_tdata, x_data)
        loss_hist[m] = loss
        if verbose:
            print(f'{m:>6}  {loss:>14.6e}')

        if _check_stop(loss_hist, m, E_frac_stop, M_check_stop,
                       stop_on_increase, verbose):
            m_stop = m
            break

        a_tdata, a_0, grad_w, *_ = dyn_adjoint(
            J_fun, U_fun, w, t_sol, x_sol, t_data, resi, rtol, atol,
            h_fun, x0, x_tdata)   # checkpointing enabled

        w, x0 = dyn_update_adjoint(w, x0, alpha, beta, N, grad_w, a_0)
    else:
        m_stop = M

    x_final, *_ = dyn_forward(h_fun, w, x0, t_data, rtol, atol)
    _, loss_final = dyn_residuals(x_final, x_data)
    loss_hist[m_stop] = loss_final

    if verbose:
        print(f'{m_stop:>6}  {loss_final:>14.6e}')
        print('-' * 22)
        print('Done.\n')

    return w, x0, loss_hist[:m_stop + 1]
