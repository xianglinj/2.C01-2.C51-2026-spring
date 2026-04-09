"""
dyn_suite.run
-------------
dyn_run — end-to-end dynamical estimation pipeline
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .forward     import dyn_forward, dyn_residuals
from .sensitivity import dyn_sensitivity, dyn_update_sensitivity
from .adjoint     import dyn_adjoint, dyn_update_adjoint
from .estimation  import dyn_estimate_sensitivity, dyn_estimate_adjoint
from .plotting    import (dyn_plot_forward, dyn_plot_sensitivity,
                          dyn_plot_adjoint, dyn_plot_residual_force,
                          dyn_plot_loss)


def dyn_run(
        h_fun, J_fun, U_fun, U_ic_fun,
        w_true, x0_true,
        w_init, x0_init, S0,
        t_data,
        alpha, beta,
        M, E_frac_stop=0, M_check_stop=1, stop_on_increase=False,
        save_path=None, rtol=1e-8, atol=1e-10,
        verbose=True, plot_opts=None):
    """
    End-to-end dynamical estimation pipeline.

    Steps
    -----
    1. Forward model  (ground truth + initial guess)
    2. Sensitivity dynamics at w_init
    3. Sensitivity estimation
    4. Adjoint dynamics at w_init (with checkpointing)
    5. Adjoint estimation
    6. Save results dict to <save_path>.pkl  (if save_path given)
    7. Verbose: all plots saved to <save_path>_figures/   (if verbose)
       LaTeX comparison table printed and saved to <save_path>.tex

    Parameters
    ----------
    h_fun, J_fun, U_fun : callables (x, t, w)
    U_ic_fun : callable or None
    w_true, x0_true     : (P,), (K,)  ground truth
    w_init, x0_init     : (P,), (K,)  initial guess
    S0       : (K, P) or None
    t_data   : (N,)   sampling times
    alpha    : (P,) or scalar  learning rate for w
    beta     : (K,) or scalar  learning rate for x0
    M        : int   max iterations
    E_frac_stop, M_check_stop, stop_on_increase : stopping options
    save_path : str or None  base path for output files (no extension)
    rtol, atol : ODE tolerances
    verbose  : bool
    plot_opts : dict with keys
        idx_state   — list of 0-based state indices to plot (up to 3)
        idx_sens    — list of (k, p) pairs for S     (up to 3, 0-based)
        idx_sens_ic — list of (k, p) pairs for S_ic  ([] to skip)
        idx_adj     — list of 0-based adjoint indices (up to 3)
        idx_resi    — list of 0-based residual indices (up to 3)
        log_scale   — bool  (default True)
        fig_offset  — int   first figure number (default 1)

    Returns
    -------
    results : dict  (all computed quantities)
    """
    w_true  = np.asarray(w_true,  dtype=float).ravel()
    x0_true = np.asarray(x0_true, dtype=float).ravel()
    w_init  = np.asarray(w_init,  dtype=float).ravel()
    x0_init = np.asarray(x0_init, dtype=float).ravel()
    t_data  = np.asarray(t_data,  dtype=float).ravel()

    # =====================================================================
    #  1. Forward model
    # =====================================================================
    print('\n=== DYN_RUN: Step 1 — Forward model ===')
    x_tdata_true, t_sol_true, x_sol_true = dyn_forward(
        h_fun, w_true, x0_true, t_data, rtol, atol)
    x_data = x_tdata_true.copy()   # noiseless; add noise here in future

    x_tdata_init, t_sol_init, x_sol_init = dyn_forward(
        h_fun, w_init, x0_init, t_data, rtol, atol)
    resi_init, loss_init = dyn_residuals(x_tdata_init, x_data)
    print(f'  Initial loss  E^[0] = {loss_init:.6e}')

    # =====================================================================
    #  2. Sensitivity dynamics at w_init
    # =====================================================================
    print('\n=== DYN_RUN: Step 2 — Sensitivity dynamics at w_init ===')
    (S_tdata_init, S_ic_tdata_init, _, t_sol_sens_init, _,
     S_sol_init, S_ic_sol_init) = dyn_sensitivity(
        h_fun, J_fun, U_fun, U_ic_fun,
        w_init, x0_init, S0, t_data, rtol, atol)

    # =====================================================================
    #  3. Sensitivity estimation
    # =====================================================================
    print('\n=== DYN_RUN: Step 3 — Sensitivity estimation ===')
    w_hat_s, x0_hat_s, loss_hist_s = dyn_estimate_sensitivity(
        h_fun, J_fun, U_fun, U_ic_fun,
        w_init, x0_init, S0,
        t_data, x_data,
        alpha, beta,
        M, E_frac_stop, M_check_stop, stop_on_increase,
        rtol, atol, verbose)
    x_tdata_s, t_sol_s, x_sol_s = dyn_forward(
        h_fun, w_hat_s, x0_hat_s, t_data, rtol, atol)

    # =====================================================================
    #  4. Adjoint dynamics at w_init
    # =====================================================================
    print('\n=== DYN_RUN: Step 4 — Adjoint dynamics at w_init ===')
    (a_tdata_init, a_0_init, _, t_adj_init, a_adj_init) = dyn_adjoint(
        J_fun, U_fun, w_init, t_sol_init, x_sol_init,
        t_data, resi_init, rtol, atol,
        h_fun, x0_init, x_tdata_init)

    # =====================================================================
    #  5. Adjoint estimation
    # =====================================================================
    print('\n=== DYN_RUN: Step 5 — Adjoint estimation ===')
    w_hat_a, x0_hat_a, loss_hist_a = dyn_estimate_adjoint(
        h_fun, J_fun, U_fun,
        w_init, x0_init,
        t_data, x_data,
        alpha, beta,
        M, E_frac_stop, M_check_stop, stop_on_increase,
        rtol, atol, verbose)
    x_tdata_a, t_sol_a, x_sol_a = dyn_forward(
        h_fun, w_hat_a, x0_hat_a, t_data, rtol, atol)

    # =====================================================================
    #  6. Collect results
    # =====================================================================
    results = dict(
        w_true=w_true, x0_true=x0_true,
        w_init=w_init, x0_init=x0_init,
        t_data=t_data, x_data=x_data,
        t_sol_true=t_sol_true, x_sol_true=x_sol_true,
        t_sol_init=t_sol_init, x_sol_init=x_sol_init,
        x_tdata_init=x_tdata_init,
        resi_init=resi_init, loss_init=loss_init,
        S_tdata_init=S_tdata_init, S_ic_tdata_init=S_ic_tdata_init,
        S_sol_init=S_sol_init, S_ic_sol_init=S_ic_sol_init,
        t_sol_sens_init=t_sol_sens_init,
        a_tdata_init=a_tdata_init, a_0_init=a_0_init,
        t_adj_init=t_adj_init, a_adj_init=a_adj_init,
        w_hat_s=w_hat_s, x0_hat_s=x0_hat_s,
        loss_hist_s=loss_hist_s,
        t_sol_s=t_sol_s, x_sol_s=x_sol_s, x_tdata_s=x_tdata_s,
        w_hat_a=w_hat_a, x0_hat_a=x0_hat_a,
        loss_hist_a=loss_hist_a,
        t_sol_a=t_sol_a, x_sol_a=x_sol_a, x_tdata_a=x_tdata_a,
    )

    # Save
    if save_path is not None:
        pkl_path = save_path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'\nResults saved to: {pkl_path}')

    if not verbose:
        return results

    # =====================================================================
    #  7. Verbose: plots + LaTeX table
    # =====================================================================
    if plot_opts is None:
        plot_opts = {}

    idx_state   = plot_opts.get('idx_state',   [0])
    idx_sens    = plot_opts.get('idx_sens',     [(0, 0)])
    idx_adj     = plot_opts.get('idx_adj',      [0])
    idx_resi    = plot_opts.get('idx_resi',     [0])
    idx_sens_ic = plot_opts.get('idx_sens_ic',  None)
    log_scale   = plot_opts.get('log_scale',    True)
    fn          = plot_opts.get('fig_offset',   1)

    # Set up figures directory
    save_figs = save_path is not None
    if save_figs:
        fig_dir  = os.path.join(os.path.dirname(save_path), 'figures')
        fig_base = os.path.basename(save_path)
        os.makedirs(fig_dir, exist_ok=True)
        print(f'Figures directory: {fig_dir}')

    def save_fig(fig, suffix):
        if not save_figs:
            return
        fpath = os.path.join(fig_dir, f'{fig_base}_{suffix}.png')
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'  Saved: {fig_base}_{suffix}.png')

    FS = 16

    fig = dyn_plot_forward(t_sol_true, x_sol_true, idx_state,
                           t_data, x_data, fig_num=fn)
    fig.axes[0].set_title('Ground-truth trajectory', fontsize=FS)
    save_fig(fig, 'forward_truth');  fn += 1

    fig = dyn_plot_forward(t_sol_init, x_sol_init, idx_state,
                           t_data, x_data, fig_num=fn)
    fig.axes[0].set_title('Initial-guess trajectory vs data', fontsize=FS)
    save_fig(fig, 'forward_init');  fn += 1

    fig = dyn_plot_sensitivity(t_sol_sens_init, S_sol_init, idx_sens,
                                fig_num=fn)
    fig.axes[0].set_title(r'Sensitivity $\mathbf{s}(t)$ at initial guess',
                           fontsize=FS)
    save_fig(fig, 'sensitivity');  fn += 1

    if idx_sens_ic:
        fig = dyn_plot_sensitivity(t_sol_sens_init, S_ic_sol_init,
                                    idx_sens_ic, fig_num=fn)
        fig.axes[0].set_title(r'IC sensitivity $\mathbf{s}_\mathrm{ic}(t)$',
                               fontsize=FS)
        save_fig(fig, 'sensitivity_ic');  fn += 1

    fig = dyn_plot_adjoint(t_adj_init, a_adj_init, idx_adj,
                            t_data, fig_num=fn)
    fig.axes[0].set_title(r'Adjoint $\mathbf{a}(t)$ at initial guess',
                           fontsize=FS)
    save_fig(fig, 'adjoint');  fn += 1

    fig = dyn_plot_residual_force(t_data, resi_init, idx_resi, fig_num=fn)
    fig.axes[0].set_title(r'Residual force $\mathbf{v}(t)$ at initial guess',
                           fontsize=FS)
    save_fig(fig, 'residual_force');  fn += 1

    fig = dyn_plot_loss(loss_hist_s, log_scale, fig_num=fn)
    fig.axes[0].set_title(r'Loss $\mathcal{E}^{[m]}$ — sensitivity method',
                           fontsize=FS)
    save_fig(fig, 'loss_sensitivity');  fn += 1

    fig = dyn_plot_loss(loss_hist_a, log_scale, fig_num=fn)
    fig.axes[0].set_title(r'Loss $\mathcal{E}^{[m]}$ — adjoint method',
                           fontsize=FS)
    save_fig(fig, 'loss_adjoint');  fn += 1

    fig = _plot_estimate_vs_truth(t_sol_s, x_sol_s, t_sol_true, x_sol_true,
                                   t_data, x_data, idx_state,
                                   'Sensitivity estimate vs ground truth', fn)
    save_fig(fig, 'estimate_sensitivity');  fn += 1

    fig = _plot_estimate_vs_truth(t_sol_a, x_sol_a, t_sol_true, x_sol_true,
                                   t_data, x_data, idx_state,
                                   'Adjoint estimate vs ground truth', fn)
    save_fig(fig, 'estimate_adjoint');  fn += 1

    # LaTeX table
    latex_str = _make_latex_table(w_true, x0_true,
                                   w_hat_s, x0_hat_s,
                                   w_hat_a, x0_hat_a)
    print(f'\n{latex_str}')
    if save_path is not None:
        tex_path = save_path + '.tex'
        with open(tex_path, 'w') as f:
            f.write(latex_str + '\n')
        print(f'LaTeX table saved to: {tex_path}')

    return results


# ---------------------------------------------------------------------------
#  Local helpers
# ---------------------------------------------------------------------------
def _plot_estimate_vs_truth(t_est, x_est, t_true, x_true,
                              t_data, x_data, indices, title, fig_num):
    from .plotting import COLORS, LSTYLES, MARKERS, LW, FS, _new_fig, _apply_style
    indices = list(indices)[:3]
    fig, ax = _new_fig(fig_num)

    handles_est  = []
    handles_true = []
    for i, k in enumerate(indices):
        h, = ax.plot(t_est, x_est[k, :],
                     color=COLORS[i], linestyle=LSTYLES[i], linewidth=LW,
                     label=fr'$\hat{{x}}_{{{k+1}}}(t)$')
        handles_est.append(h)
        ax.plot(t_data, x_data[k, :],
                linestyle='none', marker=MARKERS[i],
                color=COLORS[i], markerfacecolor=COLORS[i], markersize=6)
        ht, = ax.plot(t_true, x_true[k, :],
                      color=(0.55, 0.55, 0.55),
                      linestyle=LSTYLES[i], linewidth=0.8,
                      label=fr'$x_{{{k+1}}}^\mathrm{{true}}(t)$')
        handles_true.append(ht)

    ax.legend(handles=handles_est + handles_true, fontsize=FS, loc='best')
    ax.set_title(title, fontsize=FS)
    _apply_style(ax, '$t$', r'$\mathbf{x}(t)$')
    fig.tight_layout()
    return fig


def _make_latex_table(w_true, x0_true, w_hat_s, x0_hat_s, w_hat_a, x0_hat_a):
    lines = [
        r'\begin{tabular}{c r r r}',
        r'\toprule',
        r'Parameter & Ground truth & Sensitivity & Adjoint \\',
        r'\midrule',
    ]
    for p, (wt, ws, wa) in enumerate(zip(w_true, w_hat_s, w_hat_a)):
        lines.append(fr'$w_{{{p+1}}}$ & ${wt:.6g}$ & ${ws:.6g}$ & ${wa:.6g}$ \\')
    if len(x0_true) > 0:
        lines.append(r'[0.5ex]')
    for k, (xt, xs, xa) in enumerate(zip(x0_true, x0_hat_s, x0_hat_a)):
        lines.append(
            fr'$x_{{0,{k+1}}}$ & ${xt:.6g}$ & ${xs:.6g}$ & ${xa:.6g}$ \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)
