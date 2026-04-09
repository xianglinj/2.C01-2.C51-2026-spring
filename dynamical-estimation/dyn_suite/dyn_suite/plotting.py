"""
dyn_suite.plotting
------------------
dyn_plot_forward        — forward trajectory + optional data overlay
dyn_plot_sensitivity    — selected entries of S(t) or S_ic(t)
dyn_plot_adjoint        — adjoint a(t) with jump gaps at impulse times
dyn_plot_residual_force — stem plot of the residual force v(t)
dyn_plot_loss           — loss history vs iteration

Style conventions (matching the MATLAB suite):
    Component 1: solid red       Component 2: dashed blue
    Component 3: dotted dark-green
    linewidth = 2,  fontsize = 16,  grid on
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------
FS = 16          # font size
LW = 2           # line width
COLORS  = [(0.80, 0.0, 0.0),   (0.0, 0.0, 0.80),   (0.0, 0.55, 0.0)]
LSTYLES = ['-',                 '--',                ':']
MARKERS = ['o',                 's',                 'd']


def _apply_style(ax, xlabel, ylabel):
    """Apply shared style to a matplotlib Axes."""
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.grid(True)
    ax.set_xlim(ax.get_xlim())   # freeze auto limits
    for spine in ax.spines.values():
        spine.set_linewidth(1)


def _new_fig(fig_num=None):
    if fig_num is not None:
        fig = plt.figure(fig_num)
        fig.clf()
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax


# ---------------------------------------------------------------------------
def dyn_plot_forward(t_sol, x_sol, indices,
                     t_data=None, x_data=None,
                     labels=None, fig_num=None):
    """
    Plot selected components of the forward trajectory.

    Parameters
    ----------
    t_sol   : (S,)   dense time grid
    x_sol   : (K, S) full trajectory
    indices : list of int  state components to plot (up to 3, 0-based)
    t_data  : (N,)  sampling times for data overlay  (optional)
    x_data  : (K,N) measurements                     (optional)
    labels  : list of str  legend entries             (auto if None)
    fig_num : int  figure number                      (new figure if None)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    indices = list(indices)[:3]
    L = len(indices)
    if labels is None:
        labels = [fr'$x_{{{k+1}}}(t)$' for k in indices]

    fig, ax = _new_fig(fig_num)
    have_data = (t_data is not None) and (x_data is not None)

    handles = []
    for i, k in enumerate(indices):
        h, = ax.plot(t_sol, x_sol[k, :],
                     color=COLORS[i], linestyle=LSTYLES[i], linewidth=LW,
                     label=labels[i])
        handles.append(h)
        if have_data:
            ax.plot(t_data, x_data[k, :],
                    linestyle='none', marker=MARKERS[i],
                    color=COLORS[i], markerfacecolor=COLORS[i], markersize=6)

    ax.legend(handles=handles, fontsize=FS, loc='best')
    _apply_style(ax, '$t$', r'$\mathbf{x}(t)$')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def dyn_plot_sensitivity(t_sol, S_sol, indices,
                         labels=None, fig_num=None):
    """
    Plot selected entries of a sensitivity tensor.

    Parameters
    ----------
    t_sol   : (S,)       dense time grid (from dyn_sensitivity output 4)
    S_sol   : (K, M, S)  full sensitivity (output 6 or 7 of dyn_sensitivity)
    indices : list of (k, p)  0-based [row, col] pairs (up to 3)
    labels  : list of str    (auto if None)
    fig_num : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    indices = [tuple(idx) for idx in indices][:3]
    L = len(indices)
    if labels is None:
        labels = [fr'$s_{{{k+1},{p+1}}}(t)$' for k, p in indices]

    fig, ax = _new_fig(fig_num)
    handles = []
    for i, (k, p) in enumerate(indices):
        s_kp = S_sol[k, p, :]
        h, = ax.plot(t_sol, s_kp,
                     color=COLORS[i], linestyle=LSTYLES[i], linewidth=LW,
                     label=labels[i])
        handles.append(h)

    ax.legend(handles=handles, fontsize=FS, loc='best')
    _apply_style(ax, '$t$', r'$\mathbf{s}(t)$')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def dyn_plot_adjoint(t_adj, a_adj, indices,
                     t_data=None, labels=None, fig_num=None):
    """
    Plot selected components of the adjoint trajectory.

    NaN entries in t_adj / a_adj create visible gaps at jump points.
    Thin grey vertical lines mark sampling times when t_data is supplied.

    Parameters
    ----------
    t_adj   : (S,)   adjoint time grid (NaN-separated, from dyn_adjoint output 4)
    a_adj   : (K, S) adjoint trajectory (NaN-separated, output 5)
    indices : list of int  0-based  (up to 3)
    t_data  : (N,)   sampling times for vertical markers  (optional)
    labels  : list of str   (auto if None)
    fig_num : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    indices = list(indices)[:3]
    if labels is None:
        labels = [fr'$a_{{{k+1}}}(t)$' for k in indices]

    fig, ax = _new_fig(fig_num)

    if t_data is not None:
        for tn in t_data:
            ax.axvline(tn, color=(0.65, 0.65, 0.65), linestyle='--', linewidth=0.8)

    handles = []
    for i, k in enumerate(indices):
        h, = ax.plot(t_adj, a_adj[k, :],
                     color=COLORS[i], linestyle=LSTYLES[i], linewidth=LW,
                     label=labels[i])
        handles.append(h)

    ax.legend(handles=handles, fontsize=FS, loc='best')
    _apply_style(ax, '$t$', r'$\mathbf{a}(t)$')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def dyn_plot_residual_force(t_data, resi, indices,
                             labels=None, fig_num=None):
    """
    Stem plot of the residual force  v(t) = sum_n resi_n * delta(t - t_n).

    Parameters
    ----------
    t_data  : (N,)   sampling times
    resi    : (K, N) residuals from dyn_residuals
    indices : list of int  0-based  (up to 3)
    labels  : list of str  (auto if None)
    fig_num : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    indices = list(indices)[:3]
    if labels is None:
        labels = [fr'$v_{{{k+1}}}(t)$' for k in indices]

    fig, ax = _new_fig(fig_num)
    handles = []
    for i, k in enumerate(indices):
        col = COLORS[i]
        mk  = MARKERS[i]
        st  = ax.stem(t_data, resi[k, :],
                      linefmt='-', markerfmt=mk,
                      basefmt=' ')
        # Apply color manually (stem API changed across matplotlib versions)
        plt.setp(st.markerline, color=col, marker=mk, markersize=7,
                 markerfacecolor=col)
        plt.setp(st.stemlines, color=col, linewidth=LW)
        # Invisible proxy for the legend
        h, = ax.plot([], [], color=col, linestyle=LSTYLES[i],
                     linewidth=LW, marker=mk, label=labels[i])
        handles.append(h)

    ax.axhline(0, color=(0.4, 0.4, 0.4), linewidth=0.8)
    ax.legend(handles=handles, fontsize=FS, loc='best')
    _apply_style(ax, '$t$', r'$\mathbf{v}(t)$')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def dyn_plot_loss(loss_hist, log_scale=True, fig_num=None):
    """
    Plot the loss history  E^[m]  vs iteration m.

    Parameters
    ----------
    loss_hist : (m_stop+1,)  from dyn_estimate_* output 3
    log_scale : bool   log y-axis (default True)
    fig_num   : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    loss_hist = np.asarray(loss_hist, dtype=float).ravel()
    m_vals    = np.arange(loss_hist.size)

    fig, ax = _new_fig(fig_num)
    plot_fn = ax.semilogy if log_scale else ax.plot
    plot_fn(m_vals, loss_hist, '-', color=COLORS[0], linewidth=LW)

    ax.set_xlim(m_vals[0], m_vals[-1])
    _apply_style(ax, 'Iteration $m$', r'$\mathcal{E}^{[m]}$')
    fig.tight_layout()
    return fig
