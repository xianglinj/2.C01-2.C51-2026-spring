#!/usr/bin/env python3
"""
malthus_run.py
Malthusian population growth: analytical sensitivity + adjoint estimation.
No ODE solver -- all solutions are derived analytically.

Figures produced:
  1. Forward solution x(t), noisy data, and +/-5% parameter perturbations
  2. Sensitivities s(t) = dx/dw and s_ic(t) = dx/dx0
  3. Adjoint a(t) at m=0 with residual-force impulses
  4. Both loss functions vs iteration (semilog)
  5. Fitted trajectory overlay

Sign convention: eps_n = x_n - x(t_n)  [data minus model].
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Global style (mirrors MATLAB: font 16, LaTeX strings, grid on)
# --------------------------------------------------------------------------
plt.rcParams.update({
    'font.size'        : 16,
    'axes.labelsize'   : 16,
    'axes.titlesize'   : 16,
    'legend.fontsize'  : 13,
    'xtick.labelsize'  : 14,
    'ytick.labelsize'  : 14,
    'axes.grid'        : True,
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.4,
    'lines.linewidth'  : 2,
})
DPI = 150

os.makedirs('figures', exist_ok=True)
os.makedirs('data',    exist_ok=True)
os.makedirs('latex',   exist_ok=True)

# ==========================================================================
# Local function: analytical adjoint
# ==========================================================================
def adjoint_analytical(t_eval, w, resid, t_d):
    """
    Evaluate the Malthus adjoint a(t) analytically.

    Piecewise formula (eps = data - model):
        a(t) = -sum_{j=n}^{N} eps_j * exp(-w*(t - t_j))
        for  t_{n-1} < t <= t_n,  n = N,...,1  (t_0 = 0).

    Parameters
    ----------
    t_eval : float or array (K,)
    w      : float
    resid  : array (N,)
    t_d    : array (N,)   checkpoint times, sorted ascending

    Returns
    -------
    a_vals : array (K,)
    """
    N      = len(t_d)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float)).ravel()
    a_vals = np.zeros_like(t_eval)
    t_nodes = np.concatenate(([0.0], t_d))          # [t_0, t_1, ..., t_N]

    for k, t in enumerate(t_eval):
        # searchsorted('left') -> first index i where t_nodes[i] >= t
        # that index is the 1-based interval number n
        i = np.searchsorted(t_nodes, t, side='left')
        n = int(np.clip(i, 1, N))                   # interval 1..N
        # sum over future checkpoints j = n..N  (0-based: n-1..N-1)
        js = np.arange(n - 1, N)
        a_vals[k] = -np.sum(resid[js] * np.exp(-w * (t - t_d[js])))
    return a_vals

# ==========================================================================
## 0.  Ground truth and synthetic data
# ==========================================================================
w_true  = 1.5
x0_true = 2.0
T       = 3.0
N       = 20
sigma   = 0.3

rng    = np.random.default_rng(42)
t_d    = np.linspace(T / N, T, N)
x_clean = x0_true * np.exp(w_true * t_d)
x_data  = x_clean + sigma * rng.standard_normal(N)

t_fine  = np.linspace(0, T, 800)

# ==========================================================================
## 1.  Figure 1 -- Forward solution with +/-5% parameter perturbations
# ==========================================================================
p_err   = 0.05
x_w_1p  = x0_true * np.exp((1 + p_err) * w_true * t_d)
x_x0_1p = (1 - p_err) * x0_true * np.exp(w_true * t_d)

fig1, ax1 = plt.subplots(figsize=(7.9, 5.0))
ax1.plot(t_fine, x0_true * np.exp(w_true * t_fine), 'b-',  label=r'$x_0 e^{wt}$ (true)')
ax1.plot(t_d,    x_data,  'ko', ms=7, mfc='w', lw=1.5,     label=r'Data $x_n$')
ax1.plot(t_d,    x_w_1p,  'b--',                            label=r'$+5\%$ deviation in $w$')
ax1.plot(t_d,    x_x0_1p, 'b:',                             label=r'$-5\%$ deviation in $x_0$')
ax1.set_xlabel(r'Time $t$')
ax1.set_ylabel(r'Population $x(t)$')
ax1.set_title('Malthusian growth: trajectory and data')
ax1.legend(loc='upper left', fontsize=14)
fig1.tight_layout()
fig1.savefig(os.path.join('figures', 'malthus_1_forward.png'), dpi=DPI)
plt.close(fig1)

# ==========================================================================
## 2.  Figure 2 -- Sensitivities (at true parameters)
# ==========================================================================
s_fine   = x0_true * t_fine * np.exp(w_true * t_fine)
sic_fine = np.exp(w_true * t_fine)

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax2a.plot(t_fine, s_fine,   'r-')
ax2a.set_xlabel(r'Time $t$')
ax2a.set_ylabel(r'$s(t) = x_0\,t\,e^{wt}$')
ax2a.set_title(r'Sensitivity $\partial x/\partial w$')

ax2b.plot(t_fine, sic_fine, 'm-')
ax2b.set_xlabel(r'Time $t$')
ax2b.set_ylabel(r'$s_{\rm ic}(t) = e^{wt}$')
ax2b.set_title(r'Sensitivity $\partial x/\partial x_0$')

fig2.tight_layout()
fig2.savefig(os.path.join('figures', 'malthus_2_sensitivity.png'), dpi=DPI)
plt.close(fig2)

# ==========================================================================
## 3.  Figure 3 -- Adjoint a(t) at m = 0
# ==========================================================================
w_init  = 0.8
x0_init = 1.0

resid_0 = x_data - x0_init * np.exp(w_init * t_d)
a_fine0 = adjoint_analytical(t_fine, w_init, resid_0, t_d)

fig3, ax3 = plt.subplots(figsize=(7.9, 5.0))
t_nodes = np.concatenate(([0.0], t_d))
first_seg = True
for n in range(N):
    mask = (t_fine >= t_nodes[n]) & (t_fine <= t_nodes[n + 1])
    lbl  = r'$a(t)$ (piecewise analytical)' if first_seg else '_nolegend_'
    ax3.plot(t_fine[mask], a_fine0[mask], 'b-', lw=1.8, label=lbl)
    first_seg = False
markerline, stemlines, baseline = ax3.stem(
    t_d, -resid_0, linefmt='k-', markerfmt='ko', basefmt='k--')
markerline.set_markersize(5)
markerline.set_label(r'$-\varepsilon_n$ (impulses)')
ax3.axhline(0, color='k', ls='--', lw=0.8)
ax3.set_xlabel(r'Time $t$')
ax3.set_ylabel(r'$a(t)$')
ax3.set_title(r'Adjoint $a(t)$ at $m=0$')
ax3.legend(loc='lower right', fontsize=13)
fig3.tight_layout()
fig3.savefig(os.path.join('figures', 'malthus_3_adjoint.png'), dpi=DPI)
plt.close(fig3)

# ==========================================================================
## 4.  Sensitivity estimation loop
# ==========================================================================
alpha_w  = 2e-6
alpha_x0 = 8e-6
M        = 3000

w_sens   = w_init
x0_sens  = x0_init
loss_sens = np.zeros(M)

for m in range(M):
    x_mod = x0_sens * np.exp(w_sens * t_d)
    resid = x_data - x_mod
    loss_sens[m] = 0.5 * np.mean(resid**2)

    s_d   = x0_sens * t_d * np.exp(w_sens * t_d)   # s(t_n) = dx/dw
    sic_d = np.exp(w_sens * t_d)                    # s_ic(t_n) = dx/dx0

    w_sens  += alpha_w  * np.mean(resid * s_d)
    x0_sens += alpha_x0 * np.mean(resid * sic_d)

print('\n--- Sensitivity estimation ---')
print(f'True:     w = {w_true:.4f},   x0 = {x0_true:.4f}')
print(f'Estimate: w = {w_sens:.6f},   x0 = {x0_sens:.6f}')

# ==========================================================================
## 5.  Adjoint estimation loop
# ==========================================================================
# (T/N)*mean(a(t_n)*u(t_n)) = dE/dw, magnitudes match sensitivity,
# so alpha_w_adj = alpha_w.  a(0)/N = dE/dx0, so alpha_x0_adj = alpha_x0.
alpha_w_adj  = alpha_w
alpha_x0_adj = alpha_x0

w_adj   = w_init
x0_adj  = x0_init
loss_adj = np.zeros(M)

for m in range(M):
    x_mod  = x0_adj * np.exp(w_adj * t_d)
    resid  = x_data - x_mod
    loss_adj[m] = 0.5 * np.mean(resid**2)

    a_at_d = adjoint_analytical(t_d,   w_adj, resid, t_d)
    a0     = adjoint_analytical(1e-12, w_adj, resid, t_d)[0]

    gw_a  = (T / N) * np.mean(a_at_d * x_mod)   # = dE/dw
    gx0_a = a0 / N                                # = dE/dx0

    w_adj  -= alpha_w_adj  * gw_a
    x0_adj -= alpha_x0_adj * gx0_a

print('\n--- Adjoint estimation ---')
print(f'True:     w = {w_true:.4f},   x0 = {x0_true:.4f}')
print(f'Estimate: w = {w_adj:.6f},   x0 = {x0_adj:.6f}')

# ==========================================================================
## 6.  Figure 4 -- Both loss functions on the same semilog axes
# ==========================================================================
iters = np.arange(1, M + 1)
fig4, ax4 = plt.subplots(figsize=(7.9, 5.0))
ax4.semilogy(iters, loss_sens, 'b-',  label='Sensitivity')
ax4.semilogy(iters, loss_adj,  'r--', label='Adjoint')
ax4.set_xlabel(r'Iteration $m$')
ax4.set_ylabel(r'Loss $\mathcal{E}^{[m]}$')
ax4.set_title(r'Convergence: sensitivity vs.\ adjoint')
ax4.legend(loc='upper right')
fig4.tight_layout()
fig4.savefig(os.path.join('figures', 'malthus_4_loss.png'), dpi=DPI)
plt.close(fig4)

# ==========================================================================
## 7.  Figure 5 -- Fitted trajectory overlay
# ==========================================================================
fig5, ax5 = plt.subplots(figsize=(9.4, 5.2))
ax5.plot(t_fine, x0_true * np.exp(w_true * t_fine), 'b-',  label='True')
ax5.plot(t_fine, x0_init * np.exp(w_init * t_fine), 'r--', label='Initial guess')
ax5.plot(t_fine, x0_sens * np.exp(w_sens * t_fine), 'g-',  label='Sensitivity est.')
ax5.plot(t_fine, x0_adj  * np.exp(w_adj  * t_fine), 'm--', label='Adjoint est.')
ax5.plot(t_d,    x_data, 'ko', ms=6, mfc='w', lw=1.5,      label='Data')
ax5.set_xlabel(r'Time $t$')
ax5.set_ylabel(r'Population $x(t)$')
ax5.set_title('Fitted trajectories')
ax5.legend(loc='upper left', fontsize=13)
fig5.tight_layout()
fig5.savefig(os.path.join('figures', 'malthus_5_fitted.png'), dpi=DPI)
plt.close(fig5)

# ==========================================================================
## 8.  Save data to data/malthus_results.npz
# ==========================================================================
np.savez(os.path.join('data', 'malthus_results.npz'),
         t_d=t_d, x_data=x_data, x_clean=x_clean, t_fine=t_fine,
         w_true=w_true, x0_true=x0_true, T=T, N=N, sigma=sigma,
         w_init=w_init, x0_init=x0_init,
         w_sens=w_sens,  x0_sens=x0_sens,  loss_sens=loss_sens,
         w_adj=w_adj,    x0_adj=x0_adj,    loss_adj=loss_adj)
print('\nData saved to data/malthus_results.npz')

# ==========================================================================
## 9.  LaTeX comparison table -> latex/malthus_table.tex
# ==========================================================================
with open(os.path.join('latex', 'malthus_table.tex'), 'w') as fid:
    fid.write('%% Malthus estimation results -- auto-generated by malthus_run.py\n')
    fid.write('\\begin{table}[htbp]\n')
    fid.write('  \\centering\n')
    fid.write('  \\caption{Malthusian parameter estimation: ground truth vs.\\ estimates.}\n')
    fid.write('  \\label{tab:malthus-estimates}\n')
    fid.write('  \\begin{tabular}{@{}lccc@{}}\n')
    fid.write('    \\toprule\n')
    fid.write('    Parameter & Ground truth & Sensitivity & Adjoint \\\\\n')
    fid.write('    \\midrule\n')
    fid.write(f'    $w$   & ${w_true:.4f}$ & ${w_sens:.6f}$ & ${w_adj:.6f}$ \\\\\n')
    fid.write(f'    $x_0$ & ${x0_true:.4f}$ & ${x0_sens:.6f}$ & ${x0_adj:.6f}$ \\\\\n')
    fid.write('    \\bottomrule\n')
    fid.write('  \\end{tabular}\n')
    fid.write('\\end{table}\n')
print('LaTeX table saved to latex/malthus_table.tex')
