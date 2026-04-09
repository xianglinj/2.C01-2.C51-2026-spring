"""
lv_test.py — Python test driver for the dyn_suite on the Lotka-Volterra system.

Run from the py_suite directory:
    python lv_test.py

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from lv_system import lv_h, lv_J, lv_U
from dyn_suite import dyn_run, dyn_estimate_adjoint
from dyn_suite.plotting import dyn_plot_loss

# ============================================================
#  System functions
# ============================================================
h_fun    = lv_h
J_fun    = lv_J
U_fun    = lv_U
U_ic_fun = None

# ============================================================
#  Ground truth
# ============================================================
w_true  = np.array([1.0, 1.0, 0.7, 0.7])   # [w11, w13, w21, w23]
x0_true = np.array([2.0, 1.0])

x1eq   = w_true[2] / w_true[3]
x2eq   = w_true[0] / w_true[1]
T_osc  = 2 * np.pi / np.sqrt(w_true[0] * w_true[2])
print(f'Equilibrium:    (x1*, x2*) = ({x1eq:.3f}, {x2eq:.3f})')
print(f'Approx. period: T_osc     = {T_osc:.3f}  time units')

# ============================================================
#  Initial guess
# ============================================================
w_init  = np.array([0.5, 1.5, 1.0, 0.4])
x0_init = x0_true.copy()
S0      = None

# ============================================================
#  Sampling times (~2.5 oscillation periods)
# ============================================================
t_data = np.linspace(0.5, 20.0, 120)

# ============================================================
#  Optimization settings
# ============================================================
alpha = 0.008 * np.ones(4)
beta  = np.zeros(2)         # IC is known; do not update x0

M                = 600
E_frac_stop      = 1e-6
M_check_stop     = 15
stop_on_increase = False

# ============================================================
#  ODE tolerances
# ============================================================
rtol = 1e-8
atol = 1e-10

# ============================================================
#  Plot options  (0-based indices)
# ============================================================
plot_opts = dict(
    idx_state   = [0, 1],
    idx_sens    = [(0, 0), (0, 1), (1, 3)],  # S_{1,w11}, S_{1,w13}, S_{2,w23}
    idx_sens_ic = None,
    idx_adj     = [0, 1],
    idx_resi    = [0, 1],
    log_scale   = True,
    fig_offset  = 1,
)

# ============================================================
#  RUN
# ============================================================
results = dyn_run(
    h_fun, J_fun, U_fun, U_ic_fun,
    w_true, x0_true,
    w_init, x0_init, S0,
    t_data,
    alpha, beta,
    M, E_frac_stop, M_check_stop, stop_on_increase,
    save_path='lv_results',
    rtol=rtol, atol=atol,
    verbose=True,
    plot_opts=plot_opts,
)

# ============================================================
#  Warm-start adjoint (from sensitivity estimate)
# ============================================================
print('\n=== Warm-start adjoint (from sensitivity estimate) ===')

w_hat_aw, x0_hat_aw, loss_hist_aw = dyn_estimate_adjoint(
    h_fun, J_fun, U_fun,
    results['w_hat_s'], results['x0_hat_s'],   # <-- sensitivity as IC
    t_data, results['x_data'],
    alpha, beta,
    M, E_frac_stop, M_check_stop, stop_on_increase,
    rtol=rtol, atol=atol, verbose=True)

# Loss plot for warm-start adjoint
fig_ws = dyn_plot_loss(loss_hist_aw, log_scale=True)
fig_ws.axes[0].set_title(r'Loss $\mathcal{E}^{[m]}$ — adjoint (warm start)',
                          fontsize=16)
fig_ws.savefig('figures/lv_results_loss_adjoint_warm.png', dpi=300,
               bbox_inches='tight')

# ============================================================
#  Summary table
# ============================================================
names = ['w_{11}', 'w_{13}', 'w_{21}', 'w_{23}']
print(f'\n{"="*62}')
print(f'{"Parameter":<20}  {"True":>8}  {"Sens.":>10}  '
      f'{"Adj.(cold)":>10}  {"Adj.(warm)":>10}')
print(f'{"-"*62}')
for p, name in enumerate(names):
    print(f'{name:<20}  {w_true[p]:>8.5f}  '
          f'{results["w_hat_s"][p]:>10.5f}  '
          f'{results["w_hat_a"][p]:>10.5f}  '
          f'{w_hat_aw[p]:>10.5f}')
print(f'{"="*62}')
print(f'Final loss (sens.)      : {results["loss_hist_s"][-1]:.4e}')
print(f'Final loss (adj. cold)  : {results["loss_hist_a"][-1]:.4e}')
print(f'Final loss (adj. warm)  : {loss_hist_aw[-1]:.4e}')
print(f'Iterations (sens.)      : {len(results["loss_hist_s"]) - 1}')
print(f'Iterations (adj. cold)  : {len(results["loss_hist_a"]) - 1}')
print(f'Iterations (adj. warm)  : {len(loss_hist_aw) - 1}')

plt.show()
