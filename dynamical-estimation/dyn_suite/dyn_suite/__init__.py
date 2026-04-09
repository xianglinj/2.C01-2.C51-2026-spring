"""
dyn_suite — Dynamical Estimation Suite (Python port)

Direct translation of the MATLAB dyn_suite into Python/NumPy/SciPy.
Solver correspondence:  ode45  <->  scipy.integrate.solve_ivp (method='RK45')
                        interp1 PCHIP  <->  scipy.interpolate.PchipInterpolator

Usage
-----
from dyn_suite import dyn_forward, dyn_residuals
from dyn_suite import dyn_sensitivity, dyn_adjoint
from dyn_suite import dyn_estimate_sensitivity, dyn_estimate_adjoint
from dyn_suite import dyn_run
from dyn_suite.plotting import (dyn_plot_forward, dyn_plot_sensitivity,
                                dyn_plot_adjoint, dyn_plot_residual_force,
                                dyn_plot_loss)
"""

from .forward      import dyn_forward, dyn_residuals
from .sensitivity  import dyn_sensitivity, dyn_update_sensitivity
from .adjoint      import dyn_adjoint, dyn_update_adjoint
from .estimation   import dyn_estimate_sensitivity, dyn_estimate_adjoint
from .run          import dyn_run

__all__ = [
    'dyn_forward', 'dyn_residuals',
    'dyn_sensitivity', 'dyn_update_sensitivity',
    'dyn_adjoint',     'dyn_update_adjoint',
    'dyn_estimate_sensitivity', 'dyn_estimate_adjoint',
    'dyn_run',
]
