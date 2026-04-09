"""
lv_system.py
------------
Plant operator, Jacobian, and plant sensitivity for the reduced (4-parameter)
Lotka-Volterra predator-prey system.

    x_dot_1 =  w[0]*x[0] - w[1]*x[0]*x[1]        (prey)
    x_dot_2 = -w[2]*x[1] + w[3]*x[0]*x[1]        (predator)

Parameter vector:  w = [w11, w13, w21, w23]  (all positive)

All functions use the calling convention  f(x, t, w)  matching the dyn_suite.
"""

import numpy as np


def lv_h(x, t, w):
    """Plant operator  h(x, t, w)  -> (2,)."""
    x1, x2 = float(x[0]), float(x[1])
    w11, w13, w21, w23 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
    return np.array([
        w11 * x1 - w13 * x1 * x2,
       -w21 * x2 + w23 * x1 * x2
    ])


def lv_J(x, t, w):
    """Jacobian  dh/dx  -> (2, 2)."""
    x1, x2 = float(x[0]), float(x[1])
    w11, w13, w21, w23 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
    return np.array([
        [w11 - w13 * x2,  -w13 * x1      ],
        [w23 * x2,         -w21 + w23 * x1]
    ])


def lv_U(x, t, w):
    """Plant sensitivity  dh/dw  -> (2, 4)."""
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        [ x1,  -x1 * x2,   0.,     0.     ],
        [ 0.,   0.,        -x2,    x1 * x2]
    ])
