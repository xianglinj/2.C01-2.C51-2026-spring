# Dynamical Estimation

Tools for parameter estimation in ODE systems. Given a model
`dx/dt = h(x, t; w)` and time-series observations, recover the unknown
parameters `w` (and optionally initial conditions `x0`) by minimising the
squared residual

```
E(w, x0) = (1 / 2N) * Σₙ ‖ x(tₙ; w, x0) − x_dataₙ ‖²
```

Two gradient-based methods are provided:

- **Sensitivity method** — augment the ODE with `S = ∂x/∂w` and update
  parameters from `S` directly.
- **Adjoint method** — integrate an adjoint ODE backwards in time and
  accumulate the gradient via trapezoidal quadrature. Lower-dimensional
  and generally more accurate for larger systems.

## Layout

```
dynamical-estimation/
├── dyn_suite/              Full Python port of the MATLAB dyn_suite
│   ├── dyn_suite/          The package itself
│   │   ├── forward.py
│   │   ├── sensitivity.py
│   │   ├── adjoint.py
│   │   ├── estimation.py
│   │   ├── run.py
│   │   └── plotting.py
│   ├── lv_system.py        Lotka–Volterra system definition
│   ├── lv_test.py          LV estimation driver (sensitivity + adjoint)
│   └── User_Manual/        LaTeX source + compiled PDF of the manual
└── malthus_exact/          Analytical teaching companion
    └── malthus_run.py      Malthus growth: closed-form sensitivity & adjoint
```

## Requirements

Python 3.10+ with `numpy`, `scipy`, `matplotlib`.

```bash
pip install numpy scipy matplotlib
```

## Running

### Lotka–Volterra (full suite)

```bash
cd dyn_suite
python lv_test.py
```

Estimates the four LV parameters `[w₁₁, w₁₃, w₂₁, w₂₃]` starting from a
deliberately bad initial guess and reports results for the sensitivity
method, a cold-start adjoint, and a warm-start adjoint (initialised from
the sensitivity estimate). Produces loss curves, sensitivity/adjoint
diagnostic plots, and a LaTeX comparison table.

### Malthus (analytical example)

```bash
cd malthus_exact
python malthus_run.py
```

Standalone, pure-analytical demonstration of both estimation methods on
`dx/dt = w·x`. No ODE solver is used — every quantity (forward solution,
sensitivities, adjoint) has a closed-form expression, so you can read
the formulas directly out of the code. Outputs five figures, a results
`.npz`, and a LaTeX table.

## Plugging in your own system

Provide three callables with signature `(x, t, w) → ndarray`:

| function | returns      | meaning                     |
|----------|--------------|-----------------------------|
| `h_fun`  | `(K,)`       | ODE right-hand side         |
| `J_fun`  | `(K, K)`     | Jacobian `∂h/∂x`            |
| `U_fun`  | `(K, P)`     | Parameter sensitivity `∂h/∂w` |

See [`dyn_suite/lv_system.py`](dyn_suite/lv_system.py) for a worked
example, then follow [`dyn_suite/lv_test.py`](dyn_suite/lv_test.py) as a
driver template.
