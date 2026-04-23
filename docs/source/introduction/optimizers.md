# Optimizers

This tutorial gives a brief overview of the optimizers available in
`topoptlab.optimizer` and how they operate. The general setting is a
constrained problem of the form

```{math}
\begin{aligned}
&\min_{x}   && f(x) \\
&\text{s.t.}  && \boldsymbol{c}_{\text{eq}}(x) = \boldsymbol{0}, \\
&             && \boldsymbol{c}_{\text{ineq}}(x) \leq \boldsymbol{0}, \\
&             && x_{\min} \leq x \leq x_{\max},
\end{aligned}
```

where $x \in \mathbb{R}^n$ are the design variables, $f$ is the objective,
and the box constraints $[x_{\min}, x_{\max}]$ are always enforced.

---

## Gradient Descent

**Module:** `topoptlab.optimizer.gradient_descent`

The most basic update rule. Given the current design $x^{(k)}$ and the
gradient $\nabla f^{(k)}$, the unconstrained step is

```{math}
x^{(k+1)} = x^{(k)} - \alpha^{(k)} \nabla f^{(k)},
```

where $\alpha^{(k)} > 0$ is the step size. After the gradient step, bounds
and a *move limit* $\Delta$ are enforced element-wise. The step size $\alpha^{(k)}$ is computed by a separate *step size function*
(see [Step Size Rules](#step-size-rules) below).

**Use when:** no explicit constraints beyond box bounds are present.

---

## Optimality Criterion (OC)

**Module:** `topoptlab.optimizer.optimality_criterion`

The OC method is a classical heuristic update scheme tailored to
compliance minimization problems. The Lagrangian of
$\min f(x)$ subject to $v(x) \leq V^*$ is

```{math}
L = f(x) + \lambda\,(v(x) - V^*).
```

The KKT stationarity condition $\frac{\partial L}{\partial x_i} = 0$ motivates
the fixed-point update

```{math}
x_i^{(k+1)} = x_i^{(k)}\,\sqrt{\frac{-\partial f/\partial x_i}{\lambda\,\partial v/\partial x_i}},
```

clipped to $[x_i^{(k)} - \Delta,\, x_i^{(k)} + \Delta] \cap [0, 1]$.
The Lagrange multiplier $\lambda$ is found by bisection so that the volume
constraint is satisfied.

Three variants are available:

| Function | Intended use |
|---|---|
| `oc_top88` | Compliance minimisation, density/sensitivity/Helmholtz filter |
| `oc_mechanism` | Compliant mechanisms; uses a damping exponent $\zeta$: $x_i^{(k+1)} \propto \left(-\partial f / \partial x_i \big/ \lambda\,\partial v / \partial x_i\right)^\zeta$ |
| `oc_generalized` | Same as `oc_mechanism`, kept for experimentation |

**Note:** Works reasonably well for compliance problems and volume constraint, but the plain OC heuristic breaks down when stronger nonlinearity is encountered. MMA is almost always the better solution.

---

## Method of Moving Asymptotes (MMA / GCMMA)

**Module:** `topoptlab.optimizer.mma_utils`

MMA (Svanberg 1987) replaces the original problem at each iteration $k$ by a
separable convex sub-problem built around *moving asymptotes*
$L_i^{(k)} < x_i^{(k)} < U_i^{(k)}$. For a minimization problem the
sub-problem approximates $f$ by

```{math}
\tilde{f}^{(k)}(x) = r_0^{(k)}
  + \sum_i \left(\frac{p_{0i}^{(k)}}{U_i^{(k)} - x_i}
           + \frac{q_{0i}^{(k)}}{x_i - L_i^{(k)}}\right),
```

where the coefficients $p_{0i}, q_{0i} \geq 0$ are chosen so that the
approximation is a first-order Taylor match of $f$ at $x^{(k)}$. The
asymptotes adapt between iterations:

- if the sign of $x_i^{(k)} - x_i^{(k-1)}$ *changes* (oscillation),
  move the asymptotes closer: $U_i \leftarrow x_i + \gamma_{\text{decr}}(U_i - x_i)$;
- if the sign is *consistent* (monotone progress), move them further:
  $U_i \leftarrow x_i + \gamma_{\text{incr}}(U_i - x_i)$.

`mma_utils` exposes two helpers:

- `mma_defaultkws` — default parameters for standard MMA.
- `gcmma_defaultkws` — default parameters for GCMMA (globally convergent
  variant by Svanberg 1995), which additionally performs an inner
  conservative check before accepting each iterate.

Both helpers return a dictionary that is passed directly to `mmasub` from the
`mmapy` package. Constraints are passed as $\boldsymbol{c}(x) \leq 0$.

**Use when:** problems contains inequality constraints. MMA is the default recommended solver and GCMMA should only be used if MMA fails.  Equality constraints can be handled by conversion to two inequalities.

---

## Augmented Lagrangian Method (ALM)

**Module:** `topoptlab.optimizer.augmented_lagrangian`

The ALM handles equality *and* inequality constraints by augmenting the
Lagrangian with a quadratic penalty:

```{math}
\mathcal{L}_\rho(x, \boldsymbol{\lambda}, \boldsymbol{\mu})
  = f(x)
    + \boldsymbol{\lambda}^\top \boldsymbol{c}_{\text{eq}}
    + \frac{\rho}{2}\|\boldsymbol{c}_{\text{eq}}\|^2
    + \boldsymbol{\mu}^\top \boldsymbol{c}_{\text{ineq}}
    + \frac{\rho}{2}\|\boldsymbol{c}_{\text{ineq}}\|^2.
```

Each outer iteration consists of two steps:

**1. Primal step.** One gradient-descent step on $\mathcal{L}_\rho$ with
respect to $x$. The gradient is assembled as

```{math}
\nabla_x \mathcal{L}_\rho
  = \nabla f
    + J_{\text{eq}}^\top \left(\boldsymbol{\lambda} + \rho\,\boldsymbol{c}_{\text{eq}}\right)
    + J_{\text{ineq}}^\top \left(\boldsymbol{\mu} + \rho\,\max(\boldsymbol{c}_{\text{ineq}}, \boldsymbol{0})\right),
```

where $J_{\text{eq}}$ and $J_{\text{ineq}}$ are the constraint Jacobians.

**2. Dual update.** The multipliers are updated as

```{math}
\boldsymbol{\lambda}^{(k+1)} = \boldsymbol{\lambda}^{(k)} + \rho\,\boldsymbol{c}_{\text{eq}},
\qquad
\boldsymbol{\mu}^{(k+1)} = \max\!\left(\boldsymbol{0},\, \boldsymbol{\mu}^{(k)} + \rho\,\boldsymbol{c}_{\text{ineq}}\right).
```

The non-negativity projection on $\boldsymbol{\mu}$ enforces complementary
slackness for the inequality constraints.

**Use when:** This method is experimental so far and needs further stabilization. Constrained problems of interest are both equality and inequality constraints where a lightweight first-order method is preferred over MMA or if the number of constraints becomes very large.

---

## Step Size Rules

**Module:** `topoptlab.optimizer.stepsize`

All gradient-based optimizers share the same step-size interface. The
available rules are:

| Function | Formula | Notes |
|---|---|---|
| `constant` | $\alpha = \alpha_0$ | Fixed, user-prescribed |
| `barzilai_borwein_long` | $\alpha_{\text{BB1}} = \dfrac{\Delta x^\top \Delta x}{\Delta x^\top \Delta g}$ | Longer steps, can oscillate |
| `barzilai_borwein_short` | $\alpha_{\text{BB2}} = \dfrac{\Delta x^\top \Delta g}{\Delta g^\top \Delta g}$ | Shorter, more stable (default) |
| `barzilai_borwein_stabilized` | $\alpha = \min\!\left(\alpha_{\text{BB1}},\,\|\Delta x\|\right)$ | Falls back to BB1 with a norm bound |

where $\Delta x = x^{(k)} - x^{(k-1)}$ and $\Delta g = \nabla f^{(k)} - \nabla f^{(k-1)}$. The BB rules require two consecutive iterates; for the very first iteration a small default value ($10^{-5}$) is used.
