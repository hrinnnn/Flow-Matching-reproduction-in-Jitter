"""Unit tests for native Jittor ODE solvers (Euler, Midpoint, Heun3, RK4, Dopri5) & Likelihood.

Mirrors ``facebookresearch/flow_matching/tests/solver/test_ode_solver.py`` AND verifies:
1. Polynomial & exponential ODE accuracy across all solvers
2. Empirical convergence order of Euler (~1), Midpoint (~2), RK4 (~4), and adaptive Dopri5
3. Exact NFE counting
4. Exact divergence & Hutchinson divergence log-likelihood on analytical linear flows
5. End-to-end BPD dequantization pipeline
"""

import math
import numpy as np
import jittor as jt
from jittor import nn

from jfm.likelihood import compute_bpd_batch
from jfm.solvers import ODESolver
from jfm.utils import ModelWrapper, gaussian_log_prob


class QuadraticTimeModel(ModelWrapper):
    """dx/dt = 3 t^2 => x(1) - x(0) = 1."""

    def __init__(self):
        super().__init__(None)

    def execute(self, x: jt.Var, t: jt.Var, **extras) -> jt.Var:
        self.nfe_counter += 1
        t_val = t.reshape(-1)[0]
        return jt.ones_like(x) * (3.0 * t_val * t_val)


class LearnableConstModel(ModelWrapper):
    """dx/dt = a => d(sum x(1))/da = dim."""

    def __init__(self):
        super().__init__(None)
        self.a = jt.array([1.0])

    def execute(self, x: jt.Var, t: jt.Var, **extras) -> jt.Var:
        self.nfe_counter += 1
        return x * 0.0 + self.a


class LinearExpansionModel(ModelWrapper):
    """dx/dt = c * x => x(1) = e^c x(0), so p_1(x_1) = N(0, e^{2c} I)."""

    def __init__(self, c: float = 0.4):
        super().__init__(None)
        self.c = c

    def execute(self, x: jt.Var, t: jt.Var, **extras) -> jt.Var:
        self.nfe_counter += 1
        return self.c * x


def test_all_solvers_solve_polynomial_ode_and_count_nfe():
    model = QuadraticTimeModel()
    solver = ODESolver(velocity_model=model)
    x_init = jt.array([[1.0, 0.0]])

    expected_nfe = {"euler": 100, "midpoint": 200, "heun3": 300, "rk4": 400}
    for method, exp_nfe in expected_nfe.items():
        res = solver.sample(x_init=x_init, step_size=0.01, method=method, time_grid=[0.0, 1.0])
        np.testing.assert_allclose(res.numpy(), np.array([[2.0, 1.0]]), atol=2e-2)
        assert solver.last_nfe == exp_nfe, f"{method}: expected {exp_nfe}, got {solver.last_nfe}"

    # Dopri5 integrates degree-2 polynomial to machine precision in minimal steps
    res_dp = solver.sample(x_init=x_init, step_size=None, method="dopri5", atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(res_dp.numpy(), np.array([[2.0, 1.0]]), atol=1e-5)
    assert solver.last_nfe > 0


def test_solver_convergence_orders_and_dopri5_accuracy():
    """On dx/dt = c * x (exact x(1) = exp(c) * x(0)), check empirical order of accuracy."""
    c = 0.8
    model = LinearExpansionModel(c=c)
    solver = ODESolver(velocity_model=model)
    x0 = jt.array([[1.5, -0.7]], dtype=jt.float64)
    exact = x0.numpy() * math.exp(c)

    def err_at_steps(method: str, n_steps: int) -> float:
        pred = solver.sample(x_init=x0, step_size=1.0 / n_steps, method=method).numpy()
        return float(np.max(np.abs(pred - exact)))

    for method, min_order in [("euler", 0.95), ("midpoint", 1.95), ("rk4", 3.90)]:
        e1 = err_at_steps(method, 10)
        e2 = err_at_steps(method, 20)
        order = math.log2(e1 / e2)
        assert order >= min_order, f"{method} empirical order {order:.2f} < {min_order}"

    pred_dp = solver.sample(x_init=x0, step_size=None, method="dopri5", atol=1e-7, rtol=1e-7).numpy()
    np.testing.assert_allclose(pred_dp, exact, rtol=1e-6, atol=1e-6)


def test_differentiable_sampling_gradients():
    model = LearnableConstModel()
    solver = ODESolver(velocity_model=model)
    x_init = jt.array([[1.0, 0.0]])

    for method in ["euler", "midpoint", "rk4", "dopri5"]:
        res = solver.sample(
            x_init=x_init,
            step_size=0.1 if method != "dopri5" else None,
            method=method,
            enable_grad=True,
        )
        g = jt.grad(res.sum(), model.a)
        np.testing.assert_allclose(g.numpy(), np.array([2.0]), atol=1e-4)


def test_exact_and_hutchinson_likelihood_match_analytical_density():
    """For dx/dt = c * x in R^d, p_1(x_1) = N(x_1 | 0, e^{2c} I) => log p_1(x_1) = log p_0(e^{-c} x_1) - d * c."""
    c = 0.35
    d = 4
    model = LinearExpansionModel(c=c)
    solver = ODESolver(velocity_model=model)

    jt.misc.set_global_seed(123)
    x_1 = jt.randn((8, d))
    x_0_true = x_1 * math.exp(-c)
    true_log_p1 = gaussian_log_prob(x_0_true).numpy() - d * c

    x0_rec, exact_log_p1 = solver.compute_likelihood(
        x_1=x_1,
        log_p0=gaussian_log_prob,
        step_size=None,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        exact_divergence=True,
    )
    np.testing.assert_allclose(x0_rec.numpy(), x_0_true.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(exact_log_p1.numpy(), true_log_p1, rtol=1e-4, atol=1e-4)

    # Hutchinson estimator averaged over probe draws is unbiased (here J = c * I so z^T J z == d * c on EVERY draw!)
    _, hutch_log_p1 = solver.compute_likelihood(
        x_1=x_1,
        log_p0=gaussian_log_prob,
        step_size=None,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        exact_divergence=False,
    )
    np.testing.assert_allclose(hutch_log_p1.numpy(), true_log_p1, rtol=1e-4, atol=1e-4)


def test_bpd_pipeline_runs():
    model = LinearExpansionModel(c=0.0)  # Zero velocity => identity flow
    solver = ODESolver(velocity_model=model)
    fake_imgs = jt.rand((2, 3, 8, 8)) * 2.0 - 1.0
    out = compute_bpd_batch(solver, fake_imgs, num_dequant_samples=2, method="midpoint", step_size=0.5)
    assert out["bpd"].shape[0] == 2
    assert np.isfinite(float(out["bpd_mean"].item()))
