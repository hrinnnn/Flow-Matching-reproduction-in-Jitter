"""Parity of GroupNorm32 and the torchdiffeq-style integrator."""

import math

import jittor as jt
import numpy as np

from jfm.models.nn import GroupNorm32
from jittor_ode import odeint, odeint_adjoint


def test_group_norm_matches_two_pass_not_the_cancelled_formula():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(2, 32, 4, 4)).astype(np.float32) + 40.0
    weight = np.ones(32, dtype=np.float32)
    bias = np.zeros(32, dtype=np.float32)
    grouped = data.reshape(2, 32, 1, -1)
    mean = grouped.mean(axis=(2, 3), keepdims=True)
    var = ((grouped - mean) ** 2).mean(axis=(2, 3), keepdims=True)
    reference = ((grouped - mean) / np.sqrt(var + 1e-5)).reshape(data.shape)

    layer = GroupNorm32(32, 32)
    layer.weight.assign(jt.array(weight))
    layer.bias.assign(jt.array(bias))
    ours = layer(jt.array(data)).numpy()
    np.testing.assert_allclose(ours, reference, rtol=1e-4, atol=2e-5)


def test_odeint_rk4_matches_the_exponential():
    solution = odeint(
        lambda t, y: -y,
        jt.array([1.0, 2.0]),
        [0.0, 1.0],
        method="rk4",
        options={"step_size": 0.01},
    )
    expected = np.array([math.exp(-1.0), 2.0 * math.exp(-1.0)], dtype=np.float32)
    np.testing.assert_allclose(solution.numpy()[-1], expected, rtol=1e-5, atol=1e-5)


def test_adjoint_matches_the_scalar_linear_ode():
    rate = jt.array([0.25])

    def dynamics(t, y):
        return rate * y

    y0 = jt.array([1.5, -0.5])
    trajectory = odeint_adjoint(
        dynamics,
        y0,
        [0.0, 1.0],
        rate,
        method="rk4",
        options={"step_size": 0.05},
    )
    loss = trajectory[-1].sum()
    grad = jt.grad(loss, rate)
    # d/dc sum(y0 * exp(c)) = sum(y0) * exp(c)
    expected = float(y0.numpy().sum()) * math.exp(0.25)
    np.testing.assert_allclose(grad.numpy(), np.array([expected], dtype=np.float32), rtol=2e-3, atol=2e-3)
