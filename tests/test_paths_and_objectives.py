"""Unit tests for Jittor probability paths, schedulers, conversions, and objectives.

Mirrors ``facebookresearch/flow_matching/tests/path/test_path.py`` & ``test_scheduler.py``,
plus finite-difference derivative checks and VP Eq. (19) / Eq. (46) equivalence tests.
"""

import numpy as np
import jittor as jt

from jfm.datasets.checkerboard import sample_checkerboard
from jfm.models.mlp import ToyMLP
from jfm.objectives.losses import build_objective
from jfm.paths import (
    AffineProbPath,
    CondOTProbPath,
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    VPProbPath,
    VPScheduler,
)
from jfm.utils.model_wrapper import CFGScaledModel, EpsilonToVelocityWrapper, ScoreToVelocityWrapper


def test_schedulers_finite_difference_derivatives():
    schedulers = [
        CondOTScheduler(sigma_min=0.0),
        CondOTScheduler(sigma_min=1e-3),
        PolynomialConvexScheduler(n=2.0),
        VPScheduler(beta_min=0.1, beta_max=20.0),
        LinearVPScheduler(),
        CosineScheduler(),
    ]
    t_np = np.linspace(0.08, 0.92, 17, dtype=np.float64)
    eps = 1e-6

    for sched in schedulers:
        out = sched(jt.array(t_np))
        out_plus = sched(jt.array(t_np + eps))
        out_minus = sched(jt.array(t_np - eps))

        fd_alpha = (out_plus.alpha_t.numpy() - out_minus.alpha_t.numpy()) / (2.0 * eps)
        fd_sigma = (out_plus.sigma_t.numpy() - out_minus.sigma_t.numpy()) / (2.0 * eps)

        np.testing.assert_allclose(out.d_alpha_t.numpy(), fd_alpha, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out.d_sigma_t.numpy(), fd_sigma, rtol=1e-4, atol=1e-4)

        # Verify SNR inverse round-trip
        snr = out.alpha_t / out.sigma_t
        t_rec = sched.snr_inverse(snr)
        np.testing.assert_allclose(t_rec.numpy(), t_np, rtol=1e-4, atol=1e-4)


def test_affine_prob_path_conversions():
    for path in [CondOTProbPath(sigma_min=0.0), CondOTProbPath(sigma_min=1e-4), VPProbPath()]:
        x_0 = jt.randn((16, 5))
        x_1 = jt.randn((16, 5))
        t = jt.array(np.linspace(0.1, 0.9, 16, dtype=np.float32))

        sample = path.sample(x_0=x_0, x_1=x_1, t=t)
        assert list(sample.x_t.shape) == [16, 5]
        assert list(sample.dx_t.shape) == [16, 5]

        # Theorem 3 velocity must match dx_t = d_alpha_t x_1 + d_sigma_t x_0
        vel_from_target = path.target_to_velocity(x_1=x_1, x_t=sample.x_t, t=t)
        np.testing.assert_allclose(vel_from_target.numpy(), sample.dx_t.numpy(), rtol=1e-4, atol=1e-4)

        # Round-trip target <-> velocity <-> epsilon
        x1_rec = path.velocity_to_target(sample.dx_t, sample.x_t, t)
        np.testing.assert_allclose(x1_rec.numpy(), x_1.numpy(), rtol=1e-4, atol=1e-4)

        eps_rec = path.velocity_to_epsilon(sample.dx_t, sample.x_t, t)
        np.testing.assert_allclose(eps_rec.numpy(), x_0.numpy(), rtol=1e-4, atol=1e-4)

        vel_from_eps = path.epsilon_to_velocity(eps_rec, sample.x_t, t)
        np.testing.assert_allclose(vel_from_eps.numpy(), sample.dx_t.numpy(), rtol=1e-4, atol=1e-4)


def test_vp_score_and_ddpm_drift_match_conditional_fm_velocity():
    """Verify Paper Eq. (19) == Eq. (46) when oracle score / oracle noise is plugged in."""
    vp_path = VPProbPath(beta_min=0.1, beta_max=20.0, time_eps=1e-5)
    x_0 = jt.randn((12, 4))
    x_1 = jt.randn((12, 4))
    t = jt.array(np.linspace(0.05, 0.95, 12, dtype=np.float32))
    sample = vp_path.sample(x_0=x_0, x_1=x_1, t=t)
    sched = vp_path.scheduler(t)
    sigma_t = sched.sigma_t.reshape(-1, 1)

    class OracleScoreModel(jt.nn.Module):
        def execute(self, x, t_in, **kwargs):
            return x_0 / sigma_t

    class OracleEpsModel(jt.nn.Module):
        def execute(self, x, t_in, **kwargs):
            return x_0

    sm_vel = ScoreToVelocityWrapper(OracleScoreModel(), beta_min=0.1, beta_max=20.0)(sample.x_t, t)
    ddpm_vel = EpsilonToVelocityWrapper(OracleEpsModel(), path=vp_path)(sample.x_t, t)

    np.testing.assert_allclose(sm_vel.numpy(), sample.dx_t.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ddpm_vel.numpy(), sample.dx_t.numpy(), rtol=1e-4, atol=1e-4)


def test_all_objectives_backward_and_checkerboard_support():
    pts = sample_checkerboard(batch_size=2000).numpy()
    scaled = pts * 0.45
    in_box = (
        (scaled[:, 0] >= -2.0)
        & (scaled[:, 0] < 2.0)
        & (scaled[:, 1] >= -2.0)
        & (scaled[:, 1] < 2.0)
    )
    ix = np.floor(scaled[:, 0]).astype(int)
    iy = np.floor(scaled[:, 1]).astype(int)
    assert np.all(in_box & (((ix + iy) % 2) == 0)), "All checkerboard samples must lie in valid squares!"

    for method in ["fm_ot", "fm_dif", "sm_dif", "scoreflow", "ddpm"]:
        model = ToyMLP(input_dim=2, hidden_dim=32, num_layers=3)
        optim = jt.optim.Adam(model.parameters(), lr=1e-3)
        bundle = build_objective(method=method)
        x_1 = sample_checkerboard(batch_size=64)
        loss = bundle.loss_fn(model, x_1=x_1)
        assert np.isfinite(float(loss.item()))
        optim.step(loss)


def test_cfg_scaled_model_nfe_count():
    class CondDummy(jt.nn.Module):
        def execute(self, x, t, y=None, **kwargs):
            if y is None:
                return x * 1.0
            return x * 2.0

    wrapped = CFGScaledModel(CondDummy(), cfg_scale=1.5)
    x = jt.ones((4, 2))
    t = jt.array([0.5])
    y = jt.array([0, 1, 2, 3])
    out = wrapped(x, t, label=y)
    # (1 + 1.5) * 2 - 1.5 * 1 = 3.5
    np.testing.assert_allclose(out.numpy(), np.full((4, 2), 3.5), rtol=1e-6)
    assert wrapped.get_nfe() == 2
