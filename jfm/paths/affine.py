"""Affine Gaussian conditional probability paths ported from facebookresearch/flow_matching."""

from abc import ABC, abstractmethod

import jittor as jt

from jfm.paths.path_sample import PathSample
from jfm.paths.scheduler import CondOTScheduler, Scheduler, VPScheduler
from jfm.utils.tensor_utils import expand_tensor_like


def _match_coeff(coeff: jt.Var, ref: jt.Var) -> jt.Var:
    if coeff.ndim == 1 and ref.ndim > 1:
        return expand_tensor_like(coeff, ref)
    return coeff


class ProbPath(ABC):
    """Abstract conditional probability path transforming p_0 (t=0) into p_1 (t=1)."""

    @abstractmethod
    def sample(self, x_0: jt.Var, x_1: jt.Var, t: jt.Var) -> PathSample:
        ...

    def assert_sample_shape(self, x_0: jt.Var, x_1: jt.Var, t: jt.Var) -> None:
        assert t.ndim == 1, f"Time vector t must have shape [batch_size], got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Batch dimension mismatch: t={t.shape}, x_0={x_0.shape}, x_1={x_1.shape}."


class AffineProbPath(ProbPath):
    """Gaussian affine conditional probability path X_t = alpha_t X_1 + sigma_t X_0.

    Ported 1-to-1 from ``facebookresearch/flow_matching/path/affine.py``.
    """

    def __init__(self, scheduler: Scheduler, time_eps: float = 0.0):
        self.scheduler = scheduler
        self.time_eps = float(time_eps)

    def sample_time(self, batch_size: int, dtype=jt.float32) -> jt.Var:
        """Sample uniform training times in ``[0, 1 - time_eps]``."""
        u = jt.rand((batch_size,)).cast(dtype)
        if self.time_eps > 0.0:
            u = u * (1.0 - self.time_eps)
        return u

    def sample(self, x_0: jt.Var, x_1: jt.Var, t: jt.Var) -> PathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        sched = self.scheduler(t)

        alpha_t = expand_tensor_like(sched.alpha_t, x_1)
        sigma_t = expand_tensor_like(sched.sigma_t, x_1)
        d_alpha_t = expand_tensor_like(sched.d_alpha_t, x_1)
        d_sigma_t = expand_tensor_like(sched.d_sigma_t, x_1)

        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: jt.Var, x_t: jt.Var, t: jt.Var) -> jt.Var:
        """Compute conditional velocity u_t(x_t | x_1) via Theorem 3 (Eq. 15)."""
        sched = self.scheduler(t)
        alpha_t = _match_coeff(sched.alpha_t, x_t)
        d_alpha_t = _match_coeff(sched.d_alpha_t, x_t)
        sigma_t = _match_coeff(sched.sigma_t, x_t)
        d_sigma_t = _match_coeff(sched.d_sigma_t, x_t)

        a_t = d_sigma_t / sigma_t
        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t
        return a_t * x_t + b_t * x_1

    def epsilon_to_velocity(self, epsilon: jt.Var, x_t: jt.Var, t: jt.Var) -> jt.Var:
        """Convert noise prediction eps to velocity u_t."""
        sched = self.scheduler(t)
        alpha_t = _match_coeff(sched.alpha_t, x_t)
        d_alpha_t = _match_coeff(sched.d_alpha_t, x_t)
        sigma_t = _match_coeff(sched.sigma_t, x_t)
        d_sigma_t = _match_coeff(sched.d_sigma_t, x_t)

        a_t = d_alpha_t / alpha_t
        b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t
        return a_t * x_t + b_t * epsilon

    def velocity_to_target(self, velocity: jt.Var, x_t: jt.Var, t: jt.Var) -> jt.Var:
        """Convert velocity u_t to target prediction x_1."""
        sched = self.scheduler(t)
        alpha_t = _match_coeff(sched.alpha_t, x_t)
        d_alpha_t = _match_coeff(sched.d_alpha_t, x_t)
        sigma_t = _match_coeff(sched.sigma_t, x_t)
        d_sigma_t = _match_coeff(sched.d_sigma_t, x_t)

        denom = d_alpha_t * sigma_t - d_sigma_t * alpha_t
        a_t = -d_sigma_t / denom
        b_t = sigma_t / denom
        return a_t * x_t + b_t * velocity

    def epsilon_to_target(self, epsilon: jt.Var, x_t: jt.Var, t: jt.Var) -> jt.Var:
        """Convert noise prediction eps to target prediction x_1."""
        sched = self.scheduler(t)
        alpha_t = _match_coeff(sched.alpha_t, x_t)
        sigma_t = _match_coeff(sched.sigma_t, x_t)
        return (x_t - sigma_t * epsilon) / alpha_t

    def velocity_to_epsilon(self, velocity: jt.Var, x_t: jt.Var, t: jt.Var) -> jt.Var:
        """Convert velocity u_t to noise prediction eps."""
        sched = self.scheduler(t)
        alpha_t = _match_coeff(sched.alpha_t, x_t)
        d_alpha_t = _match_coeff(sched.d_alpha_t, x_t)
        sigma_t = _match_coeff(sched.sigma_t, x_t)
        d_sigma_t = _match_coeff(sched.d_sigma_t, x_t)

        denom = d_sigma_t * alpha_t - d_alpha_t * sigma_t
        a_t = -d_alpha_t / denom
        b_t = alpha_t / denom
        return a_t * x_t + b_t * velocity

    def target_to_epsilon(self, x_1: jt.Var, x_t: jt.Var, t: jt.Var) -> jt.Var:
        """Convert target x_1 to noise eps = (x_t - alpha_t x_1) / sigma_t."""
        sched = self.scheduler(t)
        alpha_t = _match_coeff(sched.alpha_t, x_t)
        sigma_t = _match_coeff(sched.sigma_t, x_t)
        return (x_t - alpha_t * x_1) / sigma_t


class CondOTProbPath(AffineProbPath):
    """Conditional Optimal Transport probability path (FM-OT)."""

    def __init__(self, sigma_min: float = 0.0):
        super().__init__(scheduler=CondOTScheduler(sigma_min=sigma_min), time_eps=0.0)


class VPProbPath(AffineProbPath):
    """Variance Preserving diffusion probability path (FM-Dif / SM-Dif / DDPM).

    Follows Lipman et al. (2023) Appendix E.1: time is sampled in ``[0, 1 - eps]``
    with ``eps = 1e-5`` to prevent numerical blow-up near ``t -> 1``.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        time_eps: float = 1e-5,
    ):
        super().__init__(
            scheduler=VPScheduler(beta_min=beta_min, beta_max=beta_max),
            time_eps=time_eps,
        )
