"""Probability path schedulers ported from facebookresearch/flow_matching to Jittor.

Includes:
- CondOTScheduler (supports both sigma_min=0.0 from official repo and sigma_min>0 from Paper Eq. 20)
- VPScheduler (Variance Preserving diffusion path from Paper Eq. 18-19 & Appendix E.1)
- PolynomialConvexScheduler, LinearVPScheduler, CosineScheduler
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import jittor as jt


@dataclass
class SchedulerOutput:
    """Time-dependent affine coefficients and their time derivatives."""

    alpha_t: jt.Var = field(metadata={"help": "alpha_t"})
    sigma_t: jt.Var = field(metadata={"help": "sigma_t"})
    d_alpha_t: jt.Var = field(metadata={"help": "d/dt alpha_t"})
    d_sigma_t: jt.Var = field(metadata={"help": "d/dt sigma_t"})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: jt.Var) -> SchedulerOutput:
        ...

    @abstractmethod
    def snr_inverse(self, snr: jt.Var) -> jt.Var:
        ...


class ConvexScheduler(Scheduler):
    """Scheduler base class for convex interpolants."""

    @abstractmethod
    def kappa_inverse(self, kappa: jt.Var) -> jt.Var:
        ...

    def snr_inverse(self, snr: jt.Var) -> jt.Var:
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """Conditional Optimal Transport (OT) displacement scheduler.

    Follows Lipman et al. (2023) Eq. (20):
        alpha_t = t,   sigma_t = 1 - (1 - sigma_min) * t.
    When ``sigma_min == 0.0`` (default in facebookresearch/flow_matching),
    this reduces to ``alpha_t = t, sigma_t = 1 - t``.
    """

    def __init__(self, sigma_min: float = 0.0) -> None:
        assert 0.0 <= sigma_min < 1.0, f"Expected sigma_min in [0, 1), got {sigma_min}."
        self.sigma_min = float(sigma_min)

    def __call__(self, t: jt.Var) -> SchedulerOutput:
        ones = jt.ones_like(t)
        c = 1.0 - self.sigma_min
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1.0 - c * t,
            d_alpha_t=ones,
            d_sigma_t=-c * ones,
        )

    def kappa_inverse(self, kappa: jt.Var) -> jt.Var:
        # alpha_t / sigma_t = t / (1 - c*t) = snr => t = snr / (1 + c*snr)
        # Since kappa = snr / (1 + snr), snr = kappa / (1 - kappa)
        if self.sigma_min == 0.0:
            return kappa
        c = 1.0 - self.sigma_min
        snr = kappa / (1.0 - kappa).maximum(1e-12)
        return snr / (1.0 + c * snr)


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial convex scheduler alpha_t = t^n, sigma_t = 1 - t^n."""

    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(n, (float, int)) and n > 0, f"`n` must be positive number, got {n}."
        self.n = float(n)

    def __call__(self, t: jt.Var) -> SchedulerOutput:
        n = self.n
        return SchedulerOutput(
            alpha_t=jt.pow(t, n),
            sigma_t=1.0 - jt.pow(t, n),
            d_alpha_t=n * jt.pow(t, n - 1.0),
            d_sigma_t=-n * jt.pow(t, n - 1.0),
        )

    def kappa_inverse(self, kappa: jt.Var) -> jt.Var:
        return jt.pow(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    """Variance Preserving (VP) diffusion scheduler (noise at t=0 -> data at t=1).

    Matches Lipman et al. (2023) Eq. (18)-(19) & Appendix E.1 and
    ``facebookresearch/flow_matching/path/scheduler/scheduler.py``:
        T(s) = s * beta_min + 0.5 * s^2 * (beta_max - beta_min),  s = 1 - t
        alpha_t = exp(-0.5 * T(1 - t))
        sigma_t = sqrt(1 - exp(-T(1 - t)))
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        super().__init__()
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def beta(self, s: jt.Var) -> jt.Var:
        """Linear noise scale beta(s) = beta_min + s * (beta_max - beta_min)."""
        return self.beta_min + s * (self.beta_max - self.beta_min)

    def __call__(self, t: jt.Var) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        one_minus_t = 1.0 - t
        T = 0.5 * (one_minus_t * one_minus_t) * (B - b) + one_minus_t * b
        dT = -one_minus_t * (B - b) - b

        exp_neg_T = jt.exp(-T)
        one_minus_exp = (1.0 - exp_neg_T).maximum(1e-12)
        sigma_t = jt.sqrt(one_minus_exp)
        alpha_t = jt.exp(-0.5 * T)

        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=-0.5 * dT * alpha_t,
            d_sigma_t=0.5 * dT * exp_neg_T / sigma_t,
        )

    def snr_inverse(self, snr: jt.Var) -> jt.Var:
        snr2 = snr * snr
        T = -jt.log(snr2 / (snr2 + 1.0))
        b = self.beta_min
        B = self.beta_max
        return 1.0 - ((-b + jt.sqrt(b * b + 2.0 * (B - b) * T)) / (B - b))


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler: alpha_t = t, sigma_t = sqrt(1 - t^2)."""

    def __call__(self, t: jt.Var) -> SchedulerOutput:
        denom = jt.sqrt((1.0 - t * t).maximum(1e-12))
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=denom,
            d_alpha_t=jt.ones_like(t),
            d_sigma_t=-t / denom,
        )

    def snr_inverse(self, snr: jt.Var) -> jt.Var:
        snr2 = snr * snr
        return jt.sqrt(snr2 / (1.0 + snr2))


class CosineScheduler(Scheduler):
    """Cosine Scheduler: alpha_t = sin(pi/2 * t), sigma_t = cos(pi/2 * t)."""

    def __call__(self, t: jt.Var) -> SchedulerOutput:
        half_pi = 0.5 * math.pi
        return SchedulerOutput(
            alpha_t=jt.sin(half_pi * t),
            sigma_t=jt.cos(half_pi * t),
            d_alpha_t=half_pi * jt.cos(half_pi * t),
            d_sigma_t=-half_pi * jt.sin(half_pi * t),
        )

    def snr_inverse(self, snr: jt.Var) -> jt.Var:
        return 2.0 * jt.atan(snr) / math.pi
