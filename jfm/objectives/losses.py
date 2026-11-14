"""Training losses & unified objective factory for FM-OT, FM-Dif, SM-Dif, ScoreFlow, and DDPM.

Implements all objectives compared in Lipman et al. (2023) Section 3-4 & Appendix E.1:
- ``fm_ot``: Conditional Flow Matching with Optimal Transport path (Eq. 20-23)
- ``fm_dif``: Conditional Flow Matching with VP Diffusion path (Eq. 9, 18-19)
- ``sm_dif``: Score Matching with VP Diffusion path, lambda(t) = sigma_t^2 (Eq. 43)
- ``scoreflow``: ScoreFlow with VP Diffusion path, lambda(t) = beta(1 - t) (Eq. 43)
- ``ddpm``: Noise Matching (DDPM) with VP Diffusion path (Eq. 44-45)
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jittor as jt
from jittor import nn

from jfm.paths.affine import AffineProbPath, CondOTProbPath, VPProbPath
from jfm.paths.scheduler import VPScheduler
from jfm.utils.model_wrapper import (
    CFGScaledModel,
    EpsilonToVelocityWrapper,
    ModelWrapper,
    ScoreToVelocityWrapper,
)
from jfm.utils.tensor_utils import expand_tensor_like


class CFMLoss(nn.Module):
    """Conditional Flow Matching L2 loss E[ || v_theta(x_t, t) - dx_t ||^2 ] (Eq. 9, 14).

    ``coupling="independent"`` draws ``x_0`` independently of ``x_1``, which is the
    conditional path in Lipman et al. ``coupling="minibatch"`` replaces that product
    coupling by an exact minibatch optimal-transport matching (Tong et al., 2024).
    """

    def __init__(self, path: AffineProbPath, coupling: str = "independent"):
        super().__init__()
        if coupling not in ("independent", "minibatch"):
            raise ValueError(f"coupling must be 'independent' or 'minibatch', got {coupling!r}.")
        self.path = path
        self.coupling = coupling

    def execute(
        self,
        model: nn.Module,
        x_1: jt.Var,
        x_0: Optional[jt.Var] = None,
        t: Optional[jt.Var] = None,
        **model_kwargs,
    ) -> jt.Var:
        if x_0 is None:
            x_0 = jt.randn_like(x_1)
        if self.coupling == "minibatch":
            from jfm.paths.minibatch_ot import couple_jittor

            x_0 = couple_jittor(x_0, x_1)
        if t is None:
            t = self.path.sample_time(x_1.shape[0], dtype=x_1.dtype)
        sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        pred = model(sample.x_t, sample.t, **model_kwargs)
        diff = pred - sample.dx_t
        return (diff * diff).mean()


class ScoreMatchingLoss(nn.Module):
    """Score Matching / ScoreFlow loss on VP Diffusion path (Appendix E.1 Eq. 42-43).

    In Lipman et al. Eq. (43):
        L_SM(theta) = E[ lambda(t) * || s_t(x_t) - (x_t - mu_t) / sigma_t^2 ||^2 ]
                    = E[ lambda(t) * || s_t(x_t) - x_0 / sigma_t ||^2 ].
    - ``weighting="sm"`` sets ``lambda(t) = sigma_t^2``, simplifying cleanly to
      ``E[ || sigma_t * s_t(x_t) - x_0 ||^2 ]`` (zero division risk!).
    - ``weighting="scoreflow"`` sets ``lambda(t) = beta(1 - t)``.
    """

    def __init__(self, path: VPProbPath, weighting: str = "sm"):
        super().__init__()
        assert isinstance(path.scheduler, VPScheduler), "ScoreMatchingLoss expects VPScheduler."
        assert weighting in ("sm", "scoreflow"), f"Unknown weighting {weighting!r}."
        self.path = path
        self.weighting = weighting

    def execute(
        self,
        model: nn.Module,
        x_1: jt.Var,
        x_0: Optional[jt.Var] = None,
        t: Optional[jt.Var] = None,
        **model_kwargs,
    ) -> jt.Var:
        if x_0 is None:
            x_0 = jt.randn_like(x_1)
        if t is None:
            t = self.path.sample_time(x_1.shape[0], dtype=x_1.dtype)

        sched = self.path.scheduler(t)
        alpha_t = expand_tensor_like(sched.alpha_t, x_1)
        sigma_t = expand_tensor_like(sched.sigma_t.maximum(1e-5), x_1)
        x_t = sigma_t * x_0 + alpha_t * x_1

        s_pred = model(x_t, t, **model_kwargs)
        if self.weighting == "sm":
            # lambda(t) * || s_t - x_0 / sigma_t ||^2 with lambda(t) = sigma_t^2
            err = sigma_t * s_pred - x_0
            # Note: Paper Eq. 43 uses squared Euclidean norm; matching CFMLoss mean-over-elements convention
            return (err * err).mean()
        else:
            beta_rev = self.path.scheduler.beta(1.0 - t)
            weight = expand_tensor_like(beta_rev, x_1)
            target = x_0 / sigma_t
            diff = s_pred - target
            return (weight * diff * diff).mean()


class DDPMLoss(nn.Module):
    """Noise Matching (DDPM) loss E[ || eps_theta(x_t, t) - x_0 ||^2 ] (Appendix E.1 Eq. 45)."""

    def __init__(self, path: AffineProbPath):
        super().__init__()
        self.path = path

    def execute(
        self,
        model: nn.Module,
        x_1: jt.Var,
        x_0: Optional[jt.Var] = None,
        t: Optional[jt.Var] = None,
        **model_kwargs,
    ) -> jt.Var:
        if x_0 is None:
            x_0 = jt.randn_like(x_1)
        if t is None:
            t = self.path.sample_time(x_1.shape[0], dtype=x_1.dtype)
        sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        eps_pred = model(sample.x_t, sample.t, **model_kwargs)
        diff = eps_pred - x_0
        return (diff * diff).mean()


@dataclass
class ObjectiveBundle:
    """Bundled path, training loss, sampling time interval, and ODE velocity wrapper factory."""

    method: str
    path: AffineProbPath
    loss_fn: nn.Module
    sample_t_end: float
    wrap_velocity: Callable[[nn.Module], ModelWrapper]


def build_objective(
    method: str,
    sigma_min: float = 0.0,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    time_eps: float = 1e-5,
    cfg_scale: Optional[float] = None,
    coupling: str = "independent",
) -> ObjectiveBundle:
    """Construct a complete training + ODE-sampling bundle for a named method.

    Supported ``method`` values:
        - ``"fm_ot"``: Flow Matching w/ Optimal Transport path
        - ``"fm_dif"``: Flow Matching w/ VP Diffusion path
        - ``"sm_dif"``: Score Matching w/ VP Diffusion path
        - ``"scoreflow"``: ScoreFlow w/ VP Diffusion path
        - ``"ddpm"``: DDPM Noise Matching w/ VP Diffusion path
    """
    method_key = method.lower()

    def _maybe_cfg(m: nn.Module) -> nn.Module:
        if cfg_scale is not None:
            return CFGScaledModel(m, cfg_scale=cfg_scale)
        return m

    if method_key == "fm_ot":
        path = CondOTProbPath(sigma_min=sigma_min)
        loss_fn = CFMLoss(path, coupling=coupling)
        return ObjectiveBundle(
            method=method_key,
            path=path,
            loss_fn=loss_fn,
            sample_t_end=1.0,
            wrap_velocity=lambda m: (
                CFGScaledModel(m, cfg_scale=cfg_scale)
                if cfg_scale is not None
                else ModelWrapper(m)
            ),
        )

    vp_path = VPProbPath(beta_min=beta_min, beta_max=beta_max, time_eps=time_eps)
    t_end = 1.0 - time_eps

    if method_key == "fm_dif":
        return ObjectiveBundle(
            method=method_key,
            path=vp_path,
            loss_fn=CFMLoss(vp_path, coupling=coupling),
            sample_t_end=t_end,
            wrap_velocity=lambda m: (
                CFGScaledModel(m, cfg_scale=cfg_scale)
                if cfg_scale is not None
                else ModelWrapper(m)
            ),
        )
    if method_key == "sm_dif":
        return ObjectiveBundle(
            method=method_key,
            path=vp_path,
            loss_fn=ScoreMatchingLoss(vp_path, weighting="sm"),
            sample_t_end=t_end,
            wrap_velocity=lambda m: ScoreToVelocityWrapper(
                _maybe_cfg(m), beta_min=beta_min, beta_max=beta_max
            ),
        )
    if method_key == "scoreflow":
        return ObjectiveBundle(
            method=method_key,
            path=vp_path,
            loss_fn=ScoreMatchingLoss(vp_path, weighting="scoreflow"),
            sample_t_end=t_end,
            wrap_velocity=lambda m: ScoreToVelocityWrapper(
                _maybe_cfg(m), beta_min=beta_min, beta_max=beta_max
            ),
        )
    if method_key == "ddpm":
        return ObjectiveBundle(
            method=method_key,
            path=vp_path,
            loss_fn=DDPMLoss(vp_path),
            sample_t_end=t_end,
            wrap_velocity=lambda m: EpsilonToVelocityWrapper(_maybe_cfg(m), path=vp_path),
        )

    raise ValueError(
        f"Unknown method {method!r}. Choose from ['fm_ot', 'fm_dif', 'sm_dif', 'scoreflow', 'ddpm']."
    )
