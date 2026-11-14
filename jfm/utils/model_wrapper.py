"""Model wrappers for ODE integration, NFE counting, Diffusion-to-ODE drift, and CFG."""

from typing import Optional

import jittor as jt
from jittor import nn

from jfm.utils.tensor_utils import expand_tensor_like


def _format_time_batch(t: jt.Var, batch_size: int, dtype=jt.float32) -> jt.Var:
    """Ensure time ``t`` is a 1D Jittor tensor of shape ``(batch_size,)``."""
    if not isinstance(t, jt.Var):
        t = jt.array([float(t)], dtype=dtype)
    t = t.cast(dtype).reshape(-1)
    if t.shape[0] == 1 and batch_size > 1:
        t = t.broadcast((batch_size,))
    return t


class ModelWrapper(nn.Module):
    """Base wrapper around a Jittor network with built-in NFE (function evaluation) tracking."""

    def __init__(self, model: Optional[nn.Module]):
        super().__init__()
        self.model = model
        self.nfe_counter: int = 0

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter

    def execute(self, x: jt.Var, t: jt.Var, **extras) -> jt.Var:
        t_batch = _format_time_batch(t, x.shape[0], dtype=x.dtype)
        self.nfe_counter += 1
        return self.model(x, t_batch, **extras)


class ScoreToVelocityWrapper(ModelWrapper):
    """Convert a VP Score Matching model s_theta(x, t) to Probability Flow ODE velocity.

    Follows Lipman et al. (2023) Appendix E.1 Eq. (43) & Eq. (46):
        s_theta(x, t) is trained to predict (x - mu_t) / sigma_t^2 = -grad_x log p_t(x|x_1),
        and the corresponding generating ODE velocity field is:
            u_t(x) = -0.5 * T'(1 - t) * (s_theta(x, t) - x).
    """

    def __init__(self, model: nn.Module, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__(model)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def execute(self, x: jt.Var, t: jt.Var, **extras) -> jt.Var:
        t_batch = _format_time_batch(t, x.shape[0], dtype=x.dtype)
        self.nfe_counter += 1
        score_pred = self.model(x, t_batch, **extras)
        beta_rev = self.beta_min + (1.0 - t_batch) * (self.beta_max - self.beta_min)
        beta_rev = expand_tensor_like(beta_rev, x)
        return -0.5 * beta_rev * (score_pred - x)


class EpsilonToVelocityWrapper(ModelWrapper):
    """Convert a DDPM noise-prediction model eps_theta(x, t) to ODE velocity.

    Works for any AffineProbPath (in particular VP Diffusion, matching Paper Eq. 46
    with s_t(x) = eps_t(x) / sigma_t).
    """

    def __init__(self, model: nn.Module, path, clamp_sigma: float = 1e-5):
        super().__init__(model)
        self.path = path
        self.clamp_sigma = clamp_sigma

    def execute(self, x: jt.Var, t: jt.Var, **extras) -> jt.Var:
        t_batch = _format_time_batch(t, x.shape[0], dtype=x.dtype)
        self.nfe_counter += 1
        eps_pred = self.model(x, t_batch, **extras)
        sched = self.path.scheduler(t_batch)
        alpha_t = expand_tensor_like(sched.alpha_t.maximum(self.clamp_sigma), x)
        sigma_t = expand_tensor_like(sched.sigma_t.maximum(self.clamp_sigma), x)
        d_alpha_t = expand_tensor_like(sched.d_alpha_t, x)
        d_sigma_t = expand_tensor_like(sched.d_sigma_t, x)
        a_t = d_alpha_t / alpha_t
        b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t
        return a_t * x + b_t * eps_pred


class CFGScaledModel(ModelWrapper):
    """Classifier-Free Guidance wrapper ported from facebookresearch/flow_matching."""

    def __init__(self, model: nn.Module, cfg_scale: float = 0.0):
        super().__init__(model)
        self.default_cfg_scale = cfg_scale

    def execute(
        self,
        x: jt.Var,
        t: jt.Var,
        cfg_scale: Optional[float] = None,
        label: Optional[jt.Var] = None,
        **extras,
    ) -> jt.Var:
        scale = self.default_cfg_scale if cfg_scale is None else cfg_scale
        t_batch = _format_time_batch(t, x.shape[0], dtype=x.dtype)

        if label is None:
            self.nfe_counter += 1
            return self.model(x, t_batch, y=None, **extras).float32()

        if scale != 0.0:
            # Count both conditional and unconditional network evaluations accurately in NFE
            self.nfe_counter += 2
            cond = self.model(x, t_batch, y=label, **extras)
            uncond = self.model(x, t_batch, y=None, **extras)
            out = (1.0 + scale) * cond - scale * uncond
            return out.float32()

        self.nfe_counter += 1
        return self.model(x, t_batch, y=label, **extras).float32()
