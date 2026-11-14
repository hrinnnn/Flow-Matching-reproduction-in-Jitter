"""Bits-per-dimension (BPD / NLL) estimator for image CNFs (Paper Appendix C & E.2).

Implements Equations (34)-(37) and importance-weighted uniform dequantization Eq. (47)
from Lipman et al. (2023):
    x_uint8 in {0, ..., 255}^d,   u_k ~ U(0, 1)^d,
    y_k = 2^{-7} * (x_uint8 + u_k) - 1 in [-1, 1]^d,
    log p_data(x_uint8) = log (1/K sum_{k=1}^K p_model(y_k)) - 7 * d * log(2),
    BPD = - log p_data(x_uint8) / (d * log(2)).
"""

import math
from typing import Dict, Optional

import jittor as jt
import numpy as np

from jfm.solvers.ode_solver import ODESolver


def compute_bpd_batch(
    solver: ODESolver,
    images_minus1_to_1: jt.Var,
    num_dequant_samples: int = 1,
    t_start: float = 1.0,
    method: str = "dopri5",
    step_size: Optional[float] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    **model_extras,
) -> Dict[str, jt.Var]:
    """Compute importance-weighted NLL in bits/dim (BPD) for a batch of images in [-1, 1]."""
    assert num_dequant_samples >= 1
    b = images_minus1_to_1.shape[0]
    d = int(images_minus1_to_1.numel() // b)

    # Recover discrete pixel values in {0, ..., 255} from [-1, 1]
    x_uint8 = jt.floor(jt.clamp((images_minus1_to_1 * 0.5 + 0.5) * 255.0 + 0.5, 0.0, 255.0))

    log_p_samples = []
    total_nfe = 0
    for _ in range(num_dequant_samples):
        u = jt.rand(x_uint8.shape).cast(images_minus1_to_1.dtype)
        # phi^{-1}(x + u) = (x + u) / 128 - 1 in [-1, 1]
        x_dequant = (x_uint8 + u) / 128.0 - 1.0
        _, log_phi_p0 = solver.compute_likelihood(
            x_1=x_dequant,
            method=method,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
            time_grid=[t_start, 0.0],
            exact_divergence=False,
            enable_grad=False,
            **model_extras,
        )
        total_nfe += solver.last_nfe
        log_p_samples.append(log_phi_p0)

    # Importance-weighted log-mean-exp across K uniform dequantization draws (Eq. 47)
    stacked = jt.stack(log_p_samples, dim=0)  # [K, B]
    max_val = stacked.max(dim=0, keepdims=True)
    log_mean_p = (
        max_val.squeeze(0)
        + jt.log(jt.exp(stacked - max_val).mean(dim=0).maximum(1e-30))
    )

    # Apply change-of-variables constant +7 from [-1, 1] -> [0, 256] (Eq. 36-37)
    bpd = -log_mean_p / (d * math.log(2.0)) + 7.0
    avg_nfe = float(total_nfe) / float(num_dequant_samples)
    return {
        "bpd": bpd,
        "bpd_mean": bpd.mean(),
        "avg_nfe": jt.array([avg_nfe], dtype=jt.float32),
    }
