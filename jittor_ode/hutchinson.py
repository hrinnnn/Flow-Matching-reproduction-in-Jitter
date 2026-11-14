"""Hutchinson trace estimator for a batch of vector fields.

One ``jt.grad`` call estimates ``div v(x) = E_z[z^T (Dv) z]`` for ``E[zz^T] = I``.
Rademacher probes ``z in {-1, +1}`` are the default used by FFJORD and by
Lipman et al. (2023), Appendix C, Equation 34.
"""

from typing import Optional, Tuple

import jittor as jt

from jittor_ode.tensor_utils import gradient


def rademacher_like(x: jt.Var) -> jt.Var:
    """Probe in ``{-1, +1}`` with the same shape as ``x``."""
    return ((jt.randn_like(x) < 0).float32() * 2.0 - 1.0).stop_grad()


def hutchinson_divergence(
    y: jt.Var,
    x: jt.Var,
    z: Optional[jt.Var] = None,
    retain_graph: bool = False,
) -> Tuple[jt.Var, jt.Var]:
    """Estimate ``div(dy/dx)`` for a batched map ``y = v(x)``.

    ``y`` and ``x`` have shape ``[batch, ...]``. The return value is
    ``(divergence [batch], probe z)``. Pass the same ``z`` on every step of an
    ODE if the estimator must stay continuous in time.
    """
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"Batch mismatch between y {y.shape} and x {x.shape}.")
    if z is None:
        z = rademacher_like(x)
    z = z.stop_grad()
    y_flat = y.reshape(y.shape[0], -1)
    z_flat = z.reshape(z.shape[0], -1)
    directional = (y_flat * z_flat).sum(dim=1)
    vjp = gradient(directional, x, retain_graph=retain_graph)
    divergence = (vjp.reshape(x.shape[0], -1) * z_flat).sum(dim=1)
    return divergence, z
