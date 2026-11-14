"""Small tensor helpers so ``jittor_ode`` does not depend on the Flow Matching package."""

import math
from typing import Optional

import jittor as jt


def gradient(
    output: jt.Var,
    x: jt.Var,
    grad_outputs: Optional[jt.Var] = None,
    retain_graph: bool = True,
) -> jt.Var:
    """Vector-Jacobian product of ``output`` with respect to ``x``."""
    if grad_outputs is None:
        scalar = output.sum()
    else:
        scalar = (output * grad_outputs.stop_grad()).sum()
    return jt.grad(scalar, x, retain_graph=retain_graph)


def gaussian_log_prob(x: jt.Var) -> jt.Var:
    """``log N(x | 0, I)`` summed over every dimension except the batch."""
    flat = x.reshape(x.shape[0], -1)
    dim = flat.shape[1]
    return -0.5 * (dim * math.log(2.0 * math.pi) + (flat * flat).sum(dim=1))
