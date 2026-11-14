"""Tensor helper functions ported from facebookresearch/flow_matching to Jittor."""

import math
from typing import Optional

import jittor as jt


def unsqueeze_to_match(source: jt.Var, target: jt.Var, how: str = "suffix") -> jt.Var:
    """Unsqueeze ``source`` tensor to match dimensionality of ``target``."""
    assert how in ("prefix", "suffix"), f"Unsupported mode {how!r}; expected 'prefix' or 'suffix'."
    dim_diff = target.ndim - source.ndim
    out = source
    for _ in range(dim_diff):
        if how == "prefix":
            out = out.unsqueeze(0)
        else:
            out = out.unsqueeze(-1)
    return out


def expand_tensor_like(input_tensor: jt.Var, expand_to: jt.Var) -> jt.Var:
    """Expand a 1D batch tensor ``(B,)`` to match ``expand_to`` shape ``(B, ...)``."""
    assert input_tensor.ndim == 1, f"Input tensor must be 1D, got shape {input_tensor.shape}."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"Batch dimension mismatch: {input_tensor.shape} vs {expand_to.shape}."
    dim_diff = expand_to.ndim - input_tensor.ndim
    reshaped = input_tensor.reshape(-1, *([1] * dim_diff))
    return reshaped.broadcast(expand_to.shape)


def gradient(
    output: jt.Var,
    x: jt.Var,
    grad_outputs: Optional[jt.Var] = None,
    retain_graph: bool = True,
) -> jt.Var:
    """Compute vector-Jacobian product d <output, grad_outputs> / dx in Jittor."""
    if grad_outputs is None:
        scalar = output.sum()
    else:
        scalar = (output * grad_outputs.stop_grad()).sum()
    return jt.grad(scalar, x, retain_graph=retain_graph)


def gaussian_log_prob(x: jt.Var) -> jt.Var:
    """Standard isotropic normal log-density log N(x | 0, I) summed over non-batch dims."""
    flat = x.reshape(x.shape[0], -1)
    d = flat.shape[1]
    return -0.5 * (d * math.log(2.0 * math.pi) + (flat * flat).sum(dim=1))
