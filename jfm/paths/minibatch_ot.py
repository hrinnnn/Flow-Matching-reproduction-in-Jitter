"""Minibatch optimal-transport coupling for conditional flow matching.

Independent coupling draws ``x_0`` and ``x_1`` separately. Minibatch OT (Tong et al.,
2024) instead matches them inside the current batch by minimizing the squared
Euclidean cost. The conditional path and the regression target stay the same;
only which noise is paired with which data point changes.
"""

from typing import Tuple

import numpy as np


def squared_cost(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Pairwise squared distance, shape ``[batch, batch]``."""
    a = np.asarray(x0, dtype=np.float64).reshape(x0.shape[0], -1)
    b = np.asarray(x1, dtype=np.float64).reshape(x1.shape[0], -1)
    if a.shape != b.shape:
        raise ValueError(f"Minibatch OT needs equal shapes, got {a.shape} and {b.shape}.")
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def assignment_for_targets(cost: np.ndarray) -> np.ndarray:
    """Return ``perm`` such that ``x0[perm[i]]`` is the match of ``x1[i]``."""
    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(cost)
    perm = np.empty(cost.shape[0], dtype=np.int64)
    perm[col] = row
    return perm


def sinkhorn_assignment(cost: np.ndarray, epsilon: float = 0.05, iterations: int = 40) -> np.ndarray:
    """Sinkhorn plan rounded back to a permutation by a second assignment."""
    scale = max(float(np.median(cost)), 1e-8)
    kernel = np.exp(-cost / (epsilon * scale))
    kernel = np.maximum(kernel, 1e-12)
    u = np.ones(cost.shape[0], dtype=np.float64)
    v = np.ones(cost.shape[1], dtype=np.float64)
    for _ in range(iterations):
        u = 1.0 / (kernel @ v)
        v = 1.0 / (kernel.T @ u)
    plan = u[:, None] * kernel * v[None, :]
    return assignment_for_targets(-plan)


def match_source_to_target(x0: np.ndarray, x1: np.ndarray, method: str = "exact") -> Tuple[np.ndarray, np.ndarray]:
    """Permute ``x0`` so that it is the minibatch OT match of ``x1``.

    Returns ``(matched_x0, permutation)``.
    """
    cost = squared_cost(x0, x1)
    if method == "exact":
        perm = assignment_for_targets(cost)
    elif method == "sinkhorn":
        perm = sinkhorn_assignment(cost)
    else:
        raise ValueError(f"Unknown OT method {method!r}.")
    return np.asarray(x0)[perm], perm


def couple_jittor(x0, x1, method: str = "exact"):
    """Jittor wrapper around :func:`match_source_to_target`."""
    import jittor as jt

    matched, perm = match_source_to_target(x0.numpy(), x1.numpy(), method=method)
    out = jt.array(matched.astype(np.float32))
    if tuple(out.shape) != tuple(x0.shape):
        out = out.reshape(x0.shape)
    return out
