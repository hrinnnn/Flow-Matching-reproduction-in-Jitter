"""5-layer Swish MLP for 2D Flow Matching / Score Matching (Paper Appendix E).

Ported 1-to-1 from ``facebookresearch/flow_matching/examples/2d_flow_matching.ipynb``:
5 Linear layers with ``hidden_dim=512`` and Swish (SiLU) activations.
"""

from typing import Optional

import jittor as jt
from jittor import nn


class Swish(nn.Module):
    """Swish activation x * sigmoid(x)."""

    def execute(self, x: jt.Var) -> jt.Var:
        return jt.sigmoid(x) * x


class ToyMLP(nn.Module):
    """Time-conditioned 5-layer MLP for low-dimensional velocity / score / noise prediction."""

    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 5,
    ):
        super().__init__()
        assert num_layers >= 2
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        layers = [nn.Linear(input_dim + time_dim, hidden_dim), Swish()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), Swish()])
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.main = nn.Sequential(*layers)

    def execute(self, x: jt.Var, t: jt.Var, y: Optional[jt.Var] = None, **extras) -> jt.Var:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        if not isinstance(t, jt.Var):
            t = jt.array([float(t)], dtype=x.dtype)
        t_flat = t.cast(x.dtype).reshape(-1, 1)
        if t_flat.shape[0] == 1 and x_flat.shape[0] > 1:
            t_flat = t_flat.broadcast((x_flat.shape[0], 1))
        h = jt.concat([x_flat, t_flat], dim=1)
        out = self.main(h)
        return out.reshape(orig_shape)
