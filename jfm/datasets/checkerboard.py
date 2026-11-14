"""2D Checkerboard toy distribution ported 1-to-1 from facebookresearch/flow_matching.

In ``examples/2d_flow_matching.ipynb``:
    x1 ~ U(-2, 2)
    x2_ = U(0, 1) - 2 * Bernoulli(0.5)
    x2 = x2_ + (floor(x1) % 2)
    data = concat([x1, x2]) / 0.45   (supported on 8 alternating squares in [-4.44, 4.44]^2,
                                      ground-truth uniform density = (0.45^2) / 8 ≈ 0.0253125).
"""

import jittor as jt


def sample_checkerboard(batch_size: int = 200) -> jt.Var:
    """Draw ``batch_size`` i.i.d. points from the 2D checkerboard distribution."""
    x1 = jt.rand((batch_size,)) * 4.0 - 2.0
    bern = jt.randint(0, 2, shape=(batch_size,)).float32()
    x2_ = jt.rand((batch_size,)) - bern * 2.0
    # Python/PyTorch floor(x1) % 2 is always in {0.0, 1.0} even for negative x1;
    # note C fmod on negative numbers can be -1, so use ((floor(x1) % 2) + 2) % 2!
    parity = jt.floor(x1).int32()
    parity = ((parity % 2) + 2) % 2
    x2 = x2_ + parity.float32()
    data = jt.stack([x1, x2], dim=1) / 0.45
    return data.float32()
