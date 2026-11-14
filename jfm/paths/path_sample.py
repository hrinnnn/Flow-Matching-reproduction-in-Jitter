"""Dataclasses for path samples ported from facebookresearch/flow_matching."""

from dataclasses import dataclass, field

import jittor as jt


@dataclass
class PathSample:
    """Conditional affine probability path sample X_t = alpha_t X_1 + sigma_t X_0."""

    x_1: jt.Var = field(metadata={"help": "target samples X_1 (batch_size, ...)"})
    x_0: jt.Var = field(metadata={"help": "source noise samples X_0 (batch_size, ...)"})
    t: jt.Var = field(metadata={"help": "time samples t (batch_size,)"})
    x_t: jt.Var = field(metadata={"help": "interpolated samples X_t (batch_size, ...)"})
    dx_t: jt.Var = field(metadata={"help": "conditional velocity u_t(X_t|X_1) (batch_size, ...)"})


@dataclass
class DiscretePathSample:
    """Discrete probability path sample."""

    x_1: jt.Var = field(metadata={"help": "target samples X_1"})
    x_0: jt.Var = field(metadata={"help": "source samples X_0"})
    t: jt.Var = field(metadata={"help": "time samples t"})
    x_t: jt.Var = field(metadata={"help": "path samples X_t"})
