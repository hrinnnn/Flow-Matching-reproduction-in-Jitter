"""Deterministic random seed setup across Python, NumPy, and Jittor."""

import random

import jittor as jt
import numpy as np


def set_seed(seed: int) -> None:
    """Set global random seed for reproducible paired experiments."""
    random.seed(seed)
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)
