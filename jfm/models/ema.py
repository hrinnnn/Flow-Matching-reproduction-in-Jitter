"""Exponential moving average of a Jittor module.

The decay warmup matches ``facebookresearch/flow_matching/examples/image/models/ema.py``:
``decay = min(decay, (1 + n) / (10 + n))``. Shadow weights are ``stop_grad`` so Adam
does not update them. Call :meth:`update` after each optimizer step, and
:meth:`copy_to_model` before evaluation.
"""

from typing import List

import jittor as jt
from jittor import nn


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.model = model
        self.decay = float(decay)
        self.num_updates = jt.zeros((1,), dtype=jt.int32).stop_grad()
        self.shadows = nn.Module()
        self._shadow_vars: List[jt.Var] = []
        for i, param in enumerate(model.parameters()):
            shadow = jt.zeros_like(param).stop_grad()
            shadow.assign(param.stop_grad())
            setattr(self.shadows, f"p{i}", shadow)
            self._shadow_vars.append(shadow)
        self._backup: List[jt.Var] = []

    def update(self) -> None:
        self.num_updates.assign(self.num_updates + 1)
        step = int(self.num_updates.item())
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        one_minus = 1.0 - decay
        for shadow, param in zip(self._shadow_vars, self.model.parameters()):
            shadow.assign(shadow * decay + param.stop_grad() * one_minus)

    def copy_to_model(self) -> None:
        self._backup = []
        for shadow, param in zip(self._shadow_vars, self.model.parameters()):
            backup = jt.zeros_like(param).stop_grad()
            backup.assign(param)
            self._backup.append(backup)
            param.assign(shadow)

    def restore(self) -> None:
        if not self._backup:
            raise RuntimeError("restore() called before copy_to_model().")
        for backup, param in zip(self._backup, self.model.parameters()):
            param.assign(backup)
        self._backup = []
