"""U-Net building blocks ported from guided-diffusion / facebookresearch/flow_matching.

PyTorch ``examples/image/models/nn.py`` and ``unet.py`` are the reference.
Jittor differences that matter for the reproduction:

* ``model.train()`` / ``model.eval()`` flip ``Module.is_train``. Dropout reads that flag.
* ``nn.Dropout`` is created with ``is_train=False`` and only activates after ``train()``.
* Convolution and linear layers are re-initialized with the PyTorch default
  (kaiming uniform, a = sqrt(5), bias uniform on ``1/sqrt(fan_in)``).
"""

import math

import jittor as jt
from jittor import init, nn


class SiLU(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return jt.nn.silu(x)


class Identity(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return x


class ModuleList(nn.Module):
    """Attribute-registered list so Jittor ``parameters()`` sees every child."""

    def __init__(self, modules=None):
        super().__init__()
        self._mods = []
        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module: nn.Module) -> None:
        idx = len(self._mods)
        setattr(self, f"m{idx}", module)
        self._mods.append(module)

    def __iter__(self):
        return iter(self._mods)

    def __len__(self) -> int:
        return len(self._mods)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._mods[idx]


def _kaiming_uniform_like_torch(weight: jt.Var) -> None:
    """Match ``torch.nn.init.kaiming_uniform_(weight, a=sqrt(5))``."""
    init.kaiming_uniform_(weight, a=math.sqrt(5))


def _uniform_bias_like_torch(weight: jt.Var, bias: jt.Var) -> None:
    fan_in = 1
    for size in weight.shape[1:]:
        fan_in *= int(size)
    bound = 1.0 / math.sqrt(fan_in)
    init.uniform_(bias, -bound, bound)


def _init_conv_or_linear(module: nn.Module) -> nn.Module:
    if getattr(module, "weight", None) is not None:
        _kaiming_uniform_like_torch(module.weight)
    if getattr(module, "bias", None) is not None:
        _uniform_bias_like_torch(module.weight, module.bias)
    return module


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        layer = nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        layer = nn.Conv2d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
    return _init_conv_or_linear(layer)


def linear(*args, **kwargs) -> nn.Linear:
    return _init_conv_or_linear(nn.Linear(*args, **kwargs))


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.assign(jt.zeros_like(param))
    return module


def normalization(channels: int) -> nn.GroupNorm:
    """GroupNorm with 32 groups, matching guided-diffusion ``GroupNorm32``."""
    if channels % 32 != 0:
        raise ValueError(
            f"ADM GroupNorm expects channels divisible by 32, got {channels}."
        )
    return nn.GroupNorm(32, channels)


def timestep_embedding(timesteps: jt.Var, dim: int, max_period: int = 10000) -> jt.Var:
    """Sinusoidal embedding. ``timesteps`` may be fractional, shape ``[N]``."""
    half = dim // 2
    freqs = jt.exp(-math.log(max_period) * jt.arange(half).float32() / half)
    args = timesteps.float32().reshape(-1, 1) * freqs.reshape(1, -1)
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=1)
    if dim % 2:
        embedding = jt.concat(
            [embedding, jt.zeros((embedding.shape[0], 1), dtype=embedding.dtype)],
            dim=1,
        )
    return embedding


class AvgPool2d(nn.Module):
    """Non-overlapping average pool. Spatial size must be divisible by the kernel."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def execute(self, x: jt.Var) -> jt.Var:
        k = self.kernel_size
        n, c, h, w = x.shape
        if h % k != 0 or w % k != 0:
            raise ValueError(f"AvgPool2d got spatial {(h, w)} not divisible by {k}.")
        x = x.reshape(n, c, h // k, k, w // k, k)
        return x.mean(dims=[3, 5])
