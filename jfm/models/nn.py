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


class GroupNorm32(nn.Module):
    """GroupNorm whose variance matches PyTorch.

    ``jittor.nn.GroupNorm`` uses ``E[x^2] - E[x]^2``. That is the population
    variance in exact arithmetic, but it cancels when the activations are large,
    so a U-Net loaded from PyTorch drifts. PyTorch uses the two-pass biased
    estimator ``E[(x - E[x])^2]`` (``correction=0``). Ports such as Point2RBox-v3
    had to replace Jittor's formula for the same reason.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"{num_channels} channels is not divisible by {num_groups} groups.")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = jt.ones((num_channels,))
        self.bias = jt.zeros((num_channels,))

    def execute(self, x: jt.Var) -> jt.Var:
        n, channels = x.shape[0], x.shape[1]
        per_group = channels // self.num_groups
        spatial = 1
        for size in x.shape[2:]:
            spatial *= int(size)
        # Reduce one axis of a 3D view. A 4D reduction of the same numbers
        # aborts the g++-14 CPU kernel on some Macs.
        grouped = x.reshape(n, self.num_groups, per_group * spatial)
        mean = grouped.mean(dims=[2]).reshape(n, self.num_groups, 1)
        centered = grouped - mean
        var = (centered * centered).mean(dims=[2]).reshape(n, self.num_groups, 1)
        normed = ((centered / jt.sqrt(var + self.eps))).reshape(x.shape)
        scale = self.weight.reshape(1, channels, *([1] * (x.ndim - 2)))
        shift = self.bias.reshape(1, channels, *([1] * (x.ndim - 2)))
        return normed * scale + shift


def normalization(channels: int) -> GroupNorm32:
    """32-group GroupNorm used by the ADM U-Net, with PyTorch's variance."""
    if channels % 32 != 0:
        raise ValueError(
            f"ADM GroupNorm expects channels divisible by 32, got {channels}."
        )
    return GroupNorm32(32, channels)


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
