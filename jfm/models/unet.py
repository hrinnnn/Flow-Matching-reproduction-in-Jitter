"""ADM U-Net in Jittor, ported from facebookresearch/flow_matching examples/image/models/unet.py.

That file is itself a lightly modified guided-diffusion U-Net (Dhariwal & Nichol, 2021).
Two presets are kept separate on purpose:

* ``cifar10_paper`` follows Lipman et al. (2023) Appendix E, Table 3:
  256 channels, depth 2, multipliers (1, 2, 2, 2), 4 heads / 64 channels per head,
  attention at 16x16 (downsample rate 2), dropout 0.
* ``cifar10_repo`` is the later default in ``model_configs.py`` and is not the Table 1 setup.
"""

from typing import Optional, Sequence, Tuple, Union

import jittor as jt
from jittor import nn
from jittor.misc import chunk

from jfm.models.nn import (
    AvgPool2d,
    Identity,
    ModuleList,
    SiLU,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class TimestepBlock(nn.Module):
    def execute(self, x: jt.Var, emb: jt.Var) -> jt.Var:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = ModuleList(layers)

    def execute(self, x: jt.Var, emb: jt.Var) -> jt.Var:
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

    def execute(self, x: jt.Var) -> jt.Var:
        if x.shape[1] != self.channels:
            raise ValueError(f"Upsample expected {self.channels} channels, got {x.shape[1]}.")
        x = nn.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            if self.channels != self.out_channels:
                raise ValueError("Avg-pool downsample cannot change the channel count.")
            self.op = AvgPool2d(kernel_size=stride)

    def execute(self, x: jt.Var) -> jt.Var:
        if x.shape[1] != self.channels:
            raise ValueError(f"Downsample expected {self.channels} channels, got {x.shape[1]}.")
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_norm = normalization(channels)
        self.in_silu = SiLU()
        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = Identity()
            self.x_upd = Identity()

        self.emb_silu = SiLU()
        self.emb_linear = linear(
            emb_channels,
            2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        )
        self.out_norm = normalization(self.out_channels)
        self.out_silu = SiLU()
        self.out_drop = nn.Dropout(p=dropout)
        self.out_conv = zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))

        if self.out_channels == channels:
            self.skip_connection = Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def execute(self, x: jt.Var, emb: jt.Var) -> jt.Var:
        if self.updown:
            h = self.in_silu(self.in_norm(x))
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_conv(h)
        else:
            h = self.in_conv(self.in_silu(self.in_norm(x)))

        emb_out = self.emb_linear(self.emb_silu(emb))
        while emb_out.ndim < h.ndim:
            emb_out = emb_out.unsqueeze(-1)
        if self.use_scale_shift_norm:
            scale, shift = chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1.0 + scale) + shift
            h = self.out_conv(self.out_drop(self.out_silu(h)))
        else:
            h = h + emb_out
            h = self.out_conv(self.out_drop(self.out_silu(self.out_norm(h))))
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    """Split QKV before heads. Matches guided-diffusion ``QKVAttention``."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv: jt.Var) -> jt.Var:
        bs, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError(f"qkv width {width} is not divisible by 3 * {self.n_heads} heads.")
        ch = width // (3 * self.n_heads)
        q, k, v = chunk(qkv, 3, dim=1)
        scale = (ch ** -0.25)
        q = (q * scale).reshape(bs * self.n_heads, ch, length)
        k = (k * scale).reshape(bs * self.n_heads, ch, length)
        v = v.reshape(bs * self.n_heads, ch, length)
        weight = jt.matmul(q.transpose(1, 2), k)
        weight = nn.softmax(weight.float32(), dim=-1).cast(q.dtype)
        out = jt.matmul(v, weight.transpose(1, 2))
        return out.reshape(bs, -1, length)


class QKVAttentionLegacy(nn.Module):
    """Split heads before QKV. Matches guided-diffusion ``QKVAttentionLegacy``."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv: jt.Var) -> jt.Var:
        bs, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError(f"qkv width {width} is not divisible by 3 * {self.n_heads} heads.")
        ch = width // (3 * self.n_heads)
        qkv = qkv.reshape(bs * self.n_heads, ch * 3, length)
        q, k, v = chunk(qkv, 3, dim=1)
        scale = ch ** -0.25
        weight = jt.matmul((q * scale).transpose(1, 2), k * scale)
        weight = nn.softmax(weight.float32(), dim=-1).cast(q.dtype)
        out = jt.matmul(v, weight.transpose(1, 2))
        return out.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"channels {channels} is not divisible by num_head_channels {num_head_channels}."
                )
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def execute(self, x: jt.Var) -> jt.Var:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, -1)
        h_out = self.proj_out(self.attention(self.qkv(self.norm(flat))))
        return (flat + h_out).reshape(b, c, h, w)


PRESETS = {
    # Lipman et al. 2023, Appendix E, Table 3, CIFAR-10 column.
    "cifar10_paper": dict(
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(2,),
        dropout=0.0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
    ),
    # facebookresearch/flow_matching examples/image/models/model_configs.py "cifar10".
    "cifar10_repo": dict(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=4,
        attention_resolutions=(2,),
        dropout=0.3,
        channel_mult=(2, 2, 2),
        conv_resample=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
    ),
    # Table 3, ImageNet-32 column. Data pipeline is not trained by default.
    "imagenet32_paper": dict(
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=(2, 4),
        dropout=0.0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
    ),
    "tiny": dict(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=1,
        attention_resolutions=(2,),
        dropout=0.0,
        channel_mult=(1, 2),
        conv_resample=True,
        num_heads=4,
        num_head_channels=16,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
    ),
}


class UNetModel(nn.Module):
    """Time-conditional ADM U-Net. Optional class embedding supports CFG.

    ``num_classes=None`` is the unconditional Table 1 model. ``num_classes=10`` adds a
    learned class embedding plus one null index (``10``) used when the label is dropped.
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (2,),
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 2, 2, 2),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = True,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.attention_resolutions = tuple(attention_resolutions)
        self.channel_mult = tuple(channel_mult)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes + 1, time_embed_dim, padding_idx=num_classes)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=ch)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    @classmethod
    def from_preset(cls, name: str, num_classes: Optional[int] = None, **overrides) -> "UNetModel":
        if name not in PRESETS:
            raise ValueError(f"Unknown U-Net preset {name!r}. Choose from {sorted(PRESETS)}.")
        kwargs = dict(PRESETS[name])
        kwargs["num_classes"] = num_classes
        kwargs.update(overrides)
        return cls(**kwargs)

    def execute(
        self,
        x: jt.Var,
        timesteps: jt.Var,
        y: Optional[jt.Var] = None,
        label: Optional[jt.Var] = None,
        **_unused,
    ) -> jt.Var:
        if not isinstance(timesteps, jt.Var):
            timesteps = jt.array([float(timesteps)], dtype=x.dtype)
        timesteps = timesteps.cast(x.dtype).reshape(-1)
        if timesteps.shape[0] == 1 and x.shape[0] > 1:
            timesteps = jt.ones((x.shape[0],), dtype=x.dtype) * timesteps

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            labels = y if y is not None else label
            if labels is None:
                labels = jt.zeros((x.shape[0],), dtype=jt.int32) + int(self.num_classes)
            emb = emb + self.label_emb(labels.int32().reshape(-1))

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
