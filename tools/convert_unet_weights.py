"""Map a guided-diffusion / flow_matching PyTorch U-Net state dict onto this repo's Jittor U-Net.

The two modules compute the same ADM U-Net, but Jittor names differ:

* ``input_blocks.3.1`` becomes ``input_blocks.m3.layers.m1``
* ``in_layers.0`` / ``in_layers.2`` become ``in_norm`` / ``in_conv``
* ``emb_layers.1`` becomes ``emb_linear``
* ``out_layers.0`` / ``out_layers.3`` become ``out_norm`` / ``out_conv``

``time_embed.*``, ``label_emb.weight`` and ``out.*`` already use the same names.
No pretrained checkpoint is stored here; pass a ``.pt`` file once training or the
official weights are available.
"""

import re
from typing import Dict, Mapping

import numpy as np

_BLOCK = re.compile(r"^(input_blocks|output_blocks)\.(\d+)\.(\d+)")
_MIDDLE = re.compile(r"^middle_block\.(\d+)")
_RENAMES = (
    (".in_layers.0.", ".in_norm."),
    (".in_layers.2.", ".in_conv."),
    (".emb_layers.1.", ".emb_linear."),
    (".out_layers.0.", ".out_norm."),
    (".out_layers.3.", ".out_conv."),
)


def pytorch_key_to_jittor(key: str) -> str:
    """Rename one PyTorch parameter key to the Jittor U-Net name."""
    key = _BLOCK.sub(r"\1.m\2.layers.m\3", key)
    key = _MIDDLE.sub(r"middle_block.layers.m\1", key)
    for src, dst in _RENAMES:
        key = key.replace(src, dst)
    return key


def convert_state_dict(pytorch_state: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a new dict whose keys are Jittor parameter names."""
    converted = {}
    for key, value in pytorch_state.items():
        if key.endswith("num_batches_tracked"):
            continue
        converted[pytorch_key_to_jittor(key)] = np.asarray(value)
    return converted


def assign_jittor_model(model, arrays: Mapping[str, np.ndarray]) -> Dict[str, str]:
    """Copy converted arrays into a Jittor module. Returns a small report."""
    import jittor as jt

    named = dict(model.named_parameters())
    missing = []
    shape_mismatch = []
    used = set()
    for name, param in named.items():
        if name not in arrays:
            missing.append(name)
            continue
        value = arrays[name]
        if tuple(value.shape) != tuple(param.shape):
            if value.ndim == param.ndim + 1 and value.shape[-1] == 1:
                value = np.squeeze(value, axis=-1)
            elif param.ndim == value.ndim + 1 and param.shape[-1] == 1:
                value = np.expand_dims(value, axis=-1)
        if tuple(value.shape) != tuple(param.shape):
            shape_mismatch.append(f"{name}: checkpoint {value.shape} vs model {tuple(param.shape)}")
            continue
        param.assign(jt.array(value.astype(np.float32)))
        used.add(name)
    unused = sorted(set(arrays) - used)
    if shape_mismatch:
        raise ValueError("Shape mismatch:\n" + "\n".join(shape_mismatch))
    return {
        "loaded": str(len(used)),
        "missing": ",".join(missing),
        "unused": ",".join(unused),
    }


def load_pytorch_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load a ``.pt`` / ``.pth`` file. Requires PyTorch only at conversion time."""
    import torch

    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        blob = blob["model"]
    out = {}
    for key, value in blob.items():
        name = key[7:] if key.startswith("module.") else key
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        out[name] = np.asarray(value)
    return out
