"""CIFAR-10 loader for unconditional and class-conditional Flow Matching.

Images are returned as float32 ``CHW`` in ``[-1, 1]``. Horizontal flip is optional and
applied to both paths with the same code path, so FM-Dif and FM-OT stay paired.
"""

from typing import Callable, Optional

import numpy as np
from jittor.dataset import Dataset
from jittor.dataset.cifar import CIFAR10


def _to_minus_one_one(img, random_flip: bool) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if random_flip and np.random.rand() < 0.5:
        arr = arr[:, ::-1, :]
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return arr / 127.5 - 1.0


def build_cifar10(
    root: str,
    train: bool = True,
    download: bool = True,
    random_flip: bool = False,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
) -> Dataset:
    """Construct a Jittor CIFAR-10 dataset with the Flow Matching preprocessing."""
    dataset = CIFAR10(root=root, train=train, download=download)
    flip = bool(random_flip and train)

    def _transform(img):
        return _to_minus_one_one(img, random_flip=flip)

    dataset.transform = _transform
    dataset.set_attrs(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset
