"""Datasets and synthetic generators."""

from jfm.datasets.checkerboard import sample_checkerboard
from jfm.datasets.cifar10 import build_cifar10

__all__ = ["build_cifar10", "sample_checkerboard"]
