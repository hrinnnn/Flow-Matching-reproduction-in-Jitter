"""Tests that do not need a compiled Jittor runtime."""

import numpy as np

from jfm.metrics.fid import frechet_distance, gaussian_stats
from jfm.utils.schedule import linear_warmup_decay


def test_warmup_then_linear_decay_matches_appendix():
    peak = 5e-4
    floor = 1e-8
    total = 391000
    warmup = 45000
    first = linear_warmup_decay(1, total, warmup, peak, floor)
    assert abs(first - (floor + (peak - floor) / warmup)) < 1e-12
    assert abs(linear_warmup_decay(warmup, total, warmup, peak, floor) - peak) < 1e-12
    assert abs(linear_warmup_decay(total, total, warmup, peak, floor) - floor) < 1e-12


def test_frechet_distance_is_zero_for_identical_gaussians():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(64, 8))
    mu, sigma = gaussian_stats(feats)
    distance = frechet_distance(mu, sigma, mu, sigma)
    assert distance < 1e-6
