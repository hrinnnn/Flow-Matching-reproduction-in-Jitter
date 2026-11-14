"""Fréchet Inception Distance on precomputed feature statistics.

The distance itself is pure NumPy/SciPy. Extracting Inception features needs a separate
backend because the course environment is Jittor-first; ``scripts/eval_cifar10.py`` saves
uint8 samples so FID can be computed once an Inception model is available.
"""

import numpy as np


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """FID between two Gaussian feature distributions. Lower is better."""
    from scipy import linalg

    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def gaussian_stats(features: np.ndarray):
    """Mean and covariance of an ``[N, D]`` feature matrix."""
    feats = np.asarray(features, dtype=np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma
