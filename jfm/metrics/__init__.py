"""Evaluation metrics (FID, ODE trajectory MSE, throughput and memory profiling)."""

from jfm.metrics.fid import frechet_distance, gaussian_stats

__all__ = ["frechet_distance", "gaussian_stats"]
