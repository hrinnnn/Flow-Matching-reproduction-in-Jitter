"""Backward-compatible import path. The implementation lives in ``jittor_ode``."""

from jittor_ode import ODESolver, SUPPORTED_SOLVERS, odeint_jittor

__all__ = ["ODESolver", "SUPPORTED_SOLVERS", "odeint_jittor"]
