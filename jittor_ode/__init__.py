"""Standalone ODE integrators and divergence estimators for Jittor.

Jittor does not ship an equivalent of ``torchdiffeq``. This package is the
fixed-step and adaptive solver used by the Flow Matching reproduction, factored
out so other Jittor models can integrate a neural vector field without depending
on the rest of this repository.

Example::

    import jittor as jt
    from jittor_ode import ODESolver, odeint

    def velocity(x, t, **_):
        return -x

    solver = ODESolver(velocity)
    x1 = solver.sample(jt.randn((8, 4)), method="dopri5", atol=1e-5, rtol=1e-5)
    print(solver.last_nfe)
"""

from jittor_ode.hutchinson import hutchinson_divergence, rademacher_like
from jittor_ode.solver import ODESolver, SUPPORTED_SOLVERS, odeint_jittor
from jittor_ode.tensor_utils import gaussian_log_prob, gradient

odeint = odeint_jittor

__all__ = [
    "ODESolver",
    "SUPPORTED_SOLVERS",
    "gaussian_log_prob",
    "gradient",
    "hutchinson_divergence",
    "odeint",
    "odeint_jittor",
    "rademacher_like",
]
