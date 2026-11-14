"""torchdiffeq-style ``odeint`` on top of the Jittor solvers.

``SCU-JittorMed/Jittor-NODE`` (2025) already offers ``euler`` and ``rk4``. It does
not offer adaptive ``dopri5``, a midpoint method, or the continuous adjoint.
``odeint`` here follows the ``torchdiffeq`` calling convention so a PyTorch
Neural ODE can move over without rewriting the solver loop:

    ys = odeint(func, y0, t, method="dopri5", rtol=1e-5, atol=1e-5)

``func(t, y)`` returns ``dy/dt``. A Jittor module with ``execute(self, t, y)`` is
accepted as well. ``t`` is a 1-D increasing or decreasing sequence of output times.
The result has shape ``[len(t), *y0.shape]``.
"""

from typing import Callable, Optional, Sequence, Union

import jittor as jt

from jittor_ode.solver import odeint_jittor

ArrayLike = Union[jt.Var, Sequence[float]]


def _as_time(t: ArrayLike) -> jt.Var:
    if isinstance(t, jt.Var):
        return t.float32().reshape(-1)
    return jt.array([float(v) for v in t]).float32()


def _call_func(func: Callable, t_scalar: float, y: jt.Var) -> jt.Var:
    if hasattr(func, "execute"):
        return func.execute(t_scalar, y)
    return func(t_scalar, y)


def odeint(
    func: Callable,
    y0: jt.Var,
    t: ArrayLike,
    method: str = "dopri5",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    options: Optional[dict] = None,
) -> jt.Var:
    """Integrate ``dy/dt = func(t, y)`` and return the state at every time in ``t``."""
    options = options or {}
    times = _as_time(t)
    time_list = [float(v) for v in times.numpy().reshape(-1)]
    if len(time_list) < 2:
        raise ValueError("t must contain at least two output times.")

    def wrapped(t_scalar: float, y: jt.Var) -> jt.Var:
        return _call_func(func, t_scalar, y)

    states = odeint_jittor(
        wrapped,
        y0,
        time_list,
        method=method,
        step_size=options.get("step_size"),
        atol=atol,
        rtol=rtol,
    )
    return jt.stack(list(states), dim=0)
