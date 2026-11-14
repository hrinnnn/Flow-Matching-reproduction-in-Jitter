"""Continuous adjoint backward for a neural ODE.

``torchdiffeq.odeint_adjoint`` differentiates the solution by integrating an
adjoint equation instead of storing every Runge-Kutta stage. Jittor-NODE does
not implement this. The forward pass here keeps only the states at the requested
output times. The backward pass walks those times in reverse, adds the upstream
gradient at each output, and integrates ``(z, a, dL/dθ)`` back to the previous
output. ``z`` is recomputed by the original dynamics, so the autograd graph of
the solver stages is never stored.
"""

from typing import Callable, Optional

import jittor as jt
from jittor import Function

from jittor_ode.compat import _as_time, _call_func
from jittor_ode.solver import odeint_jittor


class _AdjointIntegrate(Function):
    def execute(self, y0: jt.Var, t: jt.Var, *params: jt.Var):
        time_list = [float(v) for v in t.numpy().reshape(-1)]
        with jt.flag_scope(no_grad=1):
            states = odeint_jittor(
                lambda ts, y: _call_func(self.func, ts, y).stop_grad(),
                y0.stop_grad(),
                time_list,
                method=self.method,
                step_size=self.step_size,
                atol=self.atol,
                rtol=self.rtol,
            )
        self.time_list = time_list
        self.states = [state.stop_grad() for state in states]
        self.param_list = list(params)
        return jt.stack(self.states, dim=0)

    def grad(self, grad_out):
        # ``param_list`` may be taped copies created by ``Function``. The ODE
        # closure still reads the original tensors, so the VJP has to be taken
        # with respect to those.
        params = getattr(self, "live_params", self.param_list)
        time_list = self.time_list
        adj = jt.zeros_like(self.states[-1]).stop_grad()
        param_grads = [jt.zeros_like(param).stop_grad() for param in params]

        def augmented(t_scalar, state):
            z, adj_state = state[0], state[1]
            with jt.flag_scope(no_grad=0):
                z_in = z.stop_grad()
                z_in.start_grad()
                was_stopped = [param.is_stop_grad() for param in params]
                for param in params:
                    param.start_grad()
                velocity = _call_func(self.func, t_scalar, z_in)
                scalar = (velocity * adj_state.stop_grad()).sum()
                vjps = jt.grad(scalar, [z_in, *params], retain_graph=False)
                for param, stopped in zip(params, was_stopped):
                    if stopped:
                        param.stop_grad()
            dz = velocity.stop_grad()
            dadj = (-vjps[0]).stop_grad()
            dparams = tuple((-grad).stop_grad() for grad in vjps[1:])
            return (dz, dadj, *dparams)

        for index in range(len(time_list) - 1, 0, -1):
            adj = (adj + grad_out[index]).stop_grad()
            zeros = tuple(jt.zeros_like(param).stop_grad() for param in params)
            solved = odeint_jittor(
                augmented,
                (self.states[index], adj, *zeros),
                [time_list[index], time_list[index - 1]],
                method=self.method,
                step_size=self.step_size,
                atol=self.atol,
                rtol=self.rtol,
            )
            adj = solved[1][-1].stop_grad()
            param_grads = [
                (acc + solved[2 + i][-1]).stop_grad() for i, acc in enumerate(param_grads)
            ]
        adj = (adj + grad_out[0]).stop_grad()
        return (adj, None, *param_grads)


def odeint_adjoint(
    func: Callable,
    y0: jt.Var,
    t,
    *params: jt.Var,
    method: str = "rk4",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    options: Optional[dict] = None,
) -> jt.Var:
    """Like :func:`jittor_ode.compat.odeint`, but backward uses the adjoint ODE.

    If ``params`` is empty and ``func`` is a module, its parameters are tracked.
    """
    options = options or {}
    if not params and hasattr(func, "parameters"):
        params = tuple(param for param in func.parameters() if not param.is_stop_grad())
    runner = _AdjointIntegrate()
    runner.func = func
    runner.live_params = list(params)
    runner.method = method
    runner.rtol = float(rtol)
    runner.atol = float(atol)
    runner.step_size = options.get("step_size")
    time = _as_time(t).stop_grad()
    return runner(y0, time, *params)
