"""Native Jittor ODE solvers & CNF log-likelihood solver.

Implements fixed-step (``euler``, ``midpoint``, ``heun3``, ``rk4``) and adaptive-step
Dormand-Prince 5(4) (``dopri5``) integrators directly on ``jittor.Var`` states without
any PyTorch / ``torchdiffeq`` dependency, mirroring ``facebookresearch/flow_matching``'s
``ODESolver`` API while recording exact NFE counts.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import jittor as jt

from jfm.utils.model_wrapper import ModelWrapper
from jfm.utils.tensor_utils import gaussian_log_prob, gradient

StateType = Union[jt.Var, Tuple[jt.Var, ...]]

SUPPORTED_SOLVERS = ("euler", "midpoint", "heun3", "rk4", "dopri5")

# Dormand-Prince 5(4) Butcher tableau (Dormand & Prince, 1980)
_DP5_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_DP5_A = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_DP5_B = _DP5_A[6]
# Error coefficients b_j - b*_j (5th-order minus embedded 4th-order solution)
_DP5_E = (
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
)


def _as_tuple(state: StateType) -> Tuple[Tuple[jt.Var, ...], bool]:
    if isinstance(state, tuple):
        return state, True
    return (state,), False


def _from_tuple(state_tuple: Tuple[jt.Var, ...], was_tuple: bool) -> StateType:
    return state_tuple if was_tuple else state_tuple[0]


def _state_add_scaled(
    base: Tuple[jt.Var, ...],
    ks: Sequence[Tuple[jt.Var, ...]],
    weights: Sequence[float],
    h: float,
) -> Tuple[jt.Var, ...]:
    out: List[jt.Var] = []
    for idx, y in enumerate(base):
        acc = y
        for k_stage, w in zip(ks, weights):
            if w != 0.0:
                acc = acc + (h * w) * k_stage[idx]
        out.append(acc)
    return tuple(out)


def _state_lincomb(
    ks: Sequence[Tuple[jt.Var, ...]],
    weights: Sequence[float],
    h: float,
) -> Tuple[jt.Var, ...]:
    n_vars = len(ks[0])
    out: List[jt.Var] = []
    for idx in range(n_vars):
        acc = None
        for k_stage, w in zip(ks, weights):
            if w != 0.0:
                term = (h * w) * k_stage[idx]
                acc = term if acc is None else (acc + term)
        if acc is None:
            acc = jt.zeros_like(ks[0][idx])
        out.append(acc)
    return tuple(out)


def _rms_norm(tensors: Sequence[jt.Var]) -> float:
    total_sq = 0.0
    total_el = 0
    for v in tensors:
        vf = v.float64()
        total_sq += float((vf * vf).sum().item())
        total_el += int(v.numel())
    return math.sqrt(total_sq / max(total_el, 1))


def _fixed_step_advance(
    func: Callable[[float, Tuple[jt.Var, ...]], Tuple[jt.Var, ...]],
    t: float,
    y: Tuple[jt.Var, ...],
    h: float,
    method: str,
) -> Tuple[jt.Var, ...]:
    if method == "euler":
        k1 = func(t, y)
        return tuple(yi + h * k1i for yi, k1i in zip(y, k1))

    if method == "midpoint":
        k1 = func(t, y)
        y_mid = tuple(yi + 0.5 * h * k1i for yi, k1i in zip(y, k1))
        k2 = func(t + 0.5 * h, y_mid)
        return tuple(yi + h * k2i for yi, k2i in zip(y, k2))

    if method == "heun3":
        k1 = func(t, y)
        y2 = tuple(yi + (h / 3.0) * k1i for yi, k1i in zip(y, k1))
        k2 = func(t + h / 3.0, y2)
        y3 = tuple(yi + (2.0 * h / 3.0) * k2i for yi, k2i in zip(y, k2))
        k3 = func(t + 2.0 * h / 3.0, y3)
        return tuple(
            yi + h * (0.25 * k1i + 0.75 * k3i) for yi, k1i, k3i in zip(y, k1, k3)
        )

    if method == "rk4":
        k1 = func(t, y)
        y2 = tuple(yi + 0.5 * h * k1i for yi, k1i in zip(y, k1))
        k2 = func(t + 0.5 * h, y2)
        y3 = tuple(yi + 0.5 * h * k2i for yi, k2i in zip(y, k2))
        k3 = func(t + 0.5 * h, y3)
        y4 = tuple(yi + h * k3i for yi, k3i in zip(y, k3))
        k4 = func(t + h, y4)
        return tuple(
            yi + (h / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)
        )

    raise ValueError(f"Unsupported fixed-step method {method!r}.")


def _select_initial_step(
    func: Callable[[float, Tuple[jt.Var, ...]], Tuple[jt.Var, ...]],
    t0: float,
    y0: Tuple[jt.Var, ...],
    f0: Tuple[jt.Var, ...],
    direction: float,
    order: int,
    atol: float,
    rtol: float,
    max_span: float,
) -> float:
    """Hairer-Norsett-Wanner II.4 automatic initial step-size selection."""
    scales = [atol + jt.abs(y) * rtol for y in y0]
    d0 = _rms_norm([y / s for y, s in zip(y0, scales)])
    d1 = _rms_norm([f / s for f, s in zip(f0, scales)])

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * (d0 / d1)
    h0 = min(h0, max_span)

    y1 = tuple(y + (direction * h0) * f for y, f in zip(y0, f0))
    f1 = func(t0 + direction * h0, y1)
    d2 = _rms_norm([(f1i - f0i) / s for f1i, f0i, s in zip(f1, f0, scales)]) / max(h0, 1e-12)

    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / float(order + 1))

    return min(100.0 * h0, h1, max_span)


def odeint_jittor(
    func: Callable[[float, StateType], StateType],
    y0: StateType,
    time_grid: Sequence[float],
    method: str = "euler",
    step_size: Optional[float] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_num_steps: int = 100000,
) -> Union[List[jt.Var], Tuple[List[jt.Var], ...]]:
    """Integrate ``dy/dt = func(t, y)`` along ``time_grid`` in pure Jittor."""
    method = method.lower()
    if method not in SUPPORTED_SOLVERS:
        raise ValueError(f"Unknown ODE solver {method!r}; supported: {SUPPORTED_SOLVERS}.")

    t_list = [float(v) for v in time_grid]
    assert len(t_list) >= 2, f"time_grid must have at least 2 points, got {t_list}."

    y_init, was_tuple = _as_tuple(y0)

    def wrapped_func(t_scalar: float, state_tuple: Tuple[jt.Var, ...]) -> Tuple[jt.Var, ...]:
        raw = func(t_scalar, _from_tuple(state_tuple, was_tuple))
        res_tuple, _ = _as_tuple(raw)
        return res_tuple

    solutions: List[Tuple[jt.Var, ...]] = [y_init]

    if method != "dopri5":
        curr_y = y_init
        for seg_idx in range(len(t_list) - 1):
            t_start = t_list[seg_idx]
            t_end = t_list[seg_idx + 1]
            span = t_end - t_start
            if abs(span) == 0.0:
                solutions.append(curr_y)
                continue

            if step_size is None:
                # Step directly across time_grid intervals when step_size is None
                curr_y = _fixed_step_advance(wrapped_func, t_start, curr_y, span, method)
                solutions.append(curr_y)
            else:
                assert step_size > 0.0, f"step_size must be positive, got {step_size}."
                direction = 1.0 if span > 0.0 else -1.0
                abs_span = abs(span)
                n_full = int(math.floor(abs_span / step_size + 1e-12))
                t_curr = t_start
                for _ in range(n_full):
                    h = direction * step_size
                    # Guard floating-point overshoot on exact multiples
                    if abs((t_end - t_curr) - h) < 1e-12:
                        h = t_end - t_curr
                    curr_y = _fixed_step_advance(wrapped_func, t_curr, curr_y, h, method)
                    t_curr += h
                rem = t_end - t_curr
                if abs(rem) > 1e-11:
                    curr_y = _fixed_step_advance(wrapped_func, t_curr, curr_y, rem, method)
                solutions.append(curr_y)
    else:
        assert step_size is None, "Adaptive solver 'dopri5' requires step_size=None."
        total_span = t_list[-1] - t_list[0]
        if abs(total_span) == 0.0:
            solutions = [y_init for _ in t_list]
        else:
            direction = 1.0 if total_span > 0.0 else -1.0
            for idx in range(len(t_list) - 1):
                assert (t_list[idx + 1] - t_list[idx]) * direction >= 0.0, (
                    f"time_grid must be monotonic; got {t_list}."
                )

            t_curr = t_list[0]
            curr_y = y_init
            k1 = wrapped_func(t_curr, curr_y)
            abs_h = _select_initial_step(
                wrapped_func,
                t0=t_curr,
                y0=curr_y,
                f0=k1,
                direction=direction,
                order=4,
                atol=atol,
                rtol=rtol,
                max_span=abs(total_span),
            )

            safety = 0.9
            ifactor = 10.0
            dfactor = 0.2
            n_steps = 0

            for target_t in t_list[1:]:
                while (target_t - t_curr) * direction > 1e-14:
                    if n_steps >= max_num_steps:
                        raise RuntimeError(
                            f"dopri5 exceeded max_num_steps={max_num_steps} at t={t_curr}."
                        )
                    rem = abs(target_t - t_curr)
                    step_abs = min(abs_h, rem)
                    clamped_to_target = step_abs >= rem - 1e-14
                    h = direction * (rem if clamped_to_target else step_abs)

                    ks: List[Tuple[jt.Var, ...]] = [k1]
                    for stage in range(1, 6):
                        y_stage = _state_add_scaled(curr_y, ks, _DP5_A[stage], h)
                        k_stage = wrapped_func(t_curr + _DP5_C[stage] * h, y_stage)
                        ks.append(k_stage)

                    y_next = _state_add_scaled(curr_y, ks, _DP5_B, h)
                    k7 = wrapped_func(t_curr + h, y_next)
                    ks.append(k7)

                    err_vec = _state_lincomb(ks, _DP5_E, h)
                    scales = [
                        atol + jt.maximum(jt.abs(y0i), jt.abs(y1i)) * rtol
                        for y0i, y1i in zip(curr_y, y_next)
                    ]
                    err_ratio = _rms_norm([ei / si for ei, si in zip(err_vec, scales)])
                    n_steps += 1

                    if err_ratio <= 1.0 or step_abs <= 1e-14:
                        t_curr = target_t if clamped_to_target else (t_curr + h)
                        curr_y = y_next
                        k1 = k7  # FSAL reuse!
                        if err_ratio == 0.0:
                            factor = ifactor
                        else:
                            factor = min(
                                ifactor,
                                max(dfactor, safety * (err_ratio ** (-0.2))),
                            )
                        if not clamped_to_target or err_ratio > 0.5:
                            abs_h = max(step_abs * factor, 1e-14)
                    else:
                        factor = max(dfactor, safety * (err_ratio ** (-0.2)))
                        abs_h = max(step_abs * factor, 1e-14)

                solutions.append(curr_y)

    if was_tuple:
        n_vars = len(y_init)
        return tuple([sol[v_idx] for sol in solutions] for v_idx in range(n_vars))
    return [sol[0] for sol in solutions]


class ODESolver:
    """Numerical ODE & exact/Hutchinson likelihood solver for Jittor velocity fields.

    Matches ``facebookresearch/flow_matching/solver/ode_solver.py`` while adding
    native Jittor ``dopri5`` / ``rk4`` / ``midpoint`` / ``euler`` and ``last_nfe``.
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        self.velocity_model = velocity_model
        self.last_nfe: int = 0

    def _parse_time_grid(self, time_grid: Optional[Union[Sequence[float], jt.Var]], default: Tuple[float, float]) -> List[float]:
        if time_grid is None:
            return list(default)
        if isinstance(time_grid, jt.Var):
            return [float(x) for x in time_grid.numpy().reshape(-1)]
        return [float(x) for x in time_grid]

    def sample(
        self,
        x_init: jt.Var,
        step_size: Optional[float] = None,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Optional[Union[Sequence[float], jt.Var]] = None,
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[jt.Var, jt.Var]:
        """Integrate ``dx/dt = v_theta(x, t)`` over ``time_grid``."""
        t_list = self._parse_time_grid(time_grid, default=(0.0, 1.0))
        local_nfe = 0
        if hasattr(self.velocity_model, "reset_nfe_counter"):
            self.velocity_model.reset_nfe_counter()

        def ode_func(t_scalar: float, x: jt.Var) -> jt.Var:
            nonlocal local_nfe
            local_nfe += 1
            t_var = jt.array([t_scalar], dtype=x.dtype)
            return self.velocity_model(x=x, t=t_var, **model_extras)

        grad_flag = 0 if enable_grad else 1
        with jt.flag_scope(no_grad=grad_flag):
            sol_list = odeint_jittor(
                ode_func,
                y0=x_init,
                time_grid=t_list,
                method=method,
                step_size=step_size,
                atol=atol,
                rtol=rtol,
            )

        if hasattr(self.velocity_model, "get_nfe") and self.velocity_model.get_nfe() > 0:
            self.last_nfe = self.velocity_model.get_nfe()
        else:
            self.last_nfe = local_nfe

        if return_intermediates:
            return jt.stack(sol_list, dim=0)
        return sol_list[-1]

    def compute_likelihood(
        self,
        x_1: jt.Var,
        log_p0: Optional[Callable[[jt.Var], jt.Var]] = None,
        step_size: Optional[float] = None,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Optional[Union[Sequence[float], jt.Var]] = None,
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Tuple[jt.Var, jt.Var]:
        """Compute log p_1(x_1) by solving the reverse augmented instantaneous change-of-variables ODE."""
        t_list = self._parse_time_grid(time_grid, default=(1.0, 0.0))
        assert t_list[0] > t_list[-1], (
            f"Likelihood integration must run backward from data (t~1) to noise (t=0); got {t_list}."
        )

        if log_p0 is None:
            log_p0 = gaussian_log_prob

        if not exact_divergence:
            # Rademacher probe z in {-1, +1}^d, fixed along the entire ODE trajectory (Paper Eq. 34)
            z = ((jt.randn_like(x_1) < 0).float32() * 2.0 - 1.0).stop_grad()

        local_nfe = 0
        if hasattr(self.velocity_model, "reset_nfe_counter"):
            self.velocity_model.reset_nfe_counter()

        def dynamics_func(t_scalar: float, states: Tuple[jt.Var, jt.Var]) -> Tuple[jt.Var, jt.Var]:
            nonlocal local_nfe
            local_nfe += 1
            xt = states[0]

            with jt.flag_scope(no_grad=0):
                if enable_grad:
                    xt_in = xt
                else:
                    xt_in = xt.clone()
                    xt_in.start_grad()

                t_var = jt.array([t_scalar], dtype=xt_in.dtype)
                ut = self.velocity_model(x=xt_in, t=t_var, **model_extras)

                if exact_divergence:
                    ut_flat = ut.reshape(ut.shape[0], -1)
                    d_dim = ut_flat.shape[1]
                    div = jt.zeros((ut.shape[0],), dtype=ut.dtype)
                    for i in range(d_dim):
                        retain = enable_grad or (i < d_dim - 1)
                        g = gradient(ut_flat[:, i], xt_in, retain_graph=retain).reshape(ut.shape[0], -1)[:, i]
                        if not enable_grad:
                            g = g.stop_grad()
                        div = div + g
                else:
                    ut_flat = ut.reshape(ut.shape[0], -1)
                    z_flat = z.reshape(z.shape[0], -1)
                    ut_dot_z = (ut_flat * z_flat).sum(dim=1)
                    grad_ut_dot_z = gradient(ut_dot_z, xt_in, retain_graph=enable_grad)
                    div = (grad_ut_dot_z.reshape(xt_in.shape[0], -1) * z_flat).sum(dim=1)

            if not enable_grad:
                ut = ut.stop_grad()
                div = div.stop_grad()
            return ut, div

        init_log_det = jt.zeros((x_1.shape[0],), dtype=x_1.dtype)
        grad_flag = 0 if enable_grad else 1
        with jt.flag_scope(no_grad=grad_flag):
            sol_list, log_det_list = odeint_jittor(
                dynamics_func,
                y0=(x_1, init_log_det),
                time_grid=t_list,
                method=method,
                step_size=step_size,
                atol=atol,
                rtol=rtol,
            )

        if hasattr(self.velocity_model, "get_nfe") and self.velocity_model.get_nfe() > 0:
            self.last_nfe = self.velocity_model.get_nfe()
        else:
            self.last_nfe = local_nfe

        x_source = sol_list[-1]
        log_p = log_p0(x_source) + log_det_list[-1]
        if return_intermediates:
            return jt.stack(sol_list, dim=0), log_p
        return x_source, log_p
