"""Compare this Jittor stack with SciPy and, when installed, PyTorch.

The checks are the ones that actually bite a Flow Matching port:

* Jittor's built-in GroupNorm uses ``E[x^2] - E[x]^2``. PyTorch uses a two-pass
  variance. On large activations those disagree. ``GroupNorm32`` is the two-pass
  version used by the U-Net.
* ``dopri5`` is the same Dormand-Prince 5(4) family as SciPy's ``RK45`` and as
  ``torchdiffeq``. Jittor-NODE only has fixed-step Euler and RK4, so this is the
  adaptive comparison those users do not get.
* A scalar linear ODE has a closed form, so the table is an error against the
  truth, not against a second copy of our own code.

    python benchmarks/compare_frameworks.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jittor as jt

from jfm.models.nn import GroupNorm32
from jittor_ode import odeint


def _two_pass_group_norm(x, num_groups, weight, bias, eps=1e-5):
    n, channels, _, _ = x.shape
    grouped = x.reshape(n, num_groups, channels // num_groups, -1)
    mean = grouped.mean(axis=(2, 3), keepdims=True)
    var = ((grouped - mean) ** 2).mean(axis=(2, 3), keepdims=True)
    normed = ((grouped - mean) / np.sqrt(var + eps)).reshape(x.shape)
    return normed * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


def compare_group_norm():
    rng = np.random.default_rng(0)
    # Offset of 50 makes E[x^2] - E[x]^2 cancel in float32.
    data = (rng.normal(size=(2, 32, 8, 8)).astype(np.float32) + 50.0)
    weight = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    bias = np.linspace(-0.2, 0.2, 32, dtype=np.float32)
    reference = _two_pass_group_norm(data, 32, weight, bias)

    layer = GroupNorm32(32, 32)
    layer.weight.assign(jt.array(weight))
    layer.bias.assign(jt.array(bias))
    ours = layer(jt.array(data)).numpy()
    rows = {
        "groupnorm32_vs_numpy_two_pass_max_abs": float(np.max(np.abs(ours - reference))),
    }
    try:
        import torch

        ref = torch.nn.functional.group_norm(
            torch.tensor(data), 32, torch.tensor(weight), torch.tensor(bias), 1e-5
        ).numpy()
        rows["groupnorm32_vs_pytorch_max_abs"] = float(np.max(np.abs(ours - ref)))
    except Exception as exc:  # torch is optional
        rows["pytorch_group_norm"] = f"unavailable: {exc}"
    return rows


def compare_ode():
    y0 = 1.0
    truth = math.exp(-1.0)
    scipy_sol = solve_ivp(lambda t, y: -y, (0.0, 1.0), [y0], method="RK45", rtol=1e-5, atol=1e-5)
    jt_sol = odeint(lambda t, y: -y, jt.array([y0]), [0.0, 1.0], method="dopri5", rtol=1e-5, atol=1e-5)
    jt_final = float(jt_sol.numpy()[-1, 0])
    rows = {
        "analytical_y1": truth,
        "scipy_rk45_abs_error": float(abs(scipy_sol.y[0, -1] - truth)),
        "scipy_rk45_nfe": int(scipy_sol.nfev),
        "jittor_dopri5_abs_error": float(abs(jt_final - truth)),
        "jittor_dopri5_value": jt_final,
    }
    rk4 = odeint(
        lambda t, y: -y,
        jt.array([y0]),
        [0.0, 1.0],
        method="rk4",
        options={"step_size": 0.01},
    )
    rows["jittor_rk4_100_steps_abs_error"] = float(abs(float(rk4.numpy()[-1, 0]) - truth))
    try:
        import torch

        y = torch.tensor([y0], dtype=torch.float32)
        step = 1.0 / 100.0
        for _ in range(100):
            k1 = -y
            k2 = -(y + 0.5 * step * k1)
            k3 = -(y + 0.5 * step * k2)
            k4 = -(y + step * k3)
            y = y + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rows["pytorch_rk4_100_steps_abs_error"] = float(abs(float(y) - truth))
        rows["jittor_minus_pytorch_rk4"] = float(abs(float(rk4.numpy()[-1, 0]) - float(y)))
    except Exception as exc:
        rows["pytorch_rk4"] = f"unavailable: {exc}"
    try:
        import torch
        from torchdiffeq import odeint as torch_odeint

        def torch_f(t, y):
            return -y

        with torch.no_grad():
            out = torch_odeint(torch_f, torch.tensor([y0]), torch.tensor([0.0, 1.0]), method="dopri5", rtol=1e-5, atol=1e-5)
        rows["torchdiffeq_dopri5_abs_error"] = float(abs(float(out[-1]) - truth))
        rows["jittor_minus_torchdiffeq"] = float(abs(jt_final - float(out[-1])))
    except Exception as exc:
        rows["torchdiffeq"] = f"unavailable: {exc}"
    return rows


def main() -> None:
    report = {"group_norm": compare_group_norm(), "ode": compare_ode()}
    out = ROOT / "benchmarks" / "framework_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
