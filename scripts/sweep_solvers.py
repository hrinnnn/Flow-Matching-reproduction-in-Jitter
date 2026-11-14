"""Compare Euler, Midpoint and RK4 at matched NFE.

Without a checkpoint this integrates the analytical field ``dx/dt = -x`` and writes
trajectory MSE against a fine RK4 reference, plus wall time. With ``--checkpoint``
it does the same for a trained 2D MLP (``--arch mlp``) or CIFAR U-Net (``--arch unet``).
FID is left empty until Inception features are computed from the saved samples.

    python scripts/sweep_solvers.py --analytical
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.solvers.ode_solver import ODESolver
from jfm.utils.load_model import load_velocity

COSTS = {"euler": 1, "midpoint": 2, "rk4": 4}


def analytical_velocity(x, t, **_kwargs):
    return -x


def main() -> None:
    parser = argparse.ArgumentParser(description="Solver NFE sweep")
    parser.add_argument("--analytical", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs/checkerboard_2d.yaml"))
    parser.add_argument("--arch", choices=("mlp", "unet"), default="mlp")
    parser.add_argument("--method", default=None)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--nfes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "outputs/solver_sweep.json"))
    args = parser.parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1

    if args.analytical or not args.checkpoint:
        velocity = analytical_velocity
        t_end = 1.0
        x0 = jt.randn((args.batch, args.dim))
    else:
        velocity, t_end = load_velocity(args)
        shape = (args.batch, args.dim) if args.arch == "mlp" else (args.batch, 3, 32, 32)
        x0 = jt.randn(shape)

    solver = ODESolver(velocity)
    reference = solver.sample(x0, method="rk4", step_size=t_end / 100.0, time_grid=[0.0, t_end]).numpy()
    rows = []
    for name, cost in COSTS.items():
        for nfe in args.nfes:
            if nfe % cost != 0:
                continue
            steps = nfe // cost
            started = time.perf_counter()
            pred = solver.sample(
                x0, method=name, step_size=t_end / float(steps), time_grid=[0.0, t_end]
            ).numpy()
            elapsed = time.perf_counter() - started
            mse = float(np.mean((pred - reference) ** 2))
            rows.append(
                {
                    "solver": name,
                    "nfe": solver.last_nfe,
                    "mse_vs_rk4_400": mse,
                    "seconds": elapsed,
                    "fid": None,
                }
            )
            print(f"{name:8s} nfe {solver.last_nfe:4d} mse {mse:.3e} {elapsed:.3f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
