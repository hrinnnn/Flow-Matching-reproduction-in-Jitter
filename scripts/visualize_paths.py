"""Save intermediate states along one ODE, the image version of paper Figure 6.

``--analytical`` integrates ``dx/dt = -x`` and writes ``.npy`` snapshots, so the
script can be checked before a GPU checkpoint exists. A real 2D or CIFAR checkpoint
uses the same time grid and the trained velocity field.
"""

import argparse
import sys
from pathlib import Path

import jittor as jt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.solvers.ode_solver import ODESolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Save ODE snapshots")
    parser.add_argument("--analytical", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs/checkerboard_2d.yaml"))
    parser.add_argument("--arch", choices=("mlp", "unet"), default="mlp")
    parser.add_argument("--panels", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--solver", default="midpoint")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--output", default=str(ROOT / "outputs/path_snapshots.npy"))
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1

    if args.analytical or not args.checkpoint:
        velocity = lambda x, t, **kwargs: -x
        t_end = 1.0
        x0 = jt.randn((args.batch, 2))
    else:
        from jfm.utils.load_model import load_velocity

        velocity, t_end = load_velocity(args)
        x0 = jt.randn((args.batch, 2) if args.arch == "mlp" else (args.batch, 3, 32, 32))

    grid = np.linspace(0.0, t_end, args.panels).tolist()
    solver = ODESolver(velocity)
    traj = solver.sample(
        x0,
        method=args.solver,
        step_size=args.step_size,
        time_grid=grid,
        return_intermediates=True,
    ).numpy()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, traj)
    print(f"saved {traj.shape} to {out}, NFE {solver.last_nfe}")


if __name__ == "__main__":
    main()
