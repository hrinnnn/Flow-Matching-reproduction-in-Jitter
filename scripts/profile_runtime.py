"""Time a few ToyMLP steps and separate the first JIT compile from the steady state.

    python scripts/profile_runtime.py --steps 4
    python scripts/profile_runtime.py --use-cuda --steps 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.datasets.checkerboard import sample_checkerboard
from jfm.models.mlp import ToyMLP
from jfm.objectives.losses import build_objective
from jfm.solvers.ode_solver import ODESolver
from jfm.utils.seed import set_seed


def _sync(value):
    return float(value.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one Flow Matching step")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--method", default="fm_ot")
    parser.add_argument("--coupling", default="independent")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "outputs/profile_runtime.json"))
    args = parser.parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1
    set_seed(0)

    model = ToyMLP(input_dim=2, hidden_dim=args.hidden_dim, num_layers=4)
    model.train()
    bundle = build_objective(args.method, coupling=args.coupling)
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        x1 = sample_checkerboard(args.batch_size)
        data_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        loss = bundle.loss_fn(model, x_1=x1)
        forward_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        optimizer.step(loss)
        backward_s = time.perf_counter() - t2
        loss_value = _sync(loss)
        rows.append(
            {
                "step": step,
                "phase": "compile" if step == 1 else "steady",
                "loss": loss_value,
                "data_s": data_s,
                "forward_s": forward_s,
                "backward_s": backward_s,
            }
        )
        print(
            f"step {step} ({rows[-1]['phase']}): "
            f"data {data_s:.3f}s forward {forward_s:.3f}s backward {backward_s:.3f}s"
        )

    model.eval()
    solver = ODESolver(bundle.wrap_velocity(model))
    x0 = jt.randn((64, 2))
    t3 = time.perf_counter()
    solver.sample(x0, method="euler", step_size=0.25, time_grid=[0.0, 1.0])
    sample_s = time.perf_counter() - t3
    report = {"train": rows, "sample_euler_4_steps_s": sample_s, "sample_nfe": solver.last_nfe}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"sample {sample_s:.3f}s NFE {solver.last_nfe}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
