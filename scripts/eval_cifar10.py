"""Sample a trained CIFAR-10 velocity field and record NFE.

dopri5 with atol=rtol=1e-5 is the Table 1 protocol. Euler, midpoint, and RK4 take
``--nfe`` as the neural-function budget (1, 2, and 4 evaluations per step).
"""

import argparse
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.models.ema import EMA
from jfm.models.unet import UNetModel
from jfm.objectives.losses import build_objective
from jfm.solvers.ode_solver import ODESolver
from jfm.utils.seed import set_seed

STEP_COST = {"euler": 1, "midpoint": 2, "heun3": 3, "rk4": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample CIFAR-10 Flow Matching")
    parser.add_argument("--config", default=str(ROOT / "configs/cifar10_paper.yaml"))
    parser.add_argument("--checkpoint", required=True, help="model.pkl or ema.pkl")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "outputs/cifar10/samples.npz"))
    parser.add_argument("--solver", default=None)
    parser.add_argument("--nfe", type=int, default=None, help="Fixed-step budget. Ignored by dopri5.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def to_uint8(samples: jt.Var) -> np.ndarray:
    images = jt.clamp(samples.float32() * 0.5 + 0.5, 0.0, 1.0)
    array = np.floor(images.numpy() * 255.0 + 0.5).astype(np.uint8)
    return np.transpose(array, (0, 2, 3, 1))


def main() -> None:
    args = parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    set_seed(int(cfg["seed"]) + 1000)

    num_classes = cfg["model"].get("num_classes")
    model = UNetModel.from_preset(cfg["model"]["preset"], num_classes=num_classes)
    if args.ema:
        ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))
        ema.load(args.checkpoint)
        ema.copy_to_model()
    else:
        model.load(args.checkpoint)
    model.eval()

    solver_name = (args.solver or cfg["eval"]["solver"]).lower()
    cfg_scale = cfg["eval"].get("cfg_scale", 0.0) if args.cfg_scale is None else args.cfg_scale
    if num_classes is None:
        cfg_scale = 0.0
    bundle = build_objective(
        method=cfg["method"],
        sigma_min=cfg["path"]["sigma_min"],
        beta_min=cfg["path"]["beta_min"],
        beta_max=cfg["path"]["beta_max"],
        time_eps=cfg["path"]["time_eps"],
        cfg_scale=None if cfg_scale == 0.0 else cfg_scale,
    )
    velocity = bundle.wrap_velocity(model)
    solver = ODESolver(velocity)
    t_end = bundle.sample_t_end

    step_size = None
    if solver_name != "dopri5":
        cost = STEP_COST[solver_name]
        nfe = args.nfe if args.nfe is not None else cost * 10
        if nfe % cost != 0:
            raise ValueError(f"{solver_name} needs an NFE divisible by {cost}, got {nfe}.")
        step_size = t_end / float(nfe // cost)

    total = int(args.num_samples or cfg["eval"]["num_samples"])
    batch_size = int(args.batch_size or cfg["eval"]["batch_size"])
    written = []
    nfes = []
    remaining = total
    while remaining > 0:
        this = min(batch_size, remaining)
        labels = None
        extras = {}
        if num_classes is not None:
            labels = jt.randint(0, int(num_classes), shape=(this,))
            extras["label"] = labels
        samples = solver.sample(
            x_init=jt.randn((this, 3, 32, 32)),
            method=solver_name,
            step_size=step_size,
            atol=float(cfg["eval"]["atol"]),
            rtol=float(cfg["eval"]["rtol"]),
            time_grid=[0.0, t_end],
            **extras,
        )
        written.append(to_uint8(samples))
        nfes.append(solver.last_nfe)
        remaining -= this
        print(f"generated {total - remaining}/{total}, last batch NFE {solver.last_nfe}")

    images = np.concatenate(written, axis=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, images=images)
    summary = {
        "num_samples": int(images.shape[0]),
        "solver": solver_name,
        "mean_nfe": float(np.mean(nfes)),
        "checkpoint": args.checkpoint,
        "samples": str(out),
    }
    summary_path = out.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
