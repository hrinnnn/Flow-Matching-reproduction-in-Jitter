"""Evaluate bits/dim on a batch of images with the Hutchinson likelihood ODE.

    python scripts/eval_bpd.py --config configs/cifar10_smoke.yaml \\
        --checkpoint outputs/cifar10/fm_ot/ema.pkl --ema --num-images 8 --solver midpoint --step-size 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import jittor as jt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.datasets.cifar10 import build_cifar10
from jfm.likelihood.bpd import compute_bpd_batch
from jfm.models.ema import EMA
from jfm.models.unet import UNetModel
from jfm.objectives.losses import build_objective
from jfm.solvers.ode_solver import ODESolver


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 bits/dim")
    parser.add_argument("--config", default=str(ROOT / "configs/cifar10_paper.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dequant-samples", type=int, default=1)
    parser.add_argument("--solver", default="dopri5")
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "outputs/cifar10/bpd.json"))
    args = parser.parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    num_classes = cfg["model"].get("num_classes")
    model = UNetModel.from_preset(cfg["model"]["preset"], num_classes=num_classes)
    if args.ema:
        ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))
        ema.load(args.checkpoint)
        ema.copy_to_model()
    else:
        model.load(args.checkpoint)
    model.eval()
    bundle = build_objective(
        method=cfg["method"],
        sigma_min=cfg["path"]["sigma_min"],
        beta_min=cfg["path"]["beta_min"],
        beta_max=cfg["path"]["beta_max"],
        time_eps=cfg["path"]["time_eps"],
    )
    solver = ODESolver(bundle.wrap_velocity(model))
    loader = build_cifar10(
        root=cfg["data"]["root"],
        train=False,
        download=True,
        random_flip=False,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    totals = []
    seen = 0
    for images, _labels in loader:
        take = min(args.batch_size, args.num_images - seen)
        batch = images[:take]
        result = compute_bpd_batch(
            solver,
            batch,
            num_dequant_samples=args.dequant_samples,
            t_start=bundle.sample_t_end,
            method=args.solver,
            step_size=args.step_size,
            atol=float(cfg["eval"]["atol"]),
            rtol=float(cfg["eval"]["rtol"]),
        )
        totals.append(result["bpd"].numpy())
        seen += take
        print(f"bpd images {seen}/{args.num_images}")
        if seen >= args.num_images:
            break
    import numpy as np

    values = np.concatenate(totals)
    report = {"bpd_mean": float(values.mean()), "num_images": int(values.shape[0]), "solver": args.solver}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
