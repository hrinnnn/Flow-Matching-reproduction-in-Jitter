"""Train a CIFAR-10 Flow Matching U-Net in Jittor.

Examples:
    python scripts/train_cifar10.py --config configs/cifar10_smoke.yaml --fake-data
    python scripts/train_cifar10.py --config configs/cifar10_paper.yaml --method fm_ot --use-cuda
    python scripts/train_cifar10.py --config configs/cifar10_paper.yaml --method fm_dif --use-cuda
    python scripts/train_cifar10.py --config configs/cifar10_cfg.yaml --use-cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jittor as jt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.datasets.cifar10 import build_cifar10
from jfm.models.ema import EMA
from jfm.models.unet import UNetModel
from jfm.objectives.losses import build_objective
from jfm.utils.schedule import linear_warmup_decay
from jfm.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 Flow Matching in Jittor")
    parser.add_argument("--config", default=str(ROOT / "configs/cifar10_paper.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/cifar10"))
    parser.add_argument("--method", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--fake-data", action="store_true", help="Skip CIFAR-10 and train on noise")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.method is not None:
        cfg["method"] = args.method
    if args.iterations is not None:
        cfg["train"]["iterations"] = args.iterations
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    return cfg


def set_lr(optimizer, lr: float) -> None:
    optimizer.lr = lr
    for group in optimizer.param_groups:
        group["lr"] = lr


def next_batch(loader, iterator, fake: bool, batch_size: int, num_classes):
    if fake:
        images = jt.randn((batch_size, 3, 32, 32))
        labels = jt.randint(0, 10, shape=(batch_size,))
        return images, labels, iterator
    try:
        images, labels = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        images, labels = next(iterator)
    return images, labels, iterator


def maybe_drop_labels(labels, num_classes, drop_prob: float):
    if num_classes is None:
        return None
    labels = labels.int32().reshape(-1)
    if drop_prob <= 0.0:
        return labels
    keep = (jt.rand(labels.shape) >= drop_prob).int32()
    null = jt.zeros_like(labels) + int(num_classes)
    return labels * keep + null * (1 - keep)


def main() -> None:
    args = parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))

    num_classes = cfg["model"].get("num_classes")
    model = UNetModel.from_preset(cfg["model"]["preset"], num_classes=num_classes)
    model.train()
    ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))
    bundle = build_objective(
        method=cfg["method"],
        sigma_min=cfg["path"]["sigma_min"],
        beta_min=cfg["path"]["beta_min"],
        beta_max=cfg["path"]["beta_max"],
        time_eps=cfg["path"]["time_eps"],
        coupling=cfg["path"].get("coupling", "independent"),
    )
    train_cfg = cfg["train"]
    optimizer = jt.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    out_dir = Path(args.output_dir) / cfg["method"]
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = None
    iterator = None
    if not args.fake_data:
        loader = build_cifar10(
            root=cfg["data"]["root"],
            train=True,
            download=bool(cfg["data"]["download"]),
            random_flip=bool(cfg["data"]["random_flip"]),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["data"].get("num_workers", 0)),
        )
        iterator = iter(loader)

    accum = int(train_cfg["accum_steps"])
    total = int(train_cfg["iterations"])
    history = []
    micro = 0
    step = 0
    t0 = time.time()
    while step < total:
        images, labels, iterator = next_batch(
            loader, iterator, args.fake_data, int(train_cfg["batch_size"]), num_classes
        )
        y = maybe_drop_labels(labels, num_classes, float(train_cfg["class_drop_prob"]))
        kwargs = {} if y is None else {"y": y}
        loss = bundle.loss_fn(model, x_1=images, **kwargs)
        optimizer.backward(loss / float(accum))
        micro += 1
        if micro % accum != 0:
            continue
        step += 1
        lr = linear_warmup_decay(
            step,
            total,
            int(train_cfg["warmup_steps"]),
            float(train_cfg["lr"]),
            float(train_cfg["min_lr"]),
        )
        set_lr(optimizer, lr)
        optimizer.step()
        ema.update()

        if step == 1 or step % int(train_cfg["log_every"]) == 0 or step == total:
            loss_val = float(loss.item())
            elapsed = time.time() - t0
            print(f"[{cfg['method']}] step {step}/{total} loss {loss_val:.4f} lr {lr:.3e} {elapsed:.1f}s")
            history.append({"step": step, "loss": loss_val, "lr": lr})
            t0 = time.time()
        if step % int(train_cfg["ckpt_every"]) == 0 or step == total:
            model.save(str(out_dir / "model.pkl"))
            ema.save(str(out_dir / "ema.pkl"))
            with open(out_dir / "train_log.json", "w", encoding="utf-8") as handle:
                json.dump({"config": cfg, "history": history}, handle, indent=2)
            print(f"saved {out_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
