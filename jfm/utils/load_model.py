"""Load a trained MLP or U-Net and wrap it as an ODE velocity."""

from pathlib import Path

import yaml

from jfm.models.ema import EMA
from jfm.models.mlp import ToyMLP
from jfm.models.unet import UNetModel
from jfm.objectives.losses import build_objective


def load_velocity(args):
    """Return ``(velocity_wrapper, t_end)`` from a checkpoint and yaml config."""
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    method = getattr(args, "method", None)
    if not method:
        raw = cfg.get("method", "fm_ot")
        method = raw if isinstance(raw, str) else "fm_ot"
    if args.arch == "mlp":
        model = ToyMLP(
            input_dim=cfg["model"].get("input_dim", 2),
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
        )
        model.load(args.checkpoint)
    else:
        model = UNetModel.from_preset(
            cfg["model"]["preset"],
            num_classes=cfg["model"].get("num_classes"),
        )
        if getattr(args, "ema", False):
            ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))
            ema.load(args.checkpoint)
            ema.copy_to_model()
        else:
            model.load(args.checkpoint)
    model.eval()
    bundle = build_objective(
        method=method,
        sigma_min=cfg["path"]["sigma_min"],
        beta_min=cfg["path"]["beta_min"],
        beta_max=cfg["path"]["beta_max"],
        time_eps=cfg["path"]["time_eps"],
        coupling=cfg["path"].get("coupling", "independent"),
    )
    return bundle.wrap_velocity(model), bundle.sample_t_end
