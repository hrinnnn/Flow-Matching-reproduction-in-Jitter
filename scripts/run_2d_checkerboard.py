"""Train & evaluate SM-Dif, FM-Dif, and FM-OT on 2D Checkerboard (Reproducing Paper Figure 4).

Usage:
    # Full Figure 4 reproduction (20k steps per method):
    python scripts/run_2d_checkerboard.py --config configs/checkerboard_2d.yaml --output_dir outputs/2d_checkerboard

    # Fast local CPU smoke / preview run:
    python scripts/run_2d_checkerboard.py --iterations 1500 --batch_size 1024 --num_samples 15000 --output_dir outputs/2d_preview
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import jittor as jt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.datasets.checkerboard import sample_checkerboard
from jfm.models.mlp import ToyMLP
from jfm.objectives.losses import build_objective
from jfm.solvers.ode_solver import ODESolver
from jfm.utils.seed import set_seed

METHOD_TITLES = {
    "sm_dif": "Score Matching w/ Diffusion (SM-Dif)",
    "fm_dif": "Flow Matching w/ Diffusion (FM-Dif)",
    "fm_ot": "Flow Matching w/ OT (FM-OT)",
    "scoreflow": "ScoreFlow w/ Diffusion",
    "ddpm": "DDPM w/ Diffusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Checkerboard Flow Matching reproduction (Figure 4)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/checkerboard_2d.yaml"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs/2d_checkerboard"))
    parser.add_argument("--methods", nargs="+", default=None, help="Override methods list")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_cuda", action="store_true", help="Enable Jittor CUDA backend")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and load checkpoints")
    parser.add_argument(
        "--eval_solver_efficiency",
        action="store_true",
        help="Also compare Euler / Midpoint / RK4 trajectory MSE vs NFE on 2D",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.methods is not None:
        cfg["methods"] = args.methods
    if args.iterations is not None:
        cfg["train"]["iterations"] = args.iterations
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.hidden_dim is not None:
        cfg["model"]["hidden_dim"] = args.hidden_dim
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.num_samples is not None:
        cfg["eval"]["num_samples"] = args.num_samples
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def train_single_method(method: str, cfg: Dict, out_dir: Path) -> ToyMLP:
    set_seed(cfg["seed"])
    model = ToyMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
    )
    bundle = build_objective(
        method=method,
        sigma_min=cfg["path"]["sigma_min"],
        beta_min=cfg["path"]["beta_min"],
        beta_max=cfg["path"]["beta_max"],
        time_eps=cfg["path"]["time_eps"],
        coupling=cfg["path"].get("coupling", "independent"),
    )

    ckpt_path = out_dir / "checkpoints" / f"{method}_mlp.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if ckpt_path.exists() and cfg.get("_eval_only", False):
        model.load(str(ckpt_path))
        print(f"[{method}] Loaded checkpoint from {ckpt_path}")
        return model

    optimizer = jt.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    iters = cfg["train"]["iterations"]
    bs = cfg["train"]["batch_size"]
    log_every = max(1, cfg["train"]["log_every"])

    history: List[Dict] = []
    t0 = time.time()
    model.train()
    for step in range(1, iters + 1):
        x_1 = sample_checkerboard(batch_size=bs)
        x_0 = jt.randn_like(x_1)
        loss = bundle.loss_fn(model, x_1=x_1, x_0=x_0)
        optimizer.step(loss)

        if step == 1 or step % log_every == 0 or step == iters:
            loss_val = float(loss.item())
            elapsed = time.time() - t0
            ms_per_step = elapsed * 1000.0 / (1 if step == 1 else log_every)
            print(f"[{method:8s}] step {step:6d}/{iters} | loss {loss_val:8.4f} | {ms_per_step:6.2f} ms/step")
            history.append({"step": step, "loss": loss_val})
            t0 = time.time()

    model.save(str(ckpt_path))
    with open(out_dir / f"train_log_{method}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return model


def _plot_hist_cell(ax, pts: np.ndarray, bins: int, bound: float) -> None:
    h, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=bins, range=((-bound, bound), (-bound, bound))
    )
    vmax = max(float(np.quantile(h, 0.99)), 1.0)
    norm = cm.colors.Normalize(vmin=0.0, vmax=vmax)
    ax.hist2d(
        pts[:, 0],
        pts[:, 1],
        bins=bins,
        range=((-bound, bound), (-bound, bound)),
        cmap="viridis",
        norm=norm,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def evaluate_and_plot_figure4(
    models: Dict[str, ToyMLP],
    cfg: Dict,
    out_dir: Path,
) -> Dict:
    methods = list(models.keys())
    num_samples = cfg["eval"]["num_samples"]
    n_panels = cfg["eval"]["trajectory_panels"]
    low_nfes = cfg["eval"]["low_nfe_list"]
    bins = cfg["eval"]["hist_bins"]
    bound = cfg["eval"]["plot_range"]
    traj_step = cfg["eval"]["trajectory_step_size"]

    # Paired initial Gaussian noise across all methods
    set_seed(cfg["seed"] + 100)
    x_init = jt.randn((num_samples, 2))

    n_rows = len(methods)
    n_cols = n_panels + len(low_nfes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.65 * n_cols, 1.85 * n_rows),
        gridspec_kw={"wspace": 0.06, "hspace": 0.18},
    )
    if n_rows == 1:
        axes = axes[None, :]

    summary_metrics: Dict[str, Dict] = {}

    for r_idx, method in enumerate(methods):
        model = models[method]
        model.eval()
        bundle = build_objective(
            method=method,
            sigma_min=cfg["path"]["sigma_min"],
            beta_min=cfg["path"]["beta_min"],
            beta_max=cfg["path"]["beta_max"],
            time_eps=cfg["path"]["time_eps"],
            coupling=cfg["path"].get("coupling", "independent"),
        )
        vel_wrapper = bundle.wrap_velocity(model)
        solver = ODESolver(velocity_model=vel_wrapper)
        t_end = bundle.sample_t_end

        # 1) Left block: high-accuracy trajectory panels across t in [0, t_end]
        t_grid = np.linspace(0.0, t_end, n_panels).tolist()
        traj = solver.sample(
            x_init=x_init,
            method="midpoint",
            step_size=traj_step,
            time_grid=t_grid,
            return_intermediates=True,
        ).numpy()

        for c_idx in range(n_panels):
            ax = axes[r_idx, c_idx]
            _plot_hist_cell(ax, traj[c_idx], bins=bins, bound=bound)
            if r_idx == 0:
                ax.set_title(f"t={t_grid[c_idx]:.2f}", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel(method.upper().replace("_", "-"), fontsize=9, fontweight="bold")

        # 2) Right block: Low-NFE sampling with Midpoint solver (2 NFEs per step)
        method_nfe_stats = {}
        for k_idx, target_nfe in enumerate(low_nfes):
            assert target_nfe % 2 == 0, f"Midpoint requires even NFE, got {target_nfe}."
            n_steps = target_nfe // 2
            step_sz = t_end / float(n_steps)
            pts = solver.sample(
                x_init=x_init,
                method="midpoint",
                step_size=step_sz,
                time_grid=[0.0, t_end],
                return_intermediates=False,
            ).numpy()
            actual_nfe = solver.last_nfe
            ax = axes[r_idx, n_panels + k_idx]
            _plot_hist_cell(ax, pts, bins=bins, bound=bound)
            for spine in ax.spines.values():
                spine.set_edgecolor("#d62728")
                spine.set_linewidth(1.2)
            if r_idx == 0:
                ax.set_title(f"NFE={actual_nfe}", fontsize=9, color="#b2182b", fontweight="bold")

            # Compute empirical checkerboard support hit-rate (fraction landing on valid squares)
            scaled = pts * 0.45
            in_box = (
                (scaled[:, 0] >= -2.0)
                & (scaled[:, 0] < 2.0)
                & (scaled[:, 1] >= -2.0)
                & (scaled[:, 1] < 2.0)
            )
            ix = np.floor(scaled[:, 0]).astype(int)
            iy = np.floor(scaled[:, 1]).astype(int)
            valid_square = in_box & (((ix + iy) % 2) == 0)
            hit_rate = float(np.mean(valid_square))
            method_nfe_stats[f"nfe_{actual_nfe}_support_acc"] = round(hit_rate, 4)

        # Also measure adaptive dopri5 NFE on a 4096 subset
        _ = solver.sample(
            x_init=x_init[:4096],
            method="dopri5",
            step_size=None,
            atol=1e-5,
            rtol=1e-5,
            time_grid=[0.0, t_end],
        )
        method_nfe_stats["dopri5_1e-5_nfe"] = solver.last_nfe
        summary_metrics[method] = method_nfe_stats
        print(f"[{method:8s}] stats: {method_nfe_stats}")

    fig_path = out_dir / "figure4_checkerboard.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 4 reproduction to {fig_path}")
    return summary_metrics


def evaluate_2d_solver_efficiency(
    models: Dict[str, ToyMLP],
    cfg: Dict,
    out_dir: Path,
) -> Dict:
    """Compare Euler / Midpoint / RK4 trajectory error vs NFE on 2D models."""
    set_seed(cfg["seed"] + 999)
    x_probe = jt.randn((2048, 2))
    nfes = [4, 8, 12, 16, 20, 32, 64]
    solver_names = [("euler", 1), ("midpoint", 2), ("rk4", 4)]

    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 3.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    results = {}
    for idx, (method, model) in enumerate(models.items()):
        model.eval()
        bundle = build_objective(
            method=method,
            sigma_min=cfg["path"]["sigma_min"],
            beta_min=cfg["path"]["beta_min"],
            beta_max=cfg["path"]["beta_max"],
            time_eps=cfg["path"]["time_eps"],
            coupling=cfg["path"].get("coupling", "independent"),
        )
        solver = ODESolver(bundle.wrap_velocity(model))
        t_end = bundle.sample_t_end

        # Reference solution at 400 NFEs with RK4 (100 steps)
        ref = solver.sample(
            x_init=x_probe,
            method="rk4",
            step_size=t_end / 100.0,
            time_grid=[0.0, t_end],
        ).numpy()

        method_res = {}
        ax = axes[idx]
        for s_name, order_cost in solver_names:
            valid_nfes = [n for n in nfes if n % order_cost == 0]
            mses = []
            for nfe in valid_nfes:
                steps = nfe // order_cost
                pred = solver.sample(
                    x_init=x_probe,
                    method=s_name,
                    step_size=t_end / float(steps),
                    time_grid=[0.0, t_end],
                ).numpy()
                mse = float(np.mean((pred - ref) ** 2))
                mses.append(mse)
            method_res[s_name] = {"nfe": valid_nfes, "mse": mses}
            ax.plot(valid_nfes, mses, marker="o", label=s_name.upper())

        ax.set_yscale("log")
        ax.set_xlabel("NFE")
        if idx == 0:
            ax.set_ylabel("Trajectory MSE vs. Reference")
        ax.set_title(METHOD_TITLES.get(method, method), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        results[method] = method_res

    eff_path = out_dir / "figure_2d_solver_efficiency.png"
    fig.savefig(eff_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2D solver efficiency plot to {eff_path}")
    return results


def main() -> None:
    args = parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1
    cfg = load_config(args)
    cfg["_eval_only"] = args.eval_only

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, ToyMLP] = {}
    for method in cfg["methods"]:
        models[method] = train_single_method(method, cfg, out_dir)

    metrics = evaluate_and_plot_figure4(models, cfg, out_dir)
    if args.eval_solver_efficiency:
        metrics["solver_efficiency"] = evaluate_2d_solver_efficiency(models, cfg, out_dir)

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
