"""Convert a PyTorch ADM U-Net checkpoint into this repository's Jittor checkpoint.

    python tools/convert_unet_weights.py \\
        --pytorch-checkpoint official.pt \\
        --preset cifar10_paper \\
        --output checkpoints/cifar10_paper_jittor.pkl
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jfm.models.unet import UNetModel
from tools.convert_unet_weights import assign_jittor_model, convert_state_dict, load_pytorch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch ADM U-Net -> Jittor")
    parser.add_argument("--pytorch-checkpoint", required=True)
    parser.add_argument("--preset", default="cifar10_paper")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = UNetModel.from_preset(args.preset, num_classes=args.num_classes)
    arrays = convert_state_dict(load_pytorch_checkpoint(args.pytorch_checkpoint))
    report = assign_jittor_model(model, arrays)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(report)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
