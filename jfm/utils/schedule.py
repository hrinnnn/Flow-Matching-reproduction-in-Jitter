"""Learning-rate schedule used for the CIFAR-10 runs in Appendix E.

Table 3 calls it polynomial decay. The appendix text is more specific: linear warmup
from ``1e-8`` to the peak learning rate, then linear decay back to ``1e-8``.
"""


def linear_warmup_decay(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float = 1e-8,
) -> float:
    """Return the learning rate at a 1-indexed optimizer step."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    step = max(1, min(int(step), int(total_steps)))
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / float(warmup_steps))
    if total_steps <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    return peak_lr + (min_lr - peak_lr) * progress
