# Jittor 与 PyTorch / SciPy 对照

这些数字来自本机 CPU，命令是 `python benchmarks/compare_frameworks.py`，原始结果在 `framework_parity.json`。没有安装 `torchdiffeq`，所以自适应求解器对的是 SciPy 的 `RK45`（同样是 Dormand–Prince 5(4)）和闭式解，定步长 RK4 直接对 PyTorch 手写的同一步法。

激活值整体偏移 50，用来放大 GroupNorm 方差算法的差别。`GroupNorm32` 用两遍公式 `E[(x-E[x])^2]`，和 PyTorch `F.group_norm` 的最大绝对误差是 `1.7e-5`。Jittor 自带的 `nn.GroupNorm` 用 `E[x^2]-E[x]^2]`，大激活时会相消，U-Net 没有用它。

| 对照 | 结果 |
|---|---|
| GroupNorm32 对 NumPy 两遍公式 | 最大绝对误差 `1.6e-5` |
| GroupNorm32 对 PyTorch | 最大绝对误差 `1.7e-5` |
| Jittor dopri5 对 `exp(-1)`，容差 `1e-5` | 绝对误差 `3.0e-6` |
| SciPy RK45 对 `exp(-1)`，同样容差 | 绝对误差 `2.3e-6`，26 次函数调用 |
| Jittor RK4，100 步，对 `exp(-1)` | 绝对误差 `1.6e-7` |
| PyTorch 同样的 RK4 | 绝对误差 `1.6e-7`，和 Jittor 的差是 `0` |

`odeint_adjoint` 用 `dz/dt = c z` 的闭式梯度做过检查：对 `c` 的梯度相对误差在 `2e-3` 以内（RK4，步长 0.05）。
