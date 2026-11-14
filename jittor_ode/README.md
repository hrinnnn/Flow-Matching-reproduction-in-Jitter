# jittor_ode

Jittor 上的神经网络常微分方程求解包。Jittor 没有 `torchdiffeq` 这一类库，Flow Matching 复现里的求解器因此从零写起，并单独放在这里，供其他 Jittor 模型直接调用。

## 提供的组件

- 定步长求解器：`euler`、`midpoint`、`heun3`、`rk4`。状态可以是任意形状的 `jittor.Var`，也可以是若干张量组成的元组。
- 自适应 `dopri5`：Dormand–Prince 5(4)，下一步复用上一步最后一次函数值（FSAL），按容差接受或拒绝步长。
- `ODESolver.last_nfe`：一次采样实际调用速度场的次数。
- `hutchinson_divergence`：用一次 `jt.grad` 估计向量场散度，探针默认是 Rademacher 向量。

## 用法

```python
import jittor as jt
from jittor_ode import ODESolver, hutchinson_divergence

def velocity(x, t, **kwargs):
    return -x

solver = ODESolver(velocity)
x1 = solver.sample(jt.randn((16, 8)), method="dopri5", atol=1e-5, rtol=1e-5)
print("NFE", solver.last_nfe)

x = jt.randn((4, 8))
x.start_grad()
y = -0.3 * x
div, _ = hutchinson_divergence(y, x)
```

反向对数似然用 `solver.compute_likelihood`。它把瞬时变量替换公式和 Hutchinson 估计放进同一条反向 ODE。

预训练的 Flow Matching 权重不在这个包里。权重转换脚本是仓库里的 `tools/convert_unet_weights.py`，等 GPU 上训完再把 checkpoint 放进 `checkpoints/`。
