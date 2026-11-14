# Flow Matching Reproduction in Jittor

清华大学《人工神经网络》课程大作业开源仓库：使用 **Jittor** 框架完整复现 Lipman et al. (ICLR 2023) [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747)，并系统完成采样效率分析、精确似然（BPD）估计、扩散基线对比与条件/最优传输扩展实验。

仓库：<https://github.com/hrinnnn/Flow-Matching-reproduction-in-Jitter>

```bash
pip install -r requirements.txt
python scripts/run_2d_checkerboard.py --iterations 1500 --batch_size 1024 --output_dir outputs/2d_preview
python scripts/train_cifar10.py --config configs/cifar10_smoke.yaml --fake-data
python scripts/train_cifar10.py --config configs/cifar10_paper.yaml --method fm_ot --use-cuda
python scripts/train_cifar10.py --config configs/cifar10_paper.yaml --method fm_dif --use-cuda
python scripts/eval_cifar10.py --config configs/cifar10_paper.yaml --checkpoint outputs/cifar10/fm_ot/ema.pkl --ema --use-cuda
```

`cifar10_paper` 用论文附录 E 表 3 的 U-Net。`cifar10_cfg` 在同一结构上加 10 类条件和 classifier-free guidance。官方库 `examples/image` 里的 CIFAR 默认配置更小，对应预设名 `cifar10_repo`，不用于表 1。

---

## 1. 任务范围对齐（课程要求全覆盖）

### 必做实验（Mandatory，对应课件 Slide 15 红框）
- **2D Checkerboard（复现论文 Figure 4）**
  - 统一 5 层 × 512 维 MLP，对比 `SM-Dif`、`FM-Dif`、`FM-OT`
  - 可视化 $t \in [0, 1]$ 概率密度/轨迹演化，以及 `Midpoint` 求解器在 `NFE = 4, 8, 10, 20` 下的低预算采样质量
- **CIFAR-10 无条件图像生成（复现论文 Table 1）**
  - 统一 ADM U-Net（`channels=256, depth=2, channel_mult=[1,2,2,2], attn=16, heads=4`）
  - 对比 `FM w/ Diffusion`（VP 路径）与 `FM w/ OT`（最优传输条件路径）的 **FID** 与自适应 `dopri5`（`atol=rtol=1e-5`）下的平均 **NFE**
  - 支持完整 `391k iters × bs 256` 配置与单卡 24G/28G GPU 等比例缩短预算配置（双路径严格同种子、同预算、同评测）

### 选做实验（Optional，对应课件 Slide 16 & Slide 5）
1. **精确对数似然（NLL / BPD，复现 Table 1 & Table 4）**：基于瞬时变量替换定理（Instantaneous Change of Variables）+ Hutchinson 随机迹估计器 + 均匀反量化（$K$ 次重要性加权）计算测试集 bits/dim。
2. **采样效率系统分析（复现 Figure 7 + 耗时 Profiling）**：复用主实验权重，对比 `Euler`、`Midpoint`、`RK4` 与自适应 `Dopri5` 在不同 NFE 预算下的数值轨迹误差（相对 1000-NFE 参考解的 MSE）、FID--NFE 曲线与 FID--Wall-clock 耗时曲线。
3. **同架构扩散基线与采样轨迹对比（复现 Table 1 Ablations & Figure 6）**：在同一套 Jittor U-Net 上实现 `DDPM`（Noise Matching）与 `Score Matching` / `ScoreFlow`，对比训练稳定性及同一初始噪声 $x_0$ 下的渐进生成轨迹。
4. **前沿扩展（Slide 16 Extensions）**：
   - **Class-Conditional Flow Matching + Classifier-Free Guidance (CFG)**（CIFAR-10 类别条件生成）
   - **Minibatch OT (OT-CFM) & Rectified Flow (Reflow)**（批内最优传输配对与直线化重流）
   - **Discrete Flow Matching**（离散状态空间流匹配验证实验）与 **ImageNet 32×32** 数据管线支持
   - **Jittor vs. PyTorch 算子编译热身、吞吐与显存对比分析**

---

## 2. 仓库目录结构

```text
Flow-Matching-reproduction-in-Jitter/
├── configs/                # 实验配置（2D、CIFAR-10 短跑/正式、Optional 评测与扩展）
├── jfm/                    # Jittor Flow Matching 核心模块
│   ├── paths/              # 条件路径：OTPath, VPDiffusionPath, MinibatchOTPath, DiscretePath
│   ├── objectives/         # 损失函数：CFMLoss, ScoreMatchingLoss, DDPMLoss, CFGLoss
│   ├── models/             # 网络模型：ToyMLP (5x512), ADMUNet (32x32), DiscreteTransformer
│   ├── solvers/            # 原生 Jittor ODE 求解器：Euler, Midpoint, RK4, Dopri5 (含精确 NFE 统计)
│   ├── likelihood/         # Hutchinson 散度估计与反向增广 ODE 的 BPD 计算器
│   ├── datasets/           # 2D Checkerboard 与 CIFAR-10 / ImageNet-32 数据加载器
│   ├── metrics/            # FID 评测、轨迹 MSE、吞吐量与显存监控
│   └── utils/              # EMA、Warmup+多项式衰减日程、随机种子控制、Checkpoint 管理
├── scripts/                # 训练、采样、评测与绘图入口脚本
└── tests/                  # CPU 无卡可跑的数学与算子单元测试
```

---

## 3. 环境安装

```bash
pip install -r requirements.txt
python -m jittor.test.test_example
```

---

## 4. 参考文献

- Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. *Flow Matching for Generative Modeling*. ICLR 2023.
- 官方 PyTorch 参考库：<https://github.com/facebookresearch/flow_matching>
