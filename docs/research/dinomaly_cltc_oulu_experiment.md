# LCF + CLTC-detach：OULU-NPU 实验

这是待验证的研究模块，没有预设性能提升或新颖性结论。入口为
`train_dinomaly_lcf_cltc_oulu.py`，参考 `train_dinomaly_lcf_face.py` 的预处理、
LCF、分组和训练默认值。不会因为文件存在而自动训练。

## 接入位置和训练

```text
冻结 DINO → 原始 F0…F7 ─────────→ 相邻层 patch cosine distance → target [B,784,7]
                  ↓                                               ↑ 比较
                LCF-A → detach → LayerNorm → Linear(C,128) → GELU → Linear(128,7)
                  ↓
             原 bottleneck → decoder → 原重建损失
```

默认开启 LCF 和 CLTC，从头训练；`detach` 固定开启。CLTC 输入只保留 patch，
CLS/register 不参加轨迹监督。目标始终取冻结 DINO 的原始层特征；即使启用
context recentering，也不改变轨迹目标的定义。层索引必须唯一且按深度递增。

`loss = reconstruction_loss + 0.1 * SmoothL1(predicted_trajectory, target)`。
CLTC 损失只更新预测头，LCF、瓶颈、decoder 继续接受原重建损失。
主干与预测头分别做梯度裁剪，避免全局 norm 裁剪引入两支路间的梯度耦合。
CLTC 初始化隔离 CPU 随机数流，保留共同模块的初始化；仍应使用多种子验证，
不能因此保证跨设备逐位相同。`--model-precision float16` 延续原脚本的 BF16
主干含义，预测头、轨迹距离、损失与分数尺度保持 FP32。

## 数据与实验协议

```text
OULU-NPU_organized/
  train/normal/    # 仅真人用于梯度训练
  dev/normal/      # 独立真人用于两支路尺度与阈值估计
  test/normal/    # 最终测试真人
  test/abnormal/  # 最终测试攻击
```

脚本要求四个目录都有图像，并拒绝路径重叠。不会自动把 test 切成验证集。
路径检查不能识别不同目录下复制的帧，运行前应确认受试者/视频来源符合你的划分。
`--calibration-dir` 可以指定其他独立真人开发目录，不能指向测试集。

采用固定训练步数的最终 checkpoint，无测试集挑 checkpoint、early stopping 或 alpha。
与旧脚本 `val_split_mode=from_test` 的结果不能直接作公平数值比较；请用本入口
`--no-use-cltc` 重跑同协议基线，或比较同一次运行导出的 reconstruction 与 fused。

这是自定义 folder **帧级实验**，不实现官方 OULU Protocol 1–4，也没有视频聚合、
攻击类别分组或官方最坏类别 APCER。若当前副本仅包含少量受试者，结果只能代表该子集。

## 运行

从仓库根目录执行。下面以 WSL/Linux 已配置好的 CUDA 环境为例。
请保持与你原 LCF 实验一致的 encoder、图像尺寸和精度；392 可改成原实验所用的 224。

如果当前环境只有 CPU 版 PyTorch，先按驱动配置安装 CUDA 依赖，例如
`uv sync --frozen --extra cu126`（其他驱动配置可选择仓库支持的 `cu130`）。
`uv run --no-sync` 会直接使用现有环境，不会自动把 CPU 版换成 GPU 版。

```bash
uv run --no-sync python train_dinomaly_lcf_cltc_oulu.py \
  --root /mnt/d/BaiduNetdiskDownload/OULU-NPU_organized \
  --encoder-name vit_giant_patch14_reg4_dinov2 \
  --image-size 392 \
  --train-batch-size 2 --eval-batch-size 2 \
  --model-precision float16 \
  --max-steps 5000 --seed 42 \
  --results-dir results/oulu_cltc_s42
```

独立 LCF 基线：使用相同参数，增加 `--no-use-cltc`，结果目录改为
`results/oulu_lcf_s42`。均值融合消融还可增加 `--no-use-lcf`。

`--resume-ckpt` 只用于恢复相同配置的训练，必须仍指定相同的模型参数，且使用新的
结果目录；不接受旧 LCF checkpoint 作为“续训 CLTC”的替代。脚本拒绝覆盖已有结果目录。
如需查看所有参数，执行 `uv run --no-sync python train_dinomaly_lcf_cltc_oulu.py --help`。

## 评分与输出

两支路均使用 resize 256 → Gaussian blur → top 1% mean 的图像评分方法。
CLTC patch 误差为七个预测距离与目标距离的绝对差均值。
在 `dev/normal` 上分别估计两个**图像分数**的 q95：

```text
fused_score = reconstruction_score / q95_rec + alpha * trajectory_score / q95_traj
alpha = 1.0（默认，测试前固定）
```

这是先分别池化再融合分数；返回的融合 anomaly_map 用于可视化，其 top 1%
池化值不一定等于上述分数。尺度为全局两个常数，不做区域校准。
尺度写入预测头 buffer，并保存在 `calibrated_last.ckpt` 中。

输出目录包含：

- `config.json`：实际参数和实验规则。
- `train/calibration/test_manifest.csv`：本次固定样本清单。
- `calibration_scores.csv`、`test_scores.csv`：路径、标签与三支路原始分数。
- `metrics.json`：三套 AUROC、近似 EER、APCER/BPCER/ACER、样本清单 SHA256 和尺度。
- `calibrated_last.ckpt`：含预测头、FP32 尺度和架构参数的最终权重。

Anomalib 不将预处理等组件保存为构造参数。独立加载该 checkpoint 时，应使用
`config.json` 中的 image/crop 尺寸重建预处理，并保持后处理关闭：

```python
from anomalib.models import Dinomaly
from train_dinomaly_lcf_face import build_pre_processor

model = Dinomaly.load_from_checkpoint(
    "results/oulu_cltc_s42/calibrated_last.ckpt",
    weights_only=False,  # 仅加载自己生成的可信 checkpoint
    pre_processor=build_pre_processor(392, None),  # 与 config.json 一致
    post_processor=False,
    evaluator=False,
    visualizer=False,
).eval()
```

标签规定为真人 0、攻击 1，分数越大越可疑。APCER 为攻击被判真人的比例，
BPCER 为真人被判攻击的比例。部署阈值仅取各支路真人 dev 分数的 q95，目标约为
dev BPCER 5%；不保证 test BPCER 也是 5%。EER 是测试 ROC 上最近交点的描述性估计，
不用于选部署阈值。指标范围均为 0–1。

第一次重点比较 `metrics.fused.AUROC` 和 `metrics.reconstruction.AUROC`，并查看
ACER/EER。CLTC 单独 AUROC 不需要超过重建分支，但应提供互补信息。
禁止看完测试结果后挑最优 alpha 并当作无调参结果；后续至少使用三个种子。

## 验证

离线单元测试使用小型伪 encoder，不下载 DINO 权重、不运行 OULU 训练：

```bash
uv run --no-sync python -m pytest tests/unit/models/image/dinomaly -q
```

覆盖 detach 梯度隔离、prefix 处理、原始轨迹目标、基线初始化及评分兼容、
独立梯度裁剪、BF16 主干/FP32 预测头、优化器包含预测头、尺度及 Lightning checkpoint 恢复。
