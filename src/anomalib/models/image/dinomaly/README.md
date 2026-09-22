# Dinomaly: 基于 Vision Transformer 的特征重建异常检测

> 本文档提供 Dinomaly 模型的完整讲解，包括原理、代码、运行方式和调参指南。

实验性 CLTC 旁路可通过 `Dinomaly(use_lcf=True, use_cltc=True)` 启用（默认关闭）。
该预测头读取 LCF 输入融合结果的 detached patch 特征，仅预测头接收轨迹损失的梯度。
OULU-NPU 运行命令、正常开发集尺度拟合及三支路评分比较，见
[CLTC 实验说明](../../../../../docs/research/dinomaly_cltc_oulu_experiment.md)。

## 目录

- [1. 模型简介](#1-模型简介)
- [2. 核心原理（小白友好版）](#2-核心原理小白友好版)
- [3. 网络架构详解](#3-网络架构详解)
- [4. 核心代码逐行讲解](#4-核心代码逐行讲解)
- [5. 如何运行代码](#5-如何运行代码)
- [6. 所有可调参数详解](#6-所有可调参数详解)
- [7. 实验建议与常见问题](#7-实验建议与常见问题)
- [8. 基准测试结果](#8-基准测试结果)

---

## 1. 模型简介

Dinomaly 是一个基于 Vision Transformer 的异常检测模型，采用**编码器-瓶颈层-解码器**的架构，通过**特征重建**的方式检测异常。

**核心思想**：只在正常样本上训练解码器重建特征，推理时重建不好的区域就是异常。

| 项目 | 说明 |
|------|------|
| 模型类型 | 分割（Segmentation） |
| 学习类型 | 单类（ONE_CLASS）/ 无监督 |
| 骨干网络 | DINOv2 Vision Transformer |
| 核心方法 | 特征重建 + 余弦相似度对比 |
| 论文来源 | [Dinomaly](https://github.com/guojiajeremy/Dinomaly) |

---

## 2. 核心原理（小白友好版）

### 2.1 一个比喻

想象你是一个美术生，老师**只让你画正常的苹果**，画了成千上万张。有一天，老师拿了一张**有虫洞的苹果**让你画，你会怎么样？

- 正常的部分你画得很像（因为练过）
- 虫洞的部分你画得不像（因为从来没见过）

**Dinomaly 就是这个思路**：
- 训练时，只给模型看**正常的图片**，让它学会"重建"正常特征
- 测试时，拿一张图让模型重建，**重建得不像的地方就是异常**

### 2.2 整体流程

```
输入图片
   ↓
[DINOv2 编码器]  ← 预训练好的，不动它，用来提取特征
   ↓  （提取中间8层的特征）
[瓶颈 MLP]      ← 带 Dropout，故意"弄坏"一点特征
   ↓
[解码器]        ← 8层 Transformer，用线性注意力
   ↓
重建的特征
   ↓
和编码器特征比差异 → 差异大的地方 = 异常
```

### 2.3 四个关键设计技巧

#### 技巧1：带噪声的瓶颈层（Noisy Bottleneck）

在瓶颈 MLP 里加 **Dropout**（随机丢掉一些神经元），相当于故意把特征"弄坏一点"。解码器必须学会从"坏掉的特征"里恢复出正常特征。这样遇到真的异常时，解码器就恢复不出来了。

> 类比：你练习画画时，老师故意把参考图模糊一点给你看，你练多了就能从模糊图里脑补出正常的样子。但如果参考图上有个你从没见过的虫洞，你脑补不出来。

#### 技巧2：线性注意力（Linear Attention）

普通的 Transformer 用 Softmax 注意力，会"聚焦"在少数重要区域。Dinomaly 故意用**线性注意力**，让注意力"分散"到整张图上。

为什么？因为如果解码器注意力太集中，它可能会"抄"输入里的异常区域，导致异常也被重建得很好。分散注意力后，解码器更依赖全局的正常模式来重建，异常区域就重建不好了。

#### 技巧3：松散重建（Loose Reconstruction）

有两层意思：

**松散约束**：不是一层对一层地重建，而是把编码器的几层特征**平均融合**成两组（低层语义 + 高层语义），解码器也输出两组，两组对比。这样解码器更自由，遇到没见过的模式时，和编码器的差异会更大。

**松散损失**：训练时，那些已经重建得很好的点（简单点），我们**降低它们的梯度贡献**（乘 0.1 倍），让模型专注于重建不好的点。防止模型"太能干"，连异常都能重建好。

#### 技巧4：多尺度异常检测

不是只看最后一层特征，而是看**两组不同语义层次的特征**（低语义 + 高语义），计算余弦相似度差异，然后融合成最终的异常图。

---

## 3. 网络架构详解

### 3.1 三大组件

| 组件 | 作用 | 是否训练 |
|------|------|---------|
| DINOv2 编码器 | 提取图片的多尺度特征 | ❌ 冻结 |
| 瓶颈 MLP | 压缩特征 + Dropout 制造噪声 | ✅ 训练 |
| 解码器 | 重建编码器的特征 | ✅ 训练 |

### 3.2 编码器配置

支持四种 DINOv2 模型大小：

| 模型 | embed_dim | num_heads | target_layers（提取哪几层） |
|------|-----------|-----------|---------------------------|
| small | 384 | 6 | [2, 3, 4, 5, 6, 7, 8, 9] |
| base | 768 | 12 | [2, 3, 4, 5, 6, 7, 8, 9] |
| large | 1024 | 16 | [4, 6, 8, 10, 12, 14, 16, 18] |
| huge | 1280 | 20 | [3, 9, 12, 15, 18, 21, 24, 27] |

### 3.3 特征融合策略

默认将 8 层特征分成两组融合：

```python
DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]
# 前4层 → 低语义特征（边缘、纹理等）
# 后4层 → 高语义特征（部件、概念等）
```

### 3.4 解码器结构

- 默认 8 层 Transformer Block
- 每层使用 **LinearAttention**（线性注意力）
- 每层包含：LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差
- MLP 比例：4 倍

---

## 4. 核心代码逐行讲解

### 4.1 模型主结构：`torch_model.py`

#### 4.1.1 `__init__` 初始化

```python
def __init__(
    self,
    encoder_name: str = "vit_base_patch14_reg4_dinov2",  # 编码器名称
    bottleneck_dropout: float = 0.2,                       # 瓶颈层 dropout
    decoder_depth: int = 8,                                 # 解码器层数
    target_layers: list[int] | None = None,                 # 提取哪些层
    fuse_layer_encoder: list[list[int]] | None = None,      # 编码器层分组
    fuse_layer_decoder: list[list[int]] | None = None,      # 解码器层分组
    remove_class_token: bool = False,                       # 是否移除 class token
    use_context_recentering: bool = False,                  # 是否启用上下文重中心化
):
```

**第一步：创建编码器（冻结）**

```python
self.encoder = TimmFeatureExtractor(
    backbone=encoder_name,
    layers=[f"blocks.{i}" for i in self.target_layers],  # 提取指定层
    pre_trained=True,        # 使用预训练权重
    requires_grad=False,     # 冻结，不参与训练
    output_fmt="NLC",        # 输出格式: [Batch, Seq_len, Channel]
    return_class_token=True, # 返回 class token
    norm=False,
    dynamic_img_size=True,
)
```

> 编码器是预训练好的 DINOv2，我们不动它，只用来提取特征。

**第二步：创建瓶颈层（带 Dropout）**

```python
bottle_neck_mlp = DinomalyMLP(
    in_features=embed_dim,
    hidden_features=embed_dim * 4,
    out_features=embed_dim,
    act_layer=nn.GELU,
    drop=bottleneck_dropout,
    bias=False,
    apply_input_dropout=True,  # 关键！输入就加 dropout，制造伪异常
)
self.bottleneck = nn.ModuleList([bottle_neck_mlp])
```

> 注意 `apply_input_dropout=True`，特征一进来就先随机丢一部分，这是"噪声瓶颈"的核心。

**第三步：创建解码器（线性注意力）**

```python
decoder = []
for _ in range(decoder_depth):  # 创建 8 层
    decoder_block = DecoderViTBlock(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        attn=LinearAttention,  # 关键！用线性注意力，不是 Softmax 注意力
    )
    decoder.append(decoder_block)
self.decoder = nn.ModuleList(decoder)
```

#### 4.1.2 核心前向传播：`get_encoder_decoder_outputs`

这是模型最核心的函数，完整流程：

```python
def get_encoder_decoder_outputs(self, x: torch.Tensor):
    # 1. 计算 patch 数量
    h_patches = x.shape[2] // self.encoder.patch_size  # 高方向 patch 数
    w_patches = x.shape[3] // self.encoder.patch_size  # 宽方向 patch 数
    # 例如输入 392x392，patch_size=14 → 28x28 = 784 个 patch

    # 2. 过编码器，提取 8 层特征
    features = self.encoder(x)
    encoder_features = [features[f"blocks.{i}"] for i in self.target_layers]
    # 每个特征 shape: [B, 785, 768]
    # 785 = 1 class token + 4 register tokens + 784 patch tokens

    # 3. 处理 class token
    if self.remove_class_token:
        # 直接去掉 class token 和 register token
        encoder_features = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in encoder_features]
    elif self.use_context_recentering:
        # Dinomaly2 技巧：patch 特征减去 class token，做上下文重中心化
        recentered = []
        for e in encoder_features:
            cls_token = e[:, 0:1, :]
            patch_start = 1 + self.encoder.num_register_tokens
            patches = e[:, patch_start:, :] - cls_token
            recentered.append(patches)
        encoder_features = recentered

    # 4. 8 层特征先融合成 1 组，过瓶颈层
    x = self._fuse_feature(encoder_features)  # 8 层取平均
    for block in self.bottleneck:
        x = block(x)  # 过瓶颈 MLP（带 dropout 噪声）

    # 5. 过解码器，收集每一层的输出
    decoder_features = []
    for block in self.decoder:
        x = block(x, attn_mask=None)
        decoder_features.append(x)
    decoder_features = decoder_features[::-1]  # 倒序！深层对浅层
    # 为什么倒序？解码器第8层（最深）对应编码器第2层（最浅）

    # 6. 编码器和解码器各自做层融合（2组）
    en = [self._fuse_feature([encoder_features[idx] for idx in idxs])
          for idxs in self.fuse_layer_encoder]
    de = [self._fuse_feature([decoder_features[idx] for idx in idxs])
          for idxs in self.fuse_layer_decoder]

    # 7. 从序列 reshape 成空间特征图
    en = self._process_features_for_spatial_output(en, h_patches, w_patches)
    de = self._process_features_for_spatial_output(de, h_patches, w_patches)
    # 从 [B, 768, 784] → [B, 768, 28, 28]

    return en, de
```

#### 4.1.3 训练 vs 推理：`forward`

```python
def forward(self, batch, global_step=None):
    en, de = self.get_encoder_decoder_outputs(batch)

    if self.training:
        # 训练模式：返回损失
        return self.loss_fn(en, de, global_step=global_step)

    # 推理模式：计算异常图
    anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)

    # 高斯平滑
    anomaly_map = self.gaussian_blur(anomaly_map)

    # 计算图片级异常分数（取前 1% 最大值的平均）
    anomaly_map_flat = anomaly_map.flatten(1)
    sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][
        :, : int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO),
    ]
    sp_score = sp_score.mean(dim=1)

    return InferenceBatch(pred_score=sp_score, anomaly_map=anomaly_map_resized)
```

#### 4.1.4 异常图计算：`calculate_anomaly_maps`

```python
@staticmethod
def calculate_anomaly_maps(source_feature_maps, target_feature_maps, out_size=392):
    anomaly_map_list = []
    for i in range(len(target_feature_maps)):
        fs = source_feature_maps[i]  # 编码器特征
        ft = target_feature_maps[i]  # 解码器特征
        # 余弦相似度：越相似越接近1，越不相似越小
        a_map = 1 - F.cosine_similarity(fs, ft)  # 1 - 相似度 = 异常度
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear")
        anomaly_map_list.append(a_map)

    # 多个尺度的异常图取平均
    anomaly_map = torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, anomaly_map_list
```

> 核心思想：**编码器特征和解码器特征的余弦相似度越低，说明越异常**。

---

### 4.2 损失函数：`components/loss.py`

#### `CosineHardMiningLoss` — 硬挖掘余弦损失

```python
def forward(self, encoder_features, decoder_features, global_step):
    self._update_p_schedule(global_step)  # 更新 p 值（渐进式增长）
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0

    for item in range(len(encoder_features)):
        en_ = encoder_features[item].detach()  # 编码器不反传梯度
        de_ = decoder_features[item]

        # 计算每个点的重建误差，找到阈值
        with torch.no_grad():
            point_dist = 1 - cos_loss(en_, de_).unsqueeze(1)
        k = max(1, int(point_dist.numel() * (1 - self.p)))  # 选前 (1-p)% 最难的点
        thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]

        # 正常计算余弦损失
        loss += torch.mean(1 - cos_loss(en_.reshape(...), de_.reshape(...)))

        # 关键！给解码器输出注册梯度钩子
        # 重建得好的点（误差 < 阈值），梯度乘以 factor（默认 0.1）
        partial_func = partial(
            self._modify_grad,
            indices_to_modify=point_dist < thresh,
            gradient_multiply_factor=self.factor,
        )
        de_.register_hook(partial_func)

    return loss / len(encoder_features)
```

**p 值调度**：

```python
def _update_p_schedule(self, global_step):
    # p 从 0 线性增长到 p_final（默认0.9），用 p_schedule_steps 步
    self.p = min(self.p_final * global_step / self.p_schedule_steps, self.p_final)
```

> 大白话：训练初期 p=0，所有点正常训练。
> 训练到 1000 步后 p=0.9，90% 重建得好的点梯度被降到 10%，
> 只有 10% 最难的点正常回传梯度。
> 防止模型"太努力"学好所有东西，连异常都能重建。

---

### 4.3 线性注意力：`components/layers.py`

#### `LinearAttention`

普通注意力：`Softmax(QK^T/√d) · V`，复杂度 O(n²)，注意力分布尖锐

线性注意力：用 `ELU(x)+1` 把 Q、K 变成正数，然后
`(Q · (K^T · V)) / (Q · K_sum)`，复杂度 O(n)，注意力分布平缓

```python
def forward(self, x):
    # 生成 Q, K, V
    qkv = self.qkv(x).reshape(...).permute(...)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # 关键！用 ELU+1 代替 Softmax
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0

    # 线性注意力计算
    kv = torch.matmul(k.transpose(-2, -1), v)   # K^T · V
    k_sum = k.sum(dim=-2, keepdim=True)
    z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)  # 归一化因子
    x = torch.matmul(q, kv) * z  # Q · (K^T · V) / z

    x = self.proj(x)
    x = self.proj_drop(x)
    return x, kv
```

> 为什么线性注意力能让注意力分散？
> Softmax 会放大最大值（马太效应），线性注意力没有 Softmax，
> 注意力分布更均匀，不会过度聚焦少数区域。

---

### 4.4 Lightning 模块：`lightning_model.py`

#### 4.4.1 冻结编码器

```python
# 所有参数先冻结
for param in self.model.parameters():
    param.requires_grad = False

# 只训练瓶颈层和解码器
for param in self.model.bottleneck.parameters():
    param.requires_grad = True
for param in self.model.decoder.parameters():
    param.requires_grad = True
```

#### 4.4.2 训练步骤

```python
def training_step(self, batch, *args, **kwargs):
    loss = self.model(batch.image, global_step=self.global_step)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    return {"loss": loss}
```

#### 4.4.3 优化器配置

```python
def configure_optimizers(self):
    # StableAdamW 优化器
    optimizer = StableAdamW(
        [{"params": self.trainable_modules.parameters()}],
        lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True
    )

    # Warmup + Cosine 学习率调度
    lr_scheduler = WarmCosineScheduler(
        optimizer,
        base_value=2e-3,      # 初始学习率
        final_value=2e-4,     # 最终学习率
        total_iters=5000,     # 总步数
        warmup_iters=100,     # warmup 步数
    )

    return [optimizer], [lr_scheduler]
```

#### 4.4.4 预处理配置

```python
@classmethod
def configure_pre_processor(cls, image_size=None, crop_size=None):
    data_transforms = Compose([
        Resize((448, 448)),          # 先 resize 到 448
        CenterCrop(392),             # 中心裁剪到 392
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return PreProcessor(transform=data_transforms)
```

> 392 是 14 的倍数（DINOv2 的 patch size 是 14）。

---

### 4.5 优化器：`components/optimizer.py`

#### `StableAdamW` — 稳定版 AdamW

在普通 AdamW 基础上加了梯度裁剪机制，防止更新步太大导致训练不稳定：

```python
lr_scale = grad / denom
lr_scale = max(1.0, self._rms(lr_scale) / group["clip_threshold"])
step_size = group["lr"] / bias_correction1 / lr_scale
```

#### `WarmCosineScheduler` — 暖启动余弦调度

学习率先线性增长（warmup），再余弦下降：

```python
warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
# warmup 阶段：线性增长

iters = np.arange(total_iters - warmup_iters)
schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
# cosine 退火阶段：余弦下降
```

---

## 5. 如何运行代码

### 5.1 方式一：命令行（最简单）

#### 训练

```bash
# 最简命令
anomalib train --model Dinomaly --data MVTecAD --data.category bottle

# 指定训练步数和 batch size
anomalib train --model Dinomaly --data MVTecAD --data.category bottle \
  --trainer.max_steps 5000 --data.train_batch_size 16

# 用配置文件
anomalib train --config examples/configs/model/dinomaly.yaml --data MVTecAD --data.category bottle
```

#### 测试

```bash
anomalib test --model Dinomaly --data MVTecAD --data.category bottle \
  --ckpt_path results/Dinomaly/MVTecAD/bottle/run/weights/lightning/model.ckpt
```

#### 推理

```bash
anomalib predict --model Dinomaly --data MVTecAD --data.category bottle \
  --ckpt_path results/Dinomaly/MVTecAD/bottle/run/weights/lightning/model.ckpt \
  --output outputs/
```

#### 导出

```bash
# 导出为 ONNX
anomalib export --model Dinomaly --export_format ONNX \
  --ckpt_path results/Dinomaly/MVTecAD/bottle/run/weights/lightning/model.ckpt

# 导出为 OpenVINO
anomalib export --model Dinomaly --export_format OPENVINO \
  --ckpt_path results/Dinomaly/MVTecAD/bottle/run/weights/lightning/model.ckpt
```

### 5.2 方式二：Python API（推荐做实验用）

#### 完整训练 + 测试

```python
from anomalib.data import MVTecAD
from anomalib.models import Dinomaly
from anomalib.engine import Engine

# 1. 准备数据集
datamodule = MVTecAD(
    root="./datasets/MVTecAD",   # 数据集路径，自动下载
    category="bottle",            # 类别
    train_batch_size=16,          # 训练 batch size
    eval_batch_size=16,           # 测试 batch size
)

# 2. 创建模型
model = Dinomaly(
    encoder_name="vit_base_patch14_reg4_dinov2",
    bottleneck_dropout=0.2,
    decoder_depth=8,
)

# 3. 创建引擎
engine = Engine(
    max_steps=5000,
    devices=1,
    accelerator="gpu",
    default_root_dir="./results",
)

# 4. 训练
engine.fit(model=model, datamodule=datamodule)

# 5. 测试
engine.test(model=model, datamodule=datamodule)

# 6. 推理
predictions = engine.predict(model=model, datamodule=datamodule)
```

#### 加载模型推理

```python
from anomalib.deploy import TorchInferencer
from PIL import Image

inferencer = TorchInferencer(
    path="results/Dinomaly/MVTecAD/bottle/run/weights/lightning/model.ckpt",
    device="cuda",
)

image = Image.open("test_image.png")
result = inferencer.predict(image=image)
print(result.pred_score)       # 图片级异常分数
print(result.anomaly_map)      # 像素级异常图
```

### 5.3 结果保存位置

```
results/
└── Dinomaly/
    └── MVTecAD/
        └── bottle/
            └── run/
                ├── weights/
                │   └── lightning/
                │       └── model.ckpt    # 模型权重
                ├── images/               # 可视化结果
                └── logs/                 # 训练日志
```

---

## 6. 所有可调参数详解

### 6.1 模型参数

| 参数名 | 默认值 | 含义 | 调参建议 |
|--------|--------|------|---------|
| `encoder_name` | `"vit_base_patch14_reg4_dinov2"` | 编码器模型名称 | 可选 small/base/large/huge，越大效果越好但越慢越吃显存 |
| `bottleneck_dropout` | `0.2` | 瓶颈层 dropout 率 | 越大正则化越强。过拟合时调大 |
| `decoder_depth` | `8` | 解码器 Transformer 层数 | 层数越多重建能力越强，但可能过拟合。一般 6-12 层 |
| `target_layers` | `None` | 编码器提取哪些层的特征 | None 时自动选 8 层。可自定义如 `[3,4,5,6,7,8,9,10]` |
| `fuse_layer_encoder` | `[[0,1,2,3],[4,5,6,7]]` | 编码器层分组方式 | 默认前4后4。可改成更多组做多尺度 |
| `fuse_layer_decoder` | `[[0,1,2,3],[4,5,6,7]]` | 解码器层分组方式 | 注意和编码器组数对应 |
| `remove_class_token` | `False` | 是否去掉 class token | 一般不用改 |
| `use_context_recentering` | `False` | 是否启用 Dinomaly2 上下文重中心化 | 多类别训练时可以设为 True。不能和 remove_class_token 同时为 True |
| `precision` | `PrecisionType.FLOAT32` | 数值精度 | 设为 `"float16"` 可省显存（实际用 bfloat16） |

### 6.2 训练超参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `lr` | `2e-3` | 初始学习率 | 最常用调参项。范围 1e-4 ~ 5e-3 |
| `betas` | `(0.9, 0.999)` | Adam 的 beta 参数 | 一般不用改 |
| `weight_decay` | `1e-4` | 权重衰减 | 正则化，防止过拟合 |
| `amsgrad` | `True` | 是否用 AMSGrad | 一般不用改 |
| `final_value` | `2e-4` | 最终学习率 | cosine 退火的最终值 |
| `warmup_iters` | `100` | warmup 步数 | 训练初期学习率从 0 增长到 lr 的步数 |
| `max_steps` | `5000` | 最大训练步数 | 数据集小可以少点，一般 3000~10000 |
| `gradient_clip_val` | `0.1` | 梯度裁剪阈值 | 防止梯度爆炸，一般不用改 |
| `batch_size` | 取决于数据集 | 批次大小 | 显存够就调大，16 或 32 常用 |

### 6.3 损失函数参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `p_final` | `0.9` | 最终降低多少比例的简单样本梯度 | 0.9 表示 90% 简单样本梯度被降低。越大越"松散" |
| `p_schedule_steps` | `1000` | p 值增长到 p_final 的步数 | 一般不用改 |
| `factor` | `0.1` | 简单样本的梯度乘以多少 | 0.1 表示降到原来的 10%。越小越"松散" |

> 注意：损失函数参数目前没有在 `Dinomaly` 类中直接暴露，如需修改需调整 `torch_model.py` 中 `CosineHardMiningLoss()` 的初始化。

### 6.4 推理后处理参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `DEFAULT_RESIZE_SIZE` | `256` | 异常图 resize 大小 | 一般不用改 |
| `DEFAULT_GAUSSIAN_KERNEL_SIZE` | `5` | 高斯模糊核大小 | 越大异常图越平滑 |
| `DEFAULT_GAUSSIAN_SIGMA` | `4` | 高斯模糊 sigma | 越大越平滑 |
| `DEFAULT_MAX_RATIO` | `0.01` | 图片级分数取前百分之多少的最大值 | 0.01 = 前 1%。越小越关注最异常区域 |

### 6.5 预处理参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `DEFAULT_IMAGE_SIZE` | `448` | resize 后图片大小 | 必须是 14 的倍数 |
| `DEFAULT_CROP_SIZE` | `392` | 中心裁剪大小 | 必须是 14 的倍数。越小越快，越大细节越多 |

---

## 7. 实验建议与常见问题

### 7.1 入门实验（先跑通）

```python
from anomalib.data import MVTecAD
from anomalib.models import Dinomaly
from anomalib.engine import Engine

# 先用默认参数跑一个类别
model = Dinomaly()
datamodule = MVTecAD(category="bottle", train_batch_size=8)
engine = Engine(max_steps=1000)  # 先少跑点试试
engine.fit(model, datamodule=datamodule)
```

### 7.2 调参实验顺序

1. **先调 batch size**：看显存能撑多大，越大训练越稳定
2. **调 max_steps**：看多少步收敛，验证集 AUROC 不涨就可以停
3. **调 learning rate**：最影响效果，试试 1e-4, 5e-4, 2e-3, 5e-3
4. **调 bottleneck_dropout**：过拟合调大（0.2→0.3→0.5），欠拟合调小
5. **调 decoder_depth**：试试 6 层、8 层、12 层
6. **换编码器**：small → base → large，看效果提升

### 7.3 常见问题

**Q: 显存不够怎么办？**
- 减小 batch size（16→8→4）
- 换小的编码器（base → small）
- 减小图片尺寸（392→224，注意要是 14 的倍数）
- 用 float16 精度

**Q: 效果不好怎么办？**
- 确认训练是否收敛（loss 下降了吗？）
- 调大 max_steps 多训一会
- 试试调小 learning rate
- 试试调大 bottleneck_dropout（防止过拟合）

**Q: 怎么看训练过程？**
- TensorBoard：`tensorboard --logdir results/`
- 看控制台输出的 train_loss 和 image_AUROC

---

## 8. 基准测试结果

所有结果使用 seed=42，max_steps=5000，batch_size=16。

### MVTec AD 数据集

#### Image-Level AUC

|  | Avg | Bottle | Cable | Capsule | Carpet | Grid | HazelNut | Leather | Metal Nut | Pill | Screw | Tile | ToothBrush | Transistor | Wood | Zipper |
|--|:---:|:------:|:-----:|:-------:|:------:|:----:|:--------:|:-------:|:---------:|:----:|:-----:|:----:|:----------:|:----------:|:----:|:------:|
| Dinomaly | 0.997 | 1.000 | 1.000 | 0.986 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 0.995 | 0.986 | 1.000 | 1.000 | 0.997 | 0.996 | 1.000 |

#### Image F1 Score

|  | Avg | Bottle | Cable | Capsule | Carpet | Grid | HazelNut | Leather | Metal Nut | Pill | Screw | Tile | ToothBrush | Transistor | Wood | Zipper |
|--|:---:|:------:|:-----:|:-------:|:------:|:----:|:--------:|:-------:|:---------:|:----:|:-----:|:----:|:----------:|:----------:|:----:|:------:|
| Dinomaly | 0.987 | 1.000 | 1.000 | 0.982 | 0.994 | 0.991 | 0.993 | 0.995 | 0.984 | 0.979 | 0.961 | 1.000 | 0.983 | 0.963 | 0.984 | 0.996 |

#### Pixel-Level AUC

|  | Avg | Bottle | Cable | Capsule | Carpet | Grid | HazelNut | Leather | Metal Nut | Pill | Screw | Tile | ToothBrush | Transistor | Wood | Zipper |
|--|:---:|:------:|:-----:|:-------:|:------:|:----:|:--------:|:-------:|:---------:|:----:|:-----:|:----:|:----------:|:----------:|:----:|:------:|
| Dinomaly | 0.984 | 0.990 | 0.985 | 0.986 | 0.993 | 0.994 | 0.994 | 0.992 | 0.966 | 0.980 | 0.997 | 0.976 | 0.989 | 0.951 | 0.977 | 0.991 |

#### Pixel F1 Score

|  | Avg | Bottle | Cable | Capsule | Carpet | Grid | HazelNut | Leather | Metal Nut | Pill | Screw | Tile | ToothBrush | Transistor | Wood | Zipper |
|--|:---:|:------:|:-----:|:-------:|:------:|:----:|:--------:|:-------:|:---------:|:----:|:-----:|:----:|:----------:|:----------:|:----:|:------:|
| Dinomaly | 0.670 | 0.822 | 0.732 | 0.568 | 0.700 | 0.556 | 0.756 | 0.467 | 0.851 | 0.687 | 0.572 | 0.736 | 0.608 | 0.615 | 0.670 | 0.716 |

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `torch_model.py` | PyTorch 模型主体实现 |
| `lightning_model.py` | Lightning 模块封装（训练流程） |
| `components/layers.py` | 线性注意力、MLP、Transformer Block |
| `components/loss.py` | 硬挖掘余弦损失 |
| `components/optimizer.py` | StableAdamW + WarmCosineScheduler |
| `__init__.py` | 模块导出 |
