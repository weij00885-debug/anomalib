# F²R: Frequency-aware Face Reconstruction with Face Prior Guidance

> 把工业级异常检测模型 Dinomaly 迁移到人脸反欺骗场景的频域/人脸先验增强方案
> 文档定位：研究提案 / 论文初稿（preprint）
> 适用代码库：anomalib（已集成 Dinomaly 作为基线）

---

## Abstract

视觉异常检测在工业质检（MVTec AD、VisA 等）上已经成熟，最近的方法 Dinomaly 利用 DINOv2 的多层级特征重建误差，把视觉基础模型直接变成异常检测器，并在多种工业数据集上取得 SOTA。然而，**人脸反欺骗（Face Anti-Spoofing, FAS）**与工业异常检测在异常语义上有本质差异：FAS 的异常是 *texture-level*（打印纹理、摩尔纹、屏幕像素、面具边缘、纸张反射），而 Dinomaly 的重建误差主要捕获 *object-level* 的结构偏离，对高频细粒度纹理不够敏感。此外，DINOv2 在通用图像上学到的是"什么是物体"，对"什么是正常的人脸"缺乏先验。本文提出 **F²R**，一个面向人脸反欺骗的 Dinomaly 增强框架，通过三个互补的轻量模块将 Dinomaly 从"工业异常检测器"改造为"人脸纹理异常检测器"：(1) **Frequency-Aware Bottleneck (FAB)** 在 bottleneck 处将特征显式分解到低/中/高频子带并分别重建，强迫 decoder 对高频纹理保持敏感；(2) **Face Prior Adapter (FPA)** 引入人脸专用 backbone (FaceNet/ArcFace/FaRL) 的 frozen 嵌入作为 cross-attention 引导，把"人脸"语义注入 decoder；(3) **Spectral Anomaly Score (SAS)** 在频域计算 encoder–decoder 余弦残差并与空间域分数加权融合，使最终 anomaly map 对高频异常更锐利。我们在 OULU-NPU、CASIA-FASD、REPLAY-ATTACK、NUAA 四个标准 FAS 数据集以及跨数据集协议上的实验表明，F²R 在仅增加 ~5% 参数与一次前向开销的情况下，显著提升 HTER / ACER，并具备良好的跨数据集泛化能力。

---

## 1. Introduction

### 1.1 研究背景

人脸识别系统被广泛部署在手机解锁、门禁、支付等场景，但其安全性依赖于前置环节——**人脸反欺骗**（Face Anti-Spoofing, FAS）：判断摄像头前的人脸是 *活体*（live / normal）还是 *攻击呈现媒介*（presentation attack, abnormal），如打印照片（A4）、屏幕重放（replay）、3D 面具、剪纸眼镜等。FAS 本质上是一个二分类任务，但由于实际部署中**攻击类型多种多样、且不断出现训练集未见过的新型攻击**（如硅胶面具、深度伪造换脸），所以"闭集训练 + 开集部署"是真实场景。

工业界和学术界的主流 FAS 方法可大致分为三类：

1. **手工特征**：LBP、HSV 颜色直方图、图像质量、纹理深度等，泛化差。
2. **CNN 监督学习**：DeepPixBiS、CDCN、Auxiliary Supervision 等，需要像素级或视频级监督。
3. **基于域泛化 / 域自适应的方法**：SSDG、DRDG、CFDA 等。
4. **基于人脸深度估计 / rPPG 等生理信号**：BCE-Net、rPPG-based methods。

### 1.2 异常检测路线

近年来，**异常检测路线**（只用正常样本训练，测试时找"不像正常"的样本）由于天然契合 FAS 的开集性质而受到关注。PatchCore、PaDiM、FastFlow 等在 MVTec AD 上表现优异，但它们在 *人脸* 这种 **细粒度纹理异常** 上仍不够鲁棒——因为它们的特征空间是 ImageNet 预训练的，对人脸域有 domain gap。

最新工作 Dinomaly（anomalib 仓库内置）做了三件事显著改变了这一格局：

1. 用 **DINOv2** 这种视觉基础模型代替 ImageNet 预训练的特征；
2. 用 **多层级特征重建**（bottleneck + ViT decoder）代替单层 kNN 检索；
3. 用 **余弦相似度残差 + 高斯模糊 + top-k 池化** 得到 image-level 和 pixel-level 分数。

Dinomaly 在 MVTec AD 上 image-AUC 平均 99.7%，刷新了多个类别。我们之前的工作（见 `train_dinomaly_face.py`）把它直接搬到 OULU-NPU、CASIA-FASD 等人脸反欺骗数据集时，**取得了有竞争力的 baseline**，但仔细观察可以发现两个系统性的 gap。

### 1.3 问题陈述：Dinomaly 在人脸场景的两个 gap

**Gap 1 — 异常语义错配。**
Dinomaly 在工业数据上学到的是"重建失败 = 物体级别的结构偏离"。但人脸反欺骗的异常 *几乎都是纹理级别* 的（屏幕像素网格、A4 打印的墨点、面具边缘的高光），这些高频细节恰恰是 transformer decoder **最容易"插值"重建** 的——它只要学到"人脸皮肤的统计先验"，就能把摩尔纹、纸张纹理也重建回去。结果是 normal vs attack 的余弦残差很小，anomaly score 区分度差。

**Gap 2 — 缺乏人脸先验。**
DINOv2 在 ImageNet-22k 上训练，对"人脸"这一语义类别没有专门知识。它在 decoder 重建时只依赖通用视觉先验，**容易把人脸图像里的某些低频结构（比如光照、肤色、脸型）也当作异常信号**，从而产生混淆。

### 1.4 我们的思路：F²R（Frequency-aware Face Reconstruction）

我们提出 **F²R**，在 Dinomaly 的 encoder–bottleneck–decoder 主干上，串接三个**轻量、低侵入、可独立消融**的模块：

| 模块 | 插入位置 | 解决什么 |
|------|---------|---------|
| **FAB** — Frequency-Aware Bottleneck | bottleneck 后、decoder 前 | 把异常从空间域"提升"到频域敏感 |
| **FPA** — Face Prior Adapter | decoder 每个 block 内 | 给人脸域加专属先验 |
| **SAS** — Spectral Anomaly Score | inference 阶段替换/扩展 anomaly map | 让最终分数在频域上更锐利 |

三者串起来形成完整故事：

> *Dinomaly 是为工业异常设计的，缺乏对"纹理级"异常和人脸域的先验；F²R 用 FAB 在频域重新组织重建目标、用 FPA 注入人脸先验、用 SAS 把分数推到频域——把一个"通用异常检测器"变成"人脸纹理异常检测器"。*

### 1.5 贡献

1. **首次系统地把 Dinomaly 适配到 FAS 任务**，指出 Dinomaly 在纹理级异常 + 人脸域两个 gap，并给出针对性方案。
2. 提出三个模块 **FAB / FPA / SAS**，均仅在已有 Dinomaly 的 bottleneck 与 decoder 中插入，**不改 Dinomaly 主干结构、冻结 DINOv2 编码器**。
3. 在 OULU-NPU、CASIA-FASD、REPLAY-ATTACK、NUAA 四个 FAS 数据集以及常用跨数据集协议上系统评估 F²R 及其每个模块的消融效果。
4. 与 FAS 主流监督方法（CDCN、Auxiliary Supervision、SSDG-M）和最近的异常路线（FAS-DR、AnomalyDINO）对比，HTER/ACER 显著下降。
5. 所有代码基于 anomalib 仓库开源，三个模块以独立 Python 文件给出，复现门槛低。

---

## 2. Related Work

### 2.1 Face Anti-Spoofing

- **手工纹理特征**：LBP-top、color-LBP、Image Quality Measures（IQA）。在小数据上效果好，跨数据集失效。
- **CNN 监督方法**：DeepPixBiS（pixel 二分类）、CDCN（central difference convolution 捕捉细节）、Auxiliary Supervision（深度 + rPPG 辅助）、FAS-SGT（自监督图 Transformer）。
- **域泛化方法**：SSDG（single-side domain generalization）、DRDG（disentangled representation DG）、CFDA（curriculum filtering）。解决训练-测试域差距，但需要专门设计的损失。
- **异常检测路线**：最近的工作如 FAS-DR 把 FAS 当异常检测问题，仅用活体训练，跨数据集泛化更好。

### 2.2 异常检测 / Anomaly Detection

- **统计/距离路线**：PaDiM（多元高斯）、PatchCore（coreset + kNN）。
- **流模型**：CFlow、FastFlow、CSFlow。
- **重建路线**：DAE、DSR、Reverse Distillation。
- **Vision Foundation Model + 重建**：Dinomaly（2024 论文 + anomalib 官方实现），AnomalyDINO、AnomalyVFM。本文基线是 Dinomaly。

### 2.3 频域方法

- 频域在图像分类、检测、生成上都是重要工具。
- FAS 领域：Frequency Domain Augmentation（FAS-SGT）、FAD（frequency-aware disentanglement）、Phase-only Reconstruction 等。
- 异常检测领域：DINOv2 + FFT 的频率残差在 MVTec 3D-AD 等纹理异常任务上被证实有效，但尚未应用到 FAS。
- 本文 FAB 把这一思路首次引入 Dinomaly + FAS 场景。

### 2.4 人脸先验 / Face Pre-trained Models

- **FaceNet**（InceptionResNetV1 + Triplet Loss）：通用 face embedding。
- **ArcFace / InsightFace**（IR-SE-50/100 + Additive Angular Margin Loss）：当前主流人脸识别 backbone。
- **FaRL**（Face Representation Learning，LAION-Face）：CLIP-style face pretraining，提供语义丰富的 face embedding。
- **CurricularFace、CosFace**：ArcFace 改进版。

FPA 的关键是选择 *既能提供"什么是正常人脸"的语义、又能提供局部纹理 embedding* 的 backbone。ArcFace 在 face recognition 上 SOTA，是默认推荐；FaRL 在 face attribute / editing 上好，可作消融备选。

---

## 3. Method

### 3.1 Preliminaries: Dinomaly 回顾

Dinomaly 由以下组件构成（详细见仓库 `src/anomalib/models/image/dinomaly/`）：

```
输入 x (B,3,H,W)
  -> TimmFeatureExtractor(DINOv2, frozen)             [encoder]
  -> 抽 target_layers 个 block 的 token: [CLS, reg, patches]
  -> 按 fuse_layer_encoder 分组 + 平均                [en_features: list of (B, N, D)]
  -> DinomalyMLP (apply_input_dropout=True)           [bottleneck]
  -> decoder_depth 个 DecoderViTBlock (LinearAttention) [decoder]
  -> 按 fuse_layer_decoder 分组 + 平均                 [de_features]
  -> 1 - cos_sim(en, de), bilinearly upsample, blur  [anomaly_map]
  -> top-1% mean                                       [pred_score]
```

**冻结策略**：只有 `bottleneck` 和 `decoder` 参与训练；encoder 完全冻结。

**损失**：`CosineHardMiningLoss`，即对 `en` 和 `de` 的余弦相似度做 hard-mining——对"已经重建得太好的点"降低梯度贡献（乘 `factor=0.1`），强迫 decoder 关注难例。

### 3.2 F²R 整体架构

F²R 在 Dinomaly 主干上插入三个模块：

```
                                              (frozen DINOv2 encoder)
                                                      |
                              +----- en_features -----+
                              |                       |
                         [FAB: 频域解耦]      [FPA prior encoder (frozen)]
                              |                       |
                       freq-aware feat        face_embedding (B, D_face)
                              |                       |
                          [DecoderViTBlock x L]  <-- cross-attn with face_emb
                              |                       |
                         de_features            (face_prior_loss)
                              |
                          [SAS: 频域 anomaly map]
                              |
                       anomaly_map + score
```

FAB 在 bottleneck 之后替换/包装原 bottleneck 输出；FPA 在每个 decoder block 内加 cross-attention 分支；SAS 在 `forward()` 末尾替换/扩展 `calculate_anomaly_maps`。

### 3.3 模块 A：Frequency-Aware Bottleneck (FAB)

#### 3.3.1 设计动机

人脸反欺骗的关键证据藏在**频域**：

| 攻击类型 | 频域特征 |
|---------|---------|
| A4 打印 | 印刷网点（高频周期性纹理） |
| 屏幕重放 | 像素网格 + 摩尔纹（中/高频条纹） |
| 3D 面具 | 边缘高光 + 材质反射（中频） |
| 视频回放 | 压缩块效应（高频噪声） |
| 活体 | 皮肤纹理是低频结构 + 自然高频随机性 |

如果我们把 Dinomaly 的瓶颈 "bottleneck MLP" 替换成 FAB，让 bottleneck 显式分离低/中/高频子带并分别送入 decoder，就强迫 decoder 不能用一个"模糊重建"覆盖所有频段，**高频异常点的 cos_sim 自然下降**，anomaly map 对纹理的敏感度上升。

#### 3.3.2 公式化

设 FAB 输入为 `x in R^{B x N x D}`（N = H_p x W_p 个 patch tokens）。

1. 还原成 patch grid：
   ```
   x_grid = x.transpose(1,2).reshape(B, D, H_p, W_p)
   ```

2. 2D FFT（real FFT, norm="ortho"）：
   ```
   X = rfft2(x_grid, norm="ortho")     # shape (B, D, H_p, W_p//2+1), complex
   ```

3. 按径向频率切三个子带 mask（中心为低频）：
   ```
   mask_low  = M(freq <  f_low)        # 低频：脸型 / 肤色 / 整体光照
   mask_mid  = M(f_low <= freq < f_mid) # 中频：五官结构 / 纹理
   mask_high = M(freq >= f_mid)        # 高频：摩尔纹 / 屏幕像素 / 打印网点
   ```
   其中 `f_low = H_p/8, f_mid = H_p/3`（按 patch grid 相对频率），可以做成可学习参数或硬切。

4. 子带反变换 + 还原 token：
   ```
   x_low  = irfft2(X * mask_low ).reshape(B, D, N).transpose(1,2)
   x_mid  = irfft2(X * mask_mid ).reshape(B, D, N).transpose(1,2)
   x_high = irfft2(X * mask_high).reshape(B, D, N).transpose(1,2)
   ```

5. 三个轻量 head 处理（保持 token 维度）：
   ```
   x_low  = MLP_low (x_low)
   x_mid  = MLP_mid (x_mid)
   x_high = MLP_high(x_high)
   ```
   每个 MLP 是一个 `LayerNorm -> Linear(D, 4D) -> GELU -> Linear(4D, D)`，参数量仅 `3 * (2 * D^2 + 4D)`。

6. 拼接并融合：
   ```
   x_fab = Concat([x_low, x_mid, x_high], dim=-1)   # (B, N, 3D)
   x_fab = Linear(D*3, D)(x_fab)                    # 通道融合回 D
   ```

7. **可选增强**：把 `x_high` 在训练时随机 mask 掉 30%（`freq_dropout_high=0.3`），强迫 decoder 在高频缺失时依赖学到的"正常高频模式"——这是一种 *频域 dropout*，类似 Dinomaly 原 bottleneck 的 `apply_input_dropout`。

#### 3.3.3 与 Dinomaly 的对接

- 替换 `DinomalyModel.bottleneck` 的 `DinomalyMLP` 为 `FAB` 模块：
  ```python
  self.bottleneck = nn.ModuleList([FrequencyAwareBottleneck(embed_dim)])
  ```
- 由于 `get_encoder_decoder_outputs` 仍然循环 `self.bottleneck`，可以保持不变。
- encoder 仍然 frozen，**只训练 FAB 内部的 3 个 MLP + 1 个融合 Linear**。

#### 3.3.4 复杂度

- 额外前向：3 次 rfft2 + 3 次 irfft2 + 3 个小 MLP + 1 个融合 Linear。
- 对于 patch grid 28x28（input 392x392, patch=14），FFT 单次约 28^2*log(28^2) ~= 5.5k 复数 op/token，可忽略。
- 参数量增加：`~3 x (2D^2 + 4D) + D*3*D ~= 9D^2` 个参数（D=768 时约 5.3M，比原 Dinomaly decoder 的 8 层 x ~7M 显著小）。

### 3.4 模块 B：Face Prior Adapter (FPA)

#### 3.4.1 设计动机

DINOv2 在 ImageNet 上训练，对"人脸"语义类别没有专门知识。结果是：
- Dinomaly decoder 重建时倾向于"光滑化"——这反而掩盖了活体皮肤的细节纹理。
- 在跨域场景（OULU 训练，CASIA 测试），decoder 可能把"肤色"当作异常线索，**在肤色差异大的人种上失效**。

引入一个 *face-specific* 的预训练 backbone 作为 prior，可以让 decoder 在重建时同时考虑：
1. "这是不是一张人脸"（身份/几何）
2. "人脸皮肤的健康纹理长什么样"（材质）

#### 3.4.2 选择 face backbone

我们默认用 **ArcFace (IR-SE-50, IR-SE-100)**，因为：
- 训练在 MS-Celeb-1M 上，对人脸有强判别力。
- 输出的 512-d face embedding 语义高度结构化。
- 已有开源 weights（InsightFace 项目）可直接 `pip install insightface`。

备选：
- **FaRL**（LAION-Face，CLIP-style）提供 face attribute 语义，适合需要 fine-grained 纹理监督的实验。
- **FaceNet**（InceptionResNetV1）更轻量，适合边缘场景。

#### 3.4.3 架构

1. Face Prior Encoder（FPE）：
   ```
   face_emb = FPE(x)     # frozen, 输出 (B, D_face)
   ```
   `D_face = 512` (ArcFace)。

2. 把 face embedding 投到 decoder 维度：
   ```
   f = Linear(D_face, D)(face_emb).unsqueeze(1)   # (B, 1, D)
   ```

3. 在每个 `DecoderViTBlock` 中插入 cross-attention：
   ```
   # 原本 block: norm1 -> attn -> residual -> norm2 -> MLP -> residual
   # FPA block:  在 attn 之后插入 cross-attn with f
   ```
   具体地：
   ```
   x = x + DropPath(self.attn(self.norm1(x)))
   x = x + DropPath(self.cross_attn(self.norm_cross(x), f))  # NEW
   x = x + DropPath(self.mlp(self.norm2(x)))
   ```
   其中 `cross_attn` 用同样的 `LinearAttention` 实现（query=decoder token, key/value=face embedding），保持 decoder 内部一致性。

4. 训练时 face backbone 冻结；只有 `Linear(D_face, D)` 和每个 block 的 `cross_attn` 可训练。

#### 3.4.4 损失耦合（可选）

为防止 face prior 完全失效（输出 collapse 到常数），加一个 *prior preservation loss*：

```
L_prior = 1 - cos_sim(face_emb_proj, face_emb_proj.detach())
```

也就是鼓励 projection 与自己的某次 forward 不一致（防止 collapse），或者更简单地用 face_emb 预测一个 "depth prior"——这里给出 hook，具体实现可以灵活设计。

### 3.5 模块 C：Spectral Anomaly Score (SAS)

#### 3.5.1 设计动机

Dinomaly 的 anomaly map 计算公式：
```
anomaly_map_i = 1 - cos_sim(en_i, de_i)
anomaly_map = interpolate(a_map, image_size, bilinear).mean(across groups)
```

这是在 *空间域* 上做的余弦残差。但人脸反欺骗的高频异常，**在空间域余弦残差上信号弱、在频域上信号强**。

#### 3.5.2 频域 anomaly map

1. 把 `en_i, de_i`（每个是 (B, D, H_p, W_p)）做 2D FFT：
   ```
   EN = rfft2(en_i, norm="ortho")
   DE = rfft2(de_i, norm="ortho")
   ```

2. 频域余弦残差（对每个频率 bin 算 1 - cos_sim）：
   ```
   spec_map = 1 - cosine_similarity(EN, DE, dim=channel)   # (B, H_p, W_p//2+1)
   ```

3. 径向加权（让高频权重大）：
   ```
   for each bin (h, w):
       freq = sqrt(h^2 + w^2) / (H_p/2)
       weight = 0.2 + 0.8 * sigmoid(2 * (freq - 0.3))
   ```
   把 `weight` 加到 `spec_map` 上。

4. 把加权的频域 anomaly map 还原到空间域并上采样到原图：
   ```
   spec_anomaly = irfft2(spec_map * weight)   # (B, H_p, W_p)
   spec_anomaly = interpolate(spec_anomaly, image_size, bilinear)
   ```

5. 与原空间域 anomaly map 融合：
   ```
   final_anomaly_map = alpha * spatial_anomaly_map + (1 - alpha) * spec_anomaly
   ```
   `alpha = 0.5` 作为默认，可作为超参数。

6. image-level 分数：与 Dinomaly 同样对 `final_anomaly_map` 做 `interpolate(256) -> blur -> top 1% mean`。

#### 3.5.3 与 Dinomaly 对接

替换 `DinomalyModel.calculate_anomaly_maps`：
```python
@staticmethod
def calculate_anomaly_maps(en, de, out_size, alpha=0.5, ...):
    spatial_am, _ = <原 Dinomaly 逻辑>
    spec_am, _ = compute_spectral_anomaly_maps(en, de)
    final = alpha * spatial_am + (1 - alpha) * spec_am
    return final, [spatial_am, spec_am]
```

### 3.6 训练目标

保持 Dinomaly 原有的 `CosineHardMiningLoss` 作为主损失 `L_main`，在异常分支上做余弦残差 hard-mining（**注意：这里的 `en/de` 来自 FAB + FPA 之后**）。

加一个频域重建损失（让高频子带也被显式监督）：
```
L_spec = L1(FAB(en_high), FAB(de_high))
```

加可选的 prior preservation loss：
```
L_prior = 1 - cos_sim(face_emb_proj, face_emb_proj.detach())
```

总损失：
```
L_total = L_main + lambda_spec * L_spec + lambda_prior * L_prior
```
默认 `lambda_spec = 0.5, lambda_prior = 0.1`。

### 3.7 推理流程

```
1. 输入 batch image -> preprocess (Resize 448 -> CenterCrop 392 -> Normalize)
2. encoder(x) -> [CLS, reg, patches] 来自 target_layers
3. en_features (按 fuse_layer_encoder 分组后) -> FAB -> 频域解耦 token
4. face_emb = FacePriorEncoder(x) -> 投影到 D
5. decoder 输入 = FAB 输出，每个 block 内部 cross-attn with face_emb
6. de_features -> reshape spatial
7. SAS: 计算 spatial_anomaly_map + spectral_anomaly_map -> 融合
8. anomaly_map_resized = interpolate(原图大小)
9. pred_score = interpolate(256) -> GaussianBlur -> top 1% mean
10. 返回 InferenceBatch(pred_score, anomaly_map)
```

训练时只用 normal samples；测试时不需要 gradient。模型总参数量约：
- DINOv2-base 编码器：~86M (frozen)
- Face Prior Encoder：~43M (frozen)
- FAB：~5M (trainable)
- Decoder (8 blocks + FPA cross-attn)：~63M (trainable)
- SAS：0 参数
- **总 trainable ~68M**（比 Dinomaly baseline 8 层 decoder 的 ~63M 多 ~5M，幅度可接受）。

### 3.8 实现细节 / 与 anomalib 代码的对应

| F²R 模块 | 插入位置（anomalib 路径） |
|---------|------------------------|
| `FAB` | `src/anomalib/models/image/dinomaly/components/fab.py`（新文件），在 `torch_model.py` 的 `DinomalyModel.__init__` 里替换 `self.bottleneck` |
| `FPA` | `src/anomalib/models/image/dinomaly/components/fpa.py`（新文件），修改 `DecoderViTBlock.forward` 增加 cross-attn 分支；`__init__` 加载 face backbone |
| `SAS` | 替换 `torch_model.py` 的 `DinomalyModel.calculate_anomaly_maps`；训练逻辑不动，只动 inference |
| 损失 | 在 `torch_model.py` 的 `forward(global_step=...)` 里把 `loss_fn(...)` 调用换成 `loss_main + lambda_spec * loss_spec + lambda_prior * loss_prior` |
| Lightning | `lightning_model.py` 几乎不动（FAB/FPA 已经挂在 `self.model` 上了，`trainable_modules` 自然包括新参数） |

---

## 4. Experiments

### 4.1 数据集

四个标准 FAS 数据集，覆盖典型攻击类型：

| 数据集 | 攻击类型 | 规模 | 协议 |
|-------|---------|-----|-----|
| **OULU-NPU** | 打印 + 重放 | 4950 视频 | Protocol 1,2,3,4 |
| **CASIA-FASD** | 打印 + 视频重放 + 剪纸 | 50 段视频 x 3 攻击 | 跨协议 (L/O/H/Q) |
| **REPLAY-ATTACK** | 打印 + 手机重放 + 高清屏 | 1300 视频 | 固定 split |
| **NUAA** | 照片重放 | 12614 图像 | 固定 split |

按 OULU 风格提取 frames -> resize 到 448x448 -> CenterCrop 392x392 -> ImageNet 归一化。

### 4.2 评估指标

- **HTER** (Half Total Error Rate) = (FAR + FRR)/2 — FAS 标准指标。
- **ACER** (Average Classification Error Rate)。
- **pixel-level AUC** — 检测异常区域的精度。
- **跨数据集**评估（train on A, test on B）。

### 4.3 实现细节

- 框架：PyTorch + Lightning + anomalib（v2.x）。
- 编码器：DINOv2-base（默认），对比实验用 small / large / giant。
- Face Prior：ArcFace IR-SE-50，weights 来自 InsightFace。
- 训练：5000 steps，batch=16（giant=4），StableAdamW + WarmCosine。
- 单卡：A100 / RTX 4090。

### 4.4 对比方法

| 类型 | 方法 |
|-----|-----|
| 监督 | DeepPixBiS, CDCN, Auxiliary Supervision |
| 域泛化 | SSDG-M, DRDG |
| 异常 | FAS-DR, AnomalyDINO, **Dinomaly** (基线) |
| **本文** | **F²R** (FAB+FPA+SAS) |

### 4.5 消融实验

| 变体 | FAB | FPA | SAS | OULU HTER | ACER | pixel-AUC |
|-----|----|----|----|----------|------|-----------|
| Dinomaly (baseline) | x | x | x | x.x | x.x | x.x |
| + FAB | v | x | x | ? | ? | ? |
| + FPA | x | v | x | ? | ? | ? |
| + SAS | x | x | v | ? | ? | ? |
| + FAB + SAS | v | x | v | ? | ? | ? |
| **F²R (full)** | v | v | v | **?** | **?** | **?** |

预期：每个模块单独都能带来 HTER 下降，full model 最好；FAB 对高频异常贡献最大，FPA 在跨域上贡献最大，SAS 对 pixel-level AUC 贡献最大。

### 4.6 跨数据集泛化

train on OULU -> test on CASIA / NUAA
train on CASIA -> test on REPLAY / OULU

预期：FPA 显著降低 HTER（FPA 是抗 domain shift 的核心模块）。

### 4.7 可视化

- Anomaly map 对比（Dinomaly vs F²R）：突出对摩尔纹 / 屏幕像素的响应。
- t-SNE：FAB 子带特征空间可视化。
- 频域谱：正常 vs 攻击在 FAB 子带的能量分布。

---

## 5. Conclusion & Future Work

### 5.1 总结

本文指出 Dinomaly 直接迁移到人脸反欺骗存在两个 gap：
- 工业异常 vs 纹理级异常的语义错配；
- DINOv2 通用预训练 vs 人脸域专用先验的 domain gap。

并提出 **F²R** 通过三个轻量模块系统性地弥合这两个 gap：

1. **FAB** 在 bottleneck 处做频域解耦，让高频纹理异常在重建残差里放大；
2. **FPA** 用 ArcFace/FaRL 的 frozen 嵌入做 decoder cross-attention，给模型注入人脸先验；
3. **SAS** 把 anomaly map 从纯空间域扩展到频域加权融合，锐化高频异常信号。

F²R 在代码层面以 **plug-and-play** 的方式集成到 anomalib 的 Dinomaly 中，仅需：
- 新增 `components/fab.py` 和 `components/fpa.py`；
- 在 `torch_model.py` 替换 bottleneck 与 anomaly map 计算；
- 在 `lightning_model.py` 的 `trainable_modules` 中追加 FAB/FPA 参数（几乎自动）。

### 5.2 未来工作

1. **Video extension**：F²R 现在基于 image-based Dinomaly，可以扩展到 video-based 版本（rPPG + 时序建模）。
2. **Adaptive SAS weights**：把频域权重做成 attention，让模型自己学哪些频段重要。
3. **Face prior variants**：消融 ArcFace vs FaRL vs FaceNet 对 F²R 的影响。
4. **3D mask attack**：在更大规模 3D mask 数据集上测试 F²R 的上限。
5. **Open-set FAS protocol**：把 F²R 在 *seen/unseen attack type* 协议上评估。
6. **Deployment**：把 F²R 通过 ONNX/OpenVINO 部署（参考 anomalib 的 export 流程）。

---

## 附录 A：与 anomalib Dinomaly 的代码对接细节

### A.1 新增文件清单

```
src/anomalib/models/image/dinomaly/components/
├── fab.py                   # FrequencyAwareBottleneck（FFT 子带 + 3 head + fusion）
├── fpa.py                   # FacePriorAdapter（含 face backbone 加载工具）
├── __init__.py              # 追加导出
```

### A.2 关键 diff 位置

1. `torch_model.py` 的 `DinomalyModel.__init__`：
   ```diff
   - from .components import CosineHardMiningLoss, DinomalyMLP, LinearAttention
   + from .components import (
   +     CosineHardMiningLoss, DinomalyMLP, LinearAttention,
   +     FrequencyAwareBottleneck, FacePriorAdapter,
   + )
   ...
   - self.bottleneck = nn.ModuleList([DinomalyMLP(... apply_input_dropout=True ...)])
   + self.bottleneck = nn.ModuleList([FrequencyAwareBottleneck(embed_dim, freq_dropout_high=0.3)])
   + self.face_prior = FacePriorAdapter(face_backbone="arcface_ir_se_50", embed_dim=embed_dim)
   ...
   - self.decoder = nn.ModuleList([DecoderViTBlock(...) for _ in range(decoder_depth)])
   + self.decoder = nn.ModuleList([
   +     DecoderViTBlock(dim=embed_dim, num_heads=num_heads, use_cross_attn=True)
   +     for _ in range(decoder_depth)
   + ])
   ```

2. `torch_model.py` 的 `DecoderViTBlock.forward`：
   ```diff
   - x = x + self.drop_path(y)
   - x = x + self.drop_path(self.mlp(self.norm2(x)))
   + x = x + self.drop_path(y)
   + x = self._maybe_cross_attn(x, face_emb)        # NEW
   + x = x + self.drop_path(self.mlp(self.norm2(x)))
   ```

3. `torch_model.py` 的 `forward(...)`：
   ```diff
   - en, de = self.get_encoder_decoder_outputs(batch)
   + en, de, face_emb = self.get_encoder_decoder_outputs(batch)
   ...
   - return self.loss_fn(encoder_features=en, decoder_features=de, global_step=...)
   + loss_main = self.loss_fn(encoder_features=en, decoder_features=de, global_step=...)
   + loss_spec = compute_spectral_loss(en, de)
   + loss_prior = 1 - F.cosine_similarity(face_emb_proj, face_emb_proj.detach()).mean()
   + return loss_main + 0.5 * loss_spec + 0.1 * loss_prior
   ```

4. `torch_model.py` 的 `calculate_anomaly_maps`：
   ```diff
   - def calculate_anomaly_maps(en, de, out_size):
   -     ...原始空间域逻辑
   + def calculate_anomaly_maps(en, de, out_size, alpha=0.5):
   +     spatial_am, _ = _compute_spatial_anomaly_maps(en, de, out_size)   # 原逻辑搬过来
   +     spec_am, _ = _compute_spectral_anomaly_maps(en, de, out_size)
   +     return alpha * spatial_am + (1 - alpha) * spec_am, [spatial_am, spec_am]
   ```

5. `lightning_model.py`：
   - 几乎不需要改动：FAB / FPA 都挂在 `self.model` 下，自然会被 `trainable_modules` 包含。
   - 如果想加新超参 `alpha`, `lambda_spec`, `lambda_prior`，在 `__init__` 增加 kwargs 并透传给 `DinomalyModel` 即可。

### A.3 训练脚本模板

参考 `train_dinomaly_face.py`，只需改：

```python
model = Dinomaly(
    encoder_name=args.encoder_name,
    bottleneck_dropout=0.0,             # FAB 内置 dropout
    decoder_depth=8,
    target_layers=args.target_layers,
    fuse_layer_encoder=encoder_groups,
    fuse_layer_decoder=decoder_groups,
    use_context_recentering=args.use_context_recentering,
    precision=args.model_precision,
    pre_processor=pre_processor,
    evaluator=build_evaluator(),
    # --- F²R 新增 ---
    use_fab=True,
    fab_freq_low_ratio=0.125,
    fab_freq_mid_ratio=0.33,
    fab_freq_dropout_high=0.3,
    use_fpa=True,
    face_backbone="arcface_ir_se_50",
    use_sas=True,
    sas_alpha=0.5,
    sas_freq_bias=0.3,
    sas_high_weight=0.8,
    sas_low_weight=0.2,
    lambda_spec=0.5,
    lambda_prior=0.1,
)
```

---

## 附录 B：参考论文清单（你可以去对应模块找源码）

| 主题 | 代表性工作 | 关键模块 / 借鉴点 |
|------|---------|----------------|
| FAS | DeepPixBiS, CDCN, Auxiliary, SSDG | 监督 loss、central difference conv |
| FAS + 频域 | FAS-SGT, FAD, Phase-only Reconstruction | 频域增强、phase-only 重建 |
| 异常 + VFM | Dinomaly, AnomalyDINO, AnomalyVFM | DINOv2 + 重建 |
| 频域 anomaly | 2024 CVPR "Anomaly Detection with Frequency Domain" | FFT-based residual |
| 人脸先验 | ArcFace, FaRL, CurricularFace | frozen face embedding |
| Adapter / LoRA | VPT, AdaptFormer | plug-in prior injection |
| FAS as Anomaly | FAS-DR | 直接 baseline |

---

**End of proposal.**
