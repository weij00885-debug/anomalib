# LCF＋CLTC＋C（SPR）实验总记录与问答回溯

记录日期：2026-09-25。仓库：`C:\Users\heqi\Desktop\wj\anomalib`。

本文汇总截至本日已完成的四库实验、全部六项指标、配置、运行命令、数据来源及关键问答。表中数值来自已有结果文件，显示保留六位小数；文末附带数值快照和来源哈希以便回溯。后续有新实验时应追加日期和独立运行目录，不覆盖本文已记录的结果。

## 1. 当前状态与结论

- 正式方案为 **LCF＋CLTC＋完整 C（SPR）**。C 使用同一区域的多视图中位数和 MAD 稳定性惩罚，beta=0.25。
- 已完成：四库完整 C；四视图普通 TTA；同 beta 与同校准方式的融合 TTA；去掉惩罚的中位数消融；去掉惩罚的均值消融。
- 当前四组 checkpoint 上，完整 C 的 AUROC 和 HTER 均优于原 LCF＋CLTC、普通 TTA、同比例融合 TTA。
- 相比同比例融合 TTA，完整 C 在四库的六项指标都更好。
- 无惩罚中位数在四库的 AUROC 和 HTER 都优于无惩罚均值。MAD 惩罚只有很小的 HTER 增益，AUROC 两库略升、两库略降，不能称作主要或显著贡献。
- 尚未完成：固定同一区域与各视图独立选区域的对照、固定划分下的多训练种子实验、置信区间或显著性检验。
- VEC/CoT 尚未加入本轮正式模型。SLR、RCNC 不属于本轮新增内容。

本文中“原实验”特指保存的 **LCF＋CLTC**，并非纯 LCF。纯 LCF 的历史成绩未在本文核验，不能用本文直接量化 CLTC 相对纯 LCF 的增益。

## 2. 方法名称与评分公式

| 本文名称 | 输出列名 | 证据如何产生 |
|---|---|---|
| 原 LCF＋CLTC | original_lcf_cltc | 原图的重建与轨迹融合分数 S0 |
| 普通 TTA | tta_mean_4views | 原图、翻转、gamma0.9、gamma1.1 四个整图融合分数取平均 |
| 同比例融合 TTA | tta_blend_4views | 75% S0＋25% 经验证集真人校准的普通 TTA |
| 同区域均值，无惩罚 | mean_no_penalty | 固定原图 ROI，四个区域分数取平均，再与 S0 融合 |
| 同区域中位数，无惩罚 | median_no_penalty | 固定原图 ROI，四个区域分数取中位数，再与 S0 融合 |
| 完整 C | lcf_cltc_spr | 固定原图 ROI，中位数经 MAD 惩罚后，再与 S0 融合 |

### 原模型与 detach

原模型的重建分支沿用 LCF；CLTC 读取 detach 后的 LCF 特征，学习预测冻结 DINO 原始层特征的相邻层变化。此处“轨迹”指跨网络层的特征变化，不是视频时间轨迹。

```text
训练：L = 重建损失 + lambda × 轨迹预测损失
推理：S0 = 重建分数 / q_rec + alpha × 轨迹分数 / q_traj
```

CLTC 损失更新 CLTC head，不通过该分支回传到 LCF。LCF 仍通过原重建分支训练。detach 不会阻止 CLTC 在推理时提供新分数。

### 完整 C

对经过原数据预处理缩放、尚未标准化的 RGB 图像，依次使用原图、水平翻转、gamma=0.9、gamma=1.1。水平翻转产生的 patch 图翻回原坐标。

```text
M_j = 非负重建 patch 图 / q_rec + alpha × 非负轨迹 patch 图 / q_traj
Omega = 原图 M_0 最异常的约 10% patch（边界并列值分配权重）
v_j = 第 j 个视图在相同 Omega 内的加权平均分数
m = median(v_0, v_1, v_2, v_3)
d = median(abs(v_j - m))
e = m / (1 + d / (abs(m) + 1e-8))

q0 = 验证集真人 S0 的 95% 分位数
qe = 验证集真人 e 的 95% 分位数
S_C = 0.75 × S0 + 0.25 × q0 × e / qe
```

偶数中位数取中间两项的平均。C 无可学习参数，不进行新的模型训练。正式 C 的中位数消融和均值消融分别把 e 换成 m 与 mean(v_j)，各自重新拟合验证集尺度、阈值。

融合 TTA 使用相同形式，把 e 换成四个整图分数的平均，并用验证集真人的 TTA q95 校准。beta 沿用 C 的 0.25，没有根据测试集重新搜索。

## 3. 正式完整 C 的六项指标

所有数值为 0–1 小数，乘 100 即百分数。HTER 越低越好，其余五项越高越好。

| 数据集 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OULU-NPU | 0.883645 | 0.895417 | 0.937328 | 0.900240 | 0.977604 | 0.227865 |
| LCC | 0.845643 | 0.837862 | 0.909562 | 0.987229 | 0.843224 | 0.237847 |
| NUAA | 0.988455 | 0.950033 | 0.971488 | 0.994184 | 0.949806 | 0.049097 |
| Replay-Attack | 0.999064 | 0.995527 | 0.997580 | 0.998028 | 0.997134 | 0.013548 |

### 原 LCF＋CLTC 的六项指标

| 数据集 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OULU-NPU | 0.874795 | 0.890278 | 0.934678 | 0.892327 | 0.981250 | 0.246181 |
| LCC | 0.837748 | 0.829996 | 0.904871 | 0.985833 | 0.836199 | 0.257576 |
| NUAA | 0.984028 | 0.943725 | 0.967828 | 0.992409 | 0.944434 | 0.058983 |
| Replay-Attack | 0.998826 | 0.994533 | 0.997042 | 0.997668 | 0.996417 | 0.016109 |

### 完整 C 相对原 LCF＋CLTC 的变化（百分点）

计算为 100 × (C − 原结果)，不是相对增长率。

| 数据集 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OULU-NPU | +0.884983 | +0.513889 | +0.264999 | +0.791268 | -0.364583 | -1.831597 |
| LCC | +0.789475 | +0.786557 | +0.469081 | +0.139683 | +0.702533 | -1.972888 |
| NUAA | +0.442749 | +0.630810 | +0.366046 | +0.177429 | +0.537137 | -0.988568 |
| Replay-Attack | +0.023834 | +0.099404 | +0.053821 | +0.035945 | +0.071659 | -0.256094 |

## 4. 每个数据集全部方案的六项指标

以下表格把已经完成的对照与消融放在一起。普通 TTA 是整图分数平均，“同区域均值”是固定 ROI 内区域分数的平均，两者不能混淆。

### OULU-NPU

| 方案 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LCF＋CLTC | 0.874795 | 0.890278 | 0.934678 | 0.892327 | 0.981250 | 0.246181 |
| 普通 TTA | 0.877415 | 0.892639 | 0.935952 | 0.895229 | 0.980556 | 0.239236 |
| 同比例融合 TTA | 0.875670 | 0.891111 | 0.934634 | 0.899102 | 0.973090 | 0.231858 |
| 同区域均值，无惩罚 | 0.883317 | 0.895278 | 0.937386 | 0.898440 | 0.979861 | 0.231597 |
| 同区域中位数，无惩罚 | 0.883866 | 0.895556 | 0.937427 | 0.900128 | 0.977951 | 0.228038 |
| 完整 C | 0.883645 | 0.895417 | 0.937328 | 0.900240 | 0.977604 | 0.227865 |

### LCC

| 方案 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LCF＋CLTC | 0.837748 | 0.829996 | 0.904871 | 0.985833 | 0.836199 | 0.257576 |
| 普通 TTA | 0.834283 | 0.876832 | 0.933061 | 0.983210 | 0.887780 | 0.277732 |
| 同比例融合 TTA | 0.837181 | 0.833750 | 0.907167 | 0.985897 | 0.840081 | 0.255635 |
| 同区域均值，无惩罚 | 0.844862 | 0.828924 | 0.904099 | 0.987090 | 0.833980 | 0.242469 |
| 同区域中位数，无惩罚 | 0.846002 | 0.836968 | 0.909018 | 0.987216 | 0.842300 | 0.238310 |
| 完整 C | 0.845643 | 0.837862 | 0.909562 | 0.987229 | 0.843224 | 0.237847 |

### NUAA

| 方案 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LCF＋CLTC | 0.984028 | 0.943725 | 0.967828 | 0.992409 | 0.944434 | 0.058983 |
| 普通 TTA | 0.969576 | 0.938579 | 0.964982 | 0.986646 | 0.944249 | 0.083076 |
| 同比例融合 TTA | 0.981589 | 0.943559 | 0.967754 | 0.991642 | 0.944990 | 0.061905 |
| 同区域均值，无惩罚 | 0.985612 | 0.943559 | 0.967717 | 0.992792 | 0.943878 | 0.057661 |
| 同区域中位数，无惩罚 | 0.988429 | 0.950199 | 0.971591 | 0.993993 | 0.950176 | 0.049712 |
| 完整 C | 0.988455 | 0.950033 | 0.971488 | 0.994184 | 0.949806 | 0.049097 |

### Replay-Attack

| 方案 | AUROC ↑ | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ | HTER ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 LCF＋CLTC | 0.998826 | 0.994533 | 0.997042 | 0.997668 | 0.996417 | 0.016109 |
| 普通 TTA | 0.995880 | 0.985255 | 0.992001 | 0.995311 | 0.988714 | 0.034277 |
| 同比例融合 TTA | 0.998566 | 0.992213 | 0.995781 | 0.997841 | 0.993730 | 0.016351 |
| 同区域均值，无惩罚 | 0.998835 | 0.993539 | 0.996501 | 0.998023 | 0.994984 | 0.014623 |
| 同区域中位数，无惩罚 | 0.999062 | 0.995361 | 0.997491 | 0.998027 | 0.996954 | 0.013637 |
| 完整 C | 0.999064 | 0.995527 | 0.997580 | 0.998028 | 0.997134 | 0.013548 |

## 5. 对照和消融应如何解释

### 完整 C 相对两种 TTA（百分点）

| 数据集 | C−普通TTA AUROC | C−普通TTA HTER | C−融合TTA AUROC | C−融合TTA HTER |
| --- | ---: | ---: | ---: | ---: |
| OULU-NPU | +0.623059 | -1.137153 | +0.797490 | -0.399306 |
| LCC | +1.135945 | -3.988448 | +0.846137 | -1.778767 |
| NUAA | +1.887876 | -3.397829 | +0.686646 | -1.280785 |
| Replay-Attack | +0.318361 | -2.072978 | +0.049798 | -0.280322 |

相同四视图与同融合比例下 C 仍有收益，因此当前结果不能仅由“增加视图”或“保留 75% 原分数”解释。此结论针对当前保存的 checkpoint；尚未证明跨训练种子稳定或统计显著。

### 中位数和 MAD 惩罚各自的增量（百分点）

| 数据集 | 中位数−均值 AUROC | 中位数−均值 HTER | 完整C−无惩罚中位数 AUROC | 完整C−无惩罚中位数 HTER |
| --- | ---: | ---: | ---: | ---: |
| OULU-NPU | +0.054868 | -0.355903 | -0.022051 | -0.017361 |
| LCC | +0.114024 | -0.415973 | -0.035976 | -0.046219 |
| NUAA | +0.281682 | -0.794873 | +0.002578 | -0.061478 |
| Replay-Attack | +0.022689 | -0.098531 | +0.000197 | -0.008957 |

- 比较“无惩罚中位数 vs 无惩罚均值”才是在这里观察聚合方式差异。
- 比较“完整 C vs 无惩罚中位数”是在这里观察 MAD 惩罚的差异。
- MAD 惩罚的 HTER 改善仅约 0.009–0.061 个百分点；LCC、OULU 的 AUROC 还略降。汇报中不能写成“大幅、显著、全面提升”。
- 相比普通 TTA，LCC 的 Accuracy/F1/Recall 下降，OULU 的 Recall 下降；不能只凭 AUROC、HTER 更好而说所有指标都赢。
- 相比同比例融合 TTA，当前四库的六项指标均改善。
- 此轮没有自动将正式方案切换成某个消融版本。不要按测试集成绩为不同数据集各挑一个版本。

## 6. 配置、数据量与协议

| 数据集 | 原训练步数 | 训练 batch | 保存精度 | CLTC alpha | 验证图数 | 测试图数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OULU-NPU | 5000 | 1 | float16 | 1 | 1800 | 7200 |
| LCC | 5000 | 2 | float32 | 0.5 | 1398 | 5594 |
| NUAA | 20000 | 1 | float32 | 1 | 1505 | 6024 |
| Replay-Attack | 20000 | 1 | float32 | 1 | 1508 | 6036 |

共同配置：seed=42；CLTC detach=true；lambda=0.1；输入 392×392；CLTC hidden_dim=128；C beta=0.25；ROI 比例=0.1；评估 batch=1；四个顺序视图。C 与全部缓存对照都没有新增训练。

OULU 的旧配置字符串 `float16` 实际沿用 BF16 主干＋FP32 head。LCC 使用 `vit_large_patch14_reg4_dinov2`，另外三库使用 `vit_giant_patch14_reg4_dinov2`。

路径：

- OULU：`/mnt/d/BaiduNetdiskDownload/OULU-NPU_organized`。
- 其他三库的父目录：`/mnt/c/Users/heqi/Desktop/wj/data`，分别为 `LCC_colors/rgb`、`NUAA_mtcnn_colors/rgb`、`REPLAY_mtcnn_colors/rgb`。

协议与阈值：

| 数据集 | 当前实验协议 | 验证阈值规则 |
|---|---|---|
| OULU-NPU | `legacy_lcf_from_test_20pct_frame_level_folder` | 验证集最大 F1（沿用旧实现的边界处理） |
| LCC | `legacy_normal_holdout_from_test_val` | 验证集最小 HTER（沿用旧实现的并列处理） |
| NUAA | `legacy_dedicated_normal_test_from_test_val` | 验证集最小 HTER（沿用旧实现的并列处理） |
| Replay-Attack | `legacy_normal_holdout_from_test_val` | 验证集最小 HTER（沿用旧实现的并列处理） |

所有分类判断沿用严格的 `score > threshold`。AUROC 不需要分类阈值。分数尺度只用验证集真人拟合，分类阈值用完整有标签验证集拟合，再固定到测试集。

OULU 当前沿用旧脚本从测试目录划出 20% 验证帧的协议（验证 1800、最终测试 7200）。历史讨论中曾有独立 dev 版本；本文的运行目录和 manifest 才是当前对照依据，不能将独立 dev 的旧结果混进来。NUAA 沿用独立真人 test 来源；LCC/Replay 沿用从 normal 留出真人的旧规则。新 C/TTA/消融脚本直接复用保存清单，不重新划分。

全部是旧帧级协议。OULU 成绩不是官方 Protocol 1–4 的视频级成绩。路径清单对齐不能单独证明视频/身份层面的独立性。

### 完整 C 的校准值

| 数据集 | q0 | qe | C 分类阈值 |
| --- | ---: | ---: | ---: |
| OULU-NPU | 1.949601 | 1.572682 | 1.624783 |
| LCC | 1.517523 | 1.161872 | 1.346885 |
| NUAA | 1.970380 | 1.567851 | 1.976022 |
| Replay-Attack | 1.940275 | 1.594614 | 2.041085 |

### 完整 C 的实际推理工作量与时间

| 数据集 | 验证单图前向次数 | 测试单图前向次数 | 验证耗时（分） | 测试耗时（分） |
| --- | ---: | ---: | ---: | ---: |
| OULU-NPU | 7200 | 28800 | 9.23 | 37.04 |
| LCC | 5592 | 22376 | 7.13 | 28.94 |
| NUAA | 6020 | 24096 | 20.97 | 84.84 |
| Replay-Attack | 6032 | 24144 | 21.46 | 86.14 |

以上时间来自原 SPR 日志，包含 I/O，不能直接当作严格硬件测速。普通 TTA、融合 TTA、两项消融都由缓存统计，新增模型前向次数为 0；其所代表的图像方法均使用四视图预算。

## 7. 原始文件索引

以下路径均相对仓库根目录。每份完成的运行都有 `metrics.json`；C/TTA/消融还有 `comparison.csv`、`config.json`、`calibration.json` 和逐图分数。原 LCF＋CLTC 目录保存 checkpoint 与 train/validation/test manifest。

### OULU-NPU

- 原 LCF＋CLTC：`results/oulu_lcf_cltc_legacy_s42`；[metrics.json](../../results/oulu_lcf_cltc_legacy_s42/metrics.json)。
- 完整 C：`results/oulu_lcf_cltc_spr_20260925_010043_662042`；[metrics.json](../../results/oulu_lcf_cltc_spr_20260925_010043_662042/metrics.json)；[comparison.csv](../../results/oulu_lcf_cltc_spr_20260925_010043_662042/comparison.csv)。
- 普通/融合 TTA 对照：`results/oulu_spr_tta_20260925_131448_260792`；[metrics.json](../../results/oulu_spr_tta_20260925_131448_260792/metrics.json)；[comparison.csv](../../results/oulu_spr_tta_20260925_131448_260792/comparison.csv)。
- 两项消融：`results/oulu_spr_ablation_20260925_153517_111896`；[metrics.json](../../results/oulu_spr_ablation_20260925_153517_111896/metrics.json)；[comparison.csv](../../results/oulu_spr_ablation_20260925_153517_111896/comparison.csv)。

### LCC

- 原 LCF＋CLTC：`results/lcc_lcf_cltc_final_5k_b2_s42`；[metrics.json](../../results/lcc_lcf_cltc_final_5k_b2_s42/metrics.json)。
- 完整 C：`results/lcc_lcf_cltc_spr_20260925_092943_946978`；[metrics.json](../../results/lcc_lcf_cltc_spr_20260925_092943_946978/metrics.json)；[comparison.csv](../../results/lcc_lcf_cltc_spr_20260925_092943_946978/comparison.csv)。
- 普通/融合 TTA 对照：`results/lcc_spr_tta_20260925_131446_701614`；[metrics.json](../../results/lcc_spr_tta_20260925_131446_701614/metrics.json)；[comparison.csv](../../results/lcc_spr_tta_20260925_131446_701614/comparison.csv)。
- 两项消融：`results/lcc_spr_ablation_20260925_153515_587466`；[metrics.json](../../results/lcc_spr_ablation_20260925_153515_587466/metrics.json)；[comparison.csv](../../results/lcc_spr_ablation_20260925_153515_587466/comparison.csv)。

### NUAA

- 原 LCF＋CLTC：`results/nuaa_lcf_cltc_legacy_fp32_20k_s42`；[metrics.json](../../results/nuaa_lcf_cltc_legacy_fp32_20k_s42/metrics.json)。
- 完整 C：`results/nuaa_lcf_cltc_spr_20260924_164610_163966`；[metrics.json](../../results/nuaa_lcf_cltc_spr_20260924_164610_163966/metrics.json)；[comparison.csv](../../results/nuaa_lcf_cltc_spr_20260924_164610_163966/comparison.csv)。
- 普通/融合 TTA 对照：`results/nuaa_spr_tta_20260925_131449_592189`；[metrics.json](../../results/nuaa_spr_tta_20260925_131449_592189/metrics.json)；[comparison.csv](../../results/nuaa_spr_tta_20260925_131449_592189/comparison.csv)。
- 两项消融：`results/nuaa_spr_ablation_20260925_153518_742563`；[metrics.json](../../results/nuaa_spr_ablation_20260925_153518_742563/metrics.json)；[comparison.csv](../../results/nuaa_spr_ablation_20260925_153518_742563/comparison.csv)。

### Replay-Attack

- 原 LCF＋CLTC：`results/replay_lcf_cltc_legacy_fp32_20k_s42`；[metrics.json](../../results/replay_lcf_cltc_legacy_fp32_20k_s42/metrics.json)。
- 完整 C：`results/replay_lcf_cltc_spr_20260925_101212_401331`；[metrics.json](../../results/replay_lcf_cltc_spr_20260925_101212_401331/metrics.json)；[comparison.csv](../../results/replay_lcf_cltc_spr_20260925_101212_401331/comparison.csv)。
- 普通/融合 TTA 对照：`results/replay_spr_tta_20260925_131451_221673`；[metrics.json](../../results/replay_spr_tta_20260925_131451_221673/metrics.json)；[comparison.csv](../../results/replay_spr_tta_20260925_131451_221673/comparison.csv)。
- 两项消融：`results/replay_spr_ablation_20260925_153520_455646`；[metrics.json](../../results/replay_spr_ablation_20260925_153520_455646/metrics.json)；[comparison.csv](../../results/replay_spr_ablation_20260925_153520_455646/comparison.csv)。

## 8. 可复现的一行命令

在 WSL 仓库目录、原 CUDA 可用的 `torch_env` 中运行。环境管理遵循仓库的 uv 约定；以下 python 命令使用已激活的既有环境。默认输出新时间戳目录；显式指定已存在的结果目录会报错。

### 重新评估完整 C（需要图像推理）

OULU-NPU：

```bash
python LCF-CLTC-SPR/eval_oulu_lcf_cltc_spr.py --source-run results/oulu_lcf_cltc_legacy_s42 --source-max-steps 5000 --source-train-batch-size 1 --source-model-precision float16 --source-seed 42 --source-cltc-loss-weight 0.1 --source-cltc-score-weight 1 --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

LCC：

```bash
python LCF-CLTC-SPR/eval_lcc_lcf_cltc_spr.py --source-run results/lcc_lcf_cltc_final_5k_b2_s42 --source-max-steps 5000 --source-train-batch-size 2 --source-model-precision float32 --source-seed 42 --source-cltc-loss-weight 0.1 --source-cltc-score-weight 0.5 --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

NUAA：

```bash
python LCF-CLTC-SPR/eval_nuaa_lcf_cltc_spr.py --source-run results/nuaa_lcf_cltc_legacy_fp32_20k_s42 --source-max-steps 20000 --source-train-batch-size 1 --source-model-precision float32 --source-seed 42 --source-cltc-loss-weight 0.1 --source-cltc-score-weight 1 --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

Replay-Attack：

```bash
python LCF-CLTC-SPR/eval_replay_lcf_cltc_spr.py --source-run results/replay_lcf_cltc_legacy_fp32_20k_s42 --source-max-steps 20000 --source-train-batch-size 1 --source-model-precision float32 --source-seed 42 --source-cltc-loss-weight 0.1 --source-cltc-score-weight 1 --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

### 重算普通与同比例融合 TTA（CPU 缓存统计）

```bash
python LCF-CLTC-SPR-TTA/compare_oulu_spr_tta.py --spr-run results/oulu_lcf_cltc_spr_20260925_010043_662042 --tta-reduction mean
```

```bash
python LCF-CLTC-SPR-TTA/compare_lcc_spr_tta.py --spr-run results/lcc_lcf_cltc_spr_20260925_092943_946978 --tta-reduction mean
```

```bash
python LCF-CLTC-SPR-TTA/compare_nuaa_spr_tta.py --spr-run results/nuaa_lcf_cltc_spr_20260924_164610_163966 --tta-reduction mean
```

```bash
python LCF-CLTC-SPR-TTA/compare_replay_spr_tta.py --spr-run results/replay_lcf_cltc_spr_20260925_101212_401331 --tta-reduction mean
```

每条命令同时包含普通 TTA 和同比例融合 TTA。新增列 `spr_minus_blend_pp` 用于比较 C 与融合 TTA。

### 重算两项消融（CPU 缓存统计）

```bash
python LCF-CLTC-SPR-ABLATION/ablate_oulu_spr.py --spr-run results/oulu_lcf_cltc_spr_20260925_010043_662042
```

```bash
python LCF-CLTC-SPR-ABLATION/ablate_lcc_spr.py --spr-run results/lcc_lcf_cltc_spr_20260925_092943_946978
```

```bash
python LCF-CLTC-SPR-ABLATION/ablate_nuaa_spr.py --spr-run results/nuaa_lcf_cltc_spr_20260924_164610_163966
```

```bash
python LCF-CLTC-SPR-ABLATION/ablate_replay_spr.py --spr-run results/replay_lcf_cltc_spr_20260925_101212_401331
```

每条命令同时完成中位数无惩罚和均值无惩罚两项实验。

## 9. 关键问题与答案

### 为什么没有 Epoch？20,000 步在哪里？

20,000 是原 NUAA/Replay 模型历史训练的参数更新次数。C 脚本加载已训练好的权重进行推理，没有训练轮次。训练步数与推理次数不能直接比较。以 NUAA 为例：1505×4＋6024×4=30,116 次单图前向；每张图内部做四个视图，进度条仍按原始图像 batch 计数。

### 不重新训练是否意味着模型没有学够？

不会。旧模型已经完成记录中的训练；C 是无可学习参数的推理模块。它是否有效由配对结果决定，不能根据是否新增训练来判断。

### 验证集有什么作用？

用验证集真人估计分数尺度，用完整验证集选择分类阈值。模型权重不更新。测试集只用于固定规则后的评估，不能用它挑 beta、alpha、种子或消融版本。

### detach 后 CLTC 为什么还能影响效果？

detach 阻止 CLTC 损失回传到 LCF，但 CLTC head 仍能训练，并在推理时输出额外异常分数。原图最终分数包含该分数，所以效果可能改变；并非“只监督、不参与判断”。

### lambda=0.1、alpha 和 beta 分别是什么？

lambda 控制 CLTC 训练损失权重；alpha 控制重建分数与轨迹分数的推理融合；beta 控制 C 证据与原 LCF＋CLTC 分数的融合。三者作用阶段不同，不要求数值相同。当前 lambda=0.1，LCC alpha=0.5、其他库 alpha=1，四库 beta=0.25。

### 能在评估命令中改 20,000 步、seed 或模型精度吗？

`--source-*` 是核对已保存配置，不会重新训练或转换模型。改得与源运行不一致会报错。更换已训练模型应选择对应的完整 `--source-run`。C 的 beta、ROI 等参数能影响 C 推理实验，但调参只能依靠验证集。

### 普通 TTA 和同区域均值为什么不同？

普通 TTA 平均四个视图各自的整图异常分数；同区域均值先固定原图选出的 ROI，再平均四个视图在该 ROI 内的区域分数。后者保留 C 的空间选择步骤。

### 为什么四视图 TTA 不需要重新跑 GPU？

SPR 首次推理已经把 `view_*_fused` 和 `region_*` 列保存到 validation/test CSV。TTA、同权重融合和本轮两项消融都可读取这些列计算。

### 为什么 Accuracy 上升，HTER 却变差？

Accuracy 按测试集中实际样本数加权；HTER 对真人误报率和攻击漏检率各取一半。类别数量不平衡时两者会不同。例如 LCC 普通 TTA 的 Accuracy 高于 C，但真人误报更多，HTER 反而更高。当前 Recall 的正类是攻击，攻击漏检率=1−Recall。

### 表里的 +0.8 是什么意思？

百分点差值=100×(新指标−旧指标)。例如 0.84 到 0.848 是增加 0.8 个百分点，不是增加 0.8 的原始分数。HTER 的负差值表示改善。

### 能否说 C 的每个步骤都被证明有效？

不能。当前中位数聚合有一致的四库 AUROC/HTER 优势；MAD 增量很小；固定区域与各视图独立选区域尚未对照。当前结果也没有证明方法新颖性或论文录用可能性。

### 命令换行、目录已存在、CUDA/NVML 警告怎么处理？

WSL 命令需保持一行，或使用行尾反斜杠续行。显式结果目录已存在时换一个新目录，避免覆盖；新脚本默认生成时间戳目录。训练/SPR 推理需使用已确认 CUDA 可用的 Python；普通 TTA/融合/缓存消融无需 GPU。NVML 警告本身不等于 CUDA 推理失败，应以 torch CUDA 检测与实际进度判断。

## 10. 已完成验证、限制和后续优先级

来源检查包括原始配置、样本路径与标签、manifest 哈希、原图分数一致性、保存指标与阈值复现。消融另外检查区域分数能否还原原中位数/MAD/C 证据。相关 TTA/融合/消融离线测试累计 15 项通过，代码风格检查通过。本记录仅汇总已有实验，没有再次进行模型推理。

当前记录是每库一个已保存 checkpoint 上的配对点估计；部分历史源运行由先前实验选择而来，不能把这些数据表述为跨种子平均。后续应按预先固定的配置与划分做多训练种子验证，报告全部结果及均值/波动，避免只挑最好种子。

后续优先级：

1. 固定原图 ROI vs 各视图独立选 ROI；现有标量缓存不够，需要另行保存/计算所需证据并重新推理。
2. 固定数据划分下的多训练种子配对验证。
3. 需要时补充按视频/身份等相关样本单位设计的置信区间分析。
4. VEC/简化 CoT 仍属于后续方案，不能写入当前已完成实验贡献。

## 11. 原始数值与来源快照（机器可读）

以下 JSON 保存全部六种方案的六项指标、原配置、校准参数、checkpoint 与 manifest 哈希以及本次读取的配置/指标文件哈希。它不是 checkpoint 或逐图数据的完整备份；复现实验仍需要第 7 节的源结果文件。

```json
{
  "record_date": "2026-09-25",
  "datasets": [
    {
      "dataset": "OULU-NPU",
      "runs": {
        "name": "OULU-NPU",
        "tta": "oulu_spr_tta_20260925_131448_260792",
        "baseline": "oulu_lcf_cltc_legacy_s42",
        "ablation": "oulu_spr_ablation_20260925_153517_111896",
        "slug": "oulu",
        "spr": "oulu_lcf_cltc_spr_20260925_010043_662042"
      },
      "source_config": {
        "root": "/mnt/d/BaiduNetdiskDownload/OULU-NPU_organized",
        "normal_dir": "train/normal",
        "normal_test_dir": "test/normal",
        "abnormal_dir": "test/abnormal",
        "val_split_ratio": 0.2,
        "encoder_name": "vit_giant_patch14_reg4_dinov2",
        "target_layers": [
          6,
          10,
          14,
          18,
          22,
          26,
          30,
          34
        ],
        "decoder_depth": 8,
        "fuse_mode": "half",
        "bottleneck_dropout": 0.2,
        "lcf_dropout": 0,
        "cltc_hidden_dim": 128,
        "cltc_loss_weight": 0.1,
        "cltc_score_weight": 1,
        "image_size": 392,
        "crop_size": null,
        "remove_class_token": false,
        "use_context_recentering": false,
        "model_precision": "float16",
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_workers": 4,
        "max_steps": 5000,
        "gradient_clip_val": 0.1,
        "accelerator": "gpu",
        "seed": 42,
        "results_dir": "results/oulu_lcf_cltc_legacy_s42",
        "resume_ckpt": null,
        "protocol": "legacy_lcf_from_test_20pct_frame_level_folder",
        "checkpoint_selection": "last_fixed_step",
        "use_lcf": true,
        "use_cltc": true,
        "cltc_detach": true,
        "calibration": "validation_normal_q95_scales_and_validation_f1_threshold",
        "score_fusion": "after_pooling"
      },
      "spr_settings": {
        "beta": 0.25,
        "roi_fraction": 0.1,
        "views": [
          "original",
          "flip",
          "gamma09",
          "gamma11"
        ],
        "protocol": "legacy_lcf_from_test_20pct_frame_level_folder"
      },
      "metrics": {
        "original_lcf_cltc": {
          "image_AUROC": 0.874795283564815,
          "image_Accuracy": 0.8902777777777777,
          "image_F1Score": 0.9346783529022656,
          "image_Precision": 0.8923271234606883,
          "image_Recall": 0.98125,
          "image_HTER": 0.24618055555555554
        },
        "tta_mean_4views": {
          "image_AUROC": 0.8774145206404321,
          "image_Accuracy": 0.8926388888888889,
          "image_F1Score": 0.935951611566824,
          "image_Precision": 0.8952290378823903,
          "image_Recall": 0.9805555555555555,
          "image_HTER": 0.2392361111111111
        },
        "tta_blend_4views": {
          "image_AUROC": 0.875670211226852,
          "image_Accuracy": 0.8911111111111111,
          "image_F1Score": 0.9346339836584959,
          "image_Precision": 0.8991017003529035,
          "image_Recall": 0.9730902777777778,
          "image_HTER": 0.23185763888888888
        },
        "mean_no_penalty": {
          "image_AUROC": 0.883316936728395,
          "image_Accuracy": 0.8952777777777777,
          "image_F1Score": 0.9373858163095832,
          "image_Precision": 0.8984399872652021,
          "image_Recall": 0.9798611111111111,
          "image_HTER": 0.2315972222222222
        },
        "median_no_penalty": {
          "image_AUROC": 0.8838656201774691,
          "image_Accuracy": 0.8955555555555555,
          "image_F1Score": 0.9374271925445166,
          "image_Precision": 0.9001278363694472,
          "image_Recall": 0.9779513888888889,
          "image_HTER": 0.22803819444444445
        },
        "lcf_cltc_spr": {
          "image_AUROC": 0.8836451099537037,
          "image_Accuracy": 0.8954166666666666,
          "image_F1Score": 0.9373283395755306,
          "image_Precision": 0.9002398081534773,
          "image_Recall": 0.9776041666666667,
          "image_HTER": 0.22786458333333334
        }
      },
      "calibration": {
        "spr": {
          "beta": 0.25,
          "q0": 1.9496013160376837,
          "qe": 1.5726823568344117,
          "threshold_policy": "validation_f1",
          "original_threshold": 1.5931082303060533,
          "spr_threshold": 1.6247827125099796,
          "checkpoint_sha256": "51c6aad6bd2e13e58774794b9da6666180d855798cf604679065b0e206f03fad",
          "validation_manifest_sha256": "e6c72a850e5aed1aea7655b8e4bccdf71f80e703b02851b18a1f8793278b05dd",
          "score_scales": [
            0.28198203444480896,
            0.04887208715081215
          ]
        },
        "tta": {
          "threshold_policy": "validation_f1",
          "tta_threshold": 1.6013441383838654,
          "tta_reduction": "mean",
          "tta_view_weights": [
            0.25,
            0.25,
            0.25,
            0.25
          ],
          "views": [
            "original",
            "flip",
            "gamma09",
            "gamma11"
          ],
          "original_threshold": 1.5931082303060533,
          "spr_threshold": 1.6247827125099796,
          "validation_manifest_sha256": "e6c72a850e5aed1aea7655b8e4bccdf71f80e703b02851b18a1f8793278b05dd",
          "matched_tta": {
            "beta": 0.25,
            "q0": 1.9496013160376837,
            "q_tta": 1.935277296602726,
            "threshold_policy": "validation_f1",
            "threshold": 1.6104949692356703
          }
        },
        "ablation": {
          "median_no_penalty": {
            "beta": 0.25,
            "q0": 1.9496013160376837,
            "qe": 1.5743256330490114,
            "threshold_policy": "validation_f1",
            "threshold": 1.6251659227520672
          },
          "mean_no_penalty": {
            "beta": 0.25,
            "q0": 1.9496013160376837,
            "qe": 1.5478021502494812,
            "threshold_policy": "validation_f1",
            "threshold": 1.621289906818567
          }
        }
      },
      "runtime_spr": {
        "validation": {
          "images": 1800,
          "network_forward_batches": 7200,
          "network_image_forwards": 7200,
          "views_per_image": 4,
          "wall_seconds_including_io": 553.5171361700632,
          "seconds_per_image_including_io": 0.3075095200944795,
          "max_abs_original_score_error": 2.3115442449039847e-7,
          "peak_cuda_allocated_bytes": 2905936384
        },
        "test": {
          "images": 7200,
          "network_forward_batches": 28800,
          "network_image_forwards": 28800,
          "views_per_image": 4,
          "wall_seconds_including_io": 2222.2343303270172,
          "seconds_per_image_including_io": 0.3086436569898635,
          "max_abs_original_score_error": 4.440892098500626e-16,
          "peak_cuda_allocated_bytes": 2905936384
        }
      },
      "manifest_sha256": {
        "train": "0785d9c30a4c2a978a19a7e2abaf79c226e74cdf867b23edb1191bda73ad1134",
        "validation": "e6c72a850e5aed1aea7655b8e4bccdf71f80e703b02851b18a1f8793278b05dd",
        "test": "a4df28158c31cbea31fa801e16833280db5949de2bd181d5a29104f47ad69859"
      },
      "checkpoint_sha256": "51c6aad6bd2e13e58774794b9da6666180d855798cf604679065b0e206f03fad",
      "artifact_sha256": {
        "results/oulu_lcf_cltc_legacy_s42/metrics.json": "ccdc843be3261b4126e4081e1363159ea5714cd9951a32ea4caf58ac82f8745d",
        "results/oulu_lcf_cltc_legacy_s42/config.json": "067e1990f9a1987a8aa75cc3a986dd6d4ad8c06f622967a133a2a289c0de80a3",
        "results/oulu_lcf_cltc_spr_20260925_010043_662042/metrics.json": "4461363087cabda0c11958796beddf2e4eb51d972ed26209020d46dd51068856",
        "results/oulu_lcf_cltc_spr_20260925_010043_662042/config.json": "9792635e9210f56eda6a77709a4f7a84836f1b267ccf60d2eecbf457fbab2bc7",
        "results/oulu_spr_tta_20260925_131448_260792/metrics.json": "2f42ae7ffbbfad0d4f580fe3688866d6d80135f7579ebc9867b38a5af49333f5",
        "results/oulu_spr_tta_20260925_131448_260792/config.json": "db4b2200ea2ef54e7c6da6cde43211e8a94798d49a17084024a08326f8854122",
        "results/oulu_spr_ablation_20260925_153517_111896/metrics.json": "c5873d9f77e25359bc0363cb4437d60a7bba60c3ad3ae4388d629db649d52c5b",
        "results/oulu_spr_ablation_20260925_153517_111896/config.json": "b49f6eeeae1a9a6d194e61c9546f22b6b691df5204c048781430da980bd9b50d"
      }
    },
    {
      "dataset": "LCC",
      "runs": {
        "name": "LCC",
        "tta": "lcc_spr_tta_20260925_131446_701614",
        "baseline": "lcc_lcf_cltc_final_5k_b2_s42",
        "ablation": "lcc_spr_ablation_20260925_153515_587466",
        "slug": "lcc",
        "spr": "lcc_lcf_cltc_spr_20260925_092943_946978"
      },
      "source_config": {
        "data_root": "/mnt/c/Users/heqi/Desktop/wj/data",
        "mode": "rgb",
        "encoder_name": "vit_large_patch14_reg4_dinov2",
        "image_size": 392,
        "model_precision": "float32",
        "train_batch_size": 2,
        "eval_batch_size": 1,
        "num_workers": 4,
        "max_steps": 5000,
        "seed": 42,
        "val_split_ratio": 0.2,
        "normal_test_ratio": 0.2,
        "cltc_hidden_dim": 128,
        "cltc_loss_weight": 0.1,
        "cltc_score_weight": 0.5,
        "accelerator": "gpu",
        "results_dir": "results/lcc_lcf_cltc_final_5k_b2_s42",
        "dataset": "LCC_colors",
        "protocol": "legacy_normal_holdout_from_test_val",
        "use_lcf": true,
        "use_cltc": true,
        "cltc_detach": true,
        "score_fusion": "after_pooling"
      },
      "spr_settings": {
        "beta": 0.25,
        "roi_fraction": 0.1,
        "views": [
          "original",
          "flip",
          "gamma09",
          "gamma11"
        ],
        "protocol": "legacy_normal_holdout_from_test_val"
      },
      "metrics": {
        "original_lcf_cltc": {
          "image_AUROC": 0.837747897648064,
          "image_Accuracy": 0.8299964247407937,
          "image_F1Score": 0.9048714614384316,
          "image_Precision": 0.9858326068003488,
          "image_Recall": 0.8361989277130708,
          "image_HTER": 0.25757621181914025
        },
        "tta_mean_4views": {
          "image_AUROC": 0.8342832016708888,
          "image_Accuracy": 0.8768323203432249,
          "image_F1Score": 0.9330613037986981,
          "image_Precision": 0.9832104832104832,
          "image_Recall": 0.8877796265483453,
          "image_HTER": 0.277731808347449
        },
        "tta_blend_4views": {
          "image_AUROC": 0.837181274452489,
          "image_Accuracy": 0.8337504469074007,
          "image_F1Score": 0.9071670992214015,
          "image_Precision": 0.9858971577348665,
          "image_Recall": 0.8400813459049732,
          "image_HTER": 0.25563500272318906
        },
        "mean_no_penalty": {
          "image_AUROC": 0.8448621666591717,
          "image_Accuracy": 0.828923846978906,
          "image_F1Score": 0.9040986070748572,
          "image_Precision": 0.987089715536105,
          "image_Recall": 0.8339804030319837,
          "image_HTER": 0.2424692579434676
        },
        "median_no_penalty": {
          "image_AUROC": 0.846002408398415,
          "image_Accuracy": 0.836968180193064,
          "image_F1Score": 0.9090183559457302,
          "image_Precision": 0.9872156013001083,
          "image_Recall": 0.8422998705860603,
          "image_HTER": 0.23830952416642934
        },
        "lcf_cltc_spr": {
          "image_AUROC": 0.8456426476393198,
          "image_Accuracy": 0.8378619949946371,
          "image_F1Score": 0.9095622694186858,
          "image_Precision": 0.9872294372294372,
          "image_Recall": 0.8432242558698465,
          "image_HTER": 0.2378473315245362
        }
      },
      "calibration": {
        "spr": {
          "beta": 0.25,
          "q0": 1.517522625571829,
          "qe": 1.1618722081184387,
          "threshold_policy": "validation_hter",
          "original_threshold": 1.3124808491738815,
          "spr_threshold": 1.3468846153719,
          "checkpoint_sha256": "99be2cc9e55335bd33ffd60524bb97fa9c41f6f79b7ec00f37f1571191a329ed",
          "validation_manifest_sha256": "56dc07e5db4ced23b2e93606505ebe39e9fe994fcc616f749c103dc9154ad3f4",
          "score_scales": [
            0.06767373532056808,
            0.02903709188103676
          ]
        },
        "tta": {
          "threshold_policy": "validation_hter",
          "tta_threshold": 1.2900993525981903,
          "tta_reduction": "mean",
          "tta_view_weights": [
            0.25,
            0.25,
            0.25,
            0.25
          ],
          "views": [
            "original",
            "flip",
            "gamma09",
            "gamma11"
          ],
          "original_threshold": 1.3124808491738815,
          "spr_threshold": 1.3468846153719,
          "validation_manifest_sha256": "56dc07e5db4ced23b2e93606505ebe39e9fe994fcc616f749c103dc9154ad3f4",
          "matched_tta": {
            "beta": 0.25,
            "q0": 1.517522625571829,
            "q_tta": 1.5151851028203964,
            "threshold_policy": "validation_hter",
            "threshold": 1.3128198520667844
          }
        },
        "ablation": {
          "median_no_penalty": {
            "beta": 0.25,
            "q0": 1.517522625571829,
            "qe": 1.1648374199867249,
            "threshold_policy": "validation_hter",
            "threshold": 1.3475467940945183
          },
          "mean_no_penalty": {
            "beta": 0.25,
            "q0": 1.517522625571829,
            "qe": 1.1618213057518005,
            "threshold_policy": "validation_hter",
            "threshold": 1.3483648129916879
          }
        }
      },
      "runtime_spr": {
        "validation": {
          "images": 1398,
          "network_forward_batches": 5592,
          "network_image_forwards": 5592,
          "views_per_image": 4,
          "wall_seconds_including_io": 427.968955207034,
          "seconds_per_image_including_io": 0.30612943863164094,
          "max_abs_original_score_error": 1.9915276050497255e-7,
          "peak_cuda_allocated_bytes": 1802720256
        },
        "test": {
          "images": 5594,
          "network_forward_batches": 22376,
          "network_image_forwards": 22376,
          "views_per_image": 4,
          "wall_seconds_including_io": 1736.4347908289637,
          "seconds_per_image_including_io": 0.3104102236018884,
          "max_abs_original_score_error": 4.440892098500626e-16,
          "peak_cuda_allocated_bytes": 1802720256
        }
      },
      "manifest_sha256": {
        "train": "e9ae1060602401e6c88fea3765dbe64753a857b0a47ee63a255d4519d41b7d94",
        "validation": "56dc07e5db4ced23b2e93606505ebe39e9fe994fcc616f749c103dc9154ad3f4",
        "test": "9ad210f4ad8774ba057dd335559195d20ccd3e7032863dfb051e606d5b1deabb"
      },
      "checkpoint_sha256": "99be2cc9e55335bd33ffd60524bb97fa9c41f6f79b7ec00f37f1571191a329ed",
      "artifact_sha256": {
        "results/lcc_lcf_cltc_final_5k_b2_s42/metrics.json": "eb3123f1f632564b76baa489da97ae864ab6ea385db9595827fcc4916a8fc863",
        "results/lcc_lcf_cltc_final_5k_b2_s42/config.json": "74240a8c74720da2b27b11652d04ecbb3138aa5aaa6e4cecfc61e600e4577827",
        "results/lcc_lcf_cltc_spr_20260925_092943_946978/metrics.json": "e11050040c919444380b044665a050be66d0c5e9c55a2d3ea8f549190759fd89",
        "results/lcc_lcf_cltc_spr_20260925_092943_946978/config.json": "71496c6e8a214c79e5228e4371a783e9bd94eb47f6598571aa13d8c6aff41138",
        "results/lcc_spr_tta_20260925_131446_701614/metrics.json": "51091c07a429fe1bebb55c091c6f7b7a47c3359fbdd0fa87ff3fcd19b61fe5f7",
        "results/lcc_spr_tta_20260925_131446_701614/config.json": "3133acb195936205cbf016f917127ec8f561d814e49c5f32b62083150c5b89aa",
        "results/lcc_spr_ablation_20260925_153515_587466/metrics.json": "b54fc39317ad282a49e62f0a4771648fc67c10326ecdf1e6c49c4f76cad9ad62",
        "results/lcc_spr_ablation_20260925_153515_587466/config.json": "d764a8d619e6e07f15f15a17c480a2c1db275e52ef447457f9f433fa8896ee2c"
      }
    },
    {
      "dataset": "NUAA",
      "runs": {
        "name": "NUAA",
        "tta": "nuaa_spr_tta_20260925_131449_592189",
        "baseline": "nuaa_lcf_cltc_legacy_fp32_20k_s42",
        "ablation": "nuaa_spr_ablation_20260925_153518_742563",
        "slug": "nuaa",
        "spr": "nuaa_lcf_cltc_spr_20260924_164610_163966"
      },
      "source_config": {
        "data_root": "/mnt/c/Users/heqi/Desktop/wj/data",
        "mode": "rgb",
        "encoder_name": "vit_giant_patch14_reg4_dinov2",
        "image_size": 392,
        "model_precision": "float32",
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_workers": 4,
        "max_steps": 20000,
        "seed": 42,
        "val_split_ratio": 0.2,
        "normal_test_ratio": 0.2,
        "cltc_hidden_dim": 128,
        "cltc_loss_weight": 0.1,
        "cltc_score_weight": 1,
        "accelerator": "gpu",
        "results_dir": "results/nuaa_lcf_cltc_legacy_fp32_20k_s42",
        "dataset": "NUAA_mtcnn_colors",
        "protocol": "legacy_dedicated_normal_test_from_test_val",
        "use_lcf": true,
        "use_cltc": true,
        "cltc_detach": true,
        "score_fusion": "after_pooling"
      },
      "spr_settings": {
        "beta": 0.25,
        "roi_fraction": 0.1,
        "views": [
          "original",
          "flip",
          "gamma09",
          "gamma11"
        ],
        "protocol": "legacy_dedicated_normal_test_from_test_val"
      },
      "metrics": {
        "original_lcf_cltc": {
          "image_AUROC": 0.9840275606593814,
          "image_Accuracy": 0.9437250996015937,
          "image_F1Score": 0.9678276549302458,
          "image_Precision": 0.9924094978590892,
          "image_Recall": 0.9444341544730506,
          "image_HTER": 0.05898292276347472
        },
        "tta_mean_4views": {
          "image_AUROC": 0.9695762919059084,
          "image_Accuracy": 0.9385790172642763,
          "image_F1Score": 0.9649820177929207,
          "image_Precision": 0.9866460228372363,
          "image_Recall": 0.9442489349879607,
          "image_HTER": 0.08307553250601964
        },
        "tta_blend_4views": {
          "image_AUROC": 0.9815885904797185,
          "image_Accuracy": 0.9435590969455512,
          "image_F1Score": 0.9677541729893778,
          "image_Precision": 0.991642371234208,
          "image_Recall": 0.9449898129283201,
          "image_HTER": 0.06190509353583997
        },
        "mean_no_penalty": {
          "image_AUROC": 0.9856124467493981,
          "image_Accuracy": 0.9435590969455512,
          "image_F1Score": 0.9677174325864033,
          "image_Precision": 0.9927917397233587,
          "image_Recall": 0.9438784960177811,
          "image_HTER": 0.05766075199110947
        },
        "median_no_penalty": {
          "image_AUROC": 0.9884292646786442,
          "image_Accuracy": 0.950199203187251,
          "image_F1Score": 0.9715909090909091,
          "image_Precision": 0.9939934121294323,
          "image_Recall": 0.9501759585108354,
          "image_HTER": 0.04971202074458233
        },
        "lcf_cltc_spr": {
          "image_AUROC": 0.9884550472309687,
          "image_Accuracy": 0.9500332005312085,
          "image_F1Score": 0.9714881121530737,
          "image_Precision": 0.9941837921675067,
          "image_Recall": 0.9498055195406557,
          "image_HTER": 0.049097240229672164
        }
      },
      "calibration": {
        "spr": {
          "beta": 0.25,
          "q0": 1.9703804337689361,
          "qe": 1.567850649356842,
          "threshold_policy": "validation_hter",
          "original_threshold": 1.9487128349079528,
          "spr_threshold": 1.9760224333348781,
          "checkpoint_sha256": "f7e11ff028985ebfe30e7d0779504f1f2aa836c727a0eed8c69d9ba91007f73f",
          "validation_manifest_sha256": "f6438bd11a9c3810aae7822943ac0c9b4d05fb7963e07fb32622e8e719ce3c2d",
          "score_scales": [
            0.05590798333287239,
            0.02614019811153412
          ]
        },
        "tta": {
          "threshold_policy": "validation_hter",
          "tta_threshold": 1.9664831757545471,
          "tta_reduction": "mean",
          "tta_view_weights": [
            0.25,
            0.25,
            0.25,
            0.25
          ],
          "views": [
            "original",
            "flip",
            "gamma09",
            "gamma11"
          ],
          "original_threshold": 1.9487128349079528,
          "spr_threshold": 1.9760224333348781,
          "validation_manifest_sha256": "f6438bd11a9c3810aae7822943ac0c9b4d05fb7963e07fb32622e8e719ce3c2d",
          "matched_tta": {
            "beta": 0.25,
            "q0": 1.9703804337689361,
            "q_tta": 2.046763129532337,
            "threshold_policy": "validation_hter",
            "threshold": 1.935240527393068
          }
        },
        "ablation": {
          "median_no_penalty": {
            "beta": 0.25,
            "q0": 1.9703804337689361,
            "qe": 1.569780558347702,
            "threshold_policy": "validation_hter",
            "threshold": 1.9755755225724707
          },
          "mean_no_penalty": {
            "beta": 0.25,
            "q0": 1.9703804337689361,
            "qe": 1.5999688804149628,
            "threshold_policy": "validation_hter",
            "threshold": 1.9633273431689715
          }
        }
      },
      "runtime_spr": {
        "validation": {
          "images": 1505,
          "network_forward_batches": 6020,
          "network_image_forwards": 6020,
          "views_per_image": 4,
          "wall_seconds_including_io": 1258.2468138580443,
          "seconds_per_image_including_io": 0.8360443945900626,
          "max_abs_original_score_error": 2.360982072069362e-7,
          "peak_cuda_allocated_bytes": 5804556288
        },
        "test": {
          "images": 6024,
          "network_forward_batches": 24096,
          "network_image_forwards": 24096,
          "views_per_image": 4,
          "wall_seconds_including_io": 5090.682346881018,
          "seconds_per_image_including_io": 0.8450667906508994,
          "max_abs_original_score_error": 4.440892098500626e-16,
          "peak_cuda_allocated_bytes": 5804556288
        }
      },
      "manifest_sha256": {
        "train": "827b24be721ebf2cf35ac468e1b81cb8dc490ce5df4749295c00c59c1b2deff1",
        "validation": "f6438bd11a9c3810aae7822943ac0c9b4d05fb7963e07fb32622e8e719ce3c2d",
        "test": "4caa9d1d20a4fdb12bdef0970a1d52d0f8e2b624d5e43c07b7520a46c7be9ac9"
      },
      "checkpoint_sha256": "f7e11ff028985ebfe30e7d0779504f1f2aa836c727a0eed8c69d9ba91007f73f",
      "artifact_sha256": {
        "results/nuaa_lcf_cltc_legacy_fp32_20k_s42/metrics.json": "fd7f437fd00c88fe3ef3e4efa57b662e3d6b132661b40a34210ab79554a140d6",
        "results/nuaa_lcf_cltc_legacy_fp32_20k_s42/config.json": "11fcd85a9227533b2f9b5516ca1aa12586386e4a02b7c160d3c6a599183199c0",
        "results/nuaa_lcf_cltc_spr_20260924_164610_163966/metrics.json": "f630cdca2e99f807fe39f4e4e4191b2f28a69b5c9dd91f37fa89b44204d943bb",
        "results/nuaa_lcf_cltc_spr_20260924_164610_163966/config.json": "baf9ed83ee6c0abdfa8ed983b039e5a0073bb05db405ee7da14e56cca4605460",
        "results/nuaa_spr_tta_20260925_131449_592189/metrics.json": "5ee8cc1abcd2e8a6a63cdac4f74d8d4cc3532f62327d0192e239607c412eb3d5",
        "results/nuaa_spr_tta_20260925_131449_592189/config.json": "6a96886e24f21d928984b947bf09c4ecbe45a783d12989444d9b0fde5d7e5360",
        "results/nuaa_spr_ablation_20260925_153518_742563/metrics.json": "680d51588b91eabf33627043e24428e258845b9b2d17ff802df7347f06dae888",
        "results/nuaa_spr_ablation_20260925_153518_742563/config.json": "ef125ac5784eaa1eb3f2bcccc1223d1d4e2971720b20f741d455fc384029f722"
      }
    },
    {
      "dataset": "Replay-Attack",
      "runs": {
        "name": "Replay-Attack",
        "tta": "replay_spr_tta_20260925_131451_221673",
        "baseline": "replay_lcf_cltc_legacy_fp32_20k_s42",
        "ablation": "replay_spr_ablation_20260925_153520_455646",
        "slug": "replay",
        "spr": "replay_lcf_cltc_spr_20260925_101212_401331"
      },
      "source_config": {
        "data_root": "/mnt/c/Users/heqi/Desktop/wj/data",
        "mode": "rgb",
        "encoder_name": "vit_giant_patch14_reg4_dinov2",
        "image_size": 392,
        "model_precision": "float32",
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "num_workers": 4,
        "max_steps": 20000,
        "seed": 42,
        "val_split_ratio": 0.2,
        "normal_test_ratio": 0.2,
        "cltc_hidden_dim": 128,
        "cltc_loss_weight": 0.1,
        "cltc_score_weight": 1,
        "accelerator": "gpu",
        "results_dir": "results/replay_lcf_cltc_legacy_fp32_20k_s42",
        "dataset": "REPLAY_mtcnn_colors",
        "protocol": "legacy_normal_holdout_from_test_val",
        "use_lcf": true,
        "use_cltc": true,
        "cltc_detach": true,
        "score_fusion": "after_pooling"
      },
      "spr_settings": {
        "beta": 0.25,
        "roi_fraction": 0.1,
        "views": [
          "original",
          "flip",
          "gamma09",
          "gamma11"
        ],
        "protocol": "legacy_normal_holdout_from_test_val"
      },
      "metrics": {
        "original_lcf_cltc": {
          "image_AUROC": 0.9988256778790227,
          "image_Accuracy": 0.9945328031809145,
          "image_F1Score": 0.9970422156493681,
          "image_Precision": 0.9976681614349776,
          "image_Recall": 0.9964170548190613,
          "image_HTER": 0.016108653207209455
        },
        "tta_mean_4views": {
          "image_AUROC": 0.9958804022368942,
          "image_Accuracy": 0.9852551358515573,
          "image_F1Score": 0.9920014379437404,
          "image_Precision": 0.9953110910730387,
          "image_Recall": 0.988713722680043,
          "image_HTER": 0.03427749989345868
        },
        "tta_blend_4views": {
          "image_AUROC": 0.9985660327326507,
          "image_Accuracy": 0.9922133863485753,
          "image_F1Score": 0.9957813481734136,
          "image_Precision": 0.9978413383702105,
          "image_Recall": 0.9937298459333572,
          "image_HTER": 0.016350936064158394
        },
        "mean_no_penalty": {
          "image_AUROC": 0.998835148218708,
          "image_Accuracy": 0.9935387673956262,
          "image_F1Score": 0.9965013007984211,
          "image_Precision": 0.9980233602875113,
          "image_Recall": 0.9949838767466858,
          "image_HTER": 0.014622599071591033
        },
        "median_no_penalty": {
          "image_AUROC": 0.9990620417736683,
          "image_Accuracy": 0.9953611663353215,
          "image_F1Score": 0.9974905897114178,
          "image_Precision": 0.9980272596843616,
          "image_Recall": 0.9969544965962021,
          "image_HTER": 0.013637289146832883
        },
        "lcf_cltc_spr": {
          "image_AUROC": 0.9990640147611027,
          "image_Accuracy": 0.9955268389662028,
          "image_F1Score": 0.9975804283537951,
          "image_Precision": 0.9980276134122288,
          "image_Recall": 0.997133643855249,
          "image_HTER": 0.013547715517309414
        }
      },
      "calibration": {
        "spr": {
          "beta": 0.25,
          "q0": 1.9402751757349839,
          "qe": 1.5946141719818114,
          "threshold_policy": "validation_hter",
          "original_threshold": 2.029927656915426,
          "spr_threshold": 2.0410847258007996,
          "checkpoint_sha256": "dae9dfbcbc8864b458207d738c3cf4eb67285f74a873b4972d12e3ee18da990b",
          "validation_manifest_sha256": "2d58b0059b098e321c846b2431a02a668ecbc30aff5ac450d7c9754f1681b3ca",
          "score_scales": [
            0.04693065956234932,
            0.024929698556661606
          ]
        },
        "tta": {
          "threshold_policy": "validation_hter",
          "tta_threshold": 2.1595420837402344,
          "tta_reduction": "mean",
          "tta_view_weights": [
            0.25,
            0.25,
            0.25,
            0.25
          ],
          "views": [
            "original",
            "flip",
            "gamma09",
            "gamma11"
          ],
          "original_threshold": 2.029927656915426,
          "spr_threshold": 2.0410847258007996,
          "validation_manifest_sha256": "2d58b0059b098e321c846b2431a02a668ecbc30aff5ac450d7c9754f1681b3ca",
          "matched_tta": {
            "beta": 0.25,
            "q0": 1.9402751757349839,
            "q_tta": 2.1567203640937804,
            "threshold_policy": "validation_hter",
            "threshold": 2.032190503035901
          }
        },
        "ablation": {
          "median_no_penalty": {
            "beta": 0.25,
            "q0": 1.9402751757349839,
            "qe": 1.6040783405303956,
            "threshold_policy": "validation_hter",
            "threshold": 2.041102186559888
          },
          "mean_no_penalty": {
            "beta": 0.25,
            "q0": 1.9402751757349839,
            "qe": 1.7420053720474242,
            "threshold_policy": "validation_hter",
            "threshold": 2.02291045993237
          }
        }
      },
      "runtime_spr": {
        "validation": {
          "images": 1508,
          "network_forward_batches": 6032,
          "network_image_forwards": 6032,
          "views_per_image": 4,
          "wall_seconds_including_io": 1287.6175373740261,
          "seconds_per_image_including_io": 0.8538577834045267,
          "max_abs_original_score_error": 3.046967549380497e-7,
          "peak_cuda_allocated_bytes": 5804556288
        },
        "test": {
          "images": 6036,
          "network_forward_batches": 24144,
          "network_image_forwards": 24144,
          "views_per_image": 4,
          "wall_seconds_including_io": 5168.509528643917,
          "seconds_per_image_including_io": 0.8562805713459107,
          "max_abs_original_score_error": 8.881784197001252e-16,
          "peak_cuda_allocated_bytes": 5804556288
        }
      },
      "manifest_sha256": {
        "train": "a3b72b54d414b4806ed2eb68138a44c389b91e41bd0ca817b88227d1988d5f97",
        "validation": "2d58b0059b098e321c846b2431a02a668ecbc30aff5ac450d7c9754f1681b3ca",
        "test": "f49d8c519d06ca04e2e5cb56f0b0f12940e3307b8071b73f81371c5f445ff8b5"
      },
      "checkpoint_sha256": "dae9dfbcbc8864b458207d738c3cf4eb67285f74a873b4972d12e3ee18da990b",
      "artifact_sha256": {
        "results/replay_lcf_cltc_legacy_fp32_20k_s42/metrics.json": "8a72bf83395f8d148c047cdef32d558e081aa729a030db4984fa6c86e8083d50",
        "results/replay_lcf_cltc_legacy_fp32_20k_s42/config.json": "0dd0a53f22bd0876ced102033164e9251d1369babbd5b1eb8779898352407d44",
        "results/replay_lcf_cltc_spr_20260925_101212_401331/metrics.json": "c954f744cc656f7cab5809e5837d52d26255827d27038ea5c87215152070b49c",
        "results/replay_lcf_cltc_spr_20260925_101212_401331/config.json": "ec85495b8a282fd1be29313455700ca146951438e79acaf5a4d4b4ae0cebed31",
        "results/replay_spr_tta_20260925_131451_221673/metrics.json": "0eba1a94f7d8b80bb0c2f575d7145ac9255252b4333f08f4bad8db6c340dee90",
        "results/replay_spr_tta_20260925_131451_221673/config.json": "cdd993883b0bab96788ccff7bc9bdc6514fbcf4af1abb78f22ac6fb2872a88d8",
        "results/replay_spr_ablation_20260925_153520_455646/metrics.json": "67c90f549079e8a835ff6cd460096222c0ac29e50ec826a4407b41eae5bdd725",
        "results/replay_spr_ablation_20260925_153520_455646/config.json": "a7a5188d6c0c99a830d70fe14c2ee02086726db75563b6a97ac88d7b06d2f220"
      }
    }
  ]
}
```
