# 原 LCF＋CLTC / 普通 TTA / 同比例融合 TTA / C（SPR）对照

四个脚本分别读取已经完成的 SPR 结果，用 CPU 计算分数和指标。无需训练、图片推理、GPU 或 checkpoint 加载。

## 四条独立命令

在项目目录、原 `torch_env` 环境中分别运行，以下每个代码块都是一行。

LCC：

```bash
python LCF-CLTC-SPR-TTA/compare_lcc_spr_tta.py --spr-run results/lcc_lcf_cltc_spr_20260925_092943_946978 --tta-reduction mean
```

OULU-NPU：

```bash
python LCF-CLTC-SPR-TTA/compare_oulu_spr_tta.py --spr-run results/oulu_lcf_cltc_spr_20260925_010043_662042 --tta-reduction mean
```

NUAA：

```bash
python LCF-CLTC-SPR-TTA/compare_nuaa_spr_tta.py --spr-run results/nuaa_lcf_cltc_spr_20260924_164610_163966 --tta-reduction mean
```

Replay-Attack：

```bash
python LCF-CLTC-SPR-TTA/compare_replay_spr_tta.py --spr-run results/replay_lcf_cltc_spr_20260925_101212_401331 --tta-reduction mean
```

`--spr-run` 指向已经完成的 C 实验目录，里面需要有 `config.json`、`metrics.json`、`calibration.json` 以及两份分数 CSV。运行前还会核对原 LCF＋CLTC 的配置、清单和分数；原结果目录默认从 SPR 配置读取。若该目录搬走，可传 `--baseline-run 新路径`。不读取图片或 checkpoint 文件。

模型训练步数、训练 batch、精度、seed、CLTC 权重以及 SPR 的 beta、区域比例都会从记录中读取并打印。它们已经体现在缓存分数中，本次统计不能修改这些参数。

## 计算规则

```text
TTA = (原图 fused + 翻转 fused + gamma0.9 fused + gamma1.1 fused) / 4
```

每个视图的 fused 都已经使用原 LCF＋CLTC 的分支尺度和 alpha。直接等权平均，不再拟合视图权重。此次预先固定使用 mean。

脚本现在还会自动计算同融合比例的对照 `tta_blend_4views`，命令保持不变：

```text
beta = 原 SPR 实验保存的 beta（当前四库均为 0.25）
q0 = 验证集真人原始分数的 95% 分位数
q_tta = 验证集真人 TTA 平均分的 95% 分位数
融合 TTA = (1-beta) × 原始分数 + beta × q0 × TTA / q_tta
```

这与 C 采用相同的原分数保留比例和真人 q95 校准方式，用普通 TTA 替换 C 的区域复查证据。beta 直接沿用原 C 实验，未重新搜索；新增对照也在原验证集按旧规则独立确定阈值，再固定用于测试集。这样可以检查 C 的收益是否仅来自保留原分数。

1. 根据图片路径、标签、清单哈希核对缓存，检查原结果和 SPR 指标能否复现。
2. 计算验证集的 TTA 平均分。OULU 沿用验证集最大 F1 的阈值规则，其他三库沿用验证集最小 HTER；沿用旧实现的并列处理与严格 `score > threshold`。
3. 保存普通 TTA 和融合 TTA 的固定阈值与校准参数，再计算测试集指标。原模型和 C 直接使用核对过的已保存指标。
4. 输出四方对比。阈值仅由验证集决定，测试结果不参与聚合方式或参数选择。

TTA 和 C 都复用相同四个视图、同一 checkpoint、同一划分。每张图片的模型推理次数均为 4；这里比较的是相同视图计算预算，不代表二者后处理耗时完全相等。这次脚本新增模型推理次数为 0。

## 输出

默认写入 `results/<数据集>_spr_tta_<时间戳>/`，也可以通过 `--results-dir` 指定一个尚未存在的目录。

- `comparison.csv`：四方六项指标及百分点差值。
- `metrics.json`：指标、阈值规则和模型推理次数说明。
- `calibration.json`：验证集选出的 TTA 阈值、融合 TTA 的 beta/q0/q_tta/阈值，以及原模型/C 阈值。
- `validation_scores.csv`、`test_scores.csv`：逐图原分数、四视图分数、TTA 平均分、融合 TTA 分数、C 分数。
- `config.json`：数据来源、原配置、输入文件哈希。

重点看 `spr_minus_tta_pp`：AUROC/Accuracy/F1/Precision/Recall 为正表示 C 更好；HTER 为负表示 C 更好。比如 AUROC `+0.3` 表示 C 比普通 TTA 高 0.3 个百分点。四库仍然使用已有帧级协议。

新增的 `spr_minus_blend_pp` 表示 C 减去同融合比例 TTA，符号判断同上。C 若仍然更好，可排除“仅仅因为保留 75% 原始分数”这一解释；这仍是 C 整体证据构造的对照，尚未单独分离区域选择、中位数和 MAD 惩罚各自的作用，也需要后续多 checkpoint 验证。

只核对输入、不计算 TTA 指标也不写文件时，可以在任意命令末尾追加 `--check-inputs-only`。
