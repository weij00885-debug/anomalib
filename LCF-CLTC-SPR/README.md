# LCF＋CLTC＋SPR（C 模块）四库独立评估

本轮只筛查 **C 与你已经跑过的 LCF＋CLTC 的差别**。不训练，不重跑一轮独立 baseline，不启用 VEC，也不计算普通 TTA 的综合指标。

每个脚本从旧运行目录读取 `config.json`、`calibrated_last.ckpt`、`metrics.json`、三份 manifest 和原 validation/test 分数。直接使用原验证集与测试集，不重新划分数据。

## 四条独立运行命令

在 WSL 项目目录中，使用之前可以正常使用 CUDA 的 `torch_env`。每条命令只跑一个数据集，可按需单独复制。

```bash
python LCF-CLTC-SPR/eval_lcc_lcf_cltc_spr.py \
  --source-run results/lcc_lcf_cltc_final_5k_b2_s42 \
  --source-max-steps 5000 --source-train-batch-size 2 --source-model-precision float32 --source-seed 42 \
  --source-cltc-loss-weight 0.1 --source-cltc-score-weight 0.5 \
  --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

```bash
python LCF-CLTC-SPR/eval_oulu_lcf_cltc_spr.py \
  --source-run results/oulu_lcf_cltc_legacy_s42 \
  --source-max-steps 5000 --source-train-batch-size 1 --source-model-precision float16 --source-seed 42 \
  --source-cltc-loss-weight 0.1 --source-cltc-score-weight 1.0 \
  --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

```bash
python LCF-CLTC-SPR/eval_nuaa_lcf_cltc_spr.py \
  --source-run results/nuaa_lcf_cltc_legacy_fp32_20k_s42 \
  --source-max-steps 20000 --source-train-batch-size 1 --source-model-precision float32 --source-seed 42 \
  --source-cltc-loss-weight 0.1 --source-cltc-score-weight 1.0 \
  --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

```bash
python LCF-CLTC-SPR/eval_replay_lcf_cltc_spr.py \
  --source-run results/replay_lcf_cltc_legacy_fp32_20k_s42 \
  --source-max-steps 20000 --source-train-batch-size 1 --source-model-precision float32 --source-seed 42 \
  --source-cltc-loss-weight 0.1 --source-cltc-score-weight 1.0 \
  --eval-batch-size 1 --spr-beta 0.25 --roi-fraction 0.1
```

默认读取以下旧运行；`--source-run` 可以换成同一数据集、同一旧协议的其他完整运行目录：

|脚本|默认源目录（相对于项目）|
|---|---|
|LCC|`results/lcc_lcf_cltc_final_5k_b2_s42`|
|OULU|`results/oulu_lcf_cltc_legacy_s42`|
|NUAA|`results/nuaa_lcf_cltc_legacy_fp32_20k_s42`|
|Replay|`results/replay_lcf_cltc_legacy_fp32_20k_s42`|

命令中的 `--source-*` 参数会核对旧配置，确保你看到的步数、训练 batch、随机种子、模型精度、CLTC 权重确实属于所加载的 checkpoint。它们**不会修改已经训练好的模型**；若想换训练配置，应先得到相应的新 checkpoint，再修改 `--source-run` 及对应的 `--source-*` 参数。可以直接调整本次评估的 `--spr-beta`、`--roi-fraction` 和 `--eval-batch-size`。旧 OULU 的 `float16` 实际是 BF16 主干＋FP32 head，脚本会保留该设置。

例如指定另一次 LCC 实验并固定输出目录：

```bash
python LCF-CLTC-SPR/eval_lcc_lcf_cltc_spr.py --source-run results/lcc_lcf_cltc_final_5k_b2_s42 --spr-beta 0.25 --roi-fraction 0.1 --eval-batch-size 1 --results-dir results/lcc_spr_beta025_run1
```

不写 `--results-dir` 会自动生成带时间戳的新目录，避免重复运行撞上旧结果。显式指定的目录若已存在则报错，不覆盖结果。

数据没有移动时不用传路径。如果搬了数据，三库可以使用 `--data-root /mnt/c/Users/heqi/Desktop/wj/data`，OULU 使用 `--root /mnt/d/BaiduNetdiskDownload/OULU-NPU_organized`。这些参数只替换清单路径根目录，不改变样本成员。首版只支持 RGB 源运行。

默认设备是当前可见的 `cuda:0`；如需固定物理卡，可在命令前加 `CUDA_VISIBLE_DEVICES=0`。不要在尚未恢复的故障显卡上运行。

## 实际流程

1. 校验原 manifest 哈希、validation/test 分数与标签、不同子集间路径不重合，确认文件存在。
2. 用保存的超参数建立模型，严格恢复全部 state dict，再设为冻结、eval。源 checkpoint 需要是自己训练的可信文件。
3. 每张验证图顺序运行原图、翻转、gamma=0.9、gamma=1.1。第一遍同时获得原始分数和两路原生 patch 图，不额外跑一次 baseline。
4. 原图融合图选 top-10% patch，边界并列值均分权重。所有视图都复查这一个 ROI；翻转图先翻回原坐标。
5. 仅用验证真人估计 C 的分数尺度，用完整有标签验证集按旧规则选新阈值，保存并固定校准结果。
6. 在原测试清单上运行 C。输出 C 指标，并与旧 `metrics.json` 中的结果比较。

四个视图顺序推理，共四次网络计算，峰值激活不会同时保存四份。第一遍原图是 C 定位所必需，并非重新做一个独立 baseline 实验。

## 首版计算

```text
M_j = R_j / q_rec + alpha × U_j / q_traj
Omega = 原图 M_0 的 top 10% patch（包含并列分数权重）
v_j = M_j 在同一 Omega 中的加权均值
m = median(v_0, v_1, v_2, v_3)
d = median(abs(v_j − m))
e = m / (1 + d/(abs(m)+1e-8))

q0 = 验证真人原始 LCF+CLTC 分数的 q95
qe = 验证真人 e 的 q95
SPR = (1−beta) × 已保存的原始分数 + beta × q0 × e/qe
```

默认 beta=0.25；偶数中位数取中间两个值的平均。新的 C 地图会把浮点余弦误差导致的极小负数截成零，旧模型分数计算保持原逻辑。`qe` 或 `q0` 退化时直接停止，不强行放大极小分母。

原始分数直接使用你保存的 CSV。每张图必要的原图前向分数会与 CSV 比较；默认容差 `atol=1e-4, rtol=1e-3`。不一致就停止，优先检查预处理、数据内容、精度和模型版本；不能把更换数据或模型造成的变化归因给 C。

旧脚本的数据集在 CPU 先做 `Resize(..., antialias=True)`，之后再次调用模型的导出预处理。新脚本复用这个顺序。为保持该行为，C 的视图作用于这张已缩放、尚未标准化的 `[0,1]` RGB 图像，再走原 crop/normalize；不是对标准化后的张量做 gamma。

OULU 阈值沿用“验证集最大 F1”，其他三库沿用“验证集最小 HTER”，最终分类条件都沿用 `score > threshold`，包括旧实现的边界约定。全部是旧帧级协议，不自动变成官方视频级协议。

## 输出与怎么看结果

- `comparison.csv`：旧六项指标、C 六项指标、差值（百分点）。
- `metrics.json`：同样的对照结果、校准阈值、哈希、原分数核对误差与运行成本。
- `validation_scores.csv` / `test_scores.csv`：旧分数、核对用原图前向分数、C 分数、四视图区域分数。
- `calibration.json`：仅在验证集拟合的 q0、qe 和阈值。
- `config.json`：来源配置、checkpoint SHA256、manifest SHA256、固定 C 参数。

AUROC/Accuracy/F1/Precision/Recall 越高越好；HTER 越低越好。`delta_percentage_points` 是 `100 × (C − 原始)`，例如 AUROC 为 +0.6 表示上升 0.6 个百分点。

四个视图在正常前向时已经得到的标量分数也会缓存在 CSV 的 `view_*_fused` 列。本轮不计算 TTA 聚合结果。若 C 有效，后续同视图 TTA 的分数对照可以复用这些数据，避免再次推理。

beta 等参数若要筛选，只在验证集上筛选并固定后再测 test。历史最好 seed 的增益只是该 checkpoint 上的配对结果，后续论文需固定划分做多 checkpoint 验证。
