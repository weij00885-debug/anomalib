# C（SPR）同区域缓存消融

使用已有的 `region_original`、`region_flip`、`region_gamma09`、`region_gamma11` 分数完成两项预先指定的消融。原图选定的 ROI、四个视图、checkpoint、原划分、原 SPR beta 均保持一致，每个替代方案单独在验证集真人上拟合 q95 尺度，再按原规则在完整验证集上确定阈值。

| 方法 | 四视图同区域证据 |
|---|---|
| `mean_no_penalty` | 四个区域分数的算术平均，无惩罚 |
| `median_no_penalty` | 四个区域分数的中位数，无惩罚 |
| `lcf_cltc_spr` | 中位数 / (1 + MAD / (中位数绝对值 + 1e-8))，使用原完整 C 结果 |

最终均使用 `(1-beta) × 原分数 + beta × q0 × 证据 / qe`，当前四库 beta=0.25、ROI=10%。偶数中位数取中间两项平均。中位数使用已核验的原始缓存值；均值按原区域分数的 FP32 精度计算。脚本会核对缓存的中位数、MAD 和完整 C 证据能否由四个区域分数复现。

完整 C 与无惩罚中位数的对照用于观察 MAD 惩罚；无惩罚中位数与无惩罚平均值的对照用于观察聚合方式。完整 C 与平均值同时存在两处差异，不能将差异全部归因于其中一步。

## 独立运行

在项目目录与原 `torch_env` 环境中分别运行：

```bash
python LCF-CLTC-SPR-ABLATION/ablate_lcc_spr.py
```

```bash
python LCF-CLTC-SPR-ABLATION/ablate_oulu_spr.py
```

```bash
python LCF-CLTC-SPR-ABLATION/ablate_nuaa_spr.py
```

```bash
python LCF-CLTC-SPR-ABLATION/ablate_replay_spr.py
```

每条命令同时计算该数据集的两项消融。默认 SPR 源目录与此前 TTA 脚本相同，也可以用 `--spr-run` 指定其他完整 SPR 实验。`--baseline-run` 支持搬迁后的原始结果目录。默认输出到 `results/<数据集>_spr_ablation_<时间戳>/`；`--results-dir` 可指定一个不存在的目录。

本目录复用相邻 `LCF-CLTC-SPR-TTA/_runner.py` 的清单校验和历史指标实现，运行时需保留该目录。无需 GPU、图像读取、训练或模型推理。

输出包括六项指标的 `comparison.csv`、`metrics.json`、验证集校准记录、逐图分数和输入哈希。`full_minus_median_no_penalty_pp` 是完整 C 减去无惩罚中位数，`median_minus_mean_pp` 是无惩罚中位数减去无惩罚均值。单位为百分点；AUROC 等为正表示前者更好，HTER 为负表示前者更好。

本次已完成的结果见 [RESULTS.md](RESULTS.md)。
