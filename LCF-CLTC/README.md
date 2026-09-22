# LCF + detached CLTC：三个数据集实验

从仓库根目录、已激活 CUDA 环境运行，直接使用 `python`，不要使用项目 CPU `.venv` 的 `uv run`。

```bash
python LCF-CLTC/train_lcc_lcf_cltc.py
python LCF-CLTC/train_nuaa_lcf_cltc.py
python LCF-CLTC/train_replay_lcf_cltc.py
```

三个入口默认均为 RGB、392、float16、batch size 1、5000 steps、seed 42、LCF 开启、CLTC 开启、
detach 开启、alpha=1。LCC 沿用 ViT-Large/1024；NUAA 和 Replay 沿用 ViT-Giant/1536。

数据根目录默认为 `/mnt/c/Users/heqi/Desktop/wj/data`，可用 `--data-root` 修改。每次运行使用独立结果目录：

- `results/lcc_lcf_cltc_s42`
- `results/nuaa_lcf_cltc_s42`
- `results/replay_lcf_cltc_s42`

脚本拒绝覆盖已有结果目录。需要重跑时通过 `--results-dir` 指定新目录。

划分规则：NUAA 使用现有 `test` 作为真人测试集，并按身份从训练真人中留出 10% 校准；LCC 和 Replay
的平铺 `test` 按原有脚本约定不使用，从 `normal` 按身份划分 70% 训练、10% 校准、20% 测试。
所有 `abnormal` 用作攻击测试。划分清单、身份组和 SHA256 会写入结果目录。

训练完成后查看 `metrics.json`，重点比较 `metrics.fused.AUROC` 与
`metrics.reconstruction.AUROC`。`calibration_scores.csv` 和 `test_scores.csv` 保存逐图三分支分数。
