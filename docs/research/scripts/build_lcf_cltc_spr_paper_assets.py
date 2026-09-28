# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6", "numpy==2.2.6", "scikit-learn==1.7.2"]
# ///
"""Render paper figures and assemble tables from recorded experiments (no inference)."""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research"
OUT = DOC / "assets/lcf_cltc_spr_paper"
LATEST = ROOT / "results/spr_roi_sweep_20260926_140600_193717"
SWEEP = ROOT / "results/spr_roi_sweep_20260926_012138_088241"
DATASETS = ("oulu", "lcc", "nuaa", "replay")
NAMES = dict(zip(DATASETS, ("OULU-NPU", "LCC-FASD", "NUAA", "Replay-Attack")))
REPORTED = OUT / "reported_metrics_20260928.csv"
STAGES = ("Dinomaly", "Dinomaly＋LCF", "Dinomaly＋LCF＋CLTC", "Dinomaly＋LCF＋CLTC＋SPR")
TTA = dict(zip(DATASETS, (
    "oulu_spr_tta_20260925_131448_260792", "lcc_spr_tta_20260925_131446_701614",
    "nuaa_spr_tta_20260925_131449_592189", "replay_spr_tta_20260925_131451_221673",
)))
ABL = dict(zip(DATASETS, (
    "oulu_spr_ablation_20260925_153517_111896", "lcc_spr_ablation_20260925_153515_587466",
    "nuaa_spr_ablation_20260925_153518_742563", "replay_spr_ablation_20260925_153520_455646",
)))
METRICS = ("AUROC", "Accuracy", "F1Score", "Precision", "Recall", "HTER")
SOURCES: dict[str, str] = {}
COL = {"ink": "#203448", "muted": "#5B6B7A", "line": "#B8C6D0",
       "blue": "#3979AD", "blue_bg": "#ECF3F9", "teal": "#268B87", "teal_bg": "#EAF6F3",
       "purple": "#8666AF", "purple_bg": "#F2EDF8", "orange": "#C58B38", "orange_bg": "#FCF3E6"}


def record(path: Path) -> None:
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    record(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    record(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers: list[str], rows: list[list]) -> str:
    def fmt(value: object) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(fmt, row)) + " |" for row in rows]) + "\n"


def reported_tables() -> str:
    """Preserve user-reported metrics and distinguish arithmetic checks from corrections."""
    rows = read_csv(REPORTED)
    read_json(OUT / "reported_metrics_20260928_provenance.json")
    assert len(rows) == 16
    index = {(row["dataset"], int(row["stage"])): row for row in rows}
    assert set(index) == {(ds, stage) for ds in DATASETS for stage in range(4)}
    parts = [
        "来源：2026-09-28 用户从日志抄录的结果；本次未重新训练或模型推理。"
        "用户确认不清楚评估脚本的统计方式，故保留全部原值，聚合口径待核对。"
        "下表按用户上下文作为测试结果整理，未提供新的验证指标、逐行 checkpoint、manifest 或阈值。"
        "原记录 laf/ctlc 暂对应既有实现 LCF/CLTC，未改名模型。所有数值为 0–1；"
        "AUROC、Accuracy、F1、Precision、Recall 越高越好，HTER 越低越好。\n"
    ]
    for ds in ("lcc", "oulu", "nuaa", "replay"):
        parts.extend([f"**{NAMES[ds]}**\n", table(
            ["方法", "AUROC ↑", "Accuracy ↑", "F1 ↑", "Precision ↑", "Recall ↑", "HTER ↓"],
            [[STAGES[stage], *[index[ds, stage][m] for m in METRICS]] for stage in range(4)],
        )])
    macro = [[STAGES[stage], *[
        str(sum(Decimal(index[ds, stage][m]) for ds in DATASETS) / Decimal(4))
        for m in METRICS
    ]] for stage in range(4)]
    parts.extend(["**四数据集等权宏平均（由上述汇总数计算）**\n",
                  table(["方法", *METRICS], macro),
                  "这里的宏平均仅指四个数据集等权，不表示原始日志内部采用 macro 指标；"
                  "不是合并全部测试样本的 pooled 指标，也不是多训练 seed 均值。\n"])
    deltas = []
    for ds in DATASETS:
        values = [NAMES[ds]]
        for stage in (1, 2, 3):
            for metric in ("AUROC", "HTER"):
                change = 100 * (Decimal(index[ds, stage][metric]) - Decimal(index[ds, stage-1][metric]))
                values.append(f"{change:+.2f}")
        deltas.append(values)
    parts.extend(["**逐阶段差值（百分点，相对紧邻前一配置）**\n", table(
        ["数据集", "+LCF ΔAUROC", "+LCF ΔHTER", "+CLTC ΔAUROC", "+CLTC ΔHTER", "+SPR ΔAUROC", "+SPR ΔHTER"], deltas),
        "差值是日志汇总值的算术比较，不自动证明仅改变一个模块；同拆分、骨干、训练预算和阈值策略仍须核对。\n"])
    audit = []
    for row in rows:
        p, r, actual = (Decimal(row[m]) for m in ("Precision", "Recall", "F1Score"))
        pooled = 2*p*r/(p+r)
        if abs(pooled-actual) > Decimal("0.0003"):
            audit.append([NAMES[row["dataset"]], STAGES[int(row["stage"])], row["F1Score"], f"{pooled:.6f}"])
    parts.extend(["**统计口径核对（不是替换后的成绩）**\n",
        "只有在同一批预测、同一正类、同一阈值的整体二分类计算下，才应满足 F1=2PR/(P+R)。"
        "下列行的差异超过 0.0003，不能简单归于所示精度的取整；"
        "分折后平均、类别宏平均或不同运行来源均需检查，当前不能确定原因。\n",
        table(["数据集", "方法", "日志 F1（保留）", "整体二分类公式参考值"], audit),
        "上表参考值仅用于发现口径差异，不是重新评估所得的 F1。投稿前应逐行关联原日志、"
        "评估代码、正类、阈值、样本单位和 average 参数。完整方法与旧 SPR 结果数值接近，"
        "也不意味着其他阶段继承旧实验的元数据。\n"])
    return "\n".join(parts)


def threshold(labels, scores, ds):
    """Match the saved legacy threshold rules, including strict-greater ties."""
    if ds == "oulu":
        precision, recall, candidates = precision_recall_curve(labels, scores)
        f1 = 2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-10)
        return float(candidates[int(np.argmax(f1))])
    candidates = np.unique(scores)
    normal, attack = np.sort(scores[labels == 0]), np.sort(scores[labels == 1])
    fpr = (len(normal)-np.searchsorted(normal,candidates,side="right"))/len(normal)
    fnr = np.searchsorted(attack,candidates,side="right")/len(attack)
    hter = (fpr+fnr)/2
    return float(candidates[np.flatnonzero(hter == hter.min())[-1]])


def cached_current_comparisons(final):
    """Derive matched current controls solely from saved view/region scalars."""
    views = ("original", "flip", "gamma09", "gamma11")
    title = {"original_lcf_cltc": "原 S0", "original_region": "仅原图同 ROI 均值",
             "tta_mean_4views": "普通四视图 TTA", "mean_no_penalty": "同 ROI 四视图均值",
             "median_no_penalty": "同 ROI 四视图中位数", "lcf_cltc_spr": "完整 SPR（median＋MAD）"}
    lines = ["## H. 历史归档配置的缓存对照：ROI=30%，β=1.0\n\n"
             "本节是 2026-09-27 写作整理时，由 2026-09-26 重跑保存的逐视图/区域分数离线计算；"
             "没有新增网络前向或训练。各方法仅在验证集拟合尺度与分类阈值，再评估固定测试清单。"
             "原 S0 与完整 SPR 的六指标均已与 summary.csv 核对。"
             "原图 ROI 对照只用一个视图；其余新对照使用相同四视图。"
             "β=1 的尺度对齐 TTA 与普通 TTA 排序等价。\n"]
    all_rows, calibration = [], {}
    for ds in DATASETS:
        splits = {s: read_csv(LATEST/ds/f"{s}_scores.csv") for s in ("validation", "test")}
        arrays = {}
        for split, rows in splits.items():
            val = lambda key: np.array([float(r[key]) for r in rows])
            region = np.stack([val(f"region_{v}_roi_30") for v in views],1).astype(np.float32)
            median = np.quantile(region,.5,axis=1)
            mad = np.quantile(abs(region-median[:,None]),.5,axis=1)
            np.testing.assert_allclose(median,val("spr_roi_30_median"),rtol=2e-6,atol=1e-7)
            np.testing.assert_allclose(mad,val("spr_roi_30_mad"),rtol=2e-6,atol=1e-7)
            np.testing.assert_allclose(median/(1+mad/(abs(median)+1e-8)),val("spr_roi_30_evidence"),rtol=2e-6,atol=1e-7)
            arrays[split] = {"labels":val("label").astype(int), "original_lcf_cltc":val("original_fused"),
                             "original_region":region[:,0].astype(float),
                             "tta_mean_4views":np.stack([val(f"view_{v}_fused") for v in views],1).mean(1),
                             "mean_no_penalty":region.mean(1,dtype=np.float32).astype(float),
                             "median_no_penalty":val("spr_roi_30_median"), "lcf_cltc_spr":val("spr_roi_30_evidence")}
        vy, ty = arrays["validation"]["labels"], arrays["test"]["labels"]
        q0 = float(np.quantile(arrays["validation"]["original_lcf_cltc"][vy==0],.95))
        rows, calibration[ds] = [], {}
        for method in title:
            vs, ts = arrays["validation"][method], arrays["test"][method]
            qe = float(np.quantile(vs[vy==0],.95))
            if method not in ("original_lcf_cltc", "tta_mean_4views"):
                vs, ts = q0*vs/qe, q0*ts/qe
            cut = threshold(vy,vs,ds)
            pred = ts>cut
            vals=[roc_auc_score(ty,ts),accuracy_score(ty,pred),f1_score(ty,pred),
                  precision_score(ty,pred),recall_score(ty,pred),
                  .5*(pred[ty==0].mean()+(~pred[ty==1]).mean())]
            vals=list(map(float,vals))
            if method in ("original_lcf_cltc", "lcf_cltc_spr"):
                saved = next(r for r in final if r["dataset"]==ds)
                prefix = "baseline_image_" if method=="original_lcf_cltc" else "test_image_"
                np.testing.assert_allclose(vals,[float(saved[prefix+m]) for m in METRICS],rtol=0,atol=1e-12)
            rows.append([title[method],*vals])
            all_rows.append({"dataset":ds,"method":method,"validation_AUROC":float(roc_auc_score(vy,vs)),
                             **dict(zip(METRICS,vals))})
            calibration[ds][method]={"q0":q0,"qe":qe,"threshold":cut,"policy":"validation_f1" if ds=="oulu" else "validation_hter"}
        lines.extend([f"### {NAMES[ds]}\n",table(["方案",*METRICS],rows)])
    with (OUT/"current_cached_comparisons.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT/"current_cached_calibration.json").write_text(json.dumps({"additional_network_forwards":0,
        "source_run":LATEST.relative_to(ROOT).as_posix(),"rho":.3,"beta":1.0,
        "region_mean_dtype":"float32","calibration":calibration},indent=2),encoding="utf-8")
    return lines


def evidence_tables() -> None:
    """Create a data appendix with exact file provenance, without changing results."""
    final = read_csv(LATEST / "summary.csv")
    sweep = read_csv(SWEEP / "summary.csv")
    old30 = {r["dataset"]: r for r in sweep if float(r["roi_fraction"]) == .3}
    assert all(r == old30[r["dataset"]] for r in final), "ROI-30 rerun differs from original."
    reported = reported_tables()
    paper = DOC / "dinomaly_lcf_cltc_spr_paper.md"
    start, end = "<!-- REPORTED_METRICS_START -->", "<!-- REPORTED_METRICS_END -->"
    body = paper.read_text(encoding="utf-8")
    assert body.count(start) == body.count(end) == 1
    before, remaining = body.split(start)
    _, after = remaining.split(end)
    paper.write_text(before + start + "\n\n" + reported + "\n" + end + after, encoding="utf-8")
    parts = ["# LCF＋CLTC＋SPR 论文数据附录\n",
             "版本：2026-09-28。由脚本整理，未重新训练或推理。"
             "最新日志摘录保留用户精度；A–I 为历史归档证据，显示六位小数。"
             "两类来源不可混合作配对增益。配合 [论文主文档](dinomaly_lcf_cltc_spr_paper.md) 阅读。\n",
             "## 最新日志摘录：四库四阶段结果与口径核对\n", reported,
             "## A. 历史 2026-09-26 完整方法：ROI=30%，β=1.0\n"]
    parts.append(table(["数据集", "验证 AUROC", *METRICS], [
        [NAMES[r["dataset"]], float(r["validation_AUROC"]),
         *[float(r[f"test_image_{m}"]) for m in METRICS]] for r in final]))
    parts.append("来源：`results/spr_roi_sweep_20260926_140600_193717/summary.csv`。"
                 "与上一轮的 ROI=30% 行逐字段完全一致。这是固定 checkpoint 的重复评估。\n")
    parts.append("## B. 历史同 checkpoint 的 SPR 增量（不对应最新四阶段基线）\n")
    parts.append(table(["数据集", "原 S0 AUROC", "SPR AUROC", "AUROC Δ/百分点", "原 HTER", "SPR HTER", "HTER Δ/百分点"], [
        [NAMES[r["dataset"]], float(r["baseline_image_AUROC"]), float(r["test_AUROC"]),
         100*(float(r["test_AUROC"])-float(r["baseline_image_AUROC"])),
         float(r["baseline_image_HTER"]), float(r["test_image_HTER"]),
         100*(float(r["test_image_HTER"])-float(r["baseline_image_HTER"]))] for r in final]))
    parts.append("## C. 模块与 TTA 对照：历史 ROI=10%，β=0.25\n\n"
                 "本节全部使用历史配置，不是当前 ROI=30%、β=1 的消融。对照共享四视图缓存。\n")
    combined = []
    for ds in DATASETS:
        a = {r["metric"]: r for r in read_csv(ROOT / "results" / TTA[ds] / "comparison.csv")}
        b = {r["metric"]: r for r in read_csv(ROOT / "results" / ABL[ds] / "comparison.csv")}
        entries = [("原 LCF＋CLTC", "original_lcf_cltc", a), ("普通四视图 TTA", "tta_mean_4views", a),
                   ("同比例融合 TTA", "tta_blend_4views", a), ("同 ROI 均值，无惩罚", "mean_no_penalty", b),
                   ("同 ROI 中位数，无惩罚", "median_no_penalty", b), ("完整 SPR", "lcf_cltc_spr", b)]
        rows = []
        for title, key, source in entries:
            values = [float(source[f"image_{m}"][key]) for m in METRICS]
            rows.append([title, *values])
            combined.append({"dataset": ds, "setting": "rho=.1,beta=.25", "method": key,
                             **dict(zip(METRICS, values))})
        parts.extend([f"### {NAMES[ds]}\n", table(["方案", *METRICS], rows),
                      f"来源：`results/{TTA[ds]}/comparison.csv` 与 `results/{ABL[ds]}/comparison.csv`。\n"])
    parts.append("## D. 历史 ROI 敏感性：β=1.0，四库完整指标\n")
    for ds in DATASETS:
        rows = [r for r in sweep if r["dataset"] == ds]
        parts.extend([f"### {NAMES[ds]}\n", table(["ROI/%", "验证 AUROC", *METRICS], [
            [float(r["roi_percent"]), float(r["validation_AUROC"]),
             *[float(r[f"test_image_{m}"]) for m in METRICS]] for r in rows])])
    macro = read_csv(SWEEP / "macro_summary.csv")
    parts.append(table(["ROI/%", "验证 AUROC 宏平均", "测试 AUROC 宏平均"], [
        [100*float(r["roi_fraction"]), float(r["validation_AUROC_macro"]), float(r["test_AUROC_macro"])] for r in macro]))
    parts.append("宏平均对四个数据集等权。只支持在已测五档范围内比较；不能说明 30% 是所有比例的全局最优。\n")
    parts.append("## E. 早期 CLTC 分支实验：独立正常 dev 协议\n")
    early = read_json(ROOT / "results/oulu_cltc_s42_gpu/metrics.json")
    parts.append(table(["分支", "AUROC", "近似 EER", "正常 dev q95 阈值", "APCER pooled", "BPCER", "ACER pooled"], [
        [key, *[float(val[k]) for k in ("AUROC", "EER_nearest_test_ROC", "normal_dev_q95_threshold",
                                       "APCER_pooled", "BPCER", "ACER_pooled")]] for key, val in early["metrics"].items()]))
    parts.append("来源：`results/oulu_cltc_s42_gpu/metrics.json`，协议 `custom_frame_level_folder`。"
                 "与历史 OULU legacy 清单不同，禁止与 A/B 表直接计算模块提升。"
                 "用户历史提供的纯 LCF=0.8685 尚缺配套指标、清单和配置，暂列待核验，不能充当已验证消融。\n")
    parts.append("## F. 已保存的 LCF＋CLTC 训练探索记录\n\n"
                 "以下列出结果目录顶层现有、且 metrics 为单套 image 指标的 CLTC 训练记录；"
                 "排除 SPR/TTA 后处理。不同 seed 的数据清单不一定相同，不能直接解释为固定划分多种子误差条。\n")
    inventory, full_inventory = [], []
    for path in sorted((ROOT / "results").glob("*lcf_cltc*/metrics.json")):
        cfgpath = path.parent / "config.json"
        if "spr" in path.parent.name or not cfgpath.exists():
            continue
        report, cfg = read_json(path), read_json(cfgpath)
        met = report.get("metrics", {})
        if "image_AUROC" not in met:
            continue
        hashes = report.get("manifest_sha256", {})
        inventory.append([path.parent.name, cfg.get("seed", "—"), cfg.get("max_steps", "—"),
                          cfg.get("train_batch_size", "—"), cfg.get("model_precision", "—"),
                          cfg.get("cltc_loss_weight", "—"), cfg.get("cltc_score_weight", "—"),
                          met["image_AUROC"], met["image_HTER"], hashes.get("test", "未记录")[:12]])
        full_inventory.append({"run": path.parent.name, "config": cfg, "report": report})
    parts.append(table(["运行目录", "seed", "steps", "batch", "precision", "λ", "α", "AUROC", "HTER", "test SHA256 前12位"], inventory))
    parts.append("### F.1 上述训练探索的完整六指标\n")
    parts.append(table(["运行目录", *METRICS], [
        [item["run"], *[float(item["report"]["metrics"][f"image_{m}"]) for m in METRICS]]
        for item in full_inventory
    ]))
    parts.append("上述每行全部六项指标、完整配置和 manifest 哈希另存于 "
                 "[training_inventory.json](assets/lcf_cltc_spr_paper/training_inventory.json)。"
                 "目录存在但缺少 metrics 的运行不能算已完成实验。历史探索不自动成为最终方法的配对消融。\n")
    parts.append("## G. 历史归档配置、尺度、样本数量与运行代价\n\n"
                 "下列配置属于 2026-09-26 归档模型，不是对最新日志摘录 16 行元数据的核验。\n")
    counts, configs, costs, scales = [], [], [], []
    for ds in DATASETS:
        cfg = read_json(LATEST / ds / "config.json")
        runtime = read_json(LATEST / ds / "runtime.json")
        cc = cfg["source_config"]
        source = ROOT / "results" / Path(cfg["source_run"]).name
        src_report = read_json(source / "metrics.json")
        qrec, qtraj = src_report["score_scales"]
        source_counts = []
        for split in ("train", "validation", "test"):
            path = source / f"{split}_manifest.csv"
            rows = read_csv(path)
            label_key = "label_index" if "label_index" in rows[0] else "label"
            normal = sum(int(row[label_key]) == 0 for row in rows)
            source_counts.append((len(rows), normal, len(rows)-normal))
        for split, vals in zip(("train", "validation", "test"), source_counts):
            counts.append([NAMES[ds], split, *vals])
        configs.append([NAMES[ds], cc["encoder_name"], cc["max_steps"], cc["train_batch_size"],
                        cc["model_precision"], cc["cltc_loss_weight"], cc["cltc_score_weight"], cfg["eval_batch_size"]])
        row = next(r for r in final if r["dataset"] == ds)
        scales.append([NAMES[ds], qrec, qtraj, float(row["validation_q0"]), float(row["validation_qe"]), float(row["validation_threshold"])])
        for split in ("validation", "test"):
            cost = runtime[split]
            costs.append([NAMES[ds], split, cost["images"], cost["network_image_forwards"],
                          cost["wall_seconds_including_io"]/60, cost["seconds_per_image_including_io"],
                          cost["peak_cuda_allocated_bytes"]/1024**3, cost["max_abs_original_score_error"]])
    parts.append(table(["数据集", "骨干", "steps", "训练 batch", "配置精度", "λ", "α", "本轮评估 batch"], configs))
    parts.append(table(["数据集", "划分", "总帧数", "真人", "攻击"], counts))
    parts.append(table(["数据集", "q_rec", "q_traj", "q0", "qe", "验证阈值"], scales))
    parts.append(table(["数据集", "划分", "帧数", "单图前向次数", "含 I/O 分钟", "秒/图", "峰值 CUDA GiB", "原 S0 最大绝对偏差"], costs))
    parts.append("原 S0 偏差来自本轮重新推理与归档分数的比较；重跑汇总指标一致不代表所有 GPU 浮点分数逐位一致。"
                 "时间包含 I/O，未统一预热与硬件负载，不能作为严格的速度排名。"
                 "配置中的 OULU float16 在当前实现实际对应 BF16 主干、FP32 CLTC head。\n")
    parts.extend(cached_current_comparisons(final))
    parts.append("## I. 来源清单与复现说明\n\n"
                 "机器可读的源文件 SHA256 见 [evidence_sources.json](assets/lcf_cltc_spr_paper/evidence_sources.json)。"
                 "该清单记录本次读到的文件快照；checkpoint 哈希直接引用各运行 config 中的已保存字段，未重新读取大型权重。"
                 "数据不一定已提交到 Git，提交前需单独归档。\n")
    (DOC / "dinomaly_lcf_cltc_spr_paper_data.md").write_text("\n".join(parts), encoding="utf-8")
    (OUT / "training_inventory.json").write_text(json.dumps(full_inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "evidence_sources.json").write_text(json.dumps(SOURCES, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "module_comparisons.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)


def canvas(width=200, height=120, chinese=False):
    font = "DejaVu Sans"
    if chinese:
        candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyh.ttc")]
        available = next((p for p in candidates if p.exists()), None)
        if available is None:
            raise RuntimeError("Chinese figure requires Microsoft YaHei; set the font path for this machine.")
        font_manager.fontManager.addfont(str(available))
        font = font_manager.FontProperties(fname=str(available)).get_name()
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(width/10, height/10), facecolor="white")
    ax = fig.add_axes([.01, .01, .98, .98])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, value, size=11, color=None, weight="normal", ha="center"):
    return ax.text(x, y, value, fontsize=size, color=color or COL["ink"], weight=weight,
                   ha=ha, va="center", linespacing=1.5, zorder=5)


def box(ax, x, y, w, h, label, theme="blue", size=10.5, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.1",
                              facecolor=COL[f"{theme}_bg"], edgecolor=COL[theme], linewidth=1.25,
                              linestyle="--" if dashed else "-", zorder=2))
    text(ax, x+w/2, y+h/2, label, size)


def arrow(ax, points, theme="ink", dashed=False):
    color = COL[theme]
    if len(points)>2:
        ax.plot(*zip(*points[:-1]), color=color, linewidth=1.25, linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.25, color=color, linestyle="--" if dashed else "-", zorder=1))


def panel(ax, x, y, w, h, title, theme):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=1.6",
                              facecolor="white", edgecolor=COL["line"], linewidth=.9, zorder=0))
    ax.add_patch(Rectangle((x+1.5, y+2), .6, 4, color=COL[theme], linewidth=0))
    text(ax, x+3.4, y+4, title, 12, weight="bold", ha="left")


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.12,
                    metadata={"Creator": "LCF-CLTC-SPR research figures"} if suffix == "pdf" else None)
    plt.close(fig)


def architecture():
    fig, ax = canvas(200, 130, chinese=True)
    text(ax, 4, 4, "Dinomaly + LCF + CLTC + SPR｜当前实现架构", 21, weight="bold", ha="left")
    text(ax, 4, 10, "单类人脸防伪 · 冻结 DINOv2 · 双分支正常性建模 · 同区域扰动复核", 11, ha="left", color=COL["muted"])
    panel(ax, 3, 16, 194, 24, "A  共享特征主干：LCF 在编码器之后", "blue")
    specs = [(6, 16, "RGB 人脸\n392 × 392", "blue"), (27, 23, "冻结 DINOv2\n抽取 8 层", "blue"),
             (55, 23, "F₀ … F₇\n每层 B × 789 × C", "blue"), (83, 23, "LCF：全层融合\nZ = Aθ(F₀ … F₇)", "teal"),
             (111, 23, "瓶颈 MLP\n训练 Dropout=0.2", "blue"), (139, 24, "8 层解码器\nD₀ → … → D₇", "blue"),
             (168, 25, "输出列表反转\nD̄ = [D₇ … D₀]", "blue")]
    for x,w,label,theme in specs:
        box(ax, x, 25, w, 11, label, theme, 9.5)
    for (x,w,*_), (nx,*_) in zip(specs, specs[1:]):
        arrow(ax, [(x+w,30.5),(nx,30.5)])
    panel(ax, 3, 44, 95, 39, "B  重建分支：两组老师—学生配对", "teal")
    box(ax, 7, 53, 40, 12, "老师 T₁=Aθ(F₀…F₃)\n老师 T₂=Aθ(F₄…F₇)", "teal", 10)
    box(ax, 54, 53, 40, 12, "学生 Y₁=Aθ(D₇…D₄)\n学生 Y₂=Aθ(D₃…D₀)", "teal", 10)
    box(ax, 21, 70, 60, 9, "训练：Lrec(sg(T), Y)\n测试：R(p)=两组逐 patch 余弦距离均值", "blue", 9.5)
    arrow(ax, [(27,65),(27,70)])
    arrow(ax, [(74,65),(74,70)])
    panel(ax, 102, 44, 95, 39, "C  CLTC：预测跨层关系，损失只更新预测头", "purple")
    box(ax, 106, 53, 41, 12, "原始 F 的 patch → 实际轨迹 τ\nτᵢ = 1 − cos(Fᵢ, Fᵢ₊₁)\n8 个抽取层 → 7 维距离", "blue", 9.5)
    box(ax, 153, 53, 40, 12, "Z 的 patch → detach → head\nLN → Linear(C,128) → GELU\n→ Linear(128,7) → 预测 τ̂", "purple", 9.1)
    box(ax, 118, 70, 67, 9, "训练：Ltraj = SmoothL1(τ̂, τ)\n测试：U(p) = 七个距离的绝对预测误差均值", "purple", 9.5)
    arrow(ax, [(127,65),(127,70)])
    arrow(ax, [(173,65),(173,70)])
    text(ax, 100, 87, "训练：L = Lrec + 0.1 Ltraj   ｜   五处 Aθ 共用同一个 LCF   ｜   轨迹不含 CLS / register", 11, weight="bold")
    panel(ax, 3, 93, 194, 29, "D  测试时 SPR：同一个已训练模型顺序处理四个视图，无新增可训练参数", "orange")
    flow = [(7,32,"原图 / 水平翻转\nγ=0.9 / γ=1.1"), (44,35,"每视图得到 Rⱼ、Uⱼ\nMⱼ = [Rⱼ]₊/qr + α[Uⱼ]₊/qt"),
            (84,33,"翻转图对齐原坐标\n原图选 top-30% ROI"), (122,32,"同 ROI 得到四个 vⱼ\n中位数 m + MAD d"),
            (159,34,"e = m / [1+d/(|m|+ε)]\n当前 S = q₀ e / qe")]
    for x,w,label in flow:
        box(ax,x,104,w,13,label,"orange",9.3)
    for (x,w,*_), (nx,*_) in zip(flow,flow[1:]):
        arrow(ax,[(x+w,110.5),(nx,110.5)])
    text(ax,100,126,"β=1：最终使用 SPR 标量；重建与 CLTC 仍在 Mⱼ 中。一般式：S=(1−β)S₀+β q₀ e/qe。",10,color=COL["muted"])
    save(fig, "architecture_current_zh")


def paper_overview():
    fig, ax = canvas(200, 127)
    text(ax, 4, 4, "Learning normality, then rechecking local evidence", 20, weight="bold", ha="left")
    text(ax, 4, 10, "Layer-conditional fusion  /  Cross-layer trajectory prediction  /  Same-region perturbation recheck", 10.5, ha="left", color=COL["muted"])
    panel(ax, 3, 16, 194, 43, "(a) Bona fide training: representation and cross-layer relations", "teal")
    box(ax,7,29,18,15,"Bona fide\nfaces",size=11)
    box(ax,31,29,26,15,"Frozen DINOv2\nF₀ … F₇",size=11)
    box(ax,65,29,26,15,"LCF\nPatch-wise\nlayer attention", "teal",11)
    box(ax,102,25,36,12,"Bottleneck + decoder\nReverse + group fusion","teal",10)
    box(ax,150,25,41,12,"Reconstruction loss\nLrec(sg(T), Y)","teal",10.5)
    box(ax,102,42,36,12,"CLTC on sg(Z)\nLN → MLP → 7 distances","purple",10)
    box(ax,150,42,41,12,"Trajectory loss\nSmoothL1(τ̂, τ)","purple",10.5)
    arrow(ax,[(25,36.5),(31,36.5)])
    arrow(ax,[(57,36.5),(65,36.5)])
    arrow(ax,[(91,33),(97,33),(97,31),(102,31)],"teal")
    arrow(ax,[(91,40),(97,40),(97,48),(102,48)],"purple")
    text(ax,96,53.5,"sg",9,color=COL["purple"])
    arrow(ax,[(138,31),(150,31)],"teal")
    arrow(ax,[(138,48),(150,48)],"purple")
    arrow(ax,[(48,29),(48,23),(170,23),(170,25)],"teal",True)
    text(ax,105,20.5,"T = shared LCF over encoder groups",9,color=COL["teal"])
    arrow(ax,[(44,44),(44,56.5),(172,56.5),(172,54)],"purple",True)
    text(ax,73,52.5,"τᵢ = 1 − cos(Fᵢ, Fᵢ₊₁)",9,color=COL["purple"])
    panel(ax,3,64,194,40,"(b) Inference: localize once, align views, and recheck the same region", "orange")
    nodes=[(7,28,"Four views\nOriginal · Flip\nγ = 0.9 · γ = 1.1","blue"),
           (41,31,"Shared trained model\nLCF + reconstruction\n+ CLTC","teal"),
           (78,31,"Dual evidence maps\nMⱼ = [Rⱼ]₊/qr\n+ α[Uⱼ]₊/qt","purple"),
           (115,34,"Align to original\nSelect Ω from M₀\nReuse Ω for all views","orange"),
           (155,38,"Robust regional evidence\nvⱼ → median + MAD\n→ calibrated score S","orange")]
    for x,w,label,theme in nodes:
        box(ax,x,76,w,19,label,theme,10.5)
    for (x,w,*_), (nx,*_) in zip(nodes,nodes[1:]):
        arrow(ax,[(x+w,85.5),(nx,85.5)])
    text(ax,100,99.5,"No additional training  •  Four forward passes per image  •  Fixed checkpoint and validation calibration",10,color=COL["muted"])
    text(ax,5,109,"Training:  L = Lrec + λ Ltraj,  λ = 0.1",11,ha="left")
    text(ax,105,109,"Current inference:  ρ = 0.30,  β = 1.0,  S = q₀ e / qe",11,ha="left")
    text(ax,5,114,"sg: stop-gradient    |    T and Y: shared-LCF teacher and student groups    |    e = m / [1 + d / (|m| + ε)]",9,ha="left",color=COL["muted"])
    text(ax,100,122,"Reported configurations: Dinomaly  |  + LCF  |  + LCF + CLTC  |  + LCF + CLTC + SPR",11,weight="bold")
    save(fig,"method_overview_en")


def roi_plot():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "svg.fonttype":"none", "pdf.fonttype":42})
    rows = read_csv(SWEEP / "summary.csv")
    fig, axes = plt.subplots(1,4,figsize=(14,3.6), constrained_layout=True)
    for ax, ds in zip(axes,DATASETS):
        subset=[r for r in rows if r["dataset"]==ds]
        xx=[float(r["roi_percent"]) for r in subset]
        for key,label,color,marker in [("validation_AUROC","Validation",COL["blue"],"o"),("test_AUROC","Test (exploratory)",COL["orange"],"s")]:
            ax.plot(xx,[float(r[key]) for r in subset],label=label,color=color,marker=marker,linewidth=1.8,markersize=4)
        ax.set(title=NAMES[ds],xlabel="ROI fraction (%)",ylabel="AUROC",xticks=xx)
        ax.ticklabel_format(axis="y",style="plain",useOffset=False)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Archived 2026-09-26 ROI sensitivity · β = 1.0 · fixed checkpoints · separate y-axis ranges",fontsize=12)
    save(fig,"roi_sensitivity")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    evidence_tables()
    architecture()
    paper_overview()
    roi_plot()
    print(f"Rendered figures and assembled {len(SOURCES)} evidence sources in {OUT}")
