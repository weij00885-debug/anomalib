"""
Anomalib 多模型多数据集训练脚本
支持人脸反欺诈数据集（CASIA-FASD 等）和 anomalib 内置图像模型

使用方法：
  python train_models.py --models Dinomaly --datasets Casia --max-steps 5000
  python train_models.py --models all --datasets all --max-steps 5000
  python train_models.py --list-models
  python train_models.py --list-datasets
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import list_models
from anomalib.pre_processing import PreProcessor

# ============================================================
# 可用图像模型列表（排除视频模型 AiVad, Fuvas）
# ============================================================
IMAGE_MODELS = [
    "CFM",
    "L2BT",
    "AnomalyDINO",
    "AnomalyVFM",
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Dinomaly",
    "Draem",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Fre",
    "Ganomaly",
    "GeneralAD",
    "Glass",
    "InpFormer",
    "Padim",
    "Patchcore",
    "Patchflow",
    "ReverseDistillation",
    "Stfpm",
    "SuperADD",
    "Supersimplenet",
    "Uflow",
    "UniNet",
    "VlmAd",
    "WinClip",
]

# ============================================================
# 人脸反欺诈数据集配置
# ============================================================
# 每个数据集的配置：
#   root: 数据集根目录
#   train_normal: 训练集正常图片目录（相对于 root）
#   test_normal: 测试集正常图片目录（相对于 root）
#   test_abnormal: 测试集异常图片目录（相对于 root）
#   organizer: 数据集整理函数名（如果需要先整理原始数据）
#   organizer_config: 整理函数需要的配置
# ============================================================
FACE_DATASETS = {
    "Casia": {
        "root": "/mnt/c/Users/heqi/Desktop/CASIA-FASD/casia-fasd",
        "train_normal": "train/normal",
        "test_normal": "test/normal",
        "test_abnormal": "test/abnormal",
        "organizer": "organize_casia_flat_test",
        "organizer_config": {},
    },
    "OuluNpu": {
        "root": "/mnt/d/BaiduNetdiskDownload/OULU-NPU_organized",
        "train_normal": "train/normal",
        "test_normal": "test/normal",
        "test_abnormal": "test/abnormal",
        "organizer": "organize_oulu_npu",
        "organizer_config": {
            "src_root": "/mnt/d/BaiduNetdiskDownload/OULU-NPU",
            "samples_per_folder": 5,  # 每个视频取几帧
        },
    },
}


# ============================================================
# 工具函数
# ============================================================

def print_separator(title: str = "") -> None:
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print(f"{'=' * 60}")


def organize_casia_flat_test(dataset_root: str, config: dict) -> None:
    """
    CASIA-FASD: 如果 test 目录下图片是混在一起的（通过文件名区分标签），
    就整理成 normal/ 和 abnormal/ 两个子目录。
    """
    test_dir = Path(dataset_root) / "test"
    normal_dir = test_dir / "normal"
    abnormal_dir = test_dir / "abnormal"

    # 已经整理好了
    if normal_dir.exists() and abnormal_dir.exists():
        n = len(list(normal_dir.glob("*.jpg"))) + len(list(normal_dir.glob("*.png")))
        a = len(list(abnormal_dir.glob("*.jpg"))) + len(list(abnormal_dir.glob("*.png")))
        if n > 0 and a > 0:
            print(f"  test 集已整理: normal={n}, abnormal={a}")
            return

    print("  整理 CASIA-FASD test 集...")
    normal_dir.mkdir(exist_ok=True)
    abnormal_dir.mkdir(exist_ok=True)

    # 文件名格式: {train/test}_release_{人}_{类型}_{编号}.jpg
    # 类型: 1=真人(normal), 其他=假脸(abnormal)
    pattern = re.compile(r"_(\d+|HR_\d+)_\d+\.(jpg|png)$")

    normal_count = 0
    abnormal_count = 0

    for img_path in list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")):
        match = pattern.search(img_path.name)
        if not match:
            continue

        attack_type = match.group(1)

        if attack_type == "1":
            dst = normal_dir / img_path.name
            normal_count += 1
        else:
            dst = abnormal_dir / img_path.name
            abnormal_count += 1

        shutil.move(str(img_path), str(dst))

    print(f"    完成: normal={normal_count}, abnormal={abnormal_count}")


def organize_oulu_npu(dataset_root: str, config: dict) -> None:
    """
    OULU-NPU: 从原始格式（a_b_c_d 文件夹）整理成 Folder 需要的格式。

    原始结构:
      src_root/
      ├── train/train/
      │   ├── 1_1_01_1/   ← d=1 真人
      │   ├── 1_1_01_2/   ← d=2 假脸
      │   └── ...
      ├── test/test/
      │   └── ...
      └── dev/dev/
          └── ...

    整理后:
      dataset_root/
      ├── train/normal/
      └── test/
          ├── normal/
          └── abnormal/
    """
    src_root = Path(config["src_root"])
    dst_root = Path(dataset_root)
    samples_per_folder = config.get("samples_per_folder", 5)

    # 已经整理好了
    if (dst_root / "train" / "normal").exists() and \
       (dst_root / "test" / "normal").exists() and \
       (dst_root / "test" / "abnormal").exists():

        train_n = len(list((dst_root / "train" / "normal").glob("*.jpg")))
        test_n = len(list((dst_root / "test" / "normal").glob("*.jpg")))
        test_a = len(list((dst_root / "test" / "abnormal").glob("*.jpg")))
        if train_n > 0 and test_n > 0 and test_a > 0:
            print(f"  数据集已整理: train_normal={train_n}, test_normal={test_n}, test_abnormal={test_a}")
            return

    print(f"  整理 OULU-NPU 数据集 (每文件夹取 {samples_per_folder} 帧)...")

    def _copy_split(src_split_dir: Path, dst_normal: Path, dst_abnormal: Path | None, split_name: str):
        """整理一个 split"""
        dst_normal.mkdir(parents=True, exist_ok=True)
        if dst_abnormal:
            dst_abnormal.mkdir(parents=True, exist_ok=True)

        normal_count = 0
        abnormal_count = 0

        folders = [f for f in src_split_dir.iterdir() if f.is_dir()]

        for folder in folders:
            name = folder.name
            parts = name.split("_")
            if len(parts) != 4:
                continue

            d = parts[3]  # 类型: 1=真人, 其他=假脸
            is_normal = (d == "1")

            # 训练集只用正常样本
            if dst_abnormal is None and not is_normal:
                continue

            img_files = sorted(folder.glob("*.jpg"))
            if samples_per_folder > 0:
                img_files = img_files[:samples_per_folder]

            for i, img_path in enumerate(img_files):
                new_name = f"{name}_{i:03d}.jpg"
                if is_normal:
                    shutil.copy2(img_path, dst_normal / new_name)
                    normal_count += 1
                elif dst_abnormal:
                    shutil.copy2(img_path, dst_abnormal / new_name)
                    abnormal_count += 1

        if dst_abnormal:
            print(f"    {split_name}: normal={normal_count}, abnormal={abnormal_count}")
        else:
            print(f"    {split_name}: normal={normal_count}")

    # train 集（只用正常）
    train_src = src_root / "train" / "train"
    if train_src.exists():
        _copy_split(train_src, dst_root / "train" / "normal", None, "train")

    # test 集（正常 + 异常）
    test_src = src_root / "test" / "test"
    if test_src.exists():
        _copy_split(test_src, dst_root / "test" / "normal", dst_root / "test" / "abnormal", "test")


# 整理函数字典，供数据集配置调用
ORGANIZERS = {
    "organize_casia_flat_test": organize_casia_flat_test,
    "organize_oulu_npu": organize_oulu_npu,
}


def get_model_class(model_name: str):
    """根据模型名获取模型类"""
    import anomalib.models as models_module

    # 直接匹配
    if hasattr(models_module, model_name):
        return getattr(models_module, model_name)

    # 大小写不敏感匹配
    model_lower = model_name.lower()
    for name in IMAGE_MODELS:
        if name.lower() == model_lower:
            return getattr(models_module, name)

    raise ValueError(f"未知模型: {model_name}，可用模型: {', '.join(IMAGE_MODELS)}")


def build_datamodule(dataset_name: str, image_size: tuple, batch_size: int, num_workers: int):
    """构建数据模块"""
    if dataset_name not in FACE_DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}，可用数据集: {', '.join(FACE_DATASETS.keys())}")

    cfg = FACE_DATASETS[dataset_name]
    root = Path(cfg["root"])

    # 如果有整理函数，先调用整理（自动判断是否需要整理）
    organizer_name = cfg.get("organizer")
    if organizer_name and organizer_name in ORGANIZERS:
        organizer = ORGANIZERS[organizer_name]
        organizer_config = cfg.get("organizer_config", {})
        organizer(str(root), organizer_config)

    # 自定义预处理（人脸直接 resize，不 CenterCrop）
    face_preprocessor = PreProcessor(
        transform=Compose([
            Resize(image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    datamodule = Folder(
        name=dataset_name,
        root=str(root),
        normal_dir=cfg["train_normal"],
        abnormal_dir=cfg["test_abnormal"],
        normal_test_dir=cfg["test_normal"],
        mask_dir=None,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        val_split_mode="from_train",
        val_split_ratio=0.1,
    )

    return datamodule, face_preprocessor


# ============================================================
# 单个模型 + 单个数据集的训练
# ============================================================

def train_one(model_name: str, dataset_name: str, args) -> dict:
    """训练一个模型在一个数据集上，返回测试结果"""
    print_separator(f"{model_name} + {dataset_name}")

    # 1. 数据集
    print("加载数据集...")
    datamodule, pre_processor = build_datamodule(
        dataset_name=dataset_name,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2. 模型
    print(f"创建模型: {model_name}...")
    model_class = get_model_class(model_name)

    # 不同模型参数不同，这里用通用方式传参
    # Dinomaly 支持 pre_processor 参数，其他模型可能不支持，用 try 处理
    try:
        model = model_class(pre_processor=pre_processor)
    except TypeError:
        # 模型不接受 pre_processor 参数，用默认的
        model = model_class()
        print(f"  [提示] {model_name} 不支持自定义 pre_processor，使用默认预处理")

    # 3. 引擎
    result_dir = f"{args.results_dir}/{model_name}/{dataset_name}"
    engine = Engine(
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=result_dir,
        gradient_clip_val=0.1,
        strategy="auto",
    )

    # 4. 训练
    print(f"开始训练 (max_steps={args.max_steps})...")
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_time = time.time() - t0
    print(f"训练完成，用时: {train_time/60:.1f} 分钟")

    # 5. 测试
    print("开始测试...")
    results = engine.test(model=model, datamodule=datamodule)

    # 6. 整理结果
    result = results[0] if results else {}
    result["train_time_min"] = round(train_time / 60, 1)
    result["model"] = model_name
    result["dataset"] = dataset_name

    print(f"\n结果:")
    for k, v in result.items():
        if k not in ("model", "dataset"):
            print(f"  {k}: {v}")

    return result


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Anomalib 多模型多数据集训练脚本")

    # 列表查询
    parser.add_argument("--list-models", action="store_true", help="列出所有可用图像模型")
    parser.add_argument("--list-datasets", action="store_true", help="列出所有可用数据集")

    # 模型和数据集
    parser.add_argument("--models", nargs="+", default=["Dinomaly"],
                        help='模型列表，或 "all" 使用所有图像模型（默认: Dinomaly）')
    parser.add_argument("--datasets", nargs="+", default=["Casia"],
                        help='数据集列表，或 "all" 使用所有数据集（默认: Casia）')

    # 训练参数
    parser.add_argument("--max-steps", type=int, default=5000, help="最大训练步数（默认: 5000）")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size（默认: 8）")
    parser.add_argument("--image-size", type=int, default=392, help="图片边长（默认: 392，必须是 14 的倍数）")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数（默认: 4）")

    # 硬件
    parser.add_argument("--accelerator", default="auto",
                        choices=["auto", "cpu", "gpu"], help="加速器（默认: auto）")
    parser.add_argument("--devices", type=int, default=1, help="GPU 数量（默认: 1）")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument("--results-dir", default="./results", help="结果保存目录（默认: ./results）")

    args = parser.parse_args()

    # --- 列出可用模型 ---
    if args.list_models:
        print_separator("可用图像模型")
        for i, name in enumerate(sorted(IMAGE_MODELS), 1):
            print(f"  {i:2d}. {name}")
        print(f"\n共 {len(IMAGE_MODELS)} 个图像模型（排除视频模型 AiVad, Fuvas）")
        return

    # --- 列出可用数据集 ---
    if args.list_datasets:
        print_separator("可用数据集")
        for i, name in enumerate(sorted(FACE_DATASETS), 1):
            cfg = FACE_DATASETS[name]
            print(f"  {i:2d}. {name}")
            print(f"      路径: {cfg['root']}")
        print(f"\n共 {len(FACE_DATASETS)} 个数据集")
        return

    # --- 解析模型列表 ---
    if args.models == ["all"]:
        model_names = sorted(IMAGE_MODELS)
    else:
        model_names = args.models

    # --- 解析数据集列表 ---
    if args.datasets == ["all"]:
        dataset_names = sorted(FACE_DATASETS.keys())
    else:
        dataset_names = args.datasets

    # --- 打印配置 ---
    print_separator("训练配置")
    print(f"  模型 ({len(model_names)} 个): {', '.join(model_names)}")
    print(f"  数据集 ({len(dataset_names)} 个): {', '.join(dataset_names)}")
    print(f"  最大步数: {args.max_steps}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  图片大小: {args.image_size}x{args.image_size}")
    print(f"  加速器: {args.accelerator}")
    print(f"  GPU 数量: {args.devices}")
    print(f"  随机种子: {args.seed}")
    print(f"  结果目录: {args.results_dir}")

    # 检查 GPU
    if args.accelerator in ("auto", "gpu"):
        print(f"\n  GPU 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    # --- 逐个训练 ---
    all_results = []
    total_start = time.time()

    for model_name in model_names:
        for dataset_name in dataset_names:
            try:
                result = train_one(model_name, dataset_name, args)
                all_results.append(result)
            except Exception as e:
                print(f"\n  [错误] {model_name} + {dataset_name} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "error": str(e),
                })

    # --- 汇总结果 ---
    total_time = time.time() - total_start
    print_separator("全部完成 - 结果汇总")
    print(f"  总用时: {total_time/60:.1f} 分钟")
    print()

    # 打印表格
    if all_results:
        # 表头
        header = f"{'模型':<20} {'数据集':<12} {'AUROC':>8} {'F1':>8} {'用时(min)':>10}"
        print(header)
        print("-" * len(header))

        for r in all_results:
            if "error" in r:
                print(f"{r['model']:<20} {r['dataset']:<12} {'ERROR':>8} {'':>8} {'':>10}")
            else:
                auroc = f"{r.get('image_AUROC', 'N/A'):.4f}" if isinstance(r.get('image_AUROC'), float) else str(r.get('image_AUROC', 'N/A'))
                f1 = f"{r.get('image_F1Score', 'N/A'):.4f}" if isinstance(r.get('image_F1Score'), float) else str(r.get('image_F1Score', 'N/A'))
                t = r.get('train_time_min', 'N/A')
                print(f"{r['model']:<20} {r['dataset']:<12} {auroc:>8} {f1:>8} {t:>10}")

    print()
    print(f"结果保存在: {args.results_dir}/")
    print_separator()


if __name__ == "__main__":
    main()
