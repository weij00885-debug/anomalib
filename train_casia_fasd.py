"""
CASIA-FASD 人脸反欺诈数据集 + Dinomaly 训练测试脚本

数据集结构（已整理好的）：
  casia-fasd/
  ├── train/
  │   ├── normal/      ← 真人（5434张）
  │   └── abnormal/    ← 假脸（单类训练不用）
  └── test/
      └── *.jpg        ← 混在一起，需要按文件名分类

使用方法：
  1. conda activate torch_env
  2. python train_casia_fasd.py
"""

import re
import shutil
from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Dinomaly
from anomalib.engine import Engine
from torchvision.transforms.v2 import Compose, Normalize, Resize
from anomalib.pre_processing import PreProcessor

# ============================================================
# 配置区
# ============================================================

# 数据集根目录
DATASET_ROOT = "/mnt/c/Users/heqi/Desktop/CASIA-FASD/casia-fasd"

# 结果保存路径
RESULT_DIR = "./results/casia_fasd_dinomaly"

# 训练参数
BATCH_SIZE = 8
MAX_STEPS = 5000
IMAGE_SIZE = (392, 392)  # 必须是 14 的倍数
NUM_WORKERS = 4

# GPU 配置
GPUS = 1

# 模型参数
ENCODER_NAME = "vit_base_patch14_reg4_dinov2"
BOTTLENECK_DROPOUT = 0.2
DECODER_DEPTH = 8


# ============================================================
# 整理 test 集（把混在一起的测试图片分成 normal/abnormal）
# ============================================================

def organize_test_set(dataset_root: str) -> str:
    """
    将 test 目录下混在一起的图片按文件名分成 normal 和 abnormal 两个子目录。
    返回 test 目录路径。
    """
    test_dir = Path(dataset_root) / "test"
    normal_dir = test_dir / "normal"
    abnormal_dir = test_dir / "abnormal"

    # 如果已经分好了，直接返回
    if normal_dir.exists() and abnormal_dir.exists():
        normal_count = len(list(normal_dir.glob("*.jpg")))
        abnormal_count = len(list(abnormal_dir.glob("*.jpg")))
        if normal_count > 0 and abnormal_count > 0:
            print(f"[提示] test 集已整理好: normal={normal_count}, abnormal={abnormal_count}")
            return str(test_dir)

    print("=" * 60)
    print("正在整理 test 集（按文件名分类）...")

    normal_dir.mkdir(exist_ok=True)
    abnormal_dir.mkdir(exist_ok=True)

    # 文件名格式: test_release_{人}_{类型}_{编号}.jpg
    # 类型: 1=真人, 2/3/HR_1=假脸
    pattern = re.compile(r"_(\d+|HR_\d+)_\d+\.jpg$")

    normal_count = 0
    abnormal_count = 0

    for img_path in test_dir.glob("*.jpg"):
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

    print(f"  完成: normal={normal_count} 张, abnormal={abnormal_count} 张")
    print("=" * 60)

    return str(test_dir)


# ============================================================
# 主函数
# ============================================================

def main():
    # 第一步：整理 test 集
    organize_test_set(DATASET_ROOT)

    # 第二步：自定义预处理（人脸直接 resize，不 CenterCrop）
    face_preprocessor = PreProcessor(
        transform=Compose([
            Resize(IMAGE_SIZE),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    # 第三步：创建数据集
    print("\n" + "=" * 60)
    print("加载数据集...")
    datamodule = Folder(
        name="casia_fasd",
        root=DATASET_ROOT,
        normal_dir="train/normal",       # 训练集正常图片
        abnormal_dir="test/abnormal",    # 测试集异常图片
        normal_test_dir="test/normal",   # 测试集正常图片
        mask_dir=None,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split_mode="from_train",     # 从训练集分验证集
        val_split_ratio=0.1,             # 10% 当验证集
    )
    print(f"  训练 batch size: {BATCH_SIZE}")
    print(f"  图片大小: {IMAGE_SIZE}")
    print(f"  验证集: 从训练集分 10%")
    print("=" * 60)

    # 第四步：创建模型
    print("\n创建 Dinomaly 模型...")
    model = Dinomaly(
        encoder_name=ENCODER_NAME,
        bottleneck_dropout=BOTTLENECK_DROPOUT,
        decoder_depth=DECODER_DEPTH,
        pre_processor=face_preprocessor,
    )
    print(f"  编码器: {ENCODER_NAME}")
    print(f"  瓶颈 dropout: {BOTTLENECK_DROPOUT}")
    print(f"  解码器层数: {DECODER_DEPTH}")

    # 第五步：创建训练引擎
    print("\n创建训练引擎...")
    engine = Engine(
        max_steps=MAX_STEPS,
        accelerator="gpu",
        devices=GPUS,
        default_root_dir=RESULT_DIR,
        gradient_clip_val=0.1,
        strategy="auto",
    )
    print(f"  GPU 数量: {GPUS}")
    print(f"  最大步数: {MAX_STEPS}")
    print(f"  结果目录: {RESULT_DIR}")

    # 第六步：训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    engine.fit(model=model, datamodule=datamodule)

    # 第七步：测试
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    results = engine.test(model=model, datamodule=datamodule)

    # 第八步：打印结果
    print("\n" + "=" * 60)
    print("测试结果：")
    print("=" * 60)
    for i, result in enumerate(results):
        print(f"\n  测试集 {i+1}:")
        for key, value in result.items():
            print(f"    {key}: {value}")

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"结果保存在: {RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
