"""
OULU-NPU 数据集整理脚本
把 OULU-NPU 整理成 anomalib Folder 需要的格式：

organized_root/
├── train/
│   └── normal/        ← 训练集真人
└── test/
    ├── normal/        ← 测试集真人
    └── abnormal/      ← 测试集假脸

OULU-NPU 文件夹命名: a_b_c_d
  d = 1  → 真人 (normal)
  d = 2,3,4,5 → 假脸 (abnormal)
"""

import os
import shutil
from pathlib import Path

# ============================================================
# 配置
# ============================================================
SRC_ROOT = "/mnt/d/BaiduNetdiskDownload/OULU-NPU"
DST_ROOT = "/mnt/d/BaiduNetdiskDownload/OULU-NPU_organized"

# 每个文件夹取多少张图（防止数据太多，-1 表示全部取）
SAMPLES_PER_FOLDER = 5  # 每个视频取 5 帧，够用了


def organize_split(src_dir: Path, dst_normal: Path, dst_abnormal: Path, split_name: str):
    """整理一个 split（train/dev/test）"""
    dst_normal.mkdir(parents=True, exist_ok=True)
    dst_abnormal.mkdir(parents=True, exist_ok=True)

    normal_count = 0
    abnormal_count = 0

    folders = [f for f in src_dir.iterdir() if f.is_dir()]
    print(f"  {split_name}: {len(folders)} 个文件夹")

    for folder in folders:
        name = folder.name
        parts = name.split("_")
        if len(parts) != 4:
            print(f"    [跳过] 文件夹名格式不对: {name}")
            continue

        a, b, c, d = parts  # 设备_场景_人_类型

        # d=1 是真人，其他是假脸
        is_normal = (d == "1")

        # 取这个文件夹里的图片
        img_files = sorted(folder.glob("*.jpg"))
        if SAMPLES_PER_FOLDER > 0:
            # 均匀采样，取前 SAMPLES_PER_FOLDER 张
            img_files = img_files[:SAMPLES_PER_FOLDER]

        for i, img_path in enumerate(img_files):
            # 新文件名：文件夹名 + 序号，避免重名
            new_name = f"{name}_{i:03d}.jpg"

            if is_normal:
                dst = dst_normal / new_name
                normal_count += 1
            else:
                dst = dst_abnormal / new_name
                abnormal_count += 1

            shutil.copy2(img_path, dst)

    print(f"    正常: {normal_count} 张, 异常: {abnormal_count} 张")
    return normal_count, abnormal_count


def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)

    if dst_root.exists():
        print(f"[提示] 整理后的数据集已存在: {dst_root}")
        print("       如果想重新整理，请先删除这个目录。")
        return

    print("=" * 60)
    print("正在整理 OULU-NPU 数据集...")
    print(f"  源目录: {src_root}")
    print(f"  目标目录: {dst_root}")
    print(f"  每个文件夹取 {SAMPLES_PER_FOLDER if SAMPLES_PER_FOLDER > 0 else '全部'} 张图")
    print("=" * 60)

    # train 集（只用正常样本训练）
    train_src = src_root / "train" / "train"  # 多了一层 train
    if train_src.exists():
        organize_split(train_src, dst_root / "train" / "normal", Path("/tmp/ignore"), "train")

    # test 集（正常 + 异常都有）
    test_src = src_root / "test" / "test"
    if test_src.exists():
        organize_split(test_src, dst_root / "test" / "normal", dst_root / "test" / "abnormal", "test")

    # dev 集（也可以当验证集用，这里先放到 test 里？或者单独放？）
    # 暂时不用 dev，单类检测一般从 train 里分验证集

    print("\n" + "=" * 60)
    print("整理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
