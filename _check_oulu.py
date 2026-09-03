"""查看 OULU-NPU 压缩包结构"""
import zipfile

zip_path = "/mnt/d/BaiduNetdiskDownload/OULU-NPU/train.zip"

z = zipfile.ZipFile(zip_path)
names = z.namelist()
print(f"总文件数: {len(names)}")
print(f"\n前 30 个:")
for n in names[:30]:
    print(f"  {n}")
print(f"\n...")
print(f"\n后 10 个:")
for n in names[-10:]:
    print(f"  {n}")

# 统计一下有多少个文件夹
folders = set()
for n in names:
    parts = n.split('/')
    if len(parts) >= 2 and parts[0]:
        folders.add(parts[0])
print(f"\n文件夹数量: {len(folders)}")
print(f"前 10 个文件夹: {list(folders)[:10]}")
