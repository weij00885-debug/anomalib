"""检查环境是否就绪"""
import torch
from anomalib.models import Dinomaly

print("=" * 50)
print("环境检查")
print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("anomalib 导入成功！")
print("=" * 50)
