import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available.")