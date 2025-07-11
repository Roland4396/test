import torch
import torch.nn as nn
import torch.optim as optim
import time

# ==================== 修改部分：从resnet.py导入 ====================
# 确保所有需要的类都从您的resnet.py文件中导入
from models.resnet import ResNet, BasicBlock, Bottleneck, Classifier 
# ==================== 修改结束 ====================

# --- 1. 设置和实例化 ---

# 定义模型参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 100 # 假设为CIFAR-100
# 使用类似ResNet-18的结构进行测试
resnet_layers = [18, 18, 18] 

# 实例化您的最终版ResNet模型
# 我们测试“中等模型”，即 model_level_idx=1
try:
    model = ResNet(
        block=BasicBlock,
        layers=resnet_layers,
        num_classes=num_classes,
        num_model_levels=4,
        model_level_idx=1
    ).to(device)
except NameError as e:
    print(f"导入错误: 请确保 {e.name} 类在 resnet.py 中已定义。")
    exit()


# 创建优化器和伪数据
# 为两个测试循环分别创建优化器，避免状态干扰
optimizer1 = optim.SGD(model.parameters(), lr=0.01)
optimizer2 = optim.SGD(model.parameters(), lr=0.01)

dummy_input = torch.randn(16, 3, 32, 32).to(device) # Batch Size = 16
dummy_labels = torch.randint(0, num_classes, (16,)).to(device)
criterion = nn.CrossEntropyLoss()

print(f"开始在 {device} 上进行基准测试...")
print("="*30)

# --- 2. 测试包含核范数损失的速度 ---

# 预热GPU，让GPU频率达到稳定状态
print("正在测试包含Rank Loss的训练速度...")
for _ in range(10):
    outputs, loss_rank, loss_pruning_target = model(
        dummy_input, 
        rank_sub_batch_size=4, 
        beta=0.01,
        delta=0.5,
        target_pruning_ratios=[0.7, 0.5, 0.3, 0.2] # 假设值
    )

if torch.cuda.is_available():
    torch.cuda.synchronize() # 等待所有CUDA核心完成工作
start_time = time.time()

for _ in range(500):
    optimizer1.zero_grad()
    
    # ==================== 修改：接收3个返回值 ====================
    outputs, loss_rank, loss_pruning_target = model(
        dummy_input, 
        rank_sub_batch_size=6, 
        beta=0.01,
        delta=0.5,
        target_pruning_ratios=[0.7, 0.5, 0.3, 0.2] # 假设值
    )
    # ========================================================

    loss_ce = criterion(outputs[-1], dummy_labels)
    
    # ==================== 修改：将所有损失相加 ====================
    total_loss = loss_ce + loss_rank + loss_pruning_target
    # ========================================================

    total_loss.backward()
    optimizer1.step()

if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()
new_model_time = end_time - start_time
print(f"包含Rank Loss (sub_batch=4), 100次迭代耗时: {new_model_time:.3f} 秒")

# --- 3. 测试不包含核范数损失的速度 ---

# 预热GPU
print("\n正在测试不包含Rank Loss的训练速度...")
for _ in range(10):
    outputs, loss_rank, loss_pruning_target= model(dummy_input)

if torch.cuda.is_available():
    torch.cuda.synchronize()
start_time = time.time()

for _ in range(500):
    optimizer2.zero_grad()
    
    # ==================== 修改：接收3个返回值，但只使用第一个 ====================
    outputs, _, _ = model(dummy_input) # 用 _ 忽略后两个损失
    # =====================================================================

    loss_ce = criterion(outputs[-1], dummy_labels)
    loss_ce.backward() # 只对CE loss进行反向传播
    optimizer2.step()

if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()
original_model_time = end_time - start_time
print(f"不包含Rank Loss, 100次迭代耗时: {original_model_time:.3f} 秒")
print("="*30)

# --- 4. 输出结论 ---
if original_model_time > 0:
    slowdown_factor = new_model_time / original_model_time
    print(f"性能影响：训练时长约增加为原来的 {slowdown_factor:.2f} 倍。")
else:
    print("无法计算性能影响。")