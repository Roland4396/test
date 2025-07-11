import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import numpy as np
import os

# 导入您最终版的模型代码
# 确保所有需要的类都在resnet.py中
from models.resnet import ResNet, BasicBlock, Bottleneck, Classifier 

def calculate_stage_flops_and_weights(model_config):
    """
    计算一个完整、未剪枝的ResNet模型每个Stage的FLOPs及其总和与占比。
    """
    print("--- 正在创建完整模型用于FLOPs分析 ---")
    
    # ==================== 解决方案在这里 ====================
    # 直接使用传入的 model_config，它本身就是架构参数字典
    arch_params = model_config 
    # =======================================================
    
    # 为了准确计算FLOPs，我们需要创建一个临时的、未剪枝的饱满模型
    # 这里我们定义一个临时的ResNet子类，强制其policy_params全为1
    class TempResNet(ResNet):
        def __init__(self, *args, **kwargs):
            # 确保在调用父类__init__时，model_level_idx有效，避免出错
            init_kwargs = kwargs.copy()
            if init_kwargs.get('model_level_idx', 0) >= init_kwargs.get('num_model_levels', 1):
                init_kwargs['model_level_idx'] = 0
            super().__init__(*args, **init_kwargs)
            # 强制策略为全1 (sigmoid(10)约等于1)，确保计算的是饱满模型的FLOPs
            self.policy_params.data.fill_(10.0)

    full_model = TempResNet(**arch_params).eval()

    stage_flops_list = []
    # 注意：输入尺寸应与您训练和thop库的通常用法一致
    dummy_input = torch.randn(1, 3, 32, 32)

    print("--- 正在分段计算FLOPs ---")
    
    out = F.relu(full_model.bn1(full_model.conv1(dummy_input)))
    if full_model.num_classes > 100:
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        
    for i in range(len(full_model.layers)):
        stage_module_list = full_model.layers[i]
        stage_as_sequential = nn.Sequential(*stage_module_list)

        flops, _ = profile(stage_as_sequential, inputs=(out,), verbose=False)
        stage_flops_list.append(flops)
        
        out = stage_as_sequential(out)
        print(f"  - Stage {i+1} FLOPs: {flops / 1e9:.3f} GFLOPs")
    
    stage_flops_tensor = torch.tensor(stage_flops_list, dtype=torch.float32)
    total_flops = torch.sum(stage_flops_tensor)
    flops_weights = stage_flops_tensor / total_flops
    
    print(f"\n计算完成！原始模型总FLOPs: {total_flops.item() / 1e9:.3f} GFLOPs")
    print(f"各Stage的FLOPs权重占比: {flops_weights.numpy()}")
    
    return total_flops.item(), flops_weights
def analyze_checkpoint(config):
    """
    主分析函数
    """
    # ================== 1. 加载已训练的模型 ==================
    print("\n" + "="*50)
    print(f"正在加载模型文件: {config['model_path']}")
    
    # 实例化一个主模型框架，用于加载包含完整 policy_params 的 state_dict
    analysis_model = ResNet(**config['arch_params'])
    
    # 加载权重，处理嵌套的state_dict和非严格匹配问题
    if not os.path.exists(config['model_path']):
        print(f"错误：找不到模型文件 '{config['model_path']}'。请检查路径。")
        return
        
    checkpoint = torch.load(config['model_path'], map_location='cpu')
    
    # 检查checkpoint是否包含'state_dict'键
    if 'state_dict' in checkpoint:
        model_parameters = checkpoint['state_dict']
    else:
        model_parameters = checkpoint

    # 使用 strict=False 来忽略无关的键 (如 total_ops)
    analysis_model.load_state_dict(model_parameters, strict=False)
        
    analysis_model.eval()
    print("模型加载成功！")

    # ================== 2. 分析并展示每个Stage的保留比例 ==================
    print("\n" + "="*50)
    print("分析结果：各模型等级的Stage保留比例 (stage_scales)")
    
    with torch.no_grad():
        all_stage_scales = torch.sigmoid(analysis_model.policy_params)

    for i in range(config['arch_params']['num_model_levels']):
        level_scales = all_stage_scales[i]
        print(f"  - 等级 {i} (小->大): {[f'{s.item():.3f}' for s in level_scales]}")

    # ================== 3. 分析并展示每个等级的最终FLOPs ==================
    print("\n" + "="*50)
    print("分析结果：各模型等级的最终FLOPs")

    # a. 计算原始模型的总FLOPs和各stage的权重
    original_total_flops, stage_flops_weights = calculate_stage_flops_and_weights(config['arch_params'])
    
    # b. 计算每个等级剪枝后的FLOPs
    for i in range(config['arch_params']['num_model_levels']):
        level_scales = all_stage_scales[i]
        
        # 计算该等级剩余的FLOPs比例 (加权和)
        remaining_flops_ratio = torch.sum(level_scales * stage_flops_weights).item()
        
        # 计算最终的绝对FLOPs值
        final_flops = original_total_flops * remaining_flops_ratio
        
        # 计算节省的比例
        saved_ratio = 1 - remaining_flops_ratio

        print(f"  - 等级 {i}:")
        print(f"    - 最终FLOPs: {final_flops / 1e9:.3f} GFLOPs")
        print(f"    - 相比原始模型，节省了 {saved_ratio:.2%}")

    print("\n" + "="*50)


if __name__ == '__main__':
    # ================== 用户配置区域 ==================
    # 您只需要修改这个字典即可
    
    analysis_config = {
        # 1. 修改为您训练好的模型权重路径
        "model_path": 'D:/compile/test-main/outputs/自学习阶段/save_models/checkpoint_399.pth.tar',

        # 2. 模型的架构参数 (必须与您训练时使用的完全一致)
        "arch_params": {
            "block": BasicBlock,
            "layers": [18, 18, 18],  # ResNet-110 for CIFAR
            "num_classes": 100,
            "ee_layer_locations": [35, 41, 47], # <--- 如果训练时有，请务必填入相同的列表！
            "num_model_levels": 4,   # <--- 必须与训练时一致
        }
    }
    # =================================================

    analyze_checkpoint(analysis_config)