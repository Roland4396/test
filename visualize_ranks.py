# visualize_ranks.py
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from config import Config 
# 导入项目中的必要模块
from args import arg_parser, modify_args
import models
from data_tools.dataloader import get_dataloaders, get_datasets, get_user_groups, DatasetSplit 
from utils.hrank_pruning import get_filter_ranks
import math
def visualize_model_ranks():
    """
    主函数，用于加载模型、计算秩并生成可视化图表。
    """
    # 1. 获取命令行参数
    args = arg_parser.parse_args() 
    args = modify_args(args) 
    # 必须提供一个已训练模型的路径
    if not args.resume:
        print("错误：请使用 --resume 参数提供一个已训练的模型检查点路径（.pth.tar 文件）。")
        print("例如：python visualize_ranks.py --arch resnet20 --data cifar10 --resume ./path/to/model.pth.tar")
        return

    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 3. 加载数据
    # 3. 加载数据集
    # =================================================================================
    # 这是核心修改：
    # - 我们现在调用正确的 get_datasets(args) 函数。
    # - 它只接收 args 一个参数。
    # - 它返回 train_set, val_set, test_set。
    # =================================================================================
    print("正在加载数据集...")
    train_set, val_set, test_set = get_datasets(args)
    
    # 为了稳定和简单，我们选择测试集来创建代表性数据加载器。
    # 测试集通常有固定的变换，不包含随机数据增强，更适合评估。
    if test_set is None:
        print("错误：未能加载测试数据集，无法继续。")
        return
        # =================================================================================
    # 这是核心修改点：创建一个数据子集以减少计算量
    # =================================================================================
    # 定义用于计算秩的样本数量。您可以根据需要随时调整这个值。
    # 使用1000-2000个样本通常可以在速度和评估准确性之间取得很好的平衡。
    NUM_SAMPLES_FOR_RANK = 100

    # 确保请求的样本数不超过数据集的总大小
    num_available_samples = len(test_set)
    if NUM_SAMPLES_FOR_RANK > num_available_samples:
        print(f"警告：请求的样本数量 ({NUM_SAMPLES_FOR_RANK}) 大于可用数量 ({num_available_samples})。将使用所有可用样本。")
        NUM_SAMPLES_FOR_RANK = num_available_samples

    # 从完整测试集中随机抽取一部分样本的索引
    indices = torch.randperm(num_available_samples)[:NUM_SAMPLES_FOR_RANK]
    
    # 使用抽取的索引创建子集
    rank_subset = torch.utils.data.Subset(test_set, indices)
    
    print(f"数据加载完毕。将使用 {len(rank_subset)} 个随机样本来计算秩。")
    # =================================================================================

    # 确定用于数据加载器的批处理大小
    batch_size = args.hrank_representative_batch_size if hasattr(args, 'hrank_representative_batch_size') else args.batch_size
    
    # 使用新创建的子集来构建数据加载器
    representative_dataloader = torch.utils.data.DataLoader(
        rank_subset,  # <-- 注意：这里使用的是子集
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    # 4. 加载预训练模型
    print(f"正在加载模型 '{args.arch}'...")
    config = Config() 
    model_params = config.model_params[args.data][args.arch]
    model = getattr(models, args.arch)(args, model_params)
    
    print(f"=> 正在从 '{args.resume}' 加载权重")
    checkpoint = torch.load(args.resume, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("模型加载完毕。")

    # 5. 计算所有卷积层的滤波器秩
    filter_ranks = get_filter_ranks(model, representative_dataloader)

    if not filter_ranks:
        print("未能计算滤波器秩，程序退出。")
        return
#   # =================================================================================
    # 6 & 7. 为每个卷积层绘制独立的、带Y轴缩放的条形图
    # =================================================================================
    
    num_layers = len(filter_ranks)
    num_cols = math.ceil(math.sqrt(num_layers))
    num_rows = math.ceil(num_layers / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5))
    axes = axes.flatten()

    for i, (layer_name, ranks) in enumerate(filter_ranks.items()):
        ax = axes[i]
        kernel_indices = range(len(ranks))
        
        ax.bar(kernel_indices, ranks, color=sns.color_palette('viridis', len(ranks)))
        
        ax.set_title(layer_name, fontsize=10)
        ax.set_xlabel('卷积核索引', fontsize=8)
        ax.set_ylabel('秩', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)

        # =================================================================================
        # 核心修改点：动态设置Y轴范围
        # =================================================================================
        min_rank = min(ranks)
        max_rank = max(ranks)

        # 如果层内所有秩都相同，则创建一个小的默认范围
        if min_rank == max_rank:
            ax.set_ylim(min_rank - 1, max_rank + 1)
        # 否则，根据最小和最大秩设置一个合适的“缩放”范围
        else:
            # 增加一点padding让图形顶部和底部不拥挤
            padding = (max_rank - min_rank) * 0.1 
            ax.set_ylim(min_rank - padding - 0.5, max_rank + padding + 0.5)
        # =================================================================================


    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'模型 {args.arch.upper()} 中每个卷积核的精确秩 (Y轴缩放)', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = f'rank_per_kernel_zoomed_{args.arch}_{args.data}.png'
    plt.savefig(save_path, dpi=300)
    
    print(f"\n可视化图表已成功保存到: {save_path}")


if __name__ == '__main__':
    visualize_model_ranks()