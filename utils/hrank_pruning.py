import torch
import torch.nn as nn
from thop import profile
import copy
from tqdm import tqdm
# _calculate_rank_for_single_filter_feature_map 函数已被删除

def get_filter_ranks(model, loader):
    """
    使用 tqdm 来显示细粒度的计算进度。
    """
    feature_maps = {}

    def hook(module, input, output):
        feature_maps[module] = output

    handles = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(hook))

    rank_sums = {} 
    num_samples = 0

    model.eval()
    
    # 2. 在循环中用 tqdm 包装 loader，并添加描述
    # 这会创建一个动态更新的进度条
    progress_bar = tqdm(loader, desc="HRank Calculation", unit="batch")
    for data, _ in progress_bar:
        
        if torch.cuda.is_available():
            data = data.cuda()

        feature_maps.clear()
        
        model(data)

        num_samples += data.size(0)

        for layer, f_map_batch in feature_maps.items():
            if layer not in rank_sums:
                num_filters = layer.out_channels
                rank_sums[layer] = torch.zeros(num_filters, device=f_map_batch.device)

            for sample_idx in range(f_map_batch.size(0)):
                for filter_idx in range(f_map_batch.size(1)):
                    single_feature_map = f_map_batch[sample_idx, filter_idx, :, :]
                    rank = torch.linalg.matrix_rank(single_feature_map.float()).item()
                    rank_sums[layer][filter_idx] += rank
    
    avg_ranks = {}
    layer_name_map = {module: name for name, module in model.named_modules()}

    for layer, sums in rank_sums.items():
        if num_samples > 0:
            avg_rank_per_filter = sums / num_samples
            layer_name = layer_name_map.get(layer, 'unknown_layer')
            avg_ranks[layer_name] = avg_rank_per_filter.tolist()

    for handle in handles:
        handle.remove()

    return avg_ranks

def refine_mask_by_swap(base_mask, hrank_scores, use_max_rank):
    """
    根据 HRank 分数，通过交换通道来优化剪枝掩码。

    Args:
        base_mask (torch.Tensor): 初始的二进制掩码 (0代表剪枝, 1代表保留)。
        hrank_scores (list or torch.Tensor): 对应层所有通道的 HRank 分数。
        use_max_rank (bool): 决定剪枝策略。
                             - True: 优先保留分数高的通道 (最大化秩)。
                             - False: 优先保留分数低的通道 (最小化秩)。
    """
    # ==================== 代码修改开始 ====================

    num_prune = int(len(base_mask) - torch.sum(base_mask))
    if num_prune == 0:
        return base_mask

    device = base_mask.device
    hrank_scores = torch.tensor(hrank_scores, device=device)
    
    base_mask = base_mask.bool()
    
    active_indices = torch.where(base_mask == True)[0]
    inactive_indices = torch.where(base_mask == False)[0]
    
    active_ranks = hrank_scores[active_indices]
    inactive_ranks = hrank_scores[inactive_indices]
    
    # 策略选择：根据 use_max_rank 决定排序方式和比较逻辑
    if use_max_rank:
        # **最大化秩策略 (原始逻辑)**
        # 目标：用未激活通道中分数最高的，去替换已激活通道中分数最低的。
        
        # 找到已激活通道中分数最低的
        sorted_active_ranks, sorted_active_indices = torch.sort(active_ranks)
        
        # 找到未激活通道中分数最高的
        # 这里对 rank 为 0 的特殊处理可以保留，因为它旨在避免激活完全无效的通道
        for i, rank in enumerate(inactive_ranks):
            if rank == 0:
                inactive_ranks[i] = -100 # 一个很小的值，使其在降序排序中排在最后

        sorted_inactive_ranks, sorted_inactive_indices = torch.sort(inactive_ranks, descending=True)
        
        comparison = lambda active, inactive: active < inactive

    else:
        # **最小化秩策略 (新逻辑)**
        # 目标：用未激活通道中分数最低的，去替换已激活通道中分数最高的。
        
        # 找到已激活通道中分数最高的
        sorted_active_ranks, sorted_active_indices = torch.sort(active_ranks, descending=True)
        
        # 找到未激活通道中分数最低的
        sorted_inactive_ranks, sorted_inactive_indices = torch.sort(inactive_ranks)
        
        comparison = lambda active, inactive: active > inactive

    
    refined_mask = copy.deepcopy(base_mask)
    
    # 执行交换
    # 使用 min 是为了防止索引越界，以防万一
    num_to_compare = min(len(sorted_active_ranks), len(sorted_inactive_ranks))

    for i in range(num_to_compare):
        # 使用上面定义的 comparison 函数进行比较
        if comparison(sorted_active_ranks[i], sorted_inactive_ranks[i]):
            
            # 执行交换：将原来激活的变为不激活，不激活的变为激活
            refined_mask[active_indices[sorted_active_indices[i]]] = False
            refined_mask[inactive_indices[sorted_inactive_indices[i]]] = True
        else:
            # 因为通道已经排序，如果当前最优的交换都不满足条件，后续的更不可能满足
            break
            
    return refined_mask.float() # 返回 float 类型的 tensor，与输入保持一致

    # ==================== 代码修改结束 ====================