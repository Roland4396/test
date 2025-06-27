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

def refine_mask_by_swap(base_mask, hrank_scores):
    """
    此函数保持不变，因为它只依赖于输入的 hrank_scores 列表，
    而与分数的计算方式无关。
    """
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
    
    # a `large` value that is larger than any rank
    # for those pruned channels that we do not want to make them active
    for i, rank in enumerate(inactive_ranks):
        if rank == 0:
            inactive_ranks[i] = -100 # a small value

    sorted_active_ranks, sorted_active_indices = torch.sort(active_ranks)
    sorted_inactive_ranks, sorted_inactive_indices = torch.sort(inactive_ranks, descending=True)
    
    refined_mask = copy.deepcopy(base_mask)
    
    for i in range(num_prune):
        if sorted_active_ranks[i] < sorted_inactive_ranks[i]:
            
            refined_mask[active_indices[sorted_active_indices[i]]] = False
            refined_mask[inactive_indices[sorted_inactive_indices[i]]] = True
        else:
            # if the rank of the pruned channel is not larger than the reserved channel,
            # then we do not need to swap
            break
            
    return refined_mask