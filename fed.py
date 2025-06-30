# fed.py
import copy
import datetime as dt
import os
import pickle as pkl

import numpy as np
import torch
import torch.multiprocessing as mp
from utils.hrank_pruning import get_filter_ranks, refine_mask_by_swap
import models
from data_tools.dataloader import get_client_dataloader
from predict import local_validate
from train import execute_epoch
from utils.grad_traceback import get_downscale_index
from utils.utils import save_checkpoint
import models 

mp.set_start_method('spawn', force=True)


class Federator:
    def __init__(self, global_model, args, client_groups=[]):
        self.global_model = global_model
        self.args = args 

        self.vertical_scale_ratios = args.vertical_scale_ratios
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios

        assert len(self.vertical_scale_ratios) == len(self.horizontal_scale_ratios) == len(self.client_split_ratios)

        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.num_levels = len(self.vertical_scale_ratios)
        
        # 初始化基础掩码
        print("Federator.__init__: Initializing base idx_dicts using get_downscale_index.")
        self.idx_dicts = [get_downscale_index(self.global_model, args, s) for s in self.vertical_scale_ratios]
        
        self.client_groups = client_groups
        self.use_gpu = args.use_gpu

    # 新增方法：用于运行HRank优化
    def refine_masks_with_hrank(self, representative_dataloader, args, config, use_max_rank):
        """
        使用HRank分数来优化现有的idx_dicts。
        此函数现在总是重新计算分数，并根据 use_max_rank 决定策略。
        """
        print("Federator: Starting mask refinement process using HRank...")
        
        # ==================== 代码修改：函数签名已更新 ====================
        # 新增了 use_max_rank 参数来控制剪枝策略
        print(f"Federator: Current strategy is to use {'MAX' if use_max_rank else 'MIN'} rank.")
        # =============================================================

        # 1. 获取全尺寸模型的HRank分数
        analysis_model_params = {**config.model_params[args.data][args.arch], 'scale': 1.0}
        analysis_model = getattr(models, args.arch)(args, analysis_model_params)
        
        # 加载当前训练过的模型权重以获得更准确的秩
        analysis_model.load_state_dict(self.global_model.state_dict(), strict=False)

        # 将分析模型移动到正确的设备上
        device = next(self.global_model.parameters()).device
        analysis_model.to(device)
        
        # 调用 get_filter_ranks
        filter_ranks_per_layer = get_filter_ranks(
            analysis_model, representative_dataloader
        )

        del analysis_model
        torch.cuda.empty_cache()

        if not filter_ranks_per_layer:
            print("Warning: HRank scores could not be calculated. Skipping mask refinement.")
            return

        print("  HRank scores calculated. Now refining masks...")

        # 2. 遍历现有的 idx_dicts，并用HRank分数进行优化
        refined_idx_dicts = []
        for level in range(self.num_levels):
            base_mask_dict_for_level = self.idx_dicts[level]
            refined_mask_dict_for_level = {}

            for param_name, base_mask in base_mask_dict_for_level.items():
                
                parts = param_name.split('.')
                module_name = '.'.join(parts[:-1])
                
                associated_conv_name = None
                if 'conv' in module_name:
                    associated_conv_name = module_name
                elif 'bn' in module_name or 'shortcut.1' in module_name:
                    if module_name == "bn1": associated_conv_name = "conv1"
                    elif module_name.endswith(".bn1"): associated_conv_name = module_name[:-3] + "conv1"
                    elif module_name.endswith(".bn2"): associated_conv_name = module_name[:-3] + "conv2"
                    elif module_name.endswith(".shortcut.1"): associated_conv_name = module_name[:-1] + "0"
                
                if associated_conv_name and associated_conv_name in filter_ranks_per_layer:
                    hrank_scores = filter_ranks_per_layer[associated_conv_name]
                    
                    # ==================== 代码修改：传递策略参数 ====================
                    # 将 use_max_rank 参数传递给底层的交换函数
                    refined_mask = refine_mask_by_swap(base_mask, hrank_scores, use_max_rank)
                    # ============================================================
                    
                    refined_mask_dict_for_level[param_name] = refined_mask
                else:
                    refined_mask_dict_for_level[param_name] = base_mask
            
            refined_idx_dicts.append(refined_mask_dict_for_level)

        self.idx_dicts = refined_idx_dicts
        print("Federator: Masks refined successfully with HRank priorities.")
    def fed_train(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params):
        scores = ['epoch\ttrain_loss\tval_loss\tval_acc1\tval_acc5\tlocal_val_acc1\tlocal_val_acc5' +
                  '\tlocal_val_acc1' * self.num_levels]
        best_acc1, best_round = 0.0, 0

        if not self.client_groups:
            client_idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_client_idxs = np.random.permutation(client_idxs)
            client_groups = []
            s = 0
            for ratio in self.client_split_ratios:
                e = s + int(len(shuffled_client_idxs) * ratio)
                client_groups.append(shuffled_client_idxs[s: e])
                s = e
            self.client_groups = client_groups

            with open(os.path.join(args.save_path, 'client_groups.pkl'), 'wb') as f:
                pkl.dump(self.client_groups, f)

        for round_idx in range(args.start_round, self.num_rounds):

            print(f'\n | Global Training Round : {round_idx + 1} |\n')

            train_loss, val_results, local_val_results = \
                self.execute_round(train_set, val_set, user_groups, criterion, args, batch_size,
                                   train_params, round_idx)

            val_loss, val_acc1, val_acc5, _, _ = val_results

            scores.append(('{}' + '\t{:.4f}' * int(6 + self.num_levels))
                          .format(round_idx, train_loss, val_loss, val_acc1, val_acc5,
                                  local_val_results[-1][1], local_val_results[-1][2],
                                  *[l[1] for l in local_val_results[:-1]]))

            is_best = val_acc1 > best_acc1
            if is_best:
                best_acc1 = val_acc1
                best_round = round_idx
                print('Best var_acc1 {}'.format(best_acc1))

            model_filename = 'checkpoint_%03d.pth.tar' % round_idx
            save_checkpoint({
                'round': round_idx,
                'arch': args.arch,
                'state_dict': self.global_model.state_dict(),
                'best_acc1': best_acc1,
            }, args, is_best, model_filename, scores)

        return best_acc1, best_round

    def get_level(self, client_idx):
        try:
            level = np.where([client_idx in c for c in self.client_groups])[0][0]
        except:
            level = -1
        return level

    def execute_round(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params, round_idx):
        self.global_model.train()
        m = max(int(self.sample_rate * self.num_clients), 1)
        client_idxs = np.random.choice(range(self.num_clients), m, replace=False)

        client_train_loaders = [get_client_dataloader(train_set, user_groups[0][client_idx], args, batch_size) for
                                client_idx in client_idxs]
        levels = [self.get_level(client_idx) for client_idx in client_idxs]
        scales = [self.vertical_scale_ratios[level] for level in levels]
        local_models = [self.get_local_split(levels[i], scales[i]) for i in range(len(client_idxs))]
        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels]

        pool_args = [train_set, user_groups, criterion, args, batch_size, train_params, round_idx]
        local_weights = []
        local_losses = []
        local_grad_flags = []

        pool_args.append(None) 

        for i, client_idx in enumerate(client_idxs):
            client_args = pool_args + [local_models[i], client_train_loaders[i], levels[i], scales[i], h_scale_ratios[i], client_idx]
            result = execute_client_round(client_args) 

            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            print(f'Client {i+1}/{len(client_idxs)} completely finished')

        # ---- 关键修改：修正布尔值判断 ----
        train_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0

        global_weights = self.average_weights(local_weights, local_grad_flags, levels, self.global_model)
        self.global_model.load_state_dict(global_weights)

        global_model_for_val = None
        if self.client_split_ratios[-1] == 0:
            level = np.where(self.client_split_ratios)[0].tolist()[-1]
            scale = self.vertical_scale_ratios[level]
            global_model_for_val = self.get_local_split(level, scale)
            if self.use_gpu and torch.cuda.is_available():
                global_model_for_val = global_model_for_val.cuda()
        else:
            global_model_for_val = copy.deepcopy(self.global_model)
            if self.use_gpu and torch.cuda.is_available() and next(global_model_for_val.parameters()).device.type == 'cpu':
                global_model_for_val.cuda()

        val_results, local_val_results = local_validate(self, val_set, user_groups[1], criterion, args, 512,
                                                       global_model_for_val)

        return train_loss, val_results, local_val_results

    def average_weights(self, w, grad_flags, levels, model):
        w_avg = copy.deepcopy(model.state_dict())
        
        target_device = next(model.parameters()).device 

        for key in w_avg.keys():

            if 'num_batches_tracked' in key:
                if w: w_avg[key] = w[0][key].clone().to(target_device)
                continue

            if 'running' in key:
                accumulated_tensor = torch.zeros_like(w_avg[key], device=target_device)
                for w_ in w:
                    if key in w_:
                        accumulated_tensor += w_[key].to(target_device)
                if len(w) > 0:
                     w_avg[key] = accumulated_tensor / len(w)
                continue

            tmp = torch.zeros_like(w_avg[key], device=target_device)
            count = torch.zeros_like(tmp, device=target_device)
            for i in range(len(w)):
                if key in grad_flags[i] and grad_flags[i][key] and key in w[i]:
                    idx_from_hrank = self.idx_dicts[levels[i]][key]
                    mask_on_target_device = idx_from_hrank.to(target_device)
                    idx = self.fix_idx_array(mask_on_target_device, w[i][key].shape)
                    
                    if idx.device != target_device: idx = idx.to(target_device)

                    if idx.sum().item() != w[i][key].numel():
                        raise RuntimeError(f"Critical Error in average_weights for key '{key}': Mask sum ({idx.sum().item()}) != local param numel ({w[i][key].numel()}).")
                    
                    values_to_add = w[i][key].flatten().to(target_device)
                    tmp[idx] += values_to_add
                    count[idx] += 1
            
            updated_mask = count != 0
            count[count == 0] = 1 
            w_avg[key][updated_mask] = tmp[updated_mask] / count[updated_mask]
            
        return w_avg

    def get_idx_shape(self, inp, local_shape):
        if any([s == 0 for s in inp.shape]):
            raise RuntimeError('Indexing error: input shape has zero dimension')

        if len(local_shape) == 4:
            dim_1 = inp.shape[2] // 2
            dim_2 = inp.shape[3] // 2
            idx_shape = (inp[:, 0, dim_1, dim_2].sum().item(),
                         inp[0, :, dim_1, dim_2].sum().item(), *local_shape[2:])
        elif len(local_shape) == 2:
            idx_shape = (inp[:, 0].sum().item(),
                         inp[0, :].sum().item())
        else:
            idx_shape = (inp.sum().item(),)
        return idx_shape

    # fed.py 文件中的修复后版本

    # fed.py - The fully corrected function

    def fix_idx_array(self, idx_array, local_shape):
        # --- 原有的函数逻辑保持不变 ---
        try:
            idx_shape = self.get_idx_shape(idx_array, local_shape)
        except RuntimeError:
             return torch.ones(local_shape, dtype=torch.bool, device=idx_array.device)

        if all([idx_shape[i] >= local_shape[i] for i in range(len(local_shape))]):
            pass
        else:
            if idx_array.sum(dim=1).numel() > 0 and idx_array.sum(dim=1).argmax().numel() > 0:
                # --- START: This is the corrected logic block ---
                # This block replaces the single line that was causing the crash.
                # It robustly handles masks of any dimension (e.g., 2D for linear, 4D for conv).
                original_shape = idx_array.shape
                n_dims = len(original_shape)

                # 1. Find the best slice along the first dimension by summing over all other dimensions.
                sums_along_dim0 = idx_array.sum(dim=tuple(range(1, n_dims)))
                best_slice_index = sums_along_dim0.argmax()

                # 2. Isolate the best slice.
                best_slice = idx_array[best_slice_index]

                # 3. Add a dimension back at the start and repeat it to fill the original shape.
                repeat_dims = [original_shape[0]] + [1] * (n_dims - 1)
                idx_array = best_slice.unsqueeze(0).repeat(repeat_dims)
                # --- END: Corrected logic block ---
                
                idx_shape = self.get_idx_shape(idx_array, local_shape)

        ind_list = [slice(None)] * len(idx_array.shape)
        for i in range(len(local_shape)):
            lim = idx_array.shape[i]
            while self.get_idx_shape(idx_array[tuple(ind_list)], local_shape)[i] > local_shape[i] and lim > 0:
                lim -= 1
                ind_list[i] = slice(0, lim)
        
        tmp = torch.zeros_like(idx_array, dtype=torch.bool)
        tmp[tuple(ind_list)] = idx_array[tuple(ind_list)]
        idx_array = tmp

        if len(idx_array.shape) == 4:
            dim_1 = idx_array.shape[2] // 2
            dim_2 = idx_array.shape[3] // 2
            if idx_array.sum(dim=0).sum(dim=0)[0, 0] != idx_array.sum(dim=0).sum(dim=0)[dim_1, dim_2]:
                idx_array = idx_array[:, :, dim_1, dim_2].repeat(idx_array.shape[2], idx_array.shape[3], 1, 1).permute(
                    2, 3, 0, 1)
        
        target_numel = int(np.prod(local_shape))
        current_numel = idx_array.sum().item()

        if current_numel > target_numel:
            num_to_remove = current_numel - target_numel
            true_indices_flat = torch.where(idx_array.flatten())[0]
            indices_to_remove_flat = true_indices_flat[-num_to_remove:]
            flat_mask = idx_array.flatten()
            flat_mask[indices_to_remove_flat] = False
            idx_array = flat_mask.reshape(idx_array.shape)
        elif current_numel < target_numel:
            num_to_add = target_numel - current_numel
            false_indices_flat = torch.where(~idx_array.flatten())[0]
            indices_to_add_flat = false_indices_flat[:num_to_add]
            flat_mask = idx_array.flatten()
            flat_mask[indices_to_add_flat] = True
            idx_array = flat_mask.reshape(idx_array.shape)

        return idx_array
    def get_local_split(self, level, scale):
        model = copy.deepcopy(self.global_model)

        if scale == 1:
            return model

        if not hasattr(model, 'stored_inp_kwargs'):
             raise AttributeError("The global_model does not have 'stored_inp_kwargs'.")

        model_kwargs = model.stored_inp_kwargs
        if 'scale' in model_kwargs.keys():
            model_kwargs['scale'] = scale
        elif 'params' in model_kwargs and isinstance(model_kwargs['params'], dict):
            model_kwargs['params']['scale'] = scale
        else:
             raise KeyError("Could not find 'scale' key to set for local split.")

        local_model = type(self.global_model)(**model_kwargs)
        if 'bert' in str(type(local_model)):
             if 'ee_layer_locations' in model_kwargs:
                local_model.add_exits(model_kwargs['ee_layer_locations'])

        local_state_dict = local_model.state_dict()

        for n, p in self.global_model.state_dict().items():
            if n not in local_state_dict:
                continue

            if 'num_batches_tracked' in n:
                local_state_dict[n] = p
                continue

            local_shape = local_state_dict[n].shape

            if len(p.shape) != len(local_shape):
                raise RuntimeError(f"Models not alignable for param {n}! Global {p.shape}, Local {local_shape}")

            mask_from_dict = self.idx_dicts[level][n]
            mask_on_p_device = mask_from_dict.to(p.device)
            idx_array = self.fix_idx_array(mask_on_p_device, local_shape)
            
            if idx_array.device != p.device: idx_array = idx_array.to(p.device)
            
            if idx_array.sum().item() != np.prod(local_shape):
                 raise RuntimeError(f"Mismatch for {n}: mask sum {idx_array.sum().item()} != local numel {np.prod(local_shape)}")

            local_state_dict[n] = p[idx_array].reshape(local_shape)

        local_model.load_state_dict(local_state_dict)
        return local_model


def execute_client_round(args_tuple):
    train_set, user_groups, criterion, args, batch_size, train_params, round_idx, global_model, \
    local_model, client_train_loader, level, scale, h_scale_ratio, client_idx = args_tuple

    if args.use_gpu and torch.cuda.is_available():
        local_model = local_model.cuda()

    base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k]
    exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k]

    optimizer = torch.optim.SGD([{'params': base_params},
                                 {'params': exit_params}],
                                lr=train_params['lr'],
                                momentum=train_params['momentum'],
                                weight_decay=train_params['weight_decay'])

    loss = 0.0
    for epoch in range(train_params['num_epoch']):
        print(f'{client_idx}-{epoch}-{dt.datetime.now()}')
        iter_idx = round_idx
        
        loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, iter_idx, epoch,
                             args, train_params, h_scale_ratio, level, global_model)

    print(f'Finished epochs for {client_idx}')
    
    if args.use_gpu and torch.cuda.is_available():
        local_model.cpu() 

    local_weights = {k: v.clone() for k, v in local_model.state_dict().items()}
    local_grad_flags = {k: v.grad is not None for k, v in local_model.named_parameters()}
    
    del local_model
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return local_weights, local_grad_flags, loss