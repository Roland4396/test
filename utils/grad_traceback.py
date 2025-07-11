from cmath import e
import copy
import math

import torch
import torch.nn as nn


def filter(x, dim=1, scale=1., target_shape=None, model_name=None, split=0):
    # Filter out tensor elements after vertical splitting\
    if target_shape is None:
        if scale == 1:
            return x
        out_dim = int(scale * x.shape[dim])
    else:
        if dim == 1:
            out_dim = target_shape[0]
        else:
            if model_name == 'Linear':
                out_dim = target_shape[0]
            else:
                out_dim = target_shape[-1]

    ind_list = [slice(None)] * len(x.shape)  # indexes to filter
    if split:
        ind_list[min(dim, x.ndim - 1)] = ([False for _ in range(out_dim // split)] +
                                          [True for _ in
                                           range(x.shape[min(dim, x.ndim - 1)] // split - out_dim // split)]) * split
    else:
        ind_list[min(dim, x.ndim - 1)] = slice(out_dim, x.shape[min(dim, x.ndim - 1)])
    x[tuple(ind_list)] = 0

    return x


def get_num_gen(gen):
    return sum(1 for _ in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_downscale_index(model, args, level=0):
    """
    Traces gradients to determine the parameter indices for a specific model level.
    This version is adapted for the new ResNet that uses 'model_level_idx'.
    """
    # dim carries width
    if 'bert' in str(model.__class__):
        dim = 2
    else:
        dim = 1

    def should_filter(x):
        if hasattr(x, 'old_forward'):
            return is_leaf(x) and 'modify_forward' not in str(x.forward)
        else:
            return is_leaf(x)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
            else:
                restore_forward(child)

    def modify_forward(model, local_model, split=1):
        if 'bert' in str(model.__class__) and 'classifiers' not in str(model.__class__):
            include_linear = True
        else:
            include_linear = False

        for i, child in enumerate(model.children()):
            local_child = list(local_model.children())[i]
            if should_filter(child):
                def new_forward(m, ls):
                    def lambda_forward(x):
                        if 'BatchNorm2d' in m._get_name():
                            x_ = x + m.weight[None, :, None, None] + m.bias[None, :, None, None]
                        elif 'BatchNorm1d' in m._get_name() or 'LayerNorm' in m._get_name():
                            x_ = x + m.weight + m.bias
                        elif 'GELU' in m._get_name() or 'Softmax' in m._get_name():
                            x_ = x
                        else:
                            x_ = m.old_forward(x)

                        if include_linear:
                            if any([n in m._get_name() for n in ['Conv', 'BatchNorm', 'LayerNorm', 'Linear', 'Embedding']]):
                                x_ = filter(x_, dim=dim, target_shape=ls, model_name=m._get_name(), split=split)
                        else:
                            if any([n in m._get_name() for n in ['Conv', 'BatchNorm', 'LayerNorm', 'Embedding']]):
                                x_ = filter(x_, dim=dim, target_shape=ls, split=split)
                        return x_
                    return lambda_forward

                child.old_forward = child.forward
                local_shape = getattr(local_child, 'weight', None).shape if hasattr(local_child, 'weight') else None
                child.forward = new_forward(child, local_shape)
            else:
                modify_forward(child, local_child, split)
    
    # --- 核心修改部分 ---
    
    # 1. 复制全局模型的构造参数
    model_kwargs = copy.deepcopy(model.stored_inp_kwargs)

    # 2. 创建一个用于追踪的完整模型 (使用最高等级)
    model_kwargs['model_level_idx'] = model.policy_params.shape[0] - 1
    copy_model = type(model)(**model_kwargs)
    if args.use_gpu:
        copy_model = copy_model.cuda()

    # 3. 创建一个目标等级的本地模型 (使用传入的 level)
    model_kwargs['model_level_idx'] = level 
    local_model = type(model)(**model_kwargs)
    # --- 核心修改结束 ---


    if 'bert' in args.arch:
        modify_forward(copy_model, local_model, split=getattr(model_kwargs['config'], 'num_attention_heads'))
    else:
        modify_forward(copy_model, local_model)

    state_dict = copy_model.state_dict(keep_vars=True)
    for k, v in state_dict.items():
        if 'num_batches_tracked' in k:
            continue
        if 'efficientnet' in str(copy_model.__class__):
            state_dict[k] = torch.ones_like(v) / math.sqrt(v.numel())
        else:
            state_dict[k] = torch.ones_like(v) / v.numel()
    copy_model.load_state_dict(state_dict)

    loss_obj = nn.MSELoss()
    if 'bert' in str(model.__class__):
        inp = torch.ones((1, args.image_size[1])).long()
    else:
        inp = torch.ones((1, 3, args.image_size[0], args.image_size[1]))
    if args.use_gpu:
        inp = inp.cuda()

    # 在调用forward时，使用默认参数，避免额外损失的计算
    preds, _, _ = copy_model.forward(inp)

    # 兼容模型有或没有early-exit的情况
    if not isinstance(preds, list):
        preds = [preds]
        
    target = torch.ones_like(preds[-1])
    if args.use_gpu:
        target = target.cuda()
    
    loss = sum([loss_obj(pred, target) for pred in preds])
    loss.backward()

    state_dict = copy_model.state_dict(keep_vars=True)
    idx_dict = {}
    for k, v in state_dict.items():
        if 'num_batches_tracked' in k:
            continue
        if v.grad is None:
            idx_dict[k] = torch.ones_like(v, dtype=bool)
        else:
            idx_dict[k] = v.grad != 0
            if idx_dict[k].sum() == 0:
                raise RuntimeError("Gradient tracing failed, no gradients found for parameter: " + k)

    restore_forward(copy_model)
    return idx_dict

# 在文件顶部附近确保有 is_leaf 的定义
