import glob
import math
import numpy as np
import os
import shutil
import pickle

import models
import torch
from thop import profile
from utils.op_counter import measure_model
from models.resnet import ResNet, BasicBlock 
import torch.nn.functional as F

def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save_path, 'scores.tsv')
    model_dir = os.path.join(args.save_path, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    prev_checkpoint_list = glob.glob(os.path.join(model_dir, 'checkpoint*'))
    if prev_checkpoint_list:
        os.remove(prev_checkpoint_list[0])

    torch.save(state, model_filename)

    with open(result_filename, 'a') as f:
        print(result[-1], file=f)

    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args, load_best=True):
    model_dir = os.path.join(args.save_path, 'save_models')
    if load_best:
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
    else:
        model_filename = glob.glob(os.path.join(model_dir, 'checkpoint*'))[0]

    if os.path.exists(model_filename):
        print("=> loading checkpoint '{}'".format(model_filename))
        state = torch.load(model_filename)
        print("=> loaded checkpoint '{}'".format(model_filename))
    else:
        return None

    return state


def save_user_groups(args, user_groups):
    user_groups_filename = os.path.join(args.save_path, 'user_groups.pkl')
    if not os.path.exists(user_groups_filename):
        with open(user_groups_filename, 'wb') as fout:
            pickle.dump(user_groups, fout)


def load_user_groups(args):
    user_groups_filename = os.path.join(args.save_path, 'user_groups.pkl')

    if os.path.exists(user_groups_filename):
        with open(user_groups_filename, 'rb') as fin:
            user_groups = pickle.load(fin)
    else:
        user_groups = None

    return user_groups


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, params):
    if params['lr_type'] == 'multistep':
        lr, decay_rate = params['lr'], params['decay_rate']
        if epoch >= params['decay_epochs'][1]:
            lr *= decay_rate ** 2
        elif epoch >= params['decay_epochs'][0]:
            lr *= decay_rate
    elif params['lr_type'] == 'exp':
        lr = params['lr'] * (np.power(params['decay_rate'], epoch))
    else:
        lr = params['lr']
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr


def measure_flops(args, params):
    model = getattr(models, args.arch)(args, params)
    model.eval()
    n_flops, n_params = measure_model(model, args.image_size[0], args.image_size[1])
    torch.save(n_flops, os.path.join(args.save_path, 'flops.pth'))
    del (model)


def load_state_dict(args, model):
    state_dict = torch.load(args.evaluate_from)['state_dict']
    model.load_state_dict(state_dict)

def calculate_stage_flops_weights(model, input_shape_hw):
    """
    计算一个给定的ResNet模型实例每个Stage的FLOPs占比。
    这个版本修复了对嵌套ModuleList的处理。

    :param model: 一个已经实例化的ResNet模型。
    :param input_shape_hw: 图像的高和宽，一个元组，例如 (32, 32)。
    :return: 一个包含每个stage FLOPs占比的PyTorch张量。
    """
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.randn(1, 3, *input_shape_hw).to(device)
    
    stage_flops_list = []

    print("正在分段计算FLOPs...")
    out = F.relu(model.bn1(model.conv1(dummy_input)))
    if model.num_classes > 100:
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        
    # 遍历模型的每个Stage (外层ModuleList)
    for i, stage_module_list in enumerate(model.layers):
        
        # --- 核心修改：开始 ---
        current_stage_total_flops = 0
        # 再次遍历Stage内部的子模块 (内层ModuleList中的Sequential模块)
        for sub_stage_module in stage_module_list:
            flops, _ = profile(sub_stage_module, inputs=(out,), verbose=False)
            current_stage_total_flops += flops
            # 将子模块的输出作为下一个子模块的输入
            out = sub_stage_module(out)
        # --- 核心修改：结束 ---

        stage_flops_list.append(current_stage_total_flops)
        print(f"  - Stage {i+1} FLOPs: {current_stage_total_flops / 1e9:.4f} GFLOPs")
    
    stage_flops_tensor = torch.tensor(stage_flops_list, dtype=torch.float32)
    total_flops = torch.sum(stage_flops_tensor)
    
    if total_flops == 0:
        print("警告: 总FLOPs为0。将返回均勻权重。")
        num_stages = len(model.layers)
        return torch.ones(num_stages) / num_stages if num_stages > 0 else torch.tensor([])

    flops_weights = stage_flops_tensor / total_flops
    
    print(f"\n计算完成！总FLOPs: {total_flops / 1e9:.4f} GFLOPs")
    print(f"各Stage的FLOPs权重占比: {flops_weights.numpy()}")
    
    model.train() 
    
    return flops_weights