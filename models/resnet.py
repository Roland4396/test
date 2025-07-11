import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.model_utils import conv3x3


class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes, num_conv_layers=3, reduction=1):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.reduction = reduction

        if reduction == 1:
            conv_list = [conv3x3(in_planes, in_planes) for _ in range(num_conv_layers)]
        else:
            conv_list = [conv3x3(in_planes, int(in_planes/reduction))]
            in_planes = int(in_planes/reduction)
            conv_list.extend([conv3x3(in_planes, in_planes) for _ in range(num_conv_layers-1)])

        bn_list = [nn.BatchNorm2d(in_planes, track_running_stats=False) for _ in range(num_conv_layers)]
        relu_list = [nn.ReLU() for _ in range(num_conv_layers)]
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()

        layers = []
        for i in range(num_conv_layers):
            layers.append(conv_list[i])
            layers.append(bn_list[i])
            layers.append(relu_list[i])
        layers.append(avg_pool)
        layers.append(flatten)

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, inp, pred=None):
        output = self.layers(inp)
        output = self.fc(output)
        return output

class BasicBlock(nn.Module):
    expansion = 1

    # ==================== 修改：移除__init__签名中的scale ====================
    def __init__(self, in_planes, planes, stride=1, trs=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=trs)

        # ==================== 修改：移除scaler的定义 ====================
        # self.scaler 已移除

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=trs)
            )

    def forward(self, x):
        # ==================== 修改：移除forward中的scaler调用 ====================
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    # ==================== 修改：移除__init__签名中的scale ====================
    def __init__(self, in_planes, planes, stride=1, trs=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=trs)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=trs)

        # ==================== 修改：移除scaler的定义 ====================
        # self.scaler 已移除

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=trs)
            )

    def forward(self, x):
        # ==================== 修改：移除forward中的scaler调用 ====================
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    # __init__ 方法与上一版（移除scale和scaler的版本）完全相同，此处省略
    def __init__(self, block, layers, num_classes, 
                ee_layer_locations=[], trs=False,
                num_model_levels=3, model_level_idx=0):
        super(ResNet, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        # --- 参数设定 (不变) ---
        if num_classes == 1000:
            factor = 4
        elif num_classes == 200:
            factor = 4
        else:
            factor = 1

        # --- 临时变量 (不变) ---
        self.in_planes = int(16 * factor)
        self.num_classes = num_classes
        self.trs = trs
        ee_block_list = []
        ee_layer_list = []
        for ee_layer_idx in ee_layer_locations:
            b, l = self.find_ee_block_and_layer(layers, ee_layer_idx)
            ee_block_list.append(b)
            ee_layer_list.append(l)

        # --- conv1 和 bn1 (不变) ---
        if self.num_classes > 100:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, track_running_stats=self.trs)

        # ==================== 逻辑重构：开始 ====================
        
        # 1. 先用一个临时的Python列表来创建和收集所有的stage layer
        temp_model_layers = []
        temp_ee_classifiers = []
        
        # 为了在创建时能动态获取scales，我们在这里先计算一次
        # 注意：此时num_stages还是未知的，所以我们先用len(layers)作为预估
        # 这一步是为了让_make_layer能运行，但真正的num_stages在后面才确定
        _estimated_num_stages = len(layers)
        _temp_policy_params = nn.Parameter(torch.full((num_model_levels, _estimated_num_stages), 10.0))
        with torch.no_grad():
            _temp_stage_scales = torch.sigmoid(_temp_policy_params[model_level_idx])

        # 创建前3个stage
        layer1, ee1 = self._make_layer(block, int(16 * factor), layers[0], stride=1,
                                    ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 0],
                                    stage_scale=_temp_stage_scales[0])
        temp_model_layers.append(layer1)
        temp_ee_classifiers.append(ee1)

        layer2, ee2 = self._make_layer(block, int(32 * factor), layers[1], stride=2,
                                    ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 1],
                                    stage_scale=_temp_stage_scales[1])
        temp_model_layers.append(layer2)
        temp_ee_classifiers.append(ee2)

        layer3, ee3 = self._make_layer(block, int(64 * factor), layers[2], stride=2,
                                    ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 2],
                                    stage_scale=_temp_stage_scales[2])
        temp_model_layers.append(layer3)
        temp_ee_classifiers.append(ee3)

        # 根据num_classes条件判断是否创建并添加第4个stage
        if self.num_classes > 100:
            layer4, ee4 = self._make_layer(block, int(128 * factor), layers[3], stride=2,
                                        ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 3],
                                        stage_scale=_temp_stage_scales[3])
            temp_model_layers.append(layer4)
            temp_ee_classifiers.append(ee4)
            
        # 2. 将临时的Python列表转为PyTorch的ModuleList
        self.layers = nn.ModuleList(temp_model_layers)
        self.ee_classifiers = nn.ModuleList(temp_ee_classifiers)

        # 3. 在所有layers都确定后，获取最终、正确的stage数量
        num_stages = len(self.layers)

        # 4. 用这个正确的数量来初始化我们的核心参数
        self.register_buffer('stage_physical_weights', torch.ones(num_stages) / num_stages)
        self.policy_params = nn.Parameter(torch.zeros(num_model_levels, num_stages))
        self.model_level_idx = model_level_idx

        # 5. 最后，创建与最终网络宽度匹配的分类头
        num_planes = self.in_planes
        self.linear = nn.Linear(num_planes, num_classes)
        # ==================== 逻辑重构：结束 ====================
    def _calculate_per_channel_nuclear_norm(self, tensor, sub_batch_size=4):
        """
        计算4D特征图每个通道的核范数，并返回所有通道核范数的平均值。
        """
        # (N, C, H, W)
        batch_size, num_channels, H, W = tensor.shape
        
        # 随机选择子集的索引，所有通道共享
        actual_sub_batch_size = min(batch_size, sub_batch_size)
        indices = torch.randperm(batch_size)[:actual_sub_batch_size].to(tensor.device)
        sub_batch_tensor = tensor.index_select(0, indices) # (sub_N, C, H, W)

        total_nuclear_norm = 0

        # 遍历每个通道
        for c in range(num_channels):
            # 提取单个通道的特征图 (sub_N, H, W)
            channel_feature_map = sub_batch_tensor[:, c, :, :]
            
            # 计算这个通道的核范数 (所有样本的平均)
            # torch.linalg.svdvals可以直接处理一个批次的矩阵
            # (sub_N, H, W) -> torch.linalg.svdvals -> (sub_N, min(H,W))
            s_values = torch.linalg.svdvals(channel_feature_map)
            channel_norm = torch.sum(s_values)
            total_nuclear_norm += channel_norm

        # 返回所有通道的平均核范数，再除以估算的样本数
        return total_nuclear_norm / (num_channels * actual_sub_batch_size)

   # ==================== 修改部分：开始 ====================
    # 在_make_layer的参数中加入stage_scale
    def _make_layer(self, block_type, planes, num_block, stride, ee_layer_locations, stage_scale):
        # 根据stage_scale计算本stage的实际通道数
        scaled_planes = int(planes * stage_scale)
        if scaled_planes == 0:
            scaled_planes = 1 # 确保通道数至少为1

        strides = [stride] + [1] * (num_block - 1)
        ee_layer_locations_ = ee_layer_locations + [num_block]
        layers = [[] for _ in range(len(ee_layer_locations_))]
        ee_classifiers = []
        
        if len(ee_layer_locations_) > 1:
            start_layer = 0
            counter = 0
            for i, ee_layer_idx in enumerate(ee_layer_locations_):
                for _ in range(start_layer, ee_layer_idx):
                    # 调用BasicBlock/Bottleneck时不再需要scale参数
                    layers[i].append(block_type(self.in_planes, scaled_planes, strides[counter], trs=self.trs))
                    self.in_planes = scaled_planes * block_type.expansion
                    counter += 1
                start_layer = ee_layer_idx
                
                if ee_layer_idx == 0:
                    num_planes = self.in_planes
                else:
                    num_planes = scaled_planes * block_type.expansion
                    
                if i < len(ee_layer_locations_) - 1:
                    # 调用Classifier时不再需要scale参数
                    ee_classifiers.append(Classifier(num_planes, num_classes=self.num_classes,
                                                    reduction=block_type.expansion))
        else:
            for i in range(num_block):
                # 调用BasicBlock/Bottleneck时不再需要scale参数
                layers[0].append(block_type(self.in_planes, scaled_planes, strides[i], trs=self.trs))
                self.in_planes = scaled_planes * block_type.expansion
                
        return nn.ModuleList([nn.Sequential(*l) for l in layers]), nn.ModuleList(ee_classifiers)


    @staticmethod
    def find_ee_block_and_layer(layers, layer_idx):
        temp_array = np.zeros((sum(layers)), dtype=int)
        cum_array = np.cumsum(layers)
        for i in range(1, len(cum_array)):
            temp_array[cum_array[i-1]:] += 1
        block = temp_array[layer_idx]
        if block == 0:
            layer = layer_idx
        else:
            layer = layer_idx - cum_array[block-1]
        return block, layer

    def forward(self, x, manual_early_exit_index=0, beta=0.01, rank_sub_batch_size=4,delta=0.5, target_pruning_ratios=None):
        # 获取当前模型等级的保留比例
        # ==================== 修改部分：开始 ====================
        # 在 forward 的开头动态获取保留比例
        # self.policy_params 已经被 .to(device) 移动到了正确的设备
        # 所以这里的切片和sigmoid操作产生的结果也都在正确的设备上
        current_policy_params = self.policy_params[self.model_level_idx]
        stage_scales = torch.sigmoid(current_policy_params)
        stage_nuclear_norms = []
        # ... (前向传播到各个stage的逻辑不变) ...
        final_out = F.relu(self.bn1(self.conv1(x)))
        if self.num_classes > 100:
            final_out = F.max_pool2d(final_out, kernel_size=3, stride=2, padding=1)
        
        ee_outs = []
        counter = 0

        # 遍历每个stage
        while counter < len(self.layers):
            # ... (这部分_block_forward的调用逻辑不变) ...
            if final_out is not None:
                if manual_early_exit_index > sum([len(ee) for ee in self.ee_classifiers[:counter+1]]):
                    manual_early_exit_index_ = 0
                elif manual_early_exit_index:
                    manual_early_exit_index_ = manual_early_exit_index - sum([len(ee) for ee in self.ee_classifiers[:counter]])
                else:
                    manual_early_exit_index_ = manual_early_exit_index
                final_out = self._block_forward(self.layers[counter], self.ee_classifiers[counter], final_out, ee_outs, manual_early_exit_index_)
            
            # 如果这个stage被执行了 (final_out不是None)
            if final_out is not None:
                # ==================== 修改：调用新的计算函数 ====================
                # 计算这个stage的“平均通道核范数”
                avg_channel_norm = self._calculate_per_channel_nuclear_norm(final_out, sub_batch_size=rank_sub_batch_size)
                stage_nuclear_norms.append(avg_channel_norm)
            # ==================== 修改结束 ====================

            counter += 1

        preds = ee_outs
        if final_out is not None:
            out = F.adaptive_avg_pool2d(final_out, (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            preds.append(out)
        # --- 计算秩引导损失 ---
        loss_rank = torch.tensor(0.0, device=x.device)
        if stage_nuclear_norms: # 只有在有stage被执行时才计算
            # 将核范数列表转为向量并归一化
            nuclear_norms_vec = torch.stack(stage_nuclear_norms)
            nuclear_norms_vec = nuclear_norms_vec / (torch.sum(nuclear_norms_vec) + 1e-6)

            # 我们希望核范数向量和保留比例向量正相关，即最大化点积
            # 因此最小化负点积
            # 只取被执行了的stage对应的保留比例
            executed_stage_scales = stage_scales[:len(nuclear_norms_vec)]
            loss_rank = -1  * torch.dot(nuclear_norms_vec, executed_stage_scales)

        if manual_early_exit_index:
            assert len(preds) == manual_early_exit_index
        loss_pruning_target = torch.tensor(0.0, device=x.device)
    
        if target_pruning_ratios is not None:
            # 1. 计算所有模型等级的 stage_scales (保留比例)
            all_stage_scales = torch.sigmoid(self.policy_params)
            
            # 2. 计算每个stage的剪枝比例
            stage_pruning_ratios = 1.0 - all_stage_scales
            
            # 3. 计算每个模型等级实际节省的总FLOPs比例 (加权和)
            total_saved_flops_ratio = torch.sum(stage_pruning_ratios * self.stage_physical_weights, dim=1)
            
            # 4. 目标节省比例列表转为Tensor
            target_ratios_tensor = torch.tensor(target_pruning_ratios, device=x.device, dtype=torch.float32)
            
            # 5. 计算未达标部分的损失
            # F.relu(target - current) 只保留未达标的部分作为损失
            shortfall_loss = F.relu(target_ratios_tensor - total_saved_flops_ratio)
            
            loss_pruning_target = torch.sum(shortfall_loss)
        # 返回预测结果和秩引导损失
        return preds, loss_rank, loss_pruning_target
    def _block_forward(self, layers, ee_classifiers, x, outs, early_exit=0):
        for i in range(len(layers)-1):
            x = layers[i](x)
            if outs:
                outs.append(ee_classifiers[i](x, outs[-1]))
            else:
                outs.append(ee_classifiers[i](x))
            if early_exit == i + 1:
                break
        if early_exit == 0:
            final_out = layers[-1](x)
        else:
            final_out = None
        return final_out


def resnet110_1(args, params, **kwargs):
    """
    创建一个 ResNet-110 (无提前退出分支) 的实例
    """
    return ResNet(
        block=BasicBlock, 
        layers=[18, 18, 18], 
        num_classes=args.num_classes, 
        trs=args.track_running_stats,
        **kwargs  # 将所有额外的关键字参数直接传递给ResNet类
    )

def resnet110_4(args, params, **kwargs):
    """
    创建一个 ResNet-110 (带有提前退出分支) 的实例
    """
    ee_layer_locations = params.get('ee_layer_locations', []) 
    return ResNet(
        block=BasicBlock,
        layers=[18, 18, 18], 
        num_classes=args.num_classes, 
        ee_layer_locations=ee_layer_locations,
        trs=args.track_running_stats,
        **kwargs  # 将所有额外的关键字参数直接传递给ResNet类
    )