#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders, get_datasets, get_user_groups, DatasetSplit
from fed import Federator
from models.model_utils import KDLoss
from predict import validate, local_validate
from utils.utils import load_checkpoint, measure_flops, load_state_dict, save_user_groups, load_user_groups, save_checkpoint

np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)


def main():
    global args

    model_save_dir = os.path.join(args.save_path, 'save_models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    config = Config()

    if args.data in config.model_params and args.arch in config.model_params[args.data]:
        if 'num_blocks' in config.model_params[args.data][args.arch]:
            args.num_exits = config.model_params[args.data][args.arch]['num_blocks']
    else:
        if not hasattr(args, 'num_exits'):
            args.num_exits = 1

    if args.ee_locs:
        if args.data in config.model_params and \
           args.arch in config.model_params[args.data] and \
           'ee_layer_locations' in config.model_params[args.data][args.arch]:
            config.model_params[args.data][args.arch]['ee_layer_locations'] = args.ee_locs

    initial_global_model_params = {**config.model_params[args.data][args.arch]}
    initial_global_model = getattr(models, args.arch)(args, initial_global_model_params)


    if args.use_gpu and torch.cuda.is_available():
        criterion = KDLoss(args).cuda()
    else:
        args.use_gpu = 0
        criterion = KDLoss(args)

    train_set, val_set, test_set = get_datasets(args)
    batch_size = args.batch_size if args.batch_size is not None else config.training_params[args.data][args.arch]['batch_size']
    train_loader, val_loader, test_loader = get_dataloaders(args, batch_size, (train_set, val_set, test_set))
    if val_set is None and val_loader:
        val_set = val_loader.dataset

    # 此处假设 user_groups 是在 federator 内部或 fed_train 首次调用时初始化的
    # 如果 federator 的 __init__ 需要它，则需要提前加载或生成
    user_groups = None # 将初始化推迟到 federator 内部


    if args.evalmode is not None:
        print(f"INFO: Running in evaluation mode: {args.evalmode}")
        
        model_for_eval_params = {**config.model_params[args.data][args.arch]}
        model_for_eval_skeleton = getattr(models, args.arch)(args, model_for_eval_params)
        
        if args.evaluate_from:
            print(f"INFO: Loading model for evaluation from: {args.evaluate_from}")
            load_state_dict(args, model_for_eval_skeleton)
        else:
            print("Warning: args.evalmode is set, but args.evaluate_from is not. Evaluating a randomly initialized model.")

        if args.use_gpu and torch.cuda.is_available():
            model_for_eval_skeleton.cuda()

        if 'global' in args.evalmode:
            validate(model_for_eval_skeleton, test_loader, criterion, args, save=True)
            return
        elif 'local' in args.evalmode:
            client_groups_for_eval = []
            client_groups_path = os.path.join(args.save_path, 'client_groups.pkl')
            if os.path.exists(client_groups_path):
                client_groups_for_eval = pkl.load(open(client_groups_path, 'rb'))
            else:
                print(f"Warning: client_groups.pkl not found at {client_groups_path} for local evaluation. Using empty client groups.")
            
            federator_for_eval = Federator(model_for_eval_skeleton, args, client_groups_for_eval)
            
            # 为了在评估时获得正确的用户分组，从文件中加载
            eval_user_groups = load_user_groups(args)
            if eval_user_groups is None:
                print("Warning: could not load user groups for local evaluation.")
                # 可以选择在这里停止，或者使用新生成的用户组（可能导致结果不一致）
                _, eval_val_user_groups, _ = get_user_groups(train_set, val_set, test_set, args)
                eval_user_groups = (None, eval_val_user_groups, None)


            print("INFO: For local evaluation after HRank, ensure Federator's idx_dicts are consistent with the model being evaluated.")
            local_validate(federator_for_eval, test_set, eval_user_groups[1] if eval_user_groups else None, criterion, args, batch_size, model=model_for_eval_skeleton)
            return
        else:
            raise NotImplementedError(f"Unknown evalmode: {args.evalmode}")


    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        print(args, file=f)

    federator = Federator(initial_global_model, args)
    if args.use_gpu and torch.cuda.is_available():
        federator.global_model.cuda()

    cudnn.benchmark = True

    # ---- 新的主训练循环 ----
    total_rounds = args.num_rounds
    hrank_trigger_round=args.hrank_trigger_round
    hrank_last_train=args.last_train_round
    best_acc1_overall = 0.0
    best_round_overall = 0
    scores = ['epoch\ttrain_loss\tval_loss\tval_acc1\tval_acc5\tlocal_val_acc1\tlocal_val_acc5' + '\tlocal_val_acc1' * federator.num_levels]

    # 加载 user_groups
    loaded_user_groups = load_user_groups(args)
    if loaded_user_groups:
        user_groups = loaded_user_groups
    else:
        # 仅在第一次运行时生成和保存
        train_user_groups, val_user_groups, test_user_groups = get_user_groups(train_set, val_set, test_set, args)
        user_groups = (train_user_groups, val_user_groups, test_user_groups)
        save_user_groups(args, user_groups)


    if args.resume:
        print(f"INFO: Attempting to resume training from path: {args.save_path}")
        checkpoint = load_checkpoint(args, load_best=False)
        if checkpoint is not None:
            args.start_round = checkpoint['round'] + 1
            federator.global_model.load_state_dict(checkpoint['state_dict'])
            best_acc1_overall = checkpoint.get('best_acc1', 0.0)
            best_round_overall = checkpoint.get('round', 0)
            print(f"INFO: Resumed from checkpoint. Next round to train: {args.start_round}. Best Acc1 so far: {best_acc1_overall:.4f}")
        else:
            args.start_round = 0
    else:
        args.start_round = 0

    # 主循环
    for round_idx in range(args.start_round, total_rounds):
        
        # ==================== 代码修改开始 ====================
        
        # 检查是否进入最后50轮
        is_last_rounds = round_idx >= total_rounds - hrank_last_train
        
        # 定义一个需要触发HRank的标志
        trigger_hrank = False
        
        if is_last_rounds:
            # 在最后轮，我们只在进入这个阶段的第一轮时触发一次
            if round_idx == total_rounds - hrank_last_train:
                trigger_hrank = True
                use_max_rank = True
                print(f"\nINFO: >>>>> Final {hrank_last_train} rounds started. Triggering HRank with MAX rank (no recalc) at round {round_idx} <<<<<")
        # 如果不是最后轮，则检查是否满足每轮的条件
        elif (round_idx + 1) % hrank_trigger_round == 0:
            trigger_hrank = True
            use_max_rank = False
            print(f"\nINFO: >>>>> Periodic trigger. Triggering HRank with MIN rank (recalc) at round {round_idx} <<<<<")

        # 如果标志被设置，则执行HRank相关的操作
        if trigger_hrank:
            hrank_batch_size = args.hrank_representative_batch_size if hasattr(args, 'hrank_representative_batch_size') else batch_size
            hrank_dataset_source =  train_set
            if hrank_dataset_source is None or len(hrank_dataset_source) == 0:
                raise ValueError("No dataset available for HRank representative data.")

            collate_fn_for_hrank = None
            if 'bert' in args.arch:
                from data_tools.dataloader import collate_fn as bert_collate_fn
                collate_fn_for_hrank = bert_collate_fn
            
            num_hrank_samples = min(len(hrank_dataset_source), 1000)
            hrank_indices = np.random.choice(len(hrank_dataset_source), num_hrank_samples, replace=False if len(hrank_dataset_source) >= num_hrank_samples else True)

            representative_dataloader_for_hrank = torch.utils.data.DataLoader(
                DatasetSplit(hrank_dataset_source, hrank_indices),
                batch_size=hrank_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_for_hrank
            )
            
            # 使用新的参数调用 refine_masks_with_hrank
            federator.refine_masks_with_hrank(
                representative_dataloader_for_hrank,
                args,
                config,
                use_max_rank=use_max_rank,
            )
            print("INFO: >>>>> HRank refinement finished. Proceeding with training. <<<<<")

        # ==================== 代码修改结束 ====================
        
        # 调用 Federator 的 fed_train，但其内部不再有循环
        # 我们需要一个新的单轮训练函数
        # 此处我们直接调用 execute_round
        
        # 确保 federator.client_groups 已被初始化
        if not federator.client_groups:
            print("INFO: Client groups not found in federator. Initializing now...")
            fed_train_user_groups = user_groups[0] # 训练用
            client_idxs_init = np.arange(args.num_clients)
            np.random.seed(args.seed)
            shuffled_client_idxs_init = np.random.permutation(client_idxs_init)
            client_groups_init = []
            s = 0
            for ratio in args.client_split_ratios:
                e = s + int(len(shuffled_client_idxs_init) * ratio)
                client_groups_init.append(shuffled_client_idxs_init[s: e])
                s = e
            federator.client_groups = client_groups_init
            with open(os.path.join(args.save_path, 'client_groups.pkl'), 'wb') as f:
                pkl.dump(federator.client_groups, f)
            print("INFO: Client groups initialized and saved.")

        print(f'\n | Global Training Round : {round_idx + 1} |\n')
        train_loss, val_results, local_val_results = federator.execute_round(
            train_set, val_set, user_groups, criterion, args, batch_size,
            config.training_params[args.data][args.arch], round_idx
        )
        
        # 处理结果和保存模型
        val_loss, val_acc1, val_acc5, _, _ = val_results
        scores.append(('{}' + '\t{:.4f}' * int(6 + federator.num_levels))
                      .format(round_idx, train_loss, val_loss, val_acc1, val_acc5,
                              local_val_results[-1][1], local_val_results[-1][2],
                              *[l[1] for l in local_val_results[:-1]]))

        is_best = val_acc1 > best_acc1_overall
        if is_best:
            best_acc1_overall = val_acc1
            best_round_overall = round_idx
            print('Best var_acc1 {}'.format(best_acc1_overall))
        
        # 使用原始的 save_checkpoint
        save_checkpoint({
            'round': round_idx,
            'arch': args.arch,
            'state_dict': federator.global_model.state_dict(),
            'best_acc1': best_acc1_overall,
        }, args, is_best, 'checkpoint_%03d.pth.tar' % round_idx, scores)

    print(f'\nOverall Best val_acc1: {best_acc1_overall:.4f} at round {best_round_overall}')
    
    # 最终验证
    best_model_path = os.path.join(args.save_path, 'save_models', 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        print(f"INFO: Loading overall best model from {best_model_path} for final validation.")
        map_location = torch.device('cpu') if not (args.use_gpu and torch.cuda.is_available()) else None
        checkpoint = torch.load(best_model_path, map_location=map_location)
        
        final_val_model_params = {**config.model_params[args.data][args.arch]}
        final_val_model_skeleton = getattr(models, args.arch)(args, final_val_model_params)
        final_val_model_skeleton.load_state_dict(checkpoint['state_dict'])
        if args.use_gpu and torch.cuda.is_available():
            final_val_model_skeleton.cuda()
        
        validate(final_val_model_skeleton, test_loader, criterion, args, save=True)
    else:
        print(f"Warning: Overall best model checkpoint not found at {best_model_path}. Validating current global model from Federator instead.")
        model_to_validate_final = federator.global_model
        if args.use_gpu and torch.cuda.is_available():
            if next(model_to_validate_final.parameters()).device.type == 'cpu': model_to_validate_final.cuda()
        elif not (args.use_gpu and torch.cuda.is_available()) :
                if next(model_to_validate_final.parameters()).device.type == 'cuda': model_to_validate_final.cpu()

        validate(model_to_validate_final, test_loader, criterion, args, save=True)

    print(f"INFO: Training finished. Results saved in {args.save_path}")
    return

if __name__ == '__main__':
    main()