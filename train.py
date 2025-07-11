#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from utils.utils import adjust_learning_rate, accuracy, AverageMeter


def execute_epoch(model, train_loader, criterion, optimizer, round, epoch, args, train_params,
                  h_level, level, global_model, beta, delta, target_pruning_ratios, rank_sub_batch_size):
    """
    Executes one training epoch for a client model, incorporating pruning and rank guidance losses.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()  # This will track the total loss
    top1, top5 = [], []
    for i in range(h_level):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.train()
    end = time.time()

    for i, (inp, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, round, train_params)
        data_time.update(time.time() - end)

        if args.use_gpu:
            inp = inp.cuda()
            target = target.cuda()

         # 1. Call the updated model's forward pass with new hyperparameters
        # It now returns predictions and two additional loss components.
        output, loss_rank, loss_pruning_target = model(
            inp,
            manual_early_exit_index=h_level,
            # CRITICAL FIX: Changed from 'target_pruning_ratio' to 'target_pruning_ratios'
            target_pruning_ratios=target_pruning_ratios, 
            rank_sub_batch_size=rank_sub_batch_size,
            beta=beta,
            delta=delta
        )

        if not isinstance(output, list):
            output = [output]

        # 2. Calculate the original classification/KD loss
        original_loss = 0.0
        for j in range(len(output)):
            if j == len(output) - 1:
                original_loss += criterion.ce_loss(output[j], target) * (j + 1)
            else:
                gamma_active = round > args.num_rounds * 0.25
                original_loss += criterion.loss_fn_kd(output[j], target, output[-1], gamma_active) * (j + 1)
        
        # Normalize the classification/KD loss component
        original_loss /= len(output) * (len(output) + 1) / 2

        # 3. Compute the total loss by combining all components
        # The new losses from the model are already scalars.
        total_loss = (1-beta-delta)*original_loss + beta * loss_rank + delta * loss_pruning_target

        # Update accuracy meters
        for j in range(len(output)):
            if 'bert' in args.arch:
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), inp.size(0))
            top5[j].update(prec5.item(), inp.size(0))

        # 4. Compute gradient based on total_loss and do SGD step
        optimizer.zero_grad()
        losses.update(total_loss.item(), inp.size(0))
        total_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i + 1}/{len(train_loader)}]\t\t' +
                  f'Exit: {len(output)}\t' +
                  f'Time: {batch_time.avg:.3f}\t' +
                  f'Data: {data_time.avg:.3f}\t' +
                  f'Loss: {losses.val:.4f}\t' + # This now prints total loss
                  f'Acc@1: {top1[-1].val:.4f}\t' +
                  f'Acc@5: {top5[-1].val:.4f}')

    return losses.avg