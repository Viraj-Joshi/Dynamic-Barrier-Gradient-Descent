# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modified from AttentiveNAS (https://github.com/facebookresearch/AttentiveNAS)
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
from datetime import date

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import grad

import models
from utils.config import setup
from utils.flops_counter import count_net_flops_and_params
import utils.comm as comm
import utils.saver as saver
import utils.logging as logging
from evaluate import attentive_nas_eval as attentive_nas_eval

from data.data_loader import build_data_loader
from utils.progress import AverageMeter, ProgressMeter, accuracy
from solver import build_optimizer, build_lr_scheduler
import argparse
import loss_ops as loss_ops 

parser = argparse.ArgumentParser(description='AlphaNet Training')
parser.add_argument('--config-file', default=None, type=str, 
                    help='training configuration')
parser.add_argument('--model', default='a0', type=str, choices=['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a5_1', 'a6'])
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
run_args = parser.parse_args()
logger = logging.get_logger(__name__)

def main():
    torch.cuda.empty_cache()
    args = setup(run_args.config_file)
    args.model = run_args.model
    args.gpu = run_args.gpu
    args.batch_size = args.batch_size_per_gpu
    args.models_save_dir = os.path.join(args.models_save_dir, args.exp_name)
    if not os.path.exists(args.models_save_dir):
        os.makedirs(args.models_save_dir)
    #rescale base lr
    # args.lr_scheduler.base_lr = args.lr_scheduler.base_lr * (max(1, args.batch_size_total // 256))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)

    criterion = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).cuda(args.gpu)
    soft_criterion = loss_ops.AdaptiveLossSoft(args.alpha_min, args.alpha_max, args.iw_clip).cuda(args.gpu)


    train_loader, val_loader, train_sampler = build_data_loader(args)
    args.n_iters_per_epoch = len(train_loader)

    ## build model
    logger.info("=> creating model '{}'".format(args.arch))
    args.__dict__['active_subnet'] = args.__dict__['pareto_models'][args.model]
    print(args.active_subnet)
    model = models.model_factory.create_model(args)

    model.to(args.gpu)

    logger.info( f'building optimizer and lr scheduler, \
            local rank {args.gpu}')
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = build_optimizer(args,model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    if args.resume:
        saver.load_checkpoints(args, model, optimizer, lr_scheduler, logger)

    for epoch in range(args.start_epoch, args.epochs):
        args.curr_epoch = epoch
        logger.info('Training lr {}'.format(lr_scheduler.get_lr()[0]))

        # train for one epoch
        acc1, acc5 = train_epoch(epoch, model, train_loader, optimizer, criterion ,args, soft_criterion=soft_criterion,lr_scheduler=lr_scheduler)

        if comm.is_master_process() or args.distributed:
            # validate supernet model
            validate(
                train_loader, val_loader, model, criterion, args
            )

        if comm.is_master_process():
            # save checkpoints
            saver.save_checkpoint(
                args.checkpoint_save_path, 
                model,
                optimizer,
                lr_scheduler, 
                args,
                epoch,
            )
def train_epoch(
    epoch, 
    model, 
    train_loader, 
    optimizer,
    criterion,
    args,
    soft_criterion=None,
    lr_scheduler=None, 
):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    num_updates = epoch * len(train_loader)

    for batch_idx, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # total subnets to be sampled
        num_subnet_training = max(2, getattr(args, 'num_arch_training', 2))
        optimizer.zero_grad()

        ### compute gradients using sandwich rule ###
        # step 1 sample the largest network, apply regularization to only the largest network
        drop_connect_only_last_two_stages = getattr(args, 'drop_connect_only_last_two_stages', True)
        model.sample_max_subnet()
        model.set_dropout_rate(args.dropout, args.drop_connect, drop_connect_only_last_two_stages) #dropout for supernet
        output = model(images)
        loss = criterion(output, target)
        loss.backward()

        with torch.no_grad():
            soft_logits = output.clone().detach()

        #step 2. sample the smallest network and several random networks
        sandwich_rule = getattr(args, 'sandwich_rule', True)
        model.set_dropout_rate(0, 0, drop_connect_only_last_two_stages)  #reset dropout rate
        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training-1 and sandwich_rule:
                model.sample_min_subnet()
            else:
                model.sample_active_subnet()

            # calcualting loss
            output = model(images)
            
            if soft_criterion:
                loss = soft_criterion(output, soft_logits) # g
            else:
                assert not args.inplace_distill
                loss = criterion(output, target) # g

            # loss.backward()

        #clip gradients if specfied
        # if getattr(args, 'grad_clip_value', None):
        #     torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)
        alpha,beta = 1,1

        grad_g = list(grad(loss,model.parameters(),retain_graph=True, allow_unused=True))
        
        c=.01
        f = torch.sqrt(sum([torch.sum(p**2) for p in model.parameters()]))**2

        grad_f = list(grad(f,model.parameters()))

        for p, dg, df in zip(model.parameters(), grad_g, grad_f):
            if df is None or dg is None:
                continue
            phi = min(alpha * (loss-c),beta * torch.linalg.norm(dg.flatten(),ord=2)**2)

            lam = max(0,(phi-torch.dot(dg.flatten(),df.flatten()))/torch.linalg.norm(dg.flatten(),ord=2)**2)

            delta = df+lam*dg
            # print(p.shape,dg.shape,df.shape,delta.shape)

            p.grad = delta
        optimizer.step()
            

        #accuracy measured on the local batch
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if args.distributed:
            corr1, corr5, loss = acc1*args.batch_size, acc5*args.batch_size, loss.item()*args.batch_size #just in case the batch size is different on different nodes
            stats = torch.tensor([corr1, corr5, loss, args.batch_size], device=args.gpu)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1/batch_size, corr5/batch_size, loss/batch_size
            losses.update(loss, batch_size)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
        else:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx, logger)
        del images,target

    return top1.avg, top5.avg
def validate(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    args, 
    distributed = True,
):
    subnets_to_be_evaluated = {
        'attentive_nas_min_net': {},
        'attentive_nas_max_net': {},
    }

    acc1_list, acc5_list = attentive_nas_eval.validate(
        subnets_to_be_evaluated,
        train_loader,
        val_loader, 
        model, 
        criterion,
        args,
        logger,
        bn_calibration = True,
    )
if __name__ == '__main__':
    main()