# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable
import re
import torch
import torch.nn as nn
import json
import utils
from scipy import stats

from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy, ModelEma
import csv
import torchvision
from datasets import build_pretraining_dataset
from torchvision.transforms import ToPILImage
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


def config_model(model):
    model.eval()
    # disable grad, to (re-)enable only what tent updates
    # model.requires_grad_(False)
    count_decoder = 0

    # print(len(model.modules()))
    # for name, m in model.named_modules():
    #     if re.match('module.decoder.*', name) is not None:
    #         count_decoder += 1
    #         m.requires_grad_(True)
    # print('count_decoder {}'.format(count_decoder))
    return model

def config_mlp(mlp_head):
    mlp_head.train(True)
    # disable grad, to (re-)enable only what tent updates
    return mlp_head


def contrast_multi_train_one_epoch(model: torch.nn.Module, mlp_head: torch.nn.Module, ssl_head: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, mlp_optimizer: torch.optim.Optimizer, ssl_optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):

    model = config_model(model)
    mlp_head.train(True)
    ssl_head.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()
    mlp_loss_func = nn.CrossEntropyLoss()
    contrast_transform = utils.ContrastiveTransform()
    # loss_func = nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook = model.module.encoder.head2.register_forward_hook(get_activation('encoder_head'))

    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos = batch
        bs = images.shape[0]

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        transform_bool_masked_pos = [ bool_masked_pos for i in range(4)]
        transform_bool_masked_pos = torch.vstack(transform_bool_masked_pos)
        transform_images = contrast_transform(images)
        tr_images = transform_images[0].squeeze(0)
        for i in range(1,4): 
            tr_images = torch.vstack((tr_images, transform_images[i].squeeze(0))) 
        # import pdb; pdb.set_trace()

        with torch.cuda.amp.autocast():
            
            _ = model(tr_images, transform_bool_masked_pos)
            encoder_fts = activation['encoder_head']
            ssl_output = ssl_head(encoder_fts)
            contrast_loss, _ = utils.contrastive_loss_func(ssl_output, mlp_loss_func, bs, 4)

            _ = model(images, bool_masked_pos)
            encoder_fts = activation['encoder_head']
            mlp_outputs = mlp_head(encoder_fts)
            xent_loss = mlp_loss_func(mlp_outputs, targets)

        contrast_loss_value = contrast_loss.item()
        xent_loss_value = xent_loss.item()

        if not math.isfinite(contrast_loss_value) or not math.isfinite(xent_loss_value):
            print("Loss is {} xent_loss is {}, stopping training".format(loss_value, xent_loss_value))
            sys.exit(1)

        ssl_optimizer.zero_grad()
        mlp_optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
        #                         parameters=model.parameters(), create_graph=is_second_order)
        
        # loss_scale_value = loss_scaler.state_dict()["scale"]

        is_second_order = hasattr(mlp_optimizer, 'is_second_order') and mlp_optimizer.is_second_order
        grad_norm = loss_scaler(xent_loss, mlp_optimizer, clip_grad=max_norm,
                                parameters=mlp_head.parameters(), create_graph=is_second_order)

        mlp_loss_scale_value = loss_scaler.state_dict()["scale"]

        is_second_order = hasattr(ssl_optimizer, 'is_second_order') and mlp_optimizer.is_second_order
        grad_norm = loss_scaler(contrast_loss, ssl_optimizer, clip_grad=max_norm,
                                parameters=ssl_head.parameters(), create_graph=is_second_order)

        ssl_loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss1=contrast_loss_value)
        metric_logger.update(loss2=xent_loss_value)
        metric_logger.update(loss_scale1=ssl_loss_scale_value)
        metric_logger.update(loss_scale2=mlp_loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss1=ssl_loss_value, head="loss")
            log_writer.update(loss2=xent_loss_value, head="loss")
            log_writer.update(loss_scale1=ssl_loss_scale_value, head="opt")
            log_writer.update(loss_scale2=mlp_loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def  multi_train_one_epoch(model: torch.nn.Module, mlp_head: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    mlp_optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):

    model = config_model(model)
    mlp_head = config_mlp(mlp_head)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()
    mlp_loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook = model.module.encoder.head2.register_forward_hook(get_activation('encoder_head'))

    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # clean_images, _ = cleans_batch
        # clean_images = clean_images.to(device, non_blocking=True)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs, _ = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)
            encoder_fts = activation['encoder_head']
            outputs2 = mlp_head(encoder_fts)
            xent_loss = mlp_loss_func(outputs2, targets)

        loss_value = loss.item()
        xent_loss_value = xent_loss.item()

        if not math.isfinite(loss_value) or not math.isfinite(xent_loss_value):
            print("Loss is {} xent_loss is {}, stopping training".format(loss_value, xent_loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        mlp_optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        
        loss_scale_value = loss_scaler.state_dict()["scale"]

        is_second_order = hasattr(mlp_optimizer, 'is_second_order') and mlp_optimizer.is_second_order
        grad_norm = loss_scaler(xent_loss, mlp_optimizer, clip_grad=max_norm,
                                parameters=mlp_head.parameters(), create_graph=is_second_order)

        mlp_loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss1=loss_value)
        metric_logger.update(loss2=xent_loss_value)
        metric_logger.update(loss_scale1=loss_scale_value)
        metric_logger.update(loss_scale2=mlp_loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss1=loss_value, head="loss")
            log_writer.update(loss2=xent_loss_value, head="loss")
            log_writer.update(loss_scale1=loss_scale_value, head="opt")
            log_writer.update(loss_scale2=mlp_loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def pgd_defense_aware_attack(backbone_model, mae_model, images, labels, bool_masked_pos, criterion, device, lambda_s, eps=0.3, alpha=2/255, iters=5) :
    # images = images.to(device)
    # labels = labels.to(device)
    # loss = nn.CrossEntropyLoss()
    print('attack iter: ', iters)
    ori_images = images.data

        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = backbone_model(images)
        mae_labels = get_labels(images, bool_masked_pos, device, True, 16)
        # import pdb; pdb.set_trace()
        backbone_model.zero_grad()
        s_loss = -compute_reconstruct_loss(mae_model, criterion, images, mae_labels, bool_masked_pos, no_grad=False).to(device)
        c_loss = F.cross_entropy(outputs, labels).to(device)
        cost = c_loss + lambda_s * s_loss
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def compute_defense_aware_attack(model, attack_backbone, criterion, X, targets, labels, bool_masked_pos, epsilon, alpha, attack_iters, norm, device):
    """Reverse algorithm that optimize the SSL loss via PGD"""
    
    std = torch.as_tensor(IMAGENET_DEFAULT_STD)#.cuda()[None, :, None, None]
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN)
    epsilon = (epsilon / min(std)).cuda()
    alpha = (alpha / min(std)).cuda()

    lambda_s = 0.0
    
    delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError
    #delta = clamp(delta, lower_limit - torch.mean(X, dim=0), upper_limit - torch.mean(X, dim=0))
    delta.requires_grad = True

    for _ in range(attack_iters):

        new_x = X + delta
        labels = get_labels(new_x, bool_masked_pos, device, True, 16)
        # import pdb; pdb.set_trace()
        s_loss = -compute_reconstruct_loss(model, criterion, new_x, labels, bool_masked_pos, no_grad=False)
        
        h_adv = attack_backbone(new_x)
        c_loss = F.cross_entropy(h_adv, targets)

        loss = (lambda_s * s_loss) + c_loss
        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad
        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        elif norm == "l_1":
            g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

        #d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()

    max_delta = delta.detach()
    return max_delta

def compute_reverse_attack(model, criterion, X, labels, bool_masked_pos, epsilon, alpha, attack_iters, norm, device):
    """Reverse algorithm that optimize the SSL loss via PGD"""

    delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError
    #delta = clamp(delta, lower_limit - torch.mean(X, dim=0), upper_limit - torch.mean(X, dim=0))

    delta.requires_grad = True
    for _ in range(attack_iters):

        new_x = X + delta
        labels = get_labels(new_x, bool_masked_pos, device, True, 16)
        # import pdb; pdb.set_trace()
        loss = -compute_reconstruct_loss(model, criterion, new_x, labels, bool_masked_pos, no_grad=False)
        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad
        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        elif norm == "l_1":
            g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

        #d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()
    
    max_delta = delta.detach()
    return max_delta

def compute_contrast_loss(model, ssl_head, criterion, x, bs, bool_masked_pos, no_grad=False):


    if no_grad:
        with torch.no_grad():
            _, all_head = model(x, bool_masked_pos)
    else:
        _, all_head = model(x, bool_masked_pos)

    encoder_fts = all_head
    ssl_output = ssl_head(encoder_fts)
    contrast_loss, _ = utils.contrastive_loss_func(ssl_output, criterion, bs, 4)
    return contrast_loss

def compute_reconstruct_loss(model, criterion, x, labels, bool_masked_pos, no_grad=True):
    if no_grad:
        with torch.no_grad():
            outputs, _ = model(x, bool_masked_pos)

    else:
        outputs, _ = model(x, bool_masked_pos)
    
    loss = criterion(input=outputs, target=labels)

    return loss

def compute_contrast_reverse_attack(model, ssl_head, criterion, X, bool_masked_pos, epsilon, alpha, attack_iters, norm):
    """Reverse algorithm that optimize the SSL loss via PGD"""


    transform_bool_masked_pos = [ bool_masked_pos for i in range(4)]
    transform_bool_masked_pos = torch.vstack(transform_bool_masked_pos)

    contrastive_transform = utils.ContrastiveTransform()

    delta = torch.unsqueeze(torch.zeros_like(X[0]).cuda(), 0)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError
    #delta = clamp(delta, lower_limit - torch.mean(X, dim=0), upper_limit - torch.mean(X, dim=0))
    delta.requires_grad = True
    for _ in range(attack_iters):

        new_x = X + delta
        transform_x = contrastive_transform(new_x)

        # import pdb; pdb.set_trace()
        loss = -compute_contrast_loss(model, ssl_head, criterion, transform_x, X.shape[0], transform_bool_masked_pos, no_grad=False)
        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad
        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        elif norm == "l_1":
            g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

        #d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    return max_delta


def contrast_multi_evaluate(model: torch.nn.Module, ssl_head: torch.nn.Module, mlp_head: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, normlize_target: bool = True, log_writer=None, patch_size: int = 16):
    model.eval()
    ssl_head.eval()
    mlp_head.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(1)
    print_freq = 10
    epsilon = (8 / 255.)
    pgd_alpha = (2 / 255.)
    attack_iters = 1
    norm = 'l_inf'

    # loss_func = nn.MSELoss()
    xent_loss_func = nn.CrossEntropyLoss()
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook = model.module.encoder.head2.register_forward_hook(get_activation('encoder_head'))
    index = 0
    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):


        images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.cuda.amp.autocast():
            
            delta = compute_contrast_reverse_attack(model, ssl_head, xent_loss_func, images, bool_masked_pos, epsilon, pgd_alpha, attack_iters, norm)

            _, _ = model(images+delta, bool_masked_pos)
            # loss = loss_func(input=rev_outputs, target=labels)
            encoder_fts2 = activation['encoder_head']
            rev_mlp_outputs = mlp_head(encoder_fts2)
            rev_xent_loss = xent_loss_func(rev_mlp_outputs, targets)

        loss_value = rev_xent_loss.item()
        # print(outputs2.max(1))
        # print(targets)
        # acc1, acc5 = accuracy(mlp_outputs, targets, topk=(1,5))
        acc1, acc5 = accuracy(rev_mlp_outputs, targets, topk=(1,5))
        # print('index: ', step, 'pred: ', outputs2.max(1)[1], 'labels: ',  targets)
        # if mlp_outputs.max(1)[1] != targets and rev_mlp_outputs.max(1)[1] == targets:
        #     print('index: ', step, 'pred: ', mlp_outputs.max(1)[1].item(), 'rev pred: ', rev_mlp_outputs.max(1)[1].item(),  'labels: ',  targets.item())
        #     save_reconstruct_img(images + delta, rev_outputs, bool_masked_pos, patch_size, mlp_outputs.max(1)[1], rev_mlp_outputs.max(1)[1], index, device)
        index+=1
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
      .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_labels(images, bool_masked_pos, device: torch.device, normlize_target: bool = True, patch_size: int = 16):
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        # calculate the predict label
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        unnorm_images = images * std + mean  # in [0, 1]

    if normlize_target:
        images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
        images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # we find that the mean is about 0.48 and standard deviation is about 0.08.
        images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
    else:
        images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    B, _, C = images_patch.shape
    labels = images_patch[bool_masked_pos].reshape(B, -1, C)
    return labels

def load_attack_model(checkpoint_path, pretrained_model):
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            pretrained_model.load_state_dict(checkpoint)
            print("=> load chechpoint found at {}".format(checkpoint_path))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    # No gradients needed for the network
    pretrained_model.eval()
    return pretrained_model


def noisy_img(img, n_radius):
    return img + n_radius * torch.randn_like(img)

class Noisy(torch.autograd.Function):
    @staticmethod
    def forward(self, img, n_radius):
        return noisy_img(img, n_radius=n_radius)

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None


def multi_evaluate(model: torch.nn.Module, mlp_head: torch.nn.Module, data_loader: Iterable, data_loader_orig: Iterable,
                    device: torch.device, normlize_target: bool = True, log_writer=None, patch_size: int = 16, attack_iter: int=4, epsilon: float=4/255, lambda_s: int=2, output_dir=''):
    
    pretrained_model = torchvision.models.resnet50(pretrained=True)
    pretrained_model = pretrained_model.to(device)
    attack_model = load_attack_model("/local/rcs/yunyun/SSDG-main/resnet50.pth", pretrained_model)
    model.eval()
    mlp_head.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(1)
    print_freq = 10
    # epsilon = (8 / 255.)
    pgd_alpha = ( 8/ 255.)
    raw_correct, da_correct, rec_correct = 0, 0, 0
    test_size = 0
    # attack_iters = 1
    norm = 'l_inf'
    adv_l1_dist_list = []
    orig_l1_dist_list = []
    before_loss_list = []
    noise_loss_list = []
    after_attack_loss_list = []
    after_rec_loss_list = []

    loss_func = nn.MSELoss()
    xent_loss_func = nn.CrossEntropyLoss()
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook = model.module.encoder.head2.register_forward_hook(get_activation('encoder_head'))
    index = 0
    n_radius = 0.1

    raw_dataloader_iterater = iter(data_loader_orig)
    noisy = Noisy.apply

    norm_transforms = torch.nn.Sequential(
            transforms.Normalize(mean=[ 0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225])
    )
    NMtransforms = torch.jit.script(norm_transforms)

    inv_transforms = torch.nn.Sequential(
            transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1.])
    )
    INVtransforms = torch.jit.script(inv_transforms)

    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if step < 10:
            raw_batch = next(raw_dataloader_iterater)
            raw_images, _ = raw_batch[0]
            raw_images = raw_images.to(device, non_blocking=True)

            images, bool_masked_pos = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            # images = batch['input'].to(device, non_blocking=True)
            # targets = batch['target'].to(device, non_blocking=True)
            # bool_masked_pos = batch['mask'].to(device, non_blocking=True)

            labels = get_labels(images, bool_masked_pos, device, normlize_target, patch_size)
            raw_labels = get_labels(raw_images, bool_masked_pos, device, normlize_target, patch_size)
            with torch.no_grad():
                before_outputs, _ = model(images, bool_masked_pos)
                before_loss = loss_func(input=before_outputs, target=labels)
                before_loss_list.append(before_loss.item())
                encoder_fts1 = activation['encoder_head']
                mlp_outputs = mlp_head(encoder_fts1)

            # noisy_before_outputs, _ = model(noisy(images, n_radius), bool_masked_pos)
            # noisy_encoder_fts1 = activation['encoder_head']
            # noisy_mlp_outputs = mlp_head(noisy_encoder_fts1)
            with torch.no_grad():
                before_raw_outputs, _ = model(raw_images, bool_masked_pos)
                encoder_fts2 = activation['encoder_head']
                raw_mlp_outputs = mlp_head(encoder_fts2)

            # noisy_raw_before_outputs, _ = model(noisy(raw_images, n_radius), bool_masked_pos)
            # noisy_encoder_fts2 = activation['encoder_head']
            # noisy_raw_mlp_outputs = mlp_head(noisy_encoder_fts2)

            # adv_l1_dist = torch.norm(F.softmax(mlp_outputs) - F.softmax(noisy_mlp_outputs), 1).item() 
            # orig_l1_dist = torch.norm(F.softmax(raw_mlp_outputs) - F.softmax(noisy_raw_mlp_outputs), 1).item()
            # adv_l1_dist_list.append(adv_l1_dist)
            # orig_l1_dist_list.append(orig_l1_dist)

            # print('l1 distance detection--> orig: {} / adv: {}'.format(orig_l1_dist, adv_l1_dist))

            with torch.cuda.amp.autocast():
                rec_images = images
                print('raw image range: ', images.max(), images.min())
            
                new_da_images = pgd_defense_aware_attack(attack_model, model, images, targets, bool_masked_pos, loss_func, device, lambda_s)
                    # da_delta = compute_defense_aware_attack(model, attack_model, loss_func, images, targets, labels, bool_masked_pos, epsilon, pgd_alpha, attack_iter, norm, device)
                    # da_images = images + da_delta
                    # t = torch.clamp(INVtransforms(da_images), 0, 1)
                    # norm_da_images = torch.clamp(torch.add(torch.mul(new_da_images, 255), 0.5), 0, 255).to("cpu", torch.uint8)
                    # 
                    # for j in range(new_da_images.shape[0]):
                    #     # cls_idx = class_idx[str(labels[j].item())][0]
                    #     index = step * new_da_images.shape[0] + j
                    #     # os.makedirs(os.path.join(save_attack_folder, cls_idx), exist_ok = True)
                    #     img_file_path =  './output/da_sample/' + 'img_' + str(index).zfill(5) + '.png'
                    #     print(img_file_path)
                    #     t = INVtransforms(new_da_images[j])
                    #     t = torch.clamp(torch.mul(t, 255), 0, 255).to("cpu", torch.uint8)
                    #     new_da_img = INVtransforms(ToPILImage()(t))
                    #     new_da_img.save(img_file_path)

                        # plt.imsave(transforms.ToPILImage()(t), img_file_path)
                if attack_iter != -1:
                    new_da_images = NMtransforms(new_da_images)
                    print('da image range: ', new_da_images.max(), new_da_images.min())
                    delta = compute_reverse_attack(model, loss_func, new_da_images, labels, bool_masked_pos, epsilon, pgd_alpha, attack_iter, norm, device)
                    rec_images = new_da_images + delta

                with torch.no_grad():

                    rev_outputs, _ = model(new_da_images, bool_masked_pos)
                    labels = get_labels(rec_images, bool_masked_pos, device, True, 16)
                    loss = loss_func(input=rev_outputs, target=labels)
                    after_attack_loss_list.append(loss.item())

                    rev_outputs, _ = model(rec_images, bool_masked_pos)
                    labels = get_labels(rec_images, bool_masked_pos, device, True, 16)
                    loss = loss_func(input=rev_outputs, target=labels)
                    after_rec_loss_list.append(loss.item())
                    encoder_fts2 = activation['encoder_head']
                    rev_mlp_outputs = mlp_head(encoder_fts2)
                    rev_xent_loss = xent_loss_func(rev_mlp_outputs, targets)
        
                # print(targets.item())
                # if targets.item()==0:
                    # logits_list.append([encoder_fts2.cpu().detach(), rev_mlp_outputs.max(1)[1].cpu().detach()])

            loss_value = loss.item()
            acc1, acc5 = accuracy(rev_mlp_outputs, targets, topk=(1,5))
            with torch.no_grad():
                outputs = attack_model(raw_images)
                _, pre = torch.max(outputs.data, 1)
                raw_correct += (pre == targets).sum()
            if attack_iter != -1:
                with torch.no_grad():
                    outputs = attack_model(new_da_images)
                    _, pre = torch.max(outputs.data, 1)
                    da_correct += (pre == targets).sum()

                    outputs = attack_model(rec_images)
                    _, pre = torch.max(outputs.data, 1)
                    rec_correct += (pre == targets).sum()

            test_size += images.shape[0]
            
            # print('index: ', step, 'pred: ', outputs2.max(1)[1], 'labels: ',  targets)
            # if mlp_outputs.max(1)[1] != targets and rev_mlp_outputs.max(1)[1] == targets:
                # print('index: ', step, 'pred: ', mlp_outputs.max(1)[1].item(), 'rev pred: ', rev_mlp_outputs.max(1)[1].item(),  'labels: ',  targets.item())
                # save_reconstruct_img(raw_images, images + delta, rev_outputs, bool_masked_pos, patch_size, targets.item(), mlp_outputs.max(1)[1], rev_mlp_outputs.max(1)[1], index, device)
            
            index+=1
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            batch_size = images.shape[0]
            metric_logger.update(loss=loss_value)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            print('correct before attack --> ', raw_correct.item()/test_size) 
            if attack_iter != -1:
                print('correct after defense aware attack --> ', da_correct.item()/test_size)
                print('correct after reconstruction --> ', rec_correct.item()/test_size)
            # if log_writer is not None:
            #     log_writer.update(loss=loss_value, head="loss")
            #     log_writer.update(acc1=acc1.item(), head="acc1")
            #     log_writer.update(acc5=acc5.item(), head="acc5")
            #     log_writer.set_step()
    with open(os.path.join(output_dir, 'daa_loss_ls' + str(lambda_s) + '.npy'), 'wb') as f:
        np.save(f, [before_loss_list, after_attack_loss_list, after_rec_loss_list])
    with open(os.path.join(output_dir, 'daa_acc_ls' + str(lambda_s) + '.npy'), 'wb') as f:
        np.save(f, [raw_correct.item()/test_size, da_correct.item()/test_size, rec_correct.item()/test_size])

    # gather the stats from all processes
    # with open('./output/l1_dist_cw2.npy', 'wb') as f:
    #     np.save(f, [orig_l1_dist_list, adv_l1_dist_list])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
      .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, rec_correct.item()/test_size

def anamoly_detection(args, model: torch.nn.Module, src_data_loader: Iterable,
                    device: torch.device, normlize_target: bool = True, log_writer=None, patch_size: int = 16):
    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    src_loss = inference_model(model, src_data_loader, device, normlize_target, metric_logger, patch_size)
    print('src loss: {}'.format(src_loss))
    tar_data_path = os.listdir(args.tar_data_path)
    for t in sorted(tar_data_path):
        if t != '00':
            cls_data_path = os.path.join(args.tar_data_path, t)
            dataset_tar = build_pretraining_dataset(cls_data_path, args)
            sampler_tar = torch.utils.data.RandomSampler(dataset_tar)

            data_loader_tar = torch.utils.data.DataLoader(
                    dataset_tar, sampler=sampler_tar,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                    worker_init_fn=utils.seed_worker
                )
    
            tar_loss = inference_model(model, data_loader_tar, device, normlize_target, metric_logger, patch_size)
            print(tar_loss)
            state, p_value = stats.ks_2samp(src_loss, tar_loss)

            sample_size = 4
            acc = 0.
            avg_pvalue = 0.

            threshold = 0.05
            for i in range(len(tar_loss)//sample_size):

                state, p_value = stats.ks_2samp(src_loss, tar_loss[i*sample_size: (i+1)*sample_size])
                avg_pvalue += p_value
                if p_value < threshold:
                    acc += 1

            num_step = len(tar_loss) // sample_size
            print('class : {} Anamoly detection accuracy: {}'.format(t, acc / num_step))
            path = args.output_dir + 'anamoly_detection.csv'
            with open(path, 'a') as f:
                print('write_file to ', path)
                writer = csv.writer(f)
                writer.writerow(['class: ', t, 'acc: ', acc / num_step, 'pvalue: ', avg_pvalue / num_step])
                # writer.writerow(['src loss:', src_loss])
                # writer.writerow(['class :', t , 'tar loss:', tar_loss])
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # return acc / num_step, avg_pvalue / num_step


def inference_model(model: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, normlize_target: bool = True, metric_logger=None, patch_size: int = 16):
    
    print_freq = 10
    header = 'Epoch: [{}]'.format(1)
    loss_list = []
    loss_func = nn.MSELoss()

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):


        images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs = model(images, bool_masked_pos)
            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()
        loss_list.append(loss_value)
        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        # batch_size = images.shape[0]
        # metric_logger.update(loss=loss_value)
    return loss_list

def save_reconstruct_img(raw_img, ori_img, outputs, bool_masked_pos, patch_size, target, pred, rev_pred, idx, device):

    class_idx = json.load(open("/local/rcs/yunyun/ImageNet-Data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    print('orig. predict label: ', idx2label[pred], 'reverse predict label: ', idx2label[rev_pred])
    #save original img
    # if outputs2.max(1) == targets:
    save_path = '/local/rcs/yunyun/MAE-Reverse-Adversarial/output/reconstruct_mae_v2_pretrain_model/vis_sample/cwl2_new'
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
    
    raw_img = raw_img * std + mean

    ori_img = ori_img * std + mean  # in [0, 1]

    diff = ori_img - raw_img

    att_img = raw_img + (diff*2)

    pil_img = ToPILImage()(ori_img[0, :])

    pred = pred.item()
    rev_pred = rev_pred.item()
    # pil_img.save(f"{save_path}/rev_advimg_{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg")

    img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    img_patch[bool_masked_pos] = outputs.to(torch.float32)

    #make mask
    mask = torch.ones_like(img_patch)
    mask[bool_masked_pos] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)

    #save reconstruction img
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
    rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
    rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
    # img = ToPILImage()(rec_img[0, :].clip(0,0.996))
    Rec_img = ToPILImage()(rec_img[0, :])
    # img.save(f"{save_path}/rec_rev_advimg_{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg")

    #save random mask img
    img_mask = rec_img * mask
    masked_img = ToPILImage()(img_mask[0, :])
    # img.save(f"{save_path}/mask_img_{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg")


    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

    line = plt.Line2D((.254,.254),(.05,.95), color="dimgray", linewidth=2)
    fig.add_artist(line)

    ax[0].set_title('Clean images', fontsize=18)
    ax[0].imshow(np.transpose(raw_img[0, :].cpu().detach().numpy(), (1,2,0)))
    ax[0].set_xlabel('Ground Truth: \n'+ idx2label[target], fontsize=18)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_title('Attacked images', fontsize=18)
    ax[1].imshow(np.transpose(att_img[0, :].cpu().detach().numpy(), (1,2,0)))
    ax[1].set_xlabel('Predicted class: \n'+ idx2label[pred], fontsize=18)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_title('Masked images', fontsize=18)
    ax[2].imshow(np.transpose(img_mask[0, :].cpu().detach().numpy(), (1,2,0)))
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[3].set_title('Reconstructed images', fontsize=18)
    ax[3].imshow(np.transpose(rec_img[0, :].cpu().detach().numpy(), (1,2,0)))
    ax[3].set_xlabel('Predicted class: \n'+ idx2label[rev_pred], fontsize=18)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    # ax[3].title.set_text('Attention heatmap (red region)')
    # ax[3].imshow(np.uint8(gcam_final_mask))
    # ax[3].title.set_text('Adv. Image + GradCam \n Cos. Similarity: {:.3f}'.format(ssim[0][0]))
    # ax[3].imshow(np.uint8(mix_gcam[0]))  

    plt.tight_layout()
    plt.savefig(f"{save_path}/{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg", dpi=200)
    