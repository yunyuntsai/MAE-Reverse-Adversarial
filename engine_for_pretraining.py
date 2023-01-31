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

import utils
from scipy import stats

from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy, ModelEma
import csv
from datasets import build_pretraining_dataset
from torchvision.transforms import ToPILImage
import os


def config_model(model):
    model.train(True)
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

def multi_train_one_epoch(model: torch.nn.Module, mlp_head: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    mlp_optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):

    model = config_model(model)
    mlp_head = config_mlp(mlp_head)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

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
            outputs = model(images, bool_masked_pos)
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



def compute_reverse_attack(model, criterion, X, labels, bool_masked_pos, epsilon, alpha, attack_iters, norm):
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


def compute_reconstruct_loss(model, criterion, x, labels, bool_masked_pos, no_grad=True):
    if no_grad:
        with torch.no_grad():
            outputs = model(x, bool_masked_pos)

    else:
        outputs = model(x, bool_masked_pos)
    
    loss = criterion(input=outputs, target=labels)

    return loss

def multi_evaluate(model: torch.nn.Module, mlp_head: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, normlize_target: bool = True, log_writer=None, patch_size: int = 16):
    model.eval()
    mlp_head.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(1)
    print_freq = 1
    epsilon = (8 / 255.)
    pgd_alpha = (2 / 255.)
    attack_iters = 5
    norm = 'l_inf'

    loss_func = nn.MSELoss()
    xent_loss_func = nn.CrossEntropyLoss()
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # hook = model.module.encoder.head2.register_forward_hook(get_activation('encoder_head'))
    hook = model.encoder.head2.register_forward_hook(get_activation('encoder_head'))

    index = 0
    for step, (batch, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):


        images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
            # delta = compute_reverse_attack(model, loss_func, images, labels, bool_masked_pos, epsilon, pgd_alpha, attack_iters, norm)
            
            # noise_images = utils.gaussian_noise(images)
            # rev_outputs = model(noise_images, bool_masked_pos)
            
            rev_outputs = model(images, bool_masked_pos) # new line
            loss = loss_func(input=rev_outputs, target=labels)
            encoder_fts2 = activation['encoder_head']
            rev_mlp_outputs = mlp_head(encoder_fts2)
            rev_xent_loss = xent_loss_func(rev_mlp_outputs, targets)

        loss_value = loss.item()
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
        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(acc1=acc1.item(), head="acc1")
        #     log_writer.update(acc5=acc5.item(), head="acc5")
        #     log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
      .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



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
    
    print_freq = 1
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

def save_reconstruct_img(ori_img, outputs, bool_masked_pos, patch_size, pred, rev_pred, idx, device):

    #save original img
    # if outputs2.max(1) == targets:
    save_path = '/local/rcs/yunyun/ImageNet-Data/Reverse/cw_l2'
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
    ori_img = ori_img * std + mean  # in [0, 1]
    pil_img = ToPILImage()(ori_img[0, :])

    pred = pred.item()
    rev_pred = rev_pred.item()
    pil_img.save(f"{save_path}/rev_advimg_{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg")

    img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    # print(img_patch.dtype)
    # print(outputs.dtype)
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
    img = ToPILImage()(rec_img[0, :])
    img.save(f"{save_path}/rec_rev_advimg_{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg")

    #save random mask img
    img_mask = rec_img * mask
    img = ToPILImage()(img_mask[0, :])
    img.save(f"{save_path}/mask_img_{str(idx)}_p1_{str(pred)}_p2_{str(rev_pred)}.jpg")
    