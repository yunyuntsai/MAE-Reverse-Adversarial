# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import re
from pathlib import Path
import torch.utils.data as data

from timm.models import create_model
from optim_factory import create_optimizer
import torch.nn as nn 
from datasets import build_pretraining_dataset, gen_txt, ConcatTrainDataset
from engine_for_pretraining import multi_train_one_epoch, multi_evaluate, anamoly_detection
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from collections import OrderedDict
from modeling_pretrain import Mlp
import modeling_pretrain
import random
import csv

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--two_data_paths', default=False, type=bool, help='data_path') ## NEW 
    parser.add_argument('--data_path1', default='/datasets01/imagenet_full_size/061417/train', type=str,
                        help='dataset path')
    parser.add_argument('--data_path2', default='/datasets01/imagenet_full_size/061417/train', type=str,
                        help='dataset path')

    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/train', type=str,
    #                     help='dataset path')
    
    parser.add_argument('--src_data_path', default='/datasets01/imagenet_full_size/061417/train', type=str,
                        help='dataset path')
    parser.add_argument('--tar_data_path', default='/datasets01/imagenet_full_size/061417/train', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--mlp_resume', default='', help='resume mlp from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def get_Mlp(args):
    print(f"Creating Mlp head")

    dim = 768
    mlp_ratio = 2
    drop = 0.0
    mlp_hidden_dim = int(dim * mlp_ratio)
    
    mlp_head = Mlp(
        in_features=768, 
        hidden_features=mlp_hidden_dim, 
        out_features=1000,
        act_layer=nn.GELU, 
        drop=drop
    )

    return mlp_head


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    mlp_head = get_Mlp(args)

    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]

                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()


        for k in ['fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model.keys():
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('blocks.'):
                    new_dict['backbone.' + key[9:]] = checkpoint_model[key]
            if key.startswith('encoder.'):
                new_dict['encoder.' + key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

         # interpolate position embedding
        if 'encoder.pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['encoder.pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['encoder.pos_embed'] = new_pos_embed
        print('model prefix: ', args.model_prefix)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        # model.load_state_dict(checkpoint_model, strict=False)

    # get dataset
    if args.eval:
        if args.two_data_paths == False:
            dataset_src = build_pretraining_dataset(args.data_path1, args)
        else: 
            data_list = [build_pretraining_dataset(args.data_path1, args), build_pretraining_dataset(args.data_path2, args)]
            dataset_src = data.ConcatDataset(data_list)
        
        # # TODO: dataset_src = data.ConcatDataset(train_data_list)
        # train_data_list = list()
        # # txt_train_list = os.path.join(args.output_dir, 'train_list.txt')
        # advdata_dir = args.data_path
        # img_train_clean =  advdata_dir + '/clean/train'
        # img_train_adv =  advdata_dir  + '/unknown_adv/04/train'
        # # gen_txt(txt_train_list, img_train_clean, img_train_adv)
        # train_data_list.append(ConcatTrainDataset(txt_train_list, args))
        # # multiple_dataset = data.ConcatDataset(train_data_list)


    else: 
        # dataset_trainclean = build_pretraining_dataset(os.path.join(args.data_path, 'clean'), args)
        dataset_train = build_pretraining_dataset(args.data_path, args)

        # print('len of labels: {}'.format(np.unique((dataset_train.targets))))

        # train_data_list = list()
        # txt_train_list = os.path.join(args.output_dir, 'train_list.txt')
        # advdata_dir = args.data_path
        # img_train_clean =  advdata_dir + '/clean/train'
        # img_train_adv =  advdata_dir  + '/unknown_adv/04/train'
        # gen_txt(txt_train_list, img_train_clean, img_train_adv)
        # train_data_list.append(ConcatTrainDataset(txt_train_list, args))

        # multiple_dataset = data.ConcatDataset(train_data_list)
        # example_num = len(multiple_dataset)
        # idx_input = random.sample(range(0,example_num),int(example_num)) #100% data_portion
        # sub_dataset = data.Subset(multiple_dataset, idx_input)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        if args.eval:
            num_training_steps_per_epoch = len(dataset_src) // args.batch_size // num_tasks

            sampler_src = torch.utils.data.DistributedSampler(
                # dataset_src, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
                dataset_src, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
            )
            print("Sampler_src = %s" % str(sampler_src))

            # sampler_tar = torch.utils.data.DistributedSampler(
            #     dataset_tar, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            # )
            # print("Sampler_src = %s" % str(sampler_tar))
        else:
            num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
    else:
        if args.eval:
            sampler_src = torch.utils.data.RandomSampler(dataset_src)
            # sampler_tar = torch.utils.data.RandomSampler(dataset_tar)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    

    if args.eval:
        data_loader_src = torch.utils.data.DataLoader(
            dataset_src, sampler=sampler_src,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            worker_init_fn=utils.seed_worker
        )
        # data_loader_tar = torch.utils.data.DataLoader(
        #     dataset_tar, sampler=sampler_tar,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=True,
        #     worker_init_fn=utils.seed_worker
        # )
    else:

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size, sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            worker_init_fn=utils.seed_worker
        )

        # data_loader_trainclean = torch.utils.data.DataLoader(
        #     dataset_trainclean, sampler=sampler_train,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=True,
        #     worker_init_fn=utils.seed_worker
        # )

    model.to(device)
    mlp_head.to(device)
    model_without_ddp = model
    mlp_head_without_ddp = mlp_head

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlp_n_parameters = sum(p.numel() for p in mlp_head.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    print("Mlp = %s" % str(mlp_head_without_ddp))
    print('number of params: {} M'.format(mlp_n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

        mlp_head = torch.nn.parallel.DistributedDataParallel(mlp_head, device_ids=[args.gpu], find_unused_parameters=True)
        mlp_head_without_ddp = mlp_head.module

    optimizer = create_optimizer(args, model_without_ddp)
    mlp_optimizer = create_optimizer(args, mlp_head_without_ddp)

    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    args.resume = args.mlp_resume
    utils.auto_load_model(
        args=args, model=mlp_head, model_without_ddp=mlp_head_without_ddp, optimizer=mlp_optimizer, loss_scaler=loss_scaler)


    if args.eval:
        test_stats = multi_evaluate(model, mlp_head, data_loader_src, device,  normlize_target=args.normlize_target, log_writer=log_writer,  patch_size=patch_size[0])
        # anamoly_detection(args, model, data_loader_src, device,  normlize_target=args.normlize_target, log_writer=log_writer,  patch_size=patch_size[0])
        print(f"Accuracy of the network on the {len(data_loader_src)} test images: MSE loss {test_stats['loss']:.2f}, acc1 {test_stats['acc1']:.2f}%, acc5 {test_stats['acc5']:.2f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        # train_stats = train_one_epoch(
        #     model, data_loader_train,
        #     optimizer, device, epoch, loss_scaler,
        #     args.clip_grad, log_writer=log_writer,
        #     start_steps=epoch * num_training_steps_per_epoch,
        #     lr_schedule_values=lr_schedule_values,
        #     wd_schedule_values=wd_schedule_values,
        #     patch_size=patch_size[0],
        #     normlize_target=args.normlize_target,
        # )

        train_stats = multi_train_one_epoch(
            model, mlp_head, data_loader_train,
            optimizer, mlp_optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    mlp_head=mlp_head, mlp_head_without_ddp=mlp_head_without_ddp, mlp_optimizer=mlp_optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    f('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)
