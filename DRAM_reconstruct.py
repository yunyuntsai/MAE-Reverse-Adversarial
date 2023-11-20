import torch
import torch.nn as nn
from PIL import Image
import util.misc as misc

import argparse
import numpy as np
import os

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import util.misc as misc
from util.datasets import build_dataset
from patch_attack import generate_adversarial_image
import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('DRAM_reconstruction for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def load_classifier_model(checkpoint_path, pretrained_model):
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            pretrained_model.load_state_dict(checkpoint)
            print("=> load chechpoint found at {}".format(checkpoint_path))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return pretrained_model

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

@torch.no_grad()
@torch.cuda.amp.autocast()
def eval_DRAM(classifier_model, ori_imgs, adv_imgs, rec_imgs, target):
    output_ori = classifier_model(ori_imgs)
    _, pre = torch.max(output_ori.data, 1)
    correct_ori = (pre == target).sum()
    
    output_adv = classifier_model(adv_imgs)
    _, pre = torch.max(output_adv.data, 1)
    correct_adv = (pre == target).sum()

    output_rec = classifier_model(rec_imgs)
    _, pre = torch.max(output_rec.data, 1)
    correct_rec = (pre == target).sum()

    return correct_ori, correct_adv, correct_rec

def save_tensor_as_image(tensor, filename):
    # Define the inverse transformation
    inverse_transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], 
            std=[1/0.229, 1/0.224, 1/0.255]
        ),
        transforms.ToPILImage()
    ])
    # Remove the batch dimension and apply the inverse transform
    image = inverse_transform(tensor.squeeze(0))
    # Save the image
    image.save(filename)

# TODO: For testing purpose, remove later
@torch.no_grad()
def test_DRAM(classifier_model, mae_model, device, target):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Resize the image to 224x224 pixels
        transforms.ToTensor(),             # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize for ResNet50
    ])

    ori_img = Image.open("/home/paperspace/DRAM/zhuoxuan_mae/mae/ILSVRC2012_val_00018075.JPEG")
    ori_img = transform(ori_img)
    ori_img = ori_img.unsqueeze(0) 
    print("ori_img.shape: ", ori_img.shape)
    save_tensor_as_image(ori_img, 'original_image.jpg')
    ori_img = ori_img.to(device)
    output_ori = classifier_model(ori_img)
    pre = torch.max(output_ori)
    print("ori: ", pre)
    
    adv_img = Image.open("/home/paperspace/DRAM/zhuoxuan_mae/mae/ILSVRC2012_val_00018075.png")
    adv_img = transform(adv_img)
    adv_img = adv_img.unsqueeze(0) 
    print("adv_img.shape: ", adv_img.shape)
    save_tensor_as_image(adv_img, 'adv_img.jpg')
    adv_img = adv_img.to(device)
    output_adv = classifier_model(adv_img)
    pre = torch.max(output_adv)
    print("adv: ", pre)

    rec_img = DRAM_reconstruct(mae_model, adv_img, mask_ratio=0.75)
    save_tensor_as_image(rec_img, 'rec_img.jpg')
    output_rec = classifier_model(rec_img)
    pre = torch.max(output_rec)
    print("rec: ", pre)

@torch.no_grad()
@torch.cuda.amp.autocast()
def DRAM_reconstruct(mae_model, adv_imgs, mask_ratio=0.75, iteration=200):
    # TODO: detect which patches are attacked
    attacked_patches = [2, 3, 4, 5, 6]

    for _ in range(iteration):
        # recon attacked_patches only. 
        # rec_img = adv_img_benign + rec_adv_img_attacked
        rec_imgs = mae_model(adv_imgs, attacked_patches, mask_ratio)
        # [batch, Channel=3, Height, Width]

    return rec_imgs

@torch.no_grad()
@torch.cuda.amp.autocast()
def engine_DRAM(mae_model, data_loader_adv, data_loader_ori, device):
    # set up logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # load classifier model
    classifier_model = torchvision.models.resnet50(pretrained=True)
    classifier_model = classifier_model.to(device)

    # switch to evaluation mode
    classifier_model.eval()
    mae_model.eval()

    # total correctness
    total_correct_ori = 0
    total_correct_adv = 0
    total_correct_rec = 0

    # test_size
    test_size = 0

    # iterate ori dataset along with adv
    data_loader_ori_iter = iter(data_loader_ori)

    for step, (adv_imgs, target) in enumerate(metric_logger.log_every(data_loader_adv, 10, header)):
        adv_imgs = adv_imgs.to(device, non_blocking=True)
        # [batch, Channel=3, Height, Width]
        target = target.to(device, non_blocking=True)

        ori_imgs = next(data_loader_ori_iter)[0].to(device, non_blocking=True)

        ## TODO: reconstruct imgs to eliminate patch
        rec_imgs = DRAM_reconstruct(mae_model, adv_imgs, mask_ratio=0.75)

        ## TODO: we test it on golden fish
        test_DRAM(classifier_model, mae_model, device, target)
        exit(0)

        # eval accuracy ori vs adv vs recon
        correct_ori, correct_adv, correct_rec = eval_DRAM(classifier_model, ori_imgs, 
                                                           adv_imgs, rec_imgs, target)
        batch_size = adv_imgs.shape[0]
        print("batch_size: ", batch_size)
        print("correct_ori.item(): ", correct_ori.item())
        print("correct_adv.item(): ", correct_adv.item())
        print("correct_rec.item(): ", correct_rec.item())
        
        # we test for 5 steps
        if step == 5:
            break

        print('correct before attack --> ', correct_ori.item()/batch_size) 
        print('correct after attack --> ', correct_adv.item()/batch_size)
        print('correct after reconstruction --> ', correct_rec.item()/batch_size)

        test_size += batch_size
        total_correct_ori += correct_ori
        total_correct_adv += correct_adv
        total_correct_rec += correct_rec
    
    print('total correct before attack --> ', total_correct_ori.item()/test_size) 
    print('total correct after attack --> ', total_correct_adv.item()/test_size)
    print('total correct after reconstruction --> ', total_correct_rec.item()/test_size)
    return


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # load data
    dataset_adv = build_dataset(is_train=False, args=args)
    sampler_adv = torch.utils.data.SequentialSampler(dataset_adv)

    # if args.eval:
    dataset_ori = build_dataset(is_train=False, args=args)
    sampler_ori = torch.utils.data.SequentialSampler(dataset_ori)

    data_loader_adv = torch.utils.data.DataLoader(
        dataset_adv, sampler=sampler_adv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_ori = torch.utils.data.DataLoader(
        dataset_ori, sampler=sampler_ori,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # create mae_model
    # TODO: visualize for debugging purpose
    chkpt_dir = "mae_visualize_vit_large.pth"
    mae_model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    mae_model.to(device)

    # config parameters
    model_without_ddp = mae_model
    print('Model loaded.')
    print("Model = %s" % str(model_without_ddp))

    # eval
    engine_DRAM(mae_model, data_loader_adv, data_loader_ori, device)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
