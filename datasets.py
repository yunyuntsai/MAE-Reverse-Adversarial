# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import random
# import dlib
import numpy as np
from PIL import Image


class ConcatTrainDataset(Dataset):
    def __init__(self, txt_path, args, transform = None):
        fh= open(txt_path, 'r')
        clean_imgs = []
        adv_imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            clean_imgs.append(words[0])
            adv_imgs.append(words[1])

        self.clean_imgs = clean_imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.adv_imgs = adv_imgs
        self.transform = DataAugmentationForMAE(args)

    def rotation(self, image1, image2):

        # get a random angle range from (-180, 180)
        angle = transforms.RandomRotation.get_params([-180, 180])
        # same angle rotation for image1 and image2
        image1 = image1.rotate(angle)
        image2 = image2.rotate(angle)
        
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2

    def flip(self, image1, image2):
        # 50% prob to horizontal flip and vertical flip
        if random.random() > 0.5:
            image1 = tf.hflip(image1)
            image2 = tf.hflip(image2)
        if random.random() > 0.5:
            image1 = tf.vflip(image1)
            image2 = tf.vflip(image2)
        image1 = tf.to_tensor(image1)
        image2 = tf.to_tensor(image2)
        return image1, image2
    def __getitem__(self, index):
        clean_address = self.clean_imgs[index]
        adv_address = self.adv_imgs[index]
        clean_img = preprocess_image(cv2.imread(clean_address))  
        adv_img = preprocess_image(cv2.imread(adv_address))

        if self.transform is not None:
            clean_img = self.transform(clean_img)
            adv_img = self.transform(adv_img)

        # if self.transform == 'rotation':
        #     clean_img, adv_img = self.rotation(clean_img, adv_img)
        # elif self.transform == 'flip':
        #     clean_img, adv_img = self.flip(clean_img, adv_img)
        # else:
        #     clean_img = tf.to_tensor(clean_img)
        #     adv_img = tf.to_tensor(adv_img)
        

        return clean_img, adv_img


    def __len__(self):
        return len(self.clean_imgs)

def gen_txt(txt_path, img_dir_clean, img_dir_adv):
    f = open(txt_path, 'w')
    i = 0
    for s_dirs in sorted(os.listdir(img_dir_clean)):  # 获取 train文件下各文件夹名称
        clean_dir = os.path.join(img_dir_clean, s_dirs)
        adv_dir = os.path.join(img_dir_adv, s_dirs)
        # adv_dir_new = adv_dir.replace('clean','S')
        # # print(adv_dir_new)
        
        # if not clean_dir.endswith('bmp'):
        #     continue

        line = clean_dir + ' ' + adv_dir + ' ' + str(i) + '\n'
        f.write(line)

        i = i + 1
    f.close()


def preprocess_image(image, data = 'train', cuda=False):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    # preprocess = cifar_default_data_transforms[data]

    preprocessed_image = Image.fromarray(image)
   
    
    # Add first dimension as the network expects a batch
    # preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            # transforms.RandomResizedCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(data_path, args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(data_path, transform=transform)


def build_dataset(is_train, is_diff, normalize, args):
    transform = build_transform(is_train, normalize, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        # root = args.data_path if is_train else args.eval_data_path
        if is_train:
            root = os.path.join(args.data_path, 'train')
        else:
            if is_diff:
                root = os.path.join(args.data_path, 'diff_val')
            else:
                root = os.path.join(args.data_path, 'val')

        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, normalize, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if normalize:
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
