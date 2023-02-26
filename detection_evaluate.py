import os 
import torch
import numpy as np
import argparse
import torchvision

from detection_baseline import targeted_vals, untargeted_vals


def single_metric_fpr_tpr(fpr, 
                          criterions, 
                          model, 
                          dataset, 
                          attacks, 
                          real_dir, 
                          adv_dir, 
                          n_radius, 
                          targeted_lr, 
                          t_radius,
                          untargeted_lr, 
                          u_radius, 
                          opt='targeted'):
    # if opt == 'l1':
    #     target = l1_vals(model, dataset, title, "real", real_dir, adv_dir, n_radius)
    #     threshold = criterions[fpr][0]
    #     print("this is l1 norm for real images", target)
    if opt == 'targeted':
        target = targeted_vals(model, dataset, "real", real_dir, adv_dir, targeted_lr, t_radius)
        threshold = criterions[fpr][1]
        print('number of steps for targeted attacks on real images', target)
    elif opt == 'untargeted':
        target = untargeted_vals(model, dataset, "real", real_dir, adv_dir, untargeted_lr, u_radius)
        threshold = criterions[fpr][2]
        print("number of steps for untargeted attacks for real images", target)
    else:
        raise "Not Implemented"

    fpr_accurate = len(target[target > threshold]) * 1.0 / len(target)
    print("corresponding accurate fpr of this threshold is", fpr_accurate) #TODO: find the meaning of this

    for i in range(len(attacks)):
        # if opt == "l1":
        #     a_target = l1_vals(model, dataset, attacks[i], real_dir, adv_dir, n_radius)
        #     print("This is l1 norm for ", attacks[i], a_target)
        if opt == "targeted":
            a_target = targeted_vals(model, dataset, attacks[i], real_dir, adv_dir, targeted_lr, t_radius)
            print("number of steps for targeted attack for ", attacks[i], a_target)
        elif opt == "untargeted":
            a_target = untargeted_vals(model, dataset, attacks[i], real_dir, adv_dir, untargeted_lr, u_radius)
            print("this is step of untargeted attack for ", attacks[i], a_target)
        else: 
            raise "Not Implemented"
        tpr = len(a_target[a_target > threshold]) * 1.0 / len(a_target)
        print("corresponding tpr for " + attacks[i] + " of this threshold is", tpr)

def combined_metric_fpr_tpr(fpr, 
                            criterions, 
                            model,
                            dataset, 
                            attacks, 
                            real_dir, 
                            adv_dir, 
                            n_radius, 
                            targeted_lr, 
                            t_radius, 
                            untargeted_lr, 
                            u_radius):
    # target_1 = l1_vals(model, dataset, "real", real_dir, adv_dir, n_radius)
    target_2 = targeted_vals(model, dataset, "real", real_dir, adv_dir, targeted_lr, t_radius)
    target_3 = untargeted_vals(model, dataset, "real", real_dir, adv_dir, untargeted_lr, u_radius)

    fpr_accurate = len(np.logical_or(target_2 > criterions[fpr][1]), target_3 > criterions[fpr][2]) * 1.0 / len(target_2)
    # fpr_accurate = len(target_1[np.logical_or(np.logical_or(target_1 > criterions[fpr][0], target_2 > criterions[fpr][1]), target_3 > criterions[fpr][2])]) * 1.0 / len(target_1)
    print("corresponding accurate fpr of this threshold is ", fpr_accurate)

    for i in range(len(attacks)):
        # a_target_1 =  target_1 = l1_vals(model, dataset, attacks[i], real_dir, adv_dir, n_radius)
        a_target_2 = targeted_vals(model, dataset, attacks[i], real_dir, adv_dir, targeted_lr, t_radius)
        a_target_3 = untargeted_vals(model, dataset, attacks[i], real_dir, adv_dir, targeted_lr, u_radius)
        # tpr = len(a_target_1[np.logical_or(np.logical_or(a_target_1 > criterions[fpr][0], a_target_2 > criterions[fpr][1]), a_target_3 > criterions[fpr][2])]) * 1.0 / len(a_target_1)
        
        # TODO: verify this works
        tpr = len(np.logical_or(a_target_2 > criterions[fpr][1], a_target_3 > criterions[fpr][2])) * 1.0 / len(a_target_2) 
        print("corresponding tpr for " + attacks[i] + " of this threshold is", tpr)

# parser = argparse.ArgumentParser(description="baseline detection measures")
# parser.add_argument('--real_dir', type=str, required=True, help='the folder for real images in ImageNet in .pt format')
# parser.add_argument('--adv_dir', type=str, required=True, help='the folder to store generate adversaries of ImageNet in .pt')
# # parser.add_argument('--title', type=str, required=True, help='title of your attack, should be name+step format')
# parser.add_argument('--dataset', type=str, default='imagenet', help='dataset, imagenet or cifar')
# parser.add_argument('--base', type=str, default="resnet", help='model, vgg for cifar and resnet/inception for imagenet')
# parser.add_argument('--save_dir', dest='save_dir',help='The directory where the pretrained vgg19 model is saved',default='./vgg19model/', type=str)
# # parser.add_argument('--lowbd', type=int, default=0, help='index of the first adversarial example to load')
# # parser.add_argument('--upbd', type=int, default=1000, help='index of the last adversarial example to load')
# parser.add_argument('--fpr', type=float, default=0.1, help='false positive rate for detection')
# parser.add_argument('--det_opt', type=str, default='combined',help='l1,targeted, untargeted or combined')
# args = parser.parse_args()

# model = None
# if args.dataset == "imagenet":
#     noise_radius = 0.1
#     targeted_lr = 0.005
#     targeted_radius = 0.03
#     untargeted_radius = 0.03

#     if args.base == "resnet":
#         model = torchvision.models.resnet50(pretrained=True)
#         untargeted_step_threshold = 1000
#         criterions = {0.1: (0.190, 35, untargeted_step_threshold), 0.2: (1.77, 22, untargeted_step_threshold)}
#         # they pre-define their criterion threshold
#         # we use different resnet model
#         untargeted_lr = 0.1
#     else:
#         raise Exception("No such model predefined")
# else: 
#     raise Exception("Not supported dataset")

# model.eval()

# real_d = os.path.join(args.real_dir, args.base)
# adv_d = os.path.join(args.adv_dir, args.base)
# attacks = ["fgsm_4", "fgsm_8", "pgd_8", "pgd_16"]
# if args.det_opt == "combined":
#     combined_metric_fpr_tpr(args.fpr, 
#                             criterions, 
#                             model, 
#                             args.dataset, 
#                             attacks, 
#                             real_d, 
#                             adv_d, 
#                             noise_radius, 
#                             targeted_lr, 
#                             targeted_radius, 
#                             untargeted_lr, 
#                             untargeted_radius)
# else:
#     single_metric_fpr_tpr(args.fpr, 
#                           criterions, 
#                           model, 
#                           args.dataset, 
#                           attacks, 
#                           real_d, 
#                           adv_d, 
#                           noise_radius, 
#                           targeted_lr, 
#                           targeted_radius, 
#                           untargeted_lr, 
#                           untargeted_radius, 
#                           opt=args.det_opt)
    
#     print("finish evaluation based on tuned thresholds")