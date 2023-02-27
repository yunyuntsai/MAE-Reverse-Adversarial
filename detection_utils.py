import torch 
import numpy as np

# nrormalizes the data according to imagenet
def transform(img, dataset="imagenet"):
    if dataset == "imagenet": # which in this application it always will be
        # TODO: evaluate if this is necessary; if not, transform as we did during generating notebook
        mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
            img[0]).unsqueeze(0).expand_as(img).cuda()
    else:
        print("dataset:", dataset)
        raise "dataset is not support / dataset input error"
    return (img - mean) / std

# given true label of image, return new label that is not label
def random_label(label, dataset="imagenet"):
    if dataset == 'imagenet':
        class_num = 1000
    else:
        raise "dataset input error"
    attack_label = np.random.randint(class_num)
    while label == attack_label:
        attack_label = np.random.randint(class_num)
    return attack_label

def noisy_img(img, n_radius):
    return img + n_radius * torch.randn_like(img)

class Noisy(torch.autograd.Function):
    @staticmethod
    def forward(self, img, n_radius):
        return noisy_img(img, n_radius=n_radius)
    
    @staticmethod
    def backward(self, grad_output):
        return grad_output, None