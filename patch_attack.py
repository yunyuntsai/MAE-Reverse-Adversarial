import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from attack_util import circle_transform, init_patch_circle
import numpy as np
from pretrained_models_pytorch import pretrainedmodels

print("=> creating model ")
netClassifier = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
netClassifier.cuda()
min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
mean, std = np.array(netClassifier.mean), np.array(netClassifier.std) 
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

def attack(x, patch, mask, netClassifier, target):
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]

    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    
    count = 0 
   
    while 0.9 > target_prob:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)
        adv_out = F.log_softmax(netClassifier(adv_x))

        Loss = -adv_out[0][target]
        print("loss: ", Loss)
        Loss.backward()
     
        adv_grad = adv_x.grad.clone()
        
        adv_x.grad.data.zero_()
       
        patch -= adv_grad 
        
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
 
        out = F.softmax(netClassifier(adv_x))
        target_prob = out.data[0][target]
        #y_argmax_prob = out.data.max(1)[0][0]
        
        #print(count, conf_target, target_prob, y_argmax_prob)  

        if count >= 1000:
            break


    return adv_x, mask, patch 

def generate_adversarial_image(img, label, cuda=False):
    """
    Generate an adversarial image for a given input image.

    :param image_path: Path to the input image.
    :param model_name: Name of the pre-trained model to use.
    :param max_count: Maximum number of iterations for the attack.
    :param cuda: Boolean to use CUDA if available.
    :return: Adversarial image.
    """
    patch, patch_shape = init_patch_circle(image_size=299, patch_size=0.05) 
    # Load the pre-trained model
    netClassifier.eval()

    img, label = Variable(img), Variable(label)
    
    prediction = netClassifier(img)
 
    # only computer adversarial examples on examples that are originally classified correctly        
    if prediction.data.max(1)[1][0] != label.data[0]:
        return None, False
    
     # transform path
    data_shape = img.data.cpu().numpy().shape
    patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size=299)
    patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
    patch, mask = patch.cuda(), mask.cuda()
    patch, mask = Variable(patch), Variable(mask)

    adv_x, mask, patch = attack(img, patch, mask, netClassifier, target=label.data[0] + 1)
    
    adv_label = netClassifier(adv_x).data.max(1)[1][0]
    ori_label = label.data[0]

    if adv_label != ori_label:
        print("attack success!")
        vutils.save_image(adv_x.data, "./%d_%d_adversarial.png" %(adv_label), normalize=True)
        return  adv_x, True
    

    return None, False

# def main():
#      # Load and preprocess the image
#     transform = transforms.Compose([
#         transforms.Resize(299),
#         transforms.CenterCrop(299),
#         transforms.ToTensor(),
#     ])
#     image = Image.open("/home/paperspace/DRAM/zhuoxuan_mae/mae/ILSVRC2012_val_00018075.JPEG")
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     adversarial_image = generate_adversarial_image('path/to/your/image.jpg')
