import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import models_mae

sys.path.append('..')

# define the utils
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def compute_patchwise_mse(original_patches, reconstructed_patches):
    mse_losses = []

    for i in range(original_patches.shape[1]):
        mse = ((original_patches[:, i, :] - reconstructed_patches[:, i, :]) ** 2).mean().item()
        mse_losses.append(mse)
    
    return mse_losses

def plot_mse_losses(mse_losses, max_mse=None, save_path="mse_loss_per_patch.png"):
    plt.figure(figsize=(10, 6))  # Increase the size of the figure for clarity
    plt.plot(mse_losses, linewidth=2)  # Increase linewidth for visibility
    
    plt.xlabel('Patch Number')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss of Each Patch: Original vs. Reconstructed')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add gridlines for better clarity
    
    if max_mse:
        plt.ylim(0, max_mse)  # Set the Y-axis limit
    
    plt.legend(['MSE Loss'])  # Adding a legend, can be useful if multiple lines are added later
    plt.tight_layout()  # Adjust the layout for better fit
    plt.savefig(save_path)

def save_image(image, filename="new.png"):
    """Saves the given image to the specified file."""
    # Convert image to a format suitable for saving
    image_to_save = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    image_pil = Image.fromarray(image_to_save.numpy().astype('uint8'))
    image_pil.save(filename)


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, prefix):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    _, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked images
    im_masked = x * (1 - mask)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    save_image(x[0], f"out/{prefix}_original.png")
    save_image(im_masked[0], f"out/{prefix}_masked.png")
    save_image(y[0], f"out/{prefix}_reconstruction.png")

    # obtain patches of original and reconstructed images
    ori = torch.tensor(x[0])
    recon = torch.tensor(y[0])

    # make it a batch-like
    ori = ori.unsqueeze(dim=0)
    ori = torch.einsum('nhwc->nchw', ori)
    recon = recon.unsqueeze(dim=0)
    recon = torch.einsum('nhwc->nchw', recon)

    ori_patches = model.patchify(ori)
    recon_patches = model.patchify(recon)

    return ori_patches, recon_patches

def main():
    ### load an image
    # patch_attacked img_pth
    attacked_img_pth = "images/ILSVRC2012_val_00018075.png"
    
    # original img_pth
    ori_img_pth = "images/ILSVRC2012_val_00018075.JPEG"

    attacked_img = Image.open(attacked_img_pth)
    attacked_img = attacked_img.resize((224, 224))
    attacked_img = np.array(attacked_img) / 255.

    ori_img = Image.open(ori_img_pth)
    ori_img = ori_img.resize((224, 224))
    ori_img = np.array(ori_img) / 255.

    assert attacked_img.shape == (224, 224, 3)
    assert ori_img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    attacked_img = attacked_img - imagenet_mean
    attacked_img = attacked_img / imagenet_std

    ori_img = ori_img - imagenet_mean
    ori_img = ori_img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]

    ### Load a pre-trained MAE model
    # This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)
    chkpt_dir = 'model_pths/mae_visualize_vit_large.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    ### run mae on the image
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    attacked_ori_patches, attacked_recon_patches = run_one_image(attacked_img, model_mae, prefix='attacked')
    ori_ori_patches, ori_recon_patches = run_one_image(ori_img, model_mae, prefix='ori')

    ### save mae loss graphs
    # Compute the patch-wise MSE
    attacked_losses = compute_patchwise_mse(attacked_ori_patches, attacked_recon_patches)
    ori_losses = compute_patchwise_mse(ori_ori_patches, ori_recon_patches)

    print("average attacked loss: ", np.mean(attacked_losses))
    print("average ori loss: ", np.mean(ori_losses))

    max_mse_value = max(max(attacked_losses), max(ori_losses))
    
    plot_mse_losses(attacked_losses, max_mse=max_mse_value, save_path='out/attacked_mse_loss_per_patch.png')
    plot_mse_losses(ori_losses, max_mse=max_mse_value, save_path='out/ori_mse_loss_per_patch.png')


main()