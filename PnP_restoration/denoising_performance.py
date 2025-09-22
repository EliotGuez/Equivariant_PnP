import os
from natsort import os_sorted
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_restoration import rgb2y, rotate_image_tensor, psnr, psnr_torch, imread_uint, array2tensor, tensor2array, random_transform_noise, random_transform_rotation, random_transform_subpixel_rotation, random_transform_flip, random_transform_translation
from math import pi
from argparse import ArgumentParser
import torch
from tqdm import tqdm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Initialize the denoiser and load the weights
sys.path.append('../GS_denoising/')
from lightning_GSDRUNet import GradMatch
parser2 = ArgumentParser(prog='utils_restoration.py')
parser2 = GradMatch.add_model_specific_args(parser2)
parser2 = GradMatch.add_optim_specific_args(parser2)
hparams = parser2.parse_known_args()[0]
hparams.act_mode = 'E'
hparams.grayscale = False
denoiser_model = GradMatch(hparams)
checkpoint = torch.load('../GS_denoising/ckpts/GSDRUNet.ckpt', map_location=device)
denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
denoiser_model.eval()
for i, v in denoiser_model.named_parameters():
    v.requires_grad = False
denoiser_model = denoiser_model.to(device)

def denoise(x, sigma, weight=1.):
    torch.set_grad_enabled(True)
    Dg, N, g = denoiser_model.calculate_grad(x, sigma)
    torch.set_grad_enabled(False)
    Dg = Dg.detach()
    N = N.detach()
    Dx = x - weight * Dg
    return Dx, g, Dg

def random_transform_rotation(u):
    """
    Random quarter rotation of a torch image. 
    """
    def transform(x):
        if u==0:
            Tx = x
        if u==1:
            Tx = torch.permute(x,(0,1,3,2)); Tx = torch.flip(Tx, dims = [2])
        if u==2:
            Tx = torch.flip(x, dims = [2, 3])
        if u==3:
            Tx = torch.permute(x,(0,1,3,2)); Tx = torch.flip(Tx, dims = [3])
        return Tx
    def inverse_transform(x, Tx):
        if u==0:
            y = Tx
        if u==1:
            y = torch.flip(Tx, dims = [2]); y = torch.permute(y,(0,1,3,2))
        if u==2:
            y = torch.flip(Tx, dims = [2, 3])
        if u==3:
            y = torch.flip(Tx, dims = [3]); y = torch.permute(y,(0,1,3,2))
        return y
    return transform, inverse_transform

def random_transform_flip(u):
    bool_1, bool_2 = [(0,0), (0,1), (1,0), (1,1)][u]
    def transform(x):
        Tx = x.clone()
        if bool_1 == 1:
            Tx = torch.flip(Tx,[2])
        if bool_2 == 1:
            Tx = torch.flip(Tx,[3])
        return Tx
    def inverse_transform(x,Tx):
        return transform(Tx)
    return transform, inverse_transform

dataset_path = "../datasets/"
dataset_name = "CBSD68"
input_path = os.path.join(dataset_path, dataset_name)
input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

Denoised_PSNR, Denoised_PSNR_rot, Denoised_PSNR_trans, Denoised_PSNR_sub_rot, Denoised_PSNR_flip = [], [], [], [], []

for sigma in [20 / 255]:
    print("Sigma : ", sigma)
    M = 10

    for input_path in tqdm(input_paths):
        input_im_uint = imread_uint(input_path)
        x = array2tensor(np.float32(input_im_uint / 255.)).to(device)
        _, _, n, m = x.shape

        noise = torch.normal(torch.zeros(*x.size()).to(device), std = sigma*torch.ones(*x.size()).to(device))
        x_noised = x + noise

        # Simple denoising
        x_denoised_simple, _, _ = denoise(x_noised, sigma)
        psnr_denoised_simple = psnr_torch(x, x_denoised_simple)
        Denoised_PSNR.append(float(psnr_denoised_simple.cpu()))

        # Denoising with pi/4 rotations
        x_denoised = torch.zeros(*x.size()).to(device)
        for u in range(4):
            transform, inverse_transform = random_transform_rotation(u)
            Tx_noised = transform(x_noised)
            Tx_denoised, _, _ = denoise(Tx_noised, sigma)
            x_denoised_u = inverse_transform(x, Tx_denoised)
            x_denoised += x_denoised_u
        x_denoised = x_denoised / 4
        psnr_denoised = psnr_torch(x, x_denoised)
        Denoised_PSNR_rot.append(float(psnr_denoised.cpu()))

        # Denoising with random translation
        x_denoised = torch.zeros(*x.size()).to(device)
        for _ in range(M):
            transform, inverse_transform = random_transform_translation(x.shape[2], x.shape[3])
            Tx_noised = transform(x_noised)
            Tx_denoised, _, _ = denoise(Tx_noised, sigma)
            x_denoised_u = inverse_transform(x, Tx_denoised)
            x_denoised += x_denoised_u
        x_denoised = x_denoised / M
        psnr_denoised = psnr_torch(x, x_denoised)
        Denoised_PSNR_trans.append(float(psnr_denoised.cpu()))

        # Denoising with random subpixel rotations
        x_denoised = torch.zeros(*x.size()).to(device)
        for _ in range(M):
            transform, inverse_transform = random_transform_subpixel_rotation(device)
            Tx_noised = transform(x_noised)
            Tx_denoised, _, _ = denoise(Tx_noised, sigma)
            x_denoised_u = inverse_transform(x, Tx_denoised)
            x_denoised += x_denoised_u
        x_denoised = x_denoised / M
        psnr_denoised = psnr_torch(x, x_denoised)
        Denoised_PSNR_sub_rot.append(float(psnr_denoised.cpu()))

        # Denoising with flip
        x_denoised = torch.zeros(*x.size()).to(device)
        for u in range(4):
            transform, inverse_transform = random_transform_flip(u)
            Tx_noised = transform(x_noised)
            Tx_denoised, _, _ = denoise(Tx_noised, sigma)
            x_denoised_u = inverse_transform(x, Tx_denoised)
            x_denoised += x_denoised_u
        x_denoised = x_denoised / 4
        psnr_denoised = psnr_torch(x, x_denoised)
        Denoised_PSNR_flip.append(float(psnr_denoised.cpu()))

    Denoised_PSNR, Denoised_PSNR_rot, Denoised_PSNR_trans, Denoised_PSNR_sub_rot, Denoised_PSNR_flip = np.array(Denoised_PSNR), np.array(Denoised_PSNR_rot), np.array(Denoised_PSNR_trans), np.array(Denoised_PSNR_sub_rot), np.array(Denoised_PSNR_flip)

    print("Simple denoising & {:.2f}".format(np.mean(Denoised_PSNR)))
    print("Rotation denoising & {:.2f}".format(np.mean(Denoised_PSNR_rot)))
    print("Translation denoising & {:.2f}".format(np.mean(Denoised_PSNR_trans)))
    print("Subpixel Rotation denoising & {:.2f}".format(np.mean(Denoised_PSNR_sub_rot)))
    print("Flip denoising & {:.2f}".format(np.mean(Denoised_PSNR_flip)))



# plt.imsave("im.png", tensor2array(x))
# plt.imsave("im_transform.png", np.clip(tensor2array(Tx_noised),0,1))
# plt.imsave("im_transform_denoised.png", np.clip(tensor2array(Tx_denoised),0,1))
# plt.imsave("im_inv_transform.png", np.clip(tensor2array(x_denoised),0,1))