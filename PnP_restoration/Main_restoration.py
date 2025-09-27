import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
import torch
from argparse import ArgumentParser
from utils.utils_restoration import rgb2y, psnr, psnr_torch, array2tensor, tensor2array, rotate_image_tensor, random_transform_noise, random_transform_rotation, random_transform_subpixel_rotation, random_transform_flip, random_transform_translation
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure as ssim_gpu
from skimage.restoration import estimate_sigma
from lpips import LPIPS
import sys
from matplotlib.ticker import MaxNLocator
from utils.utils_restoration import imsave, single2uint, rescale
from scipy import ndimage
from time import time
from brisque import BRISQUE


# append path 
sys.path.append('GS_denoising/')
sys.path.append('PnP_restoration/')
import warnings
warnings.filterwarnings("ignore")


loss_lpips = LPIPS(net='alex', version='0.1')
brisque = BRISQUE(url=False)

class PnP_restoration():

    def __init__(self, hparams):

        self.hparams = hparams
        self.device = torch.device('cuda:'+str(self.hparams.gpu_number) if torch.cuda.is_available() else 'cpu')
        self.initialize_cuda_denoiser()

    def initialize_cuda_denoiser(self):
        '''
        Initialize the denoiser model with the given pretrained ckpt
        '''
        if self.hparams.denoiser_type == "GSDenoiser":
            from lightning_GSDRUNet import GradMatch
            parser2 = ArgumentParser(prog='utils_restoration.py')
            parser2 = GradMatch.add_model_specific_args(parser2)
            parser2 = GradMatch.add_optim_specific_args(parser2)
            hparams = parser2.parse_known_args()[0]
            hparams.act_mode = self.hparams.act_mode_denoiser
            hparams.grayscale = self.hparams.grayscale
            if self.hparams.opt_alg == "PnP_PGD" or self.hparams.opt_alg == "SPnP_PGD":
                hparams.grad_matching = False
            self.denoiser_model = GradMatch(hparams)
            checkpoint = torch.load(self.hparams.pretrained_checkpoint, map_location=self.device)
            self.denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
            self.denoiser_model.eval()
            for i, v in self.denoiser_model.named_parameters():
                v.requires_grad = False
            self.denoiser_model = self.denoiser_model.to(self.device)
        else:
            ValueError('Denoiser not implemented')

    def denoise(self, x, sigma, weight=1.):
        if self.hparams.denoiser_type == "GSDenoiser":
            torch.set_grad_enabled(True)
            Dg, N, g = self.denoiser_model.calculate_grad(x, sigma)
            torch.set_grad_enabled(False)
            Dg = Dg.detach()
            N = N.detach()
            Dx = x - weight * Dg
            return Dx, g, Dg
        else:
            ValueError('Denoiser not implemented')

    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.hparams.degradation_mode == 'deblurring':
            k = degradation
            self.k_tensor = torch.tensor(k).to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(img, self.k_tensor, self.sf)

    def data_fidelity_prox_step(self, x, y, stepsize):
        '''
        Calculation of the proximal step on the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            if self.hparams.degradation_mode == 'deblurring':
                px = utils_sr.prox_solution_L2(x, self.FB, self.FBC, self.F2B, self.FBFy, stepsize, self.sf)
            else:
                ValueError('Degradation not treated')
        else :  
            ValueError('noise model not treated')
        return px

    def data_fidelity_grad(self, x, y):
        """
        Calculate the gradient of the data-fidelity term.
        :param x: Point where to evaluate F
        :param y: Degraded image
        """
        if self.hparams.noise_model == 'gaussian':
            if self.hparams.degradation_mode == 'deblurring' or self.hparams.degradation_mode == 'SR':
                if self.hparams.opt_alg == "RED" or self.hparams.opt_alg == "ERED" or self.hparams.opt_alg == "SNORE":
                    return utils_sr.grad_solution_L2(x, y, self.k_tensor, self.sf)
                else: 
                    return utils_sr.grad_solution_L2(x.float(), y, self.k_tensor.float(), self.sf)
            else:
                raise ValueError('degradation not implemented')
        else:
            raise ValueError('noise model not implemented')

    def data_fidelity_grad_step(self, x, y, stepsize):
        '''
        Calculation of the gradient step on the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            grad = utils_sr.grad_solution_L2(x, y, self.k_tensor, self.sf)
        else:
            raise ValueError('noise model not implemented')
        return x - stepsize*grad, grad
        
    def A(self,y):
        '''
        Calculation A*x with A the linear degradation operator 
        '''
        if self.hparams.degradation_mode == 'deblurring':
            y = utils_sr.G(y, self.k_tensor, sf=1)
        else:
            raise ValueError('degradation not implemented')
        return y  

    def At(self,y):
        '''
        Calculation A*x with A the linear degradation operator 
        '''
        if self.hparams.degradation_mode == 'deblurring':
            y = utils_sr.Gt(y, self.k_tensor, sf=1)
        else:
            raise ValueError('degradation not implemented')
        return y  


    def calculate_data_term(self,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
        deg_y = self.A(y)
        if self.hparams.noise_model == 'gaussian':
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.noise_model == 'poisson':
            f = (img*torch.log(img/deg_y + 1e-15) + deg_y - img).sum()
        return f

    def calculate_regul(self,x, g=None):
        '''
        Calculation of the regularization phi_sigma(y)
        :param x: Point where to evaluate
        :param g: Precomputed regularization function value at x
        :return: regul(x)
        '''
        if g is None:
            _,g,_ = self.denoise(x, self.sigma_denoiser)
        return g

    def calculate_F(self,x, img, g = None):
        '''
        Calculation of the objective function value f(x) + lamb*g(x)
        :param x: Point where to evaluate F
        :param img: Degraded image
        :param g: Precomputed regularization function value at x
        :return: F(x)
        '''
        regul = self.calculate_regul(x, g=g)
        if self.hparams.no_data_term:
            F = regul
            f = torch.zeros_like(F)
        else:
            f = self.calculate_data_term(x,img)
            F = f + self.lamb * regul
        return f.item(), F.item()

    def restore(self, img, init_im, clean_img, degradation, extract_results=False, sf=1):
        '''
        Compute GS-PnP restoration algorithm
        :param img: Degraded image
        :param init_im: Initialization of the algorithm
        :param clean_img: ground-truth clean image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        :param extract_results: Extract information for subsequent image or curve saving
        :param sf: Super-resolution factor
        '''
        self.sf = sf

        if extract_results:
            x_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, residual_list, estimated_noise_list =  [],  [],  [], [], [], [], []

        # initalize parameters
        if (self.hparams.opt_alg == "ERED" or self.hparams.opt_alg == "RED" or self.hparams.opt_alg == "SNORE"):
            if self.hparams.stepsize is None:
                self.stepsize = 1 / self.lamb
            else:
                self.stepsize = self.hparams.stepsize

        ### new part
        if self.hparams.opt_alg in ["PnP_PGD", "SPnP_PGD"]:
            nu = float(self.hparams.noise_level_img) / 255.0
            self.stepsize = (float(self.hparams.stepsize) if getattr(self.hparams, "stepsize", None) is not None else 1.9) * nu**2
            self.std =  nu
            self.sigma_denoiser = (float(self.hparams.sigma_denoiser) if getattr(self.hparams, "sigma_denoiser", None) is not None else self.hparams.noise_level_img) / 255.0
            self.lamb = 0.
            self.noise_stochastic = (float(self.hparams.noise_level_SPnP)/255. if self.hparams.noise_level_SPnP is not None else float(self.hparams.noise_level_img) / 255.0)

        # Initialization of the algorithm

        img_tensor = array2tensor(img).to(self.device)
        clean_img_torch = array2tensor(clean_img).to(self.device)
        self.initialize_prox(img_tensor, degradation)

        # Initialization of the algorithm
        x0 = array2tensor(init_im).to(self.device)
        if self.hparams.opt_alg == "RED" or self.hparams.opt_alg == "ERED" or self.hparams.opt_alg =="SNORE":
            x0 = self.data_fidelity_prox_step(x0, img_tensor, self.stepsize)
        x = x0

        if extract_results:  # extract np images and PSNR values
            current_x_psnr = psnr_torch(clean_img_torch, x0)
            psnr_tab.append(float(current_x_psnr.cpu()))

        #for reproducibility
        generator = torch.Generator(device = self.device)
        if self.hparams.seed != None:
            generator.manual_seed(self.hparams.seed)
        else:
            generator.manual_seed(0)

        F = float('inf')
        self.backtracking_check = True
        print(f"std used : {self.std}, stepsize used : {self.stepsize}, lambda used : {self.lamb}")
        for i in range(self.maxitr):
            F_old = F
            x_old = x
            # print(f'At iteration {i}: self.stepsize = {self.stepsize}, lamb = {self.lamb}, std = {self.std}', end='\r')

            ### algorithm 
            if self.hparams.opt_alg == "RED":
                if extract_results:
                    x_old_array = tensor2array(x_old)
                    estimated_noise_list.append(estimate_sigma(x_old_array, average_sigmas=True, channel_axis=-1))
                _,g,Dg = self.denoise(x_old, self.std)
                z = x_old - self.stepsize * self.lamb * Dg
                x = z - self.stepsize * self.data_fidelity_grad(x_old, img_tensor)
                y = z 
                # Calculate Objective
                f, F = self.calculate_F(x, img_tensor, g=g)
                residual = torch.norm(x - x_old)/torch.norm(x0)

            
            if self.hparams.opt_alg == "ERED":
                if extract_results:
                    x_old_array = tensor2array(x_old)
                    estimated_noise_list.append(estimate_sigma(x_old_array, average_sigmas=True, channel_axis=-1))
                
                if self.hparams.transformation == "subpixel_rotation":
                    transform, inverse_transform = random_transform_subpixel_rotation(self.device, generator)
                elif self.hparams.transformation == "rotation":
                    transform, inverse_transform = random_transform_rotation(self.device, generator)
                elif self.hparams.transformation == "flip":
                    transform, inverse_transform = random_transform_flip(self.device, generator)
                elif self.hparams.transformation == "translation":
                    transform, inverse_transform = random_transform_translation(x_old.shape[2], x_old.shape[3], self.device, generator)
                elif self.hparams.transformation == "all_transformations":
                    indx_transformation = np.random.randint(5)
                    if indx_transformation == 0:
                        transform, inverse_transform = random_transform_subpixel_rotation(self.device, generator)
                    elif indx_transformation == 1:
                        transform, inverse_transform = random_transform_rotation(self.device, generator)
                    elif indx_transformation == 2:
                        transform, inverse_transform = random_transform_flip(self.device, generator)
                    elif indx_transformation == 3:
                        transform, inverse_transform = random_transform_translation(x_old.shape[2], x_old.shape[3], self.device, generator)
                    else:  # indx_transformation == 4
                        transform, inverse_transform = random_transform_noise(self.std, x_old.shape, generator, self.device)
                
                x_old_transform = transform(x_old)
                _,g,Dg = self.denoise(x_old_transform, self.std)
                reg_grad = inverse_transform(x_old, Dg)                
                z = x_old - self.stepsize * self.lamb * reg_grad
                x = z - self.stepsize * self.data_fidelity_grad(x_old, img_tensor)
                y = x 
                # Calculate Objective
                f, F = self.calculate_F(x, img_tensor, g=g)
                residual = torch.norm(x - x_old)/torch.norm(x0)

            if self.hparams.opt_alg == "SNORE":
                x_old = x
                num_itr_each_ann = (self.maxitr - self.hparams.last_itr) // self.hparams.annealing_number
                if  i < self.maxitr - self.hparams.last_itr and i % num_itr_each_ann == 0:
                    self.std =  self.std_0 * (1 - i / (self.maxitr - self.hparams.last_itr)) + self.std_end * (i / (self.maxitr - self.hparams.last_itr))
                    self.lamb = self.lamb_0 * (1 - i / (self.maxitr - self.hparams.last_itr)) + self.lamb_end * (i / (self.maxitr - self.hparams.last_itr))
                if i >= self.maxitr - self.hparams.last_itr:
                    self.std = self.std_end
                    self.lamb = self.lamb_end
                # Regularization term
                g_mean = torch.tensor([0]).to(self.device).float()
                Dg_mean = torch.zeros(*x_old.size()).to(self.device)
                for ite in range(self.hparams.num_noise):
                    noise = torch.normal(torch.zeros(*x_old.size()).to(self.device), std = self.std*torch.ones(*x_old.size()).to(self.device), generator = generator)
                    x_old_noise = x_old + noise
                    if extract_results and ite==0:
                        x_old_noise_array = tensor2array(x_old_noise)
                        estimated_noise_list.append(estimate_sigma(x_old_noise_array, average_sigmas=True, channel_axis=-1))
                    _,g,Dg = self.denoise(x_old_noise, self.std)
                    g_mean += g
                    Dg_mean += Dg
                g, Dg = g_mean/self.hparams.num_noise, Dg_mean/self.hparams.num_noise
                # Total-Gradient step
                z = x_old - self.stepsize * self.lamb * Dg
                if self.hparams.opt_alg == "SNORE":
                    x = z - self.stepsize * self.data_fidelity_grad(x_old, img_tensor)
                # Calculate Objective
                f, F = self.calculate_F(x, img_tensor, g=g)
                y = x
                residual = torch.norm(x - x_old)/torch.norm(x0)


            if self.hparams.opt_alg == "PnP_PGD":
                if extract_results:
                    x_old_array = tensor2array(x_old)
                    if self.hparams.grayscale:
                        estimated_noise_list.append(estimate_sigma(x_old_array, channel_axis=-1))
                    else:
                        estimated_noise_list.append(estimate_sigma(x_old_array, average_sigmas=True, channel_axis=-1))
                    
                nu = float(self.hparams.noise_level_img) / 255.0
                grad_f = self.data_fidelity_grad(x_old, img_tensor)/ max(nu**2, 1e-8)
                z = x_old - self.stepsize* grad_f

                if self.hparams.transformation == "subpixel_rotation":
                    transform, inverse_transform = random_transform_subpixel_rotation(self.device, generator)
                elif self.hparams.transformation == "rotation":
                    transform, inverse_transform = random_transform_rotation(self.device, generator)
                elif self.hparams.transformation == "flip":
                    transform, inverse_transform = random_transform_flip(self.device, generator)
                elif self.hparams.transformation == "translation":
                    transform, inverse_transform = random_transform_translation(x_old.shape[2], x_old.shape[3], self.device, generator)
                elif self.hparams.transformation == "all_transformations":
                    indx_transformation = np.random.randint(5)
                    if indx_transformation == 0:
                        transform, inverse_transform = random_transform_subpixel_rotation(self.device, generator)
                    elif indx_transformation == 1:
                        transform, inverse_transform = random_transform_rotation(self.device, generator)
                    elif indx_transformation == 2:
                        transform, inverse_transform = random_transform_flip(self.device, generator)
                    elif indx_transformation == 3:
                        transform, inverse_transform = random_transform_translation(x_old.shape[2], x_old.shape[3], self.device, generator)
                    else:  # indx_transformation == 4
                        transform, inverse_transform = random_transform_noise(self.std, x_old.shape, generator, self.device)
                else:
                    transform = inverse_transform = None

                if transform is not None:
                    z_t = transform(z)
                    Dx, _, _ = self.denoise(z_t.float(), self.sigma_denoiser)
                    x = inverse_transform(z, Dx)  # map denoised back
                else:
                    Dx, _, _ = self.denoise(z.float(), self.sigma_denoiser)
                    x = Dx
                residual = torch.norm(x - x_old)/torch.norm(x0)

            if self.hparams.opt_alg == "SPnP_PGD":  
                grad_f = self.data_fidelity_grad(x_old, img_tensor)/ max(nu**2, 1e-8)
                z = x_old - self.stepsize* grad_f
                transform, inverse_transform = random_transform_noise(self.noise_stochastic, x_old.shape, generator, self.device)
                z_t = transform(z)
                Dx, _, _ = self.denoise(z_t.float(), self.noise_stochastic)
                x = inverse_transform(z, Dx)
                residual = torch.norm(x - x_old)/torch.norm(x0)


            ###
            # i+=1
            if self.backtracking_check : 
                if extract_results:
                    current_x_psnr = psnr_torch(clean_img_torch, x)
                    current_x_ssim_gpu = ssim_gpu(clean_img_torch, x, data_range=1.0)
                    x_list.append(x.clone())
                    psnr_tab.append(current_x_psnr)
                    residual_list.append(residual)
                    ssim_tab.append(current_x_ssim_gpu)
                    if self.hparams.lpips and not(self.hparams.grayscale):
                        current_x_lpips = loss_lpips.forward(clean_img_torch, x)
                        lpips_tab.append(current_x_lpips) 

            x = x_old
            F = F_old


        output_psnr = psnr_torch(clean_img_torch, x)
        output_ssim = ssim_gpu(clean_img_torch, x, data_range=1.0)
        if not(self.hparams.grayscale):
            clean_img_torch_cpu = clean_img_torch.cpu()
            x_cpu = x.cpu().float()
            output_lpips = loss_lpips.forward(clean_img_torch_cpu, x_cpu)
        else:
            output_brisque = output_lpips = 0
        ###
        Dy,_,_ = self.denoise(x, self.std)


        output_den_psnr = psnr_torch(clean_img_torch, Dy)
        output_den_ssim = ssim_gpu(clean_img_torch, Dy, data_range=1.0)

        if not(self.hparams.grayscale):
            output_den_lpips = loss_lpips.forward(clean_img_torch_cpu, Dy.cpu().float())
        else:
            output_den_brisque = output_den_lpips = 0

        if extract_results:
            return x, x0, output_psnr, output_ssim, output_lpips, Dy, output_den_psnr, output_den_ssim, output_den_lpips, i, x_list, psnr_tab, ssim_tab, lpips_tab, estimated_noise_list, residual_list, clean_img_torch
        else:
            return x, x0, output_psnr, output_ssim, output_lpips, Dy, output_den_psnr, output_den_ssim, output_den_lpips, i
        
        
    def initialize_curves(self):

        self.conv = []
        self.conv_F = []
        self.PSNR = []
        self.SSIM = []
        self.BRISQUE = []
        self.LPIPS = []
        self.g = []
        self.Dg = []
        self.F = []
        self.nabla_F = []
        self.f = []
        self.lamb_tab = []
        self.std_tab = []
        self.lip_algo = []
        self.lip_D = []
        self.lip_Dg = []
        self.noise_estimated_list = []
        self.residual_list = []

    def update_curves(self, x_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, Dg_list, g_list, F_list, nabla_F_list, f_list, lamb_tab, std_tab, estimated_noise_list, residual_list):
        self.F.append(F_list)
        self.nabla_F.append(nabla_F_list)
        self.f.append(f_list)
        self.g.append(g_list)
        self.Dg.append(Dg_list)
        self.PSNR.append(psnr_tab)
        self.SSIM.append(ssim_tab)
        self.BRISQUE.append(brisque_tab)
        self.LPIPS.append(lpips_tab)
        self.lamb_tab = lamb_tab
        self.std_tab = std_tab
        self.conv.append(np.array([(np.linalg.norm(x_list[k + 1] - x_list[k]) ** 2) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))
        self.lip_algo.append(np.sqrt(np.array([np.sum(np.abs(x_list[k + 1] - x_list[k]) ** 2) for k in range(1, len(x_list) - 1)]) / np.array([np.sum(np.abs(x_list[k] - x_list[k - 1]) ** 2) for k in range(1, len(x_list[:-1]))])))
        self.noise_estimated_list.append(estimated_noise_list)
        self.residual_list.append(residual_list)

    def save_curves(self, save_path):

        # plt.figure(0)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.g)):
        #     plt.plot(self.g[i], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'g.png'),bbox_inches="tight")

        # plt.figure(1)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.g)):
        #     plt.plot(self.g[i][-self.hparams.last_itr+1:], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'g_end.png'), bbox_inches="tight")

        # plt.figure(2)
        plt.figure(0)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.PSNR)):
            plt.plot(self.PSNR[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'PSNR.png'),bbox_inches="tight")

        # plt.figure(3)
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.SSIM)):
            plt.plot(self.SSIM[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'SSIM.png'),bbox_inches="tight")

        # plt.figure(4)
        plt.figure(2)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.LPIPS)):
            plt.plot(self.LPIPS[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'LPIPS.png'),bbox_inches="tight")

        # plt.figure(5)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.F)):
        #     plt.plot(self.F[i], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'F.png'), bbox_inches="tight")

        # plt.figure(6)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.F)):
        #     plt.plot(self.F[i][-self.hparams.last_itr+1:], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'F_end.png'), bbox_inches="tight")

        # plt.figure(7)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.F)):
        #     plt.plot(self.f[i], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'f.png'), bbox_inches="tight")

        # plt.figure(8)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.f)):
        #     plt.plot(self.f[i][-self.hparams.last_itr+1:], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'f_end.png'), bbox_inches="tight")

        # conv_DPIR = np.load('conv_DPIR2.npy')
        # conv_rate = self.conv[0][0]*np.array([(1/k) for k in range(1,len(self.conv[0]))])
        # plt.figure(9)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.conv)):
        #     plt.plot(self.conv[i],'-o')
        #     # plt.plot(conv_DPIR[:self.hparams.maxitr], marker=marker_list[-1], markevery=10, label='DPIR')
        # plt.plot(conv_rate, '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        # plt.semilogy()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'conv_log.png'), bbox_inches="tight")

        # plt.figure(10)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.lip_algo)):
        #     plt.plot(self.lip_algo[i],'-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'lip_algo.png'))

        # plt.figure(11)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # for i in range(len(self.BRISQUE)):
        #     plt.plot(self.BRISQUE[i], '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'BRISQUE.png'),bbox_inches="tight")

        # plt.figure(12)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # plt.plot(self.lamb_tab, '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'Lambda_list.png'),bbox_inches="tight")

        # plt.figure(13)
        # fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # plt.plot(self.std_tab, '-o')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_path, 'Std_list.png'),bbox_inches="tight")

        # plt.figure(14)
        plt.figure(3)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.noise_estimated_list)):
            plt.plot(self.noise_estimated_list[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'estimated_noise.png'),bbox_inches="tight")

        # plt.figure(15)
        plt.figure(4)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.residual_list)):
            plt.plot(self.residual_list[i], '-o')
        plt.semilogy()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'residual.png'),bbox_inches="tight")

    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_path', type=str, default='datasets')
        parser.add_argument('--gpu_number', type=int, default=0)
        parser.add_argument('--pretrained_checkpoint', type=str,default='GS_denoising/ckpts/GSDRUNet.ckpt')
        parser.add_argument('--im_init', type=str)
        parser.add_argument('--noise_model', type=str,  default='gaussian')
        parser.add_argument('--dataset_name', type=str, default='set3c')
        parser.add_argument('--noise_level_img', type=float)
        parser.add_argument('--noise_level_SPnP', type=float, default=None,help="Noise level (in [0,255] scale) for SPnP_PGD.If None, defaults to noise_level_img.")
        parser.add_argument('--maxitr', type=int)
        parser.add_argument('--stepsize', type=float)
        parser.add_argument('--lamb', type=float)
        parser.add_argument('--beta', type=float)
        parser.add_argument('--std_0', type=float)
        parser.add_argument('--std_end', type=float)
        parser.add_argument('--lamb_0', type=float)
        parser.add_argument('--lamb_end', type=float)
        parser.add_argument('--num_noise', type=int, default=1)
        parser.add_argument('--annealing_number', type=int, default=16)
        parser.add_argument('--last_itr', type=int, default=300)
        parser.add_argument('--sigma_denoiser', type=float)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--lpips', dest='lpips', action='store_true')
        parser.set_defaults(lpips=False)
        parser.add_argument('--n_images', type=int, default=68)
        parser.add_argument('--crit_conv', type=str, default='cost')
        parser.add_argument('--thres_conv', type=float, default=1e-5)
        parser.add_argument('--no_backtracking', dest='use_backtracking', action='store_false')
        parser.set_defaults(use_backtracking=False)
        parser.add_argument('--eta_backtracking', type=float, default=0.9)
        parser.add_argument('--gamma_backtracking', type=float, default=0.1)
        parser.add_argument('--inpainting_init', dest='inpainting_init', action='store_true')
        parser.set_defaults(inpainting_init=False)
        parser.add_argument('--extract_curves', dest='extract_curves', action='store_true')
        parser.set_defaults(extract_curves=False)
        parser.add_argument('--extract_images', dest='extract_images', action='store_true')
        parser.set_defaults(extract_images=False)
        parser.add_argument('--save_video', dest='save_video', action='store_true')
        parser.set_defaults(save_video=False)
        parser.add_argument('--print_each_step', dest='print_each_step', action='store_true')
        parser.set_defaults(print_each_step=False)
        parser.add_argument('--no_data_term', dest='no_data_term', action='store_true')
        parser.set_defaults(no_data_term=False)
        parser.add_argument('--use_hard_constraint', dest='use_hard_constraint', action='store_true')
        parser.set_defaults(use_hard_constraint=False)
        parser.add_argument('--rescale_for_denoising', dest='rescale_for_denoising', action='store_true')
        parser.set_defaults(rescale_for_denoising=False)
        parser.add_argument('--clip_for_denoising', dest='clip_for_denoising', action='store_true')
        parser.set_defaults(clip_for_denoising=False)
        parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')
        parser.set_defaults(use_wandb=False)
        parser.add_argument('--use_linear_init', dest='use_linear_init', action='store_true')
        parser.set_defaults(use_linear_init=False)
        parser.add_argument('--grayscale', dest='grayscale', action='store_true')
        parser.set_defaults(grayscale=False)
        parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false')
        parser.set_defaults(early_stopping=True)
        parser.add_argument('--exp_out_path', type=str, default="Result_ERED")
        parser.add_argument('--weight_Dg', type=float, default=1.)
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--act_mode_denoiser', type=str, default='E')
        parser.add_argument('--denoiser_type', type=str, default='GSDenoiser')
        parser.add_argument('--rot', dest='rot', action='store_true')
        parser.set_defaults(rot=False)
        parser.add_argument('--transformation', type=str, choices=['rotation', 'subpixel_rotation', 'flip', 'translation','all_transformations'], help='Specify random transformation for ERED algorithm.', default=None)
        parser.add_argument('--opt_alg', dest='opt_alg', choices=['SNORE', 'Data_GD', 'SNORE_Prox', 'RED_Prox', 'ARED_Prox', 'RED', 'PnP_SGD', 'PnP_PGD', 'SPnP_PGD', 'ERED', 'ERED_Prox'], help='Specify optimization algorithm')
        return parser