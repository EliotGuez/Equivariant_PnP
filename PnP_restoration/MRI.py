import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import psnr, array2tensor, tensor2array, get_parameters, create_out_dir, single2uint, imread_uint, imsave, genMask, fft2c_numpy, ifft2c_numpy
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from natsort import os_sorted
from Main_restoration import PnP_restoration
import wandb
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# # Define sweep config
# sweep_configuration = {
#     "method": "grid",
#     "name": "pnp_pgd_parameter_optimization",
#     "metric": {"goal": "maximize", "name": "output_psnr"},
#     "parameters": {
#         "stepsize" : {"values": [1.5, 1.9]},
#         "denoiser_strength": {"values": [2., 5.]},
#     },
# }

# sweep_configuration = {
#     "method": "random",
#     "name": "spnp_pgd",
#     "metric": {"goal": "maximize", "name": "output_psnr"},
#     "parameters": {
#         "stepsize": {
#             "distribution": "uniform",
#             "min": 1.5,    # Much wider range to explore
#             "max": 1.95
#         },
#         # "denoiser_strength": {
#         #     "distribution": "uniform", 
#         #     "min": 1.,    # Explore from very low to high
#         #     "max": 10.0
#         # },
#     },
# }
# Initialize sweep by passing in config.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="pnp_pgd_optimization")
def MRI():
    # hyperparameters
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--acceleration_factor', type=int)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()
    hparams.degradation_mode = 'MRI'
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    psnr_list, ssim_list, F_list = [], [], []

    if hparams.use_wandb:
        wandb.init()        
    data = []

    if PnP_module.hparams.noise_level_img == None:
        noise_list = [5., 10., 20.]
    else:
        noise_list = [PnP_module.hparams.noise_level_img]

    for noise in noise_list:
        PnP_module.hparams.noise_level_img = noise

        n_it_list, psnr_k_list, ssim_k_list = [], [], []

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        # definition of parameters setting : by default or defined by user
        PnP_module.lamb, PnP_module.std, PnP_module.maxitr, PnP_module.thres_conv, PnP_module.stepsize, PnP_module.std_0, PnP_module.std_end, PnP_module.lamb_0, PnP_module.lamb_end, PnP_module.beta = get_parameters(hparams.noise_level_img, PnP_module.hparams, degradation_mode='MRI')
        # print(PnP_module.lamb, PnP_module.stepsize)
        PnP_module.sigma_denoiser = PnP_module.std
        if hparams.use_wandb:
            if hasattr(wandb.config, "denoiser_strength"):
                hparams.sigma_denoiser = wandb.config.denoiser_strength 
            if hasattr(wandb.config, "stepsize"):
                hparams.stepsize = wandb.config.stepsize
            PnP_module.sigma_denoiser = hparams.sigma_denoiser
            PnP_module.stepsize = hparams.stepsize

        #create the folder to save experimental results
        exp_out_path = hparams.exp_out_path
        exp_out_path = create_out_dir(exp_out_path, hparams)

        for i in range(min(len(input_paths),hparams.n_images)): # For each image

            print('Restoration of image {}'.format(i))

            np.random.seed(seed=0)
            
            # load image
            input_im_uint = imread_uint(input_paths[i], n_channels=1)
            input_im = np.float32(input_im_uint / 255.)
            # Degrade image
            numline = input_im.shape[0]//hparams.acceleration_factor
            M = genMask(input_im.shape[:2], numline)
            observation = M[:,:,None] * fft2c_numpy(input_im) + (noise/255)*np.random.randn(*input_im.shape)
            pseudo_inverse = np.float32(np.real(ifft2c_numpy(M[:,:,None]*observation.copy())))

            if hparams.im_init == 'random':
                init_im = np.random.random(input_im.shape)
            elif hparams.im_init == 'oracle':
                init_im = input_im
            else:
                init_im = pseudo_inverse

            if hparams.extract_images or hparams.extract_curves:
                deblur_im_gpu, init_im_gpu, output_psnr_gpu, output_ssim_gpu, _, \
                output_den_img_gpu, output_den_psnr_gpu, output_den_ssim_gpu, _, \
                n_it, x_list_gpu, psnr_tab_gpu, ssim_tab_gpu, _, estimated_noise_list, \
                residual_tab_gpu, clean_img_torch = PnP_module.restore(observation.copy(),init_im.copy(),input_im.copy(),M, extract_results=True)
                
                deblur_im = tensor2array(deblur_im_gpu.cpu())
                init_im = tensor2array(init_im_gpu.cpu())
                output_psnr = float(output_psnr_gpu.cpu())
                output_ssim = float(output_ssim_gpu.cpu())        
                
                output_den_img = tensor2array(output_den_img_gpu.cpu())
                output_den_psnr = float(output_den_psnr_gpu.cpu())
                output_den_ssim = float(output_den_ssim_gpu.cpu())
                
                x_list = [tensor2array(x_gpu.cpu()) for x_gpu in x_list_gpu]
                psnr_tab = [float(p) for p in psnr_tab_gpu]
                # psnr_tab = [float(p.cpu()) for p in psnr_tab_gpu]
                ssim_tab = [float(s.cpu()) for s in ssim_tab_gpu]
                residual_tab = [float(r.cpu()) for r in residual_tab_gpu]
            
            else:
                # Non-extract case - still return GPU tensors, convert only final values
                (deblur_im_gpu, init_im_gpu, output_psnr_gpu, output_ssim_gpu, _,
                output_den_img_gpu, output_den_psnr_gpu, output_den_ssim_gpu, _, 
                n_it) = PnP_module.restore(observation.copy(),init_im.copy(),input_im.copy(),M)
                # Convert only final results
                deblur_im = tensor2array(deblur_im_gpu.cpu())
                output_psnr = float(output_psnr_gpu.cpu())
                output_ssim = float(output_ssim_gpu.cpu())
                
                output_den_img = tensor2array(output_den_img_gpu.cpu())
                output_den_psnr = float(output_den_psnr_gpu.cpu())
                output_den_ssim = float(output_den_ssim_gpu.cpu())
            
            # print(f'N iterations: {n_it}')
            print('PSNR / SSIM : {:.3f}dB / {:.3f}'.format(output_psnr, output_ssim))
            
            psnr_k_list.append(output_psnr)
            ssim_k_list.append(output_ssim)
            psnr_list.append(output_psnr)
            ssim_list.append(output_ssim)
            n_it_list.append(n_it)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, psnr_tab, ssim_tab, [], [], [], [], F_list, [],  [], [], [], estimated_noise_list, residual_tab)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(exp_out_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + "_restore.png"), single2uint(np.clip(deblur_im, 0, 1)))
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_observation.png'), single2uint(np.clip(np.abs(observation), 0, 1)))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(np.clip(init_im, 0, 1)))
                # print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))
                
                #save the result of the experiment
                dict = {
                        'GT' : input_im,
                        'x_list' : x_list,
                        'estimated_noise_GT' : estimate_sigma(input_im, average_sigmas=True, channel_axis=-1),
                        'Deblur' : deblur_im,
                        'Observation' : observation,
                        'PSNR_blur' : psnr(input_im, observation),
                        'SSIM_blur' : ssim(input_im, observation, data_range = 1, channel_axis = 2),
                        'Init' : init_im,
                        'SSIM_output' : output_ssim,
                        'PSNR_output' : output_psnr,
                        'maxitr' : PnP_module.maxitr,
                        'stepsize' : PnP_module.stepsize,
                        'opt_alg': PnP_module.hparams.opt_alg,
                        'psnr_tab' : psnr_tab,
                        'ssim_tab' : ssim_tab,
                        'output_den_img' : output_den_img, 
                        'output_den_psnr' : output_den_psnr, 
                        'output_den_ssim' : output_den_ssim, 
                        'estimated_noise_list' : estimated_noise_list,
                        'residual_list' : residual_tab,
                    }
                np.save(os.path.join(exp_out_path, 'dict_' + str(i) + '_results'), dict)
            
            if not(hparams.extract_images):
                #save the result of the experiment
                dict = {
                        'GT' : input_im,
                        'Deblur' : deblur_im,
                        'Blur' : observation,
                        'PSNR_blur' : psnr(input_im, observation),
                        'SSIM_blur' : ssim(input_im, observation, data_range = 1, channel_axis = 2),
                        'Init' : init_im,
                        'SSIM_output' : output_ssim,
                        'PSNR_output' : output_psnr,
                        'lamb' : PnP_module.lamb,
                        'maxitr' : PnP_module.maxitr,
                        'stepsize' : PnP_module.stepsize,
                        'opt_alg': PnP_module.hparams.opt_alg,
                        'output_den_img' : output_den_img, 
                        'output_den_psnr' : output_den_psnr, 
                        'output_den_ssim' : output_den_ssim,
                    }
                np.save(os.path.join(exp_out_path, 'dict_' + str(i) + '_results'), dict)

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(exp_out_path,'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))

        data.append([avg_k_psnr, np.mean(np.mean(n_it_list))])
    
    if hparams.use_wandb:
        wandb.log(
            {
                "stepsize": PnP_module.stepsize,
                "denoiser_strength": PnP_module.sigma_denoiser,
                "maxitr": PnP_module.maxitr,
                "output_psnr" : np.mean(np.array(psnr_list)),
                "output_ssim" : np.mean(np.array(ssim_list)),
            }
            )
    
    data = np.array(data)

    # if hparams.use_wandb :
    #     table = wandb.Table(data=data, columns=['k', 'psnr', 'n_it'])
    #     for i, metric in enumerate(['psnr', 'n_it']):
    #         wandb.log({
    #             f'{metric}_plot': wandb.plot.scatter(
    #                 table, 'k', metric,
    #                 title=f'{metric} vs. k'),
    #             f'average_{metric}': np.mean(data[:,i+1])
    #         },
    #         step = 0)
    # plt.close('all')
    # if hparams.use_wandb:
    #     wandb.finish() 
    return np.mean(np.array(psnr_list))

# # # Start sweep job.
# wandb.agent(sweep_id, function=deblur, count=8)

if __name__ == '__main__':
    MRI()