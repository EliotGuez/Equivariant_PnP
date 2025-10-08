import numpy as np
import random
from scipy.fftpack import dct, idct
import torch
import cv2
import os 
from math import pi


def get_parameters(noise_level_img, hparams, k_index=0, degradation_mode='deblur'):
    '''
    Hyperparamers have been optimized for each algorithms at noise levels 0.01 0.03 and 0.05. For other noise levels, optimality is not guaranteed.
    '''
    lamb, lamb_0, lamb_end, maxitr, std_0, std_end, stepsize, sigma_denoiser, thres_conv, beta = hparams.lamb, hparams.lamb_0, hparams.lamb_end, hparams.maxitr, hparams.std_0, hparams.std_end, hparams.stepsize, hparams.sigma_denoiser, hparams.thres_conv, hparams.beta
    std = sigma_denoiser

    if degradation_mode == 'deblur':
        if hparams.opt_alg == 'RED_Prox' or hparams.opt_alg == 'ERED_Prox' or hparams.opt_alg == 'ERED' or hparams.opt_alg == 'RED' or hparams.opt_alg == 'Data_GD':
            if lamb == None:
                if k_index == 8 :
                    lamb = 0.075
                elif k_index == 9 :
                    lamb = 0.075
                else :
                    lamb = 0.1
            sigma_k_denoiser = 1.8
            if thres_conv == None:
                thres_conv = 1e-5
            if sigma_denoiser == None:
                sigma_denoiser = sigma_k_denoiser*noise_level_img
            if maxitr == None:
                maxitr = 400
            std = sigma_denoiser / 255
        
        if hparams.opt_alg == 'RED':
            if hparams.noise_level_img == 5. or hparams.noise_level_img==10.:
                if lamb == None:
                    lamb = 0.2
                if std == None:
                    std = 1.4 * hparams.noise_level_img /255.
            if hparams.noise_level_img == 20.:
                if lamb == None:
                    lamb = 0.3
                if std == None:
                    std = 1.8 * hparams.noise_level_img /255.
            if maxitr == None:
                maxitr = 100
        
        if hparams.opt_alg == 'PnP_SGD':
            if lamb == None:
                lamb = .5
            if sigma_denoiser == None:
                std = 1. * hparams.noise_level_img /255.
            else:
                std = sigma_denoiser / 255.
            if maxitr == None:
                maxitr = 1000
            if stepsize == None:
                stepsize = 0.1
            beta = .01

        if hparams.opt_alg == 'SNORE' or hparams.opt_alg == 'SNORE_Prox' or hparams.opt_alg == 'ARED_Prox' or hparams.opt_alg == 'SNORE_Adam':
            if std_0 == None:
                std_0 = 1.8 * hparams.noise_level_img /255.
            if std_end == None:
                std_end = 0.5 * hparams.noise_level_img / 255.
            if stepsize == None:
                stepsize = 0.1
            if lamb_end == None:
                lamb_end = 1.0
            if lamb_0 == None:
                lamb_0 = 0.1
            if maxitr == None:
                maxitr = 1500
            std = sigma_denoiser /255.
    
    return lamb, std, maxitr, thres_conv, stepsize, std_0, std_end, lamb_0, lamb_end, beta


def create_out_dir(exp_out_path, hparams, k_index = 0):
    """
        Create the directory to save results.
    """
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path_new = os.path.join(exp_out_path, hparams.degradation_mode)
    if not os.path.exists(exp_out_path_new):
        os.mkdir(exp_out_path_new)
    exp_out_path_new = os.path.join(exp_out_path_new, hparams.dataset_name)
    if not os.path.exists(exp_out_path_new):
        os.mkdir(exp_out_path_new)
    if hparams.degradation_mode == 'SR':
        exp_out_path_new = os.path.join(exp_out_path_new, "sf_"+str(hparams.sf))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
        exp_out_path_new = os.path.join(exp_out_path_new, hparams.opt_alg+"_k_"+str(k_index))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.degradation_mode == 'deblurring':
        exp_out_path_new = os.path.join(exp_out_path_new, hparams.opt_alg+"_k_"+str(k_index))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.degradation_mode == 'inpainting':
        exp_out_path_new = os.path.join(exp_out_path_new, hparams.opt_alg)
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
        exp_out_path_new = os.path.join(exp_out_path_new, "mask_prop_"+str(hparams.prop_mask))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.degradation_mode == "despeckle":
        exp_out_path_new = os.path.join(exp_out_path_new, "L_"+str(hparams.L))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
        exp_out_path_new = os.path.join(exp_out_path_new, hparams.opt_alg)
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.degradation_mode == "MRI":
        exp_out_path_new = os.path.join(exp_out_path_new, "acceleration_factor_"+str(hparams.acceleration_factor))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
        exp_out_path_new = os.path.join(exp_out_path_new, hparams.opt_alg)
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.denoiser_type != "GSDenoiser":
        exp_out_path_new = os.path.join(exp_out_path_new, hparams.denoiser_type)
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    exp_out_path_new = os.path.join(exp_out_path_new, "noise_"+str(hparams.noise_level_img))
    if not os.path.exists(exp_out_path_new):
        os.mkdir(exp_out_path_new)
    if hparams.maxitr != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "maxitr_"+str(hparams.maxitr))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.seed != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "seed_"+str(hparams.seed))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.stepsize != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "stepsize_"+str(hparams.stepsize))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.lamb_0 != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "lamb_0_"+str(hparams.lamb_0))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.lamb_end != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "lamb_end_"+str(hparams.lamb_end))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.std_0 != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "std_0_"+str(hparams.std_0))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.std_end != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "std_end_"+str(hparams.std_end))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.lamb != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "lamb_"+str(hparams.lamb))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.beta != None:
        exp_out_path = os.path.join(exp_out_path, "beta_"+str(hparams.beta))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if hparams.sigma_denoiser != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "sigma_denoiser_"+str(hparams.sigma_denoiser))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    ###
    if hparams.transformation != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "transformation_"+str(hparams.transformation))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    ###
    if hparams.im_init != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "im_init_"+hparams.im_init)
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.no_data_term == True:
        exp_out_path_new = os.path.join(exp_out_path_new, "no_data_term")
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.rot == True:
        exp_out_path_new = os.path.join(exp_out_path_new, "rot")
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.annealing_number != None:
        exp_out_path_new = os.path.join(exp_out_path_new, "annealing_number_"+str(hparams.annealing_number))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.num_noise != 1:
        exp_out_path_new = os.path.join(exp_out_path_new, "num_noise_"+str(hparams.num_noise))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    if hparams.opt_alg == "ERED":
        exp_out_path_new = os.path.join(exp_out_path_new, str(hparams.transformation))
        if not os.path.exists(exp_out_path_new):
            os.mkdir(exp_out_path_new)
    return exp_out_path_new


'''
Copyright (c) 2020 Kai Zhang (cskaizhang@gmail.com)
'''

def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def randomCrop(img1,img2,width,height):
    assert img1.shape[0] >= height
    assert img1.shape[1] >= width
    x = random.randint(0, img1.shape[1] - width)
    y = random.randint(0, img1.shape[0] - height)
    img1 = img1[y:y+height, x:x+width]
    img2 = img2[y:y + height, x:x + width]
    return img1,img2

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:int(H-H_r), :int(W-W_r), :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def crop_center(img,cropx,cropy):
    y,x = img.shape[0],img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def tensor2array(img):
    img = img.cpu()
    img = img.squeeze(0).detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def rescale(img):
    mintmp = img.min()
    maxtmp = img.max()
    img = (img - mintmp) / (maxtmp - mintmp)
    return img

def single2uint(img):
    return np.uint8((img*255.).round())

def imsave(img_path,img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def rgb2y(im):
    xform = np.array([.299, .587, .114])
    y = im.dot(xform.T)
    return y


def psnr(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    # img1 = np.float64(img1)
    # img2 = np.float64(img2)
    mse = np.mean((img1 - img2)**2)
    return 20 * np.log10(1. / np.sqrt(mse))

def psnr_torch(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10( 1. / torch.sqrt(mse))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def dct2(img) :
    if img.shape[-1] == 1 :
        return dct(dct(img.T, norm='ortho').T, norm='ortho')
    else :
        out = np.zeros(img.shape)
        for i in range(img.shape[-1]):
            out[:,:,i] = dct(dct(img[:,:,i].T, norm='ortho').T, norm='ortho')
        return out

def idct2(freq) :
    if freq.shape[-1] == 1 :
        return idct(idct(freq.T, norm='ortho').T, norm='ortho')
    else :
        out = np.zeros(freq.shape)
        for i in range(freq.shape[-1]):
            out[:,:,i] = idct(idct(freq[:,:,i].T, norm='ortho').T, norm='$' \
            'rtho')
        return out

def extract_low_high_DCT_f_images(img,rho=2):
    w, h = img.shape[0], img.shape[1]
    freq = dct2(img)
    w_out = int(w / rho)
    h_out = int(h / rho)
    low_f = np.copy(freq)
    high_f = np.copy(freq)
    high_f[:w_out,:h_out] = 0
    low_f[w_out:, h_out:] = 0
    import matplotlib.pyplot as plt
    plt.imshow(np.abs(high_f))
    plt.show()
    plt.imshow(np.abs(low_f))
    plt.show()
    return idct2(low_f), idct2(high_f)

def extract_low_high_f_images(img,rho=2):
    w, h = img.shape[0], img.shape[1]
    freq = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)))
    mask = np.abs(np.ones_like(freq))
    mask[int(w/(2*rho)):int((2*rho-1)*w/(2*rho)),int(h/(2*rho)):int((2*rho-1)*h/(2*rho))] = 0
    high_f = np.fft.fftshift(freq*mask)
    low_f = np.fft.fftshift(freq*(1-mask))
    return np.real(np.fft.ifft2(low_f, axes=(0, 1))), np.real(np.fft.ifft2(high_f, axes=(0, 1)))

def decompose_DCT_pyramid(img,levels,rho,use_scaling=False, show_dyadic_DCT_pyramid=False):
    if show_dyadic_DCT_pyramid :
        show_dyadic_DCT_pyramid(img, levels, use_scaling)
    w,h = img.shape[0],img.shape[1]
    freq = dct2(img)
    pyramid = []
    for l in range(levels) :
        w_out = int(w/(rho**l))
        h_out = int(h/(rho**l))
        if use_scaling:
            scaling = np.sqrt((w_out * h_out) / (w * h))
        else:
            scaling = 1.
        out_freq = freq[:w_out,:h_out]*scaling
        pyramid.append(idct2(out_freq))
    return pyramid

def show_dyadic_DCT_pyramid(img,levels,use_scaling=False):
    w,h = img.shape[0],img.shape[1]
    freq = dct2(img)
    for l in range(levels) :
        w_out = int(w/(2**l))
        h_out = int(h/(2**l))
        if use_scaling:
            scaling = np.sqrt((w_out * h_out) / (w * h))
        else:
            scaling = 1.
        freq[:w_out,:h_out] = freq[:w_out,:h_out]*scaling
    import matplotlib.pyplot as plt
    im = 20*np.log(np.abs(freq)+1)
    plt.imshow(im)
    plt.show()

def merge_coarse(image,coarse,frec,use_scaling = False):
    freq = dct2(image)
    tmp = dct2(coarse)
    w, h = tmp.shape[0], tmp.shape[1]
    w_out, h_out = freq.shape[0], freq.shape[1]
    wrec, hrec = int(w * frec), int(h * frec)
    if use_scaling :
        scaling = np.sqrt((w_out * h_out) / (w * h))
    else :
        scaling = 1.
    freq[:wrec, :hrec] = tmp[:wrec, :hrec] * scaling
    out = idct2(freq)
    return out

def recompose_DCT_pyramid(pyramid,frec):
    img = pyramid[0]
    for l in range(1,len(pyramid)) :
        img = merge_coarse(img,pyramid[l],frec)
    return img


def get_DPIR_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0, lamb = 0.23):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: lamb*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas

def rotate_image_tensor(image_tensor, device, generator=None, angle = None):
    """
    A function to rotate a torch.tensor square image named image_tensor by a multiple of 90°. 
    If the argument angle is given then the image is rotate by -angle, else a random angle is taken.
    """
    if angle == None:
        angles = [0, 90, 180, 270]
        angle_idx = torch.randint(0, len(angles), (1,), generator=generator, device = device).item()
        angle = angles[angle_idx]
    else:
        angle = angle
    if angle == 90:
        rotated_tensor = torch.rot90(image_tensor, k=1, dims=(2, 3))
    elif angle == 180:
        rotated_tensor = torch.rot90(image_tensor, k=2, dims=(2, 3))
    elif angle == 270:
        rotated_tensor = torch.rot90(image_tensor, k=3, dims=(2, 3))
    else:
        rotated_tensor = image_tensor
    return rotated_tensor, angle

def ffttrans(u, device, tx=0.):
    """
    Subpixel signal/image translation using Fourier (Shannon) interpolation 
    (tx,ty) is the position in output image corresponding to (0,0) in input image
    in other terms, input value (0,0) is translated to (tx,ty)
    output image is defined by v(l,k) = U(k-tx,l-ty), where U is the Shannon interpolate of u
    """
    _,_,nx = u.shape
    u = u.to(torch.float32)
    mx = nx//2
    P = torch.arange(mx,mx+nx)%nx - mx
    Tx = torch.exp(-2.*1j*pi*tx*P/nx).to(device)
    v = torch.real(torch.fft.ifft(torch.fft.fft(u)*Tx[None,None,:]))
    return v

def fftshear(u, a, b, device, axis=1):
    """
    Vectorize version.
    Apply an horizontal or vertical shear to an image with Fourier interpolation
    axis (0 or 1) specifies the coordinate along which the variable translation is applied
    If axis=1, output image v is defined by
      v[y, x] = U(y, x+a(y-b))    x=0..nx-1, y=0..ny-1
    where U is the Fourier interpolate of u
    """
    _,_,ny,nx = u.shape
    v = u.to(torch.float32)
    if axis==1:
        Y = (a*(torch.arange(ny, device=device) - b)).to(device)
        w = v.to(torch.float32)
        mx = nx//2
        P = (torch.arange(mx,mx+nx, device=device)%nx - mx).to(device)
        Tx = torch.exp(-2.*1j*pi*Y[:,None]*P[None,:]/nx).to(device)
        v = torch.real(torch.fft.ifft(torch.fft.fft(w, dim = -1)*Tx[None,None,:,:],dim = -1))
    elif axis==0:
        X = (a*(torch.arange(nx, device=device) - b)).to(device)
        w = v.to(torch.float32)
        my = ny//2
        P = (torch.arange(my,my+ny, device=device)%ny - my).to(device)
        Tx = torch.exp(-2.*1j*pi*X[None,:]*P[:,None]/ny).to(device)
        v = torch.real(torch.fft.ifft(torch.fft.fft(w, dim = 2)*Tx[None,None,:,:],dim = 2))
    else:
        raise NameError("Unrecognized axis value (should be 0 or 1)")
    return v


def random_transform_subpixel_rotation(device, generator):
    theta = (torch.rand(1, generator=generator, device=device)-1/2)*pi
    bool_0 = torch.randint(0, 2, (1,), generator=generator, device=device).item()
    def transform(x):
        Tx = x.clone()
        if bool_0 == 1:
            Tx = torch.flip(x, dims = [2,3])
        x0 = (Tx.shape[2]+1)/2
        y0 = (Tx.shape[3]+1)/2 # center of the image
        x1 = fftshear(Tx,-torch.tan(theta/2), y0, device, axis=1)
        x2 = fftshear(x1, torch.sin(theta), x0, device, axis=0)
        Tx = fftshear(x2,-torch.tan(theta/2), y0, device, axis=1)
        return Tx
    def inverse_transform(x, Tx):
        Tx0 = (Tx.shape[2]+1)/2
        Ty0 = (Tx.shape[3]+1)/2 # center of the image
        Tx1 = fftshear(Tx,torch.tan(theta/2), Ty0, device, axis=1)
        Tx2 = fftshear(Tx1, -torch.sin(theta), Tx0, device, axis=0)
        y = fftshear(Tx2,torch.tan(theta/2), Ty0, device, axis=1)
        if bool_0 == 1:
            y = torch.flip(y, dims = [2,3])
        return y
    return transform, inverse_transform

def random_transform_rotation(device, generator):
    """
    Random quarter rotation of a torch image. 
    """
    u = torch.randint(0, 4, (1,), generator=generator, device=device).item()
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

def random_transform_flip(device, generator):
    bool_1 = torch.randint(0, 2, (1,), generator=generator, device=device).item()
    bool_2 = torch.randint(0, 2, (1,), generator=generator, device=device).item()
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

def random_transform_translation(n,m, device, generator):
    n1 = torch.randint(0, n, (1,), generator=generator, device=device).item()
    n2 = torch.randint(0, m, (1,), generator=generator, device = device).item()
    def transform(x):
        x1 = x.clone()
        x1[:,:,:n-n1,:] = x[:,:,n1:,:]
        x1[:,:,n-n1:,:] = x[:,:,:n1,:]
        Tx = x1.clone()
        Tx[:,:,:,:m-n2] = x1[:,:,:,n2:]
        Tx[:,:,:,m-n2:] = x1[:,:,:,:n2]
        return Tx
    def inverse_transform(x,Tx):
        x1 = Tx.clone()
        x1[:,:,:n1,:] = Tx[:,:,n-n1:,:]
        x1[:,:,n1:,:] = Tx[:,:,:n-n1,:]
        y = x1.clone()
        y[:,:,:,:n2] = x1[:,:,:,m-n2:]
        y[:,:,:,n2:] = x1[:,:,:,:m-n2]
        return y
    return transform, inverse_transform

def random_transform_noise(std, x_shape, generator, device):
    z = torch.normal(torch.zeros(x_shape).to(device), std = torch.ones(x_shape).to(device), generator = generator)
    def transform(x):
        return x + std * z
    def inverse_transform(x,Tx):
        return Tx
    return transform, inverse_transform

def genMask(imgSize, numLines, device='cpu'):
    """
    Generate a mask for MRI reconstruction in torch.
    It is a translation in torch of the code proposed in https://github.com/wustl-cig/bcred/tree/master
    
    Args:
        imgSize (tuple): (high, width) of the image; need to be a multiple of 2
        numLines (int): number of ligne to draw
        device (str): 'cpu' or 'cuda'
    
    Returns:
        torch.BoolTensor: binary mask of size imgSize
    """
    H, W = imgSize
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("image must be even sized!")
    
    center = torch.tensor([H/2 + 1, W/2 + 1], device=device)
    freqMax = np.ceil(np.sqrt((H/2)**2 + (W/2)**2))
    
    # angles of the lines
    ang = torch.linspace(0, np.pi, steps=numLines+1, device=device)[:-1]  # we remove the last one to avoid multiples
    
    mask = torch.zeros(imgSize, dtype=torch.bool, device=device)
    
    # relative coordonates
    offsets = torch.arange(-freqMax, freqMax+1, device=device)
    
    for theta in ang:
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        # Float coordonnates
        ix = center[1] + offsets * cos_t
        iy = center[0] + offsets * sin_t
        
        # Integer coordonates
        ix = torch.floor(ix + 0.5).long()
        iy = torch.floor(iy + 0.5).long()
        
        # Filter to keep the valid indexes
        valid = (ix >= 1) & (ix <= W) & (iy >= 1) & (iy <= H)
        ix = ix[valid] - 1  # to go to the 1-based index
        iy = iy[valid] - 1
        
        mask[iy, ix] = True
    
    return mask

def fft2c(x):
    """
    2D centered FFT (unitary)
    """
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            norm="ortho"
        ),
        dim=(-2, -1)
    )

def ifft2c(k):
    """
    2D centered iFFT (unitary)
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(k, dim=(-2, -1)),
            norm="ortho"
        ),
        dim=(-2, -1)
    )

