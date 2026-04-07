# Equivariant Plug-and-Play for Image Restoration

This is the code associated with the paper Equivariant Denoisers for Plug and Play Image Restoration.

We focus on deblurring and MRI reconstruction.

## Create environment
To use the code, you can create a conda environement with the proposed dependencies. To do so, execute the following code
```
conda env create -f environment.yml
```

## Load pre-trained weights
To load the pretrained weights used in our experiments, use:
- For GSDRUNet for color images follow this [link](https://plmbox.math.cnrs.fr/f/ab6829cb933743848bef/?dl=1) 
- For GSDRUNet for grauscale images follow this [link](https://plmbox.math.cnrs.fr/f/04318d36824443a6bf8d/?dl=1) 
The weights need to be save in the folder GS_denoising/ckpts/

## How to execute the code
For image deblurring you can run the following command
```
python deblur.py --dataset_name "CBSD10" --opt_alg "RED" --stepsize 1.5 --sigma_denoiser 7. --maxitr 400
```
Note that for PnP based methods (PnP, EPnP, SnoPnP), we parametrize the iterations with the gradient step $\tau$: $x_{k+1} = D_{\sigma}(x_k - \tau \nabla f(x_k))$.
In this case, $\tau = 1 / {\lambda}$ if you take the values reported in the table 6 of the paper.

## Structure of the code
```
├── datasets/        
│ ├── CBSD10/             # 10 images of the dataset CBSD68
│ ├── CBSD68/
│ └── MRI_knee/           # 10 images from the Fast MRI dataset
├── GS_denoising/         # Code to train and use neural networks
├── PnP_restoration/      # Code to run PnP algorithms with Equivariance
│ ├── deblur.py           # Code for image deblurring
│ ├── MRI.py              # Code for MRI reconstruction
└ └── Main_restoration.py # Code with the forward models and algorithms
```

## Acknowledgement
This code is based on the repository : - [Equivariant Denoiser for Image Restoration](https://github.com/Marien-RENAUD/EquivariantPnP) 
