import deepinv as dinv
from torchvision.transforms import Compose, ColorJitter, RandomErasing, Resize
import matplotlib.pyplot as plt

x = plt.imread("dataset/CBSD10/0000.png")

# Transformation
rotate = dinv.transform.Rotate(multiples=90, positive=True, n_trans=4)
transform = rotate * dinv.transform.Reflect(dim=[-1], n_trans=2)

sigma = 0.1
physics = dinv.physics.GaussianNoise(sigma=sigma)
y = physics(Resize(128)(x))

model = dinv.models.MedianFilter()
model_eq = dinv.models.EquivariantDenoiser(model, transform=transform)

plt.imsave(x, "im_gt.png")
plt.imsave(y, "im_noisy.png")
plt.imsave(x, "im_denoisy.png")
plt.imsave(x, "im_denoisy_eq.png")