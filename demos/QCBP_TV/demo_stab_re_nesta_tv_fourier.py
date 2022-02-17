import math
import torch
import numpy as np
import operators as op
import stability
import nn
from PIL import Image

with Image.open("../demo_images/test_image.png") as im:
    im.save("stab-ground-truth.png")
    X = np.asarray(im).astype(np.float64) / 255

# parameters
eta = 1e-1
N, _ = X.shape
inner_iters = 5
outer_iters = 8
mu0 = 1e-1
delta = 1e-12
r = 0.5

# generate sampling mask
mask = torch.zeros((N,N), dtype=bool)
for i in range(-N//2+1,N//2+1):
    for j in range(-N//2+1,N//2+1):
        if (max(abs(i),abs(j)) <= 0.25*(N/2)):
            mask[i+N//2,j+N//2] = True

mask = mask | (torch.rand((N,N)) < 0.1)

n_mask = mask.numpy()

(Image.fromarray(n_mask)).save("stab-mask.png")

m = torch.sum(mask)

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m)
print('Sample rate:', m/(N*N))

# generate functions for measurement and weight operators
subsampled_ft = lambda x, mode: op.fourier_2d(x,mode,N,mask)*(N/math.sqrt(m))
discrete_grad = lambda x, mode: op.discrete_gradient_2d(x,mode,N,N)
L_grad = 2.0

# define the inverse problem
noise = (eta/math.sqrt(m))*torch.randn(m)
T = torch.from_numpy(np.reshape(X.transpose(),N*N))
y = subsampled_ft(T,1) + noise

# compute worst-case perturbation 
# (see Sec. 19.4 of Adcock & Hansen)

lam = 100.0
x = torch.from_numpy(np.reshape(np.transpose(X),(N*N)))
z0 = torch.zeros(N*N,dtype=y.dtype)

def recon_fn(y):
    rec, _ = nn.restarted_nesta_wqcbp(
                y, z0,
                subsampled_ft, discrete_grad, L_grad,
                inner_iters, outer_iters,
                eta, mu0, delta, r, False)
    return rec

adv_noise, adv_rec = stability.projected_adversarial_perturbation(
    x, subsampled_ft, recon_fn, lam, 10, 1e-1, 0.9)

#assert len(adv_noise.shape) == 1
#print('l2 norm of perturbation:', torch.linalg.norm(adv_noise,2))

print(adv_noise.dtype)
print(adv_noise)

X_pert = np.transpose(np.reshape(np.real(adv_rec.numpy()),(N,N)))
X_pert = np.clip((X_pert*255),0,255).astype('uint8')

#pert_im = np.transpose(np.reshape(np.abs(adv_noise.numpy()),(N,N)))
#pert_im = np.clip(255*(pert_im-np.min(pert_im))/(np.max(pert_im)-np.min(pert_im)),0,255).astype('uint8')

Image.fromarray(X_pert).save("stab-worst-pert.png")
#Image.fromarray(pert_im).save("stab-worst-pert.png")
