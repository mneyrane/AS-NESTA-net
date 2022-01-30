# -*- coding: utf-8 -*-
"""
Compares the effect of setting eta and zeta when reconstructing an image
using restarted NESTA for measurements with no noise.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar analysis 
operator to recover an image from subsampled Fourier measurements.

-- Maksym Neyra-Nesterenko
-- mneyrane@sfu.ca
"""
import math
import operators
import sampling
import nn
import torch
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image


### fix the RNG seed for debugging
#torch.manual_seed(0)
#np.random.seed(0)


### load image

with Image.open("../demo_images/phantom_brain_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# grid parameters
eta = 10**(np.arange(2,-8,-1,dtype=float))  # noise level
zeta = 10**(np.arange(2,-8,-1,dtype=float)) # CS error parameter

# fixed parameters
sample_rate = 0.25  # sample rate
outer_iters = 15    # num of restarts + 1
r = 1/4             # decay factor
delta = 0.055       # rNSP parameter

# inferred parameters (mu and inner_iters are defined later)

N, _ = X.shape              # image size (assumed to be N by N)
m = (sample_rate/2)*N*N     # expected number of measurements


### generate sampling mask

var_hist = sampling.inverse_square_law_hist_2d(N,1)
var_probs = sampling.bernoulli_sampling_probs_2d(var_hist,N,m)
var_mask = sampling.generate_sampling_mask_from_probs(var_probs)

num_var_samples = np.sum(var_mask)
uni_mask_cond = np.random.rand(N*N-num_var_samples) <= m/(N*N-num_var_samples)

uni_mask = np.zeros((N,N), dtype=bool)
uni_mask[~var_mask] = uni_mask_cond

# logical OR the two masks
mask = uni_mask | var_mask
assert np.sum(mask) == np.sum(var_mask)+np.sum(uni_mask)

Image.fromarray(mask).save('NESTA_TV_Haar_eta_zeta_effect-mask.png')

m_exact = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().cuda()

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m_exact)
print('Target sample rate:', sample_rate)
print('Actual sample rate:', m_exact/(N*N))


### generate functions for measurement and weight operators

subsampled_ft = lambda x, mode: operators.fourier_2d(x,mode,N,mask_t,use_gpu=True)*(N/math.sqrt(m))

# define TV-Haar operator
def custom_op_tv_haar(x,mode,N,levels):
    if mode == 1:
        dgrad_x = operators.discrete_gradient_2d(x,1,N,N)
        wave_x = operators.discrete_haar_wavelet(x,1,N,levels)
        y = torch.cat([dgrad_x, wave_x])
        return y
    else: # adjoint
        x1 = x[:2*N*N]
        x2 = x[2*N*N:]
        y1 = operators.discrete_gradient_2d(x1,0,N,N)
        y2 = operators.discrete_haar_wavelet(x2,0,N,levels)
        y = y1+y2
        return y

# compute maximum Haar wavelet resolution
nlevmax = 0

while N % 2**(nlevmax+1) == 0:
    nlevmax += 1

assert nlevmax > 0

tv_haar = lambda x, mode: custom_op_tv_haar(x,mode,N,nlevmax)
sqrt_beta_tv_haar = 2.0


### compute normalizing constant for orthonormal rows of A

e1 = (torch.arange(m_exact) == 0).float().cuda()
c = torch.linalg.norm(subsampled_ft(e1,0), 2)**2
c = c.cpu()


### reconstruct image using restarted NESTA for each eta value

# create variables that are only need to be created once
X_flat_t = torch.from_numpy(np.reshape(X,N*N))
y = subsampled_ft(X_flat_t,1)

norm_fro_X = np.linalg.norm(X,'fro')

inner_iters = math.ceil(sqrt_beta_tv_haar*(4+r)/(r*math.sqrt(N)*delta))
print('Inner iterations:', inner_iters)

eta_grid, zeta_grid = np.meshgrid(eta, zeta, indexing='ij')

rel_errs = np.zeros(eta_grid.shape, dtype=float)

for i in range(len(eta)):
    for j in range(len(zeta)):
        print('(i,j) =', (i,j))
        eta_val, zeta_val = eta_grid[i,j], zeta_grid[i,j]

        ### compute restarted NESTA solution
        
        z0 = torch.zeros(N*N,dtype=y.dtype)
        
        eps0 = max(norm_fro_X, zeta_val)

        mu = []
        eps = eps0
        for k in range(outer_iters):
            mu.append(r*delta*eps)
            eps = r*eps + zeta_val

        y = y.cuda()
        z0 = z0.cuda()

        X_rec_t, _ = nn.restarted_nesta_wqcbp(
                y, z0,
                subsampled_ft, tv_haar, 
                c, sqrt_beta_tv_haar,
                inner_iters, outer_iters,
                eta_val, mu, False)

        X_rec = X_rec_t.cpu().numpy()
        X_rec = np.reshape(X_rec, (N,N))

        rel_errs[i,j] = np.linalg.norm(X-X_rec,'fro')/norm_fro_X

### plots

plt.imshow(rel_errs, norm=LogNorm(), interpolation='none')
plt.colorbar()
plt.xlabel('zeta parameter')
plt.ylabel('noise level')
plt.savefig('NESTA_TV_Haar_eta_zeta_effect-plot.png', dpi=300)
