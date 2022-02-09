# -*- coding: utf-8 -*-
"""
Generate plots of exponential decay in the image reconstruction error when
using restarted NESTA.

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
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


### fix the RNG seed for debugging
#torch.manual_seed(0)
#np.random.seed(0)


### load image

with Image.open("../demo_images/phantom_brain_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
eta = 10**(np.arange(0,-4,-1,dtype=float)) # noise level
sample_rate = 0.15          # sample rate
outer_iters = 15            # num of restarts + 1
r = 1/4                     # decay factor
zeta = 1e-12                # CS error parameter
delta = 0.055               # rNSP parameter

# inferred parameters (mu and inner_iters are defined later)
eps0 = np.linalg.norm(X,'fro')

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

Image.fromarray(mask).save('NESTA_TV_Haar_restarts-mask.png')

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

norm_fro_X = np.linalg.norm(X,'fro')

inner_iters = math.ceil(sqrt_beta_tv_haar*(4+r)/(r*math.sqrt(N)*delta))
print('Inner iterations:', inner_iters)

mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

rel_errs_dict = dict()

for noise_level in eta:
    
    ### define the inverse problem

    noise = torch.randn(m_exact) + 1j*torch.rand(m_exact)
    e = noise_level * noise / torch.linalg.norm(noise,2)

    y = subsampled_ft(X_flat_t,1) + e
    
    
    ### compute restarted NESTA solution
    
    z0 = torch.zeros(N*N,dtype=y.dtype)
        
    y = y.cuda()
    z0 = z0.cuda()

    X_rec_t, iterates = nn.restarted_nesta_wqcbp(
            y, z0,
            subsampled_ft, tv_haar, 
            c, sqrt_beta_tv_haar,
            inner_iters, outer_iters,
            noise_level, mu, True)


    ### extract restart values

    final_its = [torch.reshape(its[-1],(N,N)) for its in iterates]

    rel_errs = list()

    for X_final in final_its:
        X_final = X_final.cpu().numpy()
        rel_errs.append(np.linalg.norm(X-X_final,'fro')/norm_fro_X)

    rel_errs_dict[noise_level] = rel_errs


### plots

sns.set(context='paper', style='whitegrid')

for noise_level in eta:
    end_idx = len(rel_errs_dict[noise_level])+1
    plt.semilogy(
        range(1,end_idx), 
        rel_errs_dict[noise_level], 
        label='%.e' % noise_level,
        marker='*',
        linewidth=1)

plt.xlabel('Restart')
plt.ylabel('Relative error')
plt.legend(loc='lower left')
plt.savefig('NESTA_TV_Haar_restarts-plot.png', dpi=300)