# -*- coding: utf-8 -*-
"""
Compare the performance of NESTA with and without restarts, under a fixed
iteration budget.

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
eta = 1e-1          # noise level
sample_rate = 0.15  # sample rate
outer_iters = 15    # num of restarts + 1
r = 1/4             # decay factor 
zeta = 1e-12        # CS error parameter
delta = 0.055       # rNSP parameter

# smoothing parameter for NESTA (without restarts)
nesta_mu = 10**np.arange(-2,-6,-1, dtype=float) 

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

Image.fromarray(mask).save('NESTA_TV_Haar_compare_without_restarts-mask.png')

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


### define the inverse problem

X_flat_t = torch.from_numpy(np.reshape(X,N*N))

noise = torch.randn(m_exact) + 1j*torch.rand(m_exact)
noise = eta * noise / torch.linalg.norm(noise,2)

y = subsampled_ft(X_flat_t,1) + noise


### compute normalizing constant for orthonormal rows of A

e1 = (torch.arange(m_exact) == 0).float().cuda()
c = torch.linalg.norm(subsampled_ft(e1,0), 2)**2
c = c.cpu()


### compute restarted NESTA solution

norm_fro_X = np.linalg.norm(X,'fro')
inner_iters = math.ceil(sqrt_beta_tv_haar*(4+r)/(r*math.sqrt(N)*delta))
total_iters = outer_iters*inner_iters
print('Inner iterations:', inner_iters)
print('Total iterations:', total_iters)

z0 = torch.zeros(N*N,dtype=y.dtype)
    
mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

y = y.cuda()
z0 = z0.cuda()

# compute restarted NESTA solution and compute relative error of iterates
_, restarted_iterates = nn.restarted_nesta_wqcbp(
    y, z0,
    subsampled_ft, tv_haar, 
    c, sqrt_beta_tv_haar,
    inner_iters, outer_iters,
    eta, mu, True)

r_its = [
    torch.reshape(it,(N,N))
    for re_its in restarted_iterates for it in re_its[1:]
]

r_rel_errs = list()

for it in r_its:
    X_it = it.cpu().numpy()
    r_rel_errs.append(np.linalg.norm(X-X_it,'fro')/norm_fro_X)


# compute NESTA solutions and relative error of iterates
n_rel_errs = dict()

for n_mu in nesta_mu:
    n_rel_errs[n_mu] = list()

    _, nesta_iterates = nn.nesta_wqcbp(
        y, z0, 
        subsampled_ft, tv_haar,
        c, sqrt_beta_tv_haar,
        total_iters, 
        eta, n_mu, True)

    n_its = [
        torch.reshape(it,(N,N))
        for it in nesta_iterates[1:]
    ]

    for it in n_its:
        X_it = it.cpu().numpy()
        n_rel_errs[n_mu].append(np.linalg.norm(X-X_it,'fro')/norm_fro_X)

    del nesta_iterates

assert len(r_rel_errs) == len(n_rel_errs[nesta_mu[0]])
assert len(r_rel_errs) == total_iters


### plots

sns.set(context='paper', style='whitegrid')

plt.semilogy(
    range(1,total_iters+1), 
    r_rel_errs, 
    label='With restarts',
    linewidth=1)

for n_mu in n_rel_errs:
    plt.semilogy(
        range(1,total_iters+1), 
        n_rel_errs[n_mu], 
        label='No restarts, mu = %.e' % n_mu,
        linewidth=1)

plt.xlabel('Iteration')
plt.ylabel('Relative error')
plt.legend(loc='upper right')
plt.savefig('NESTA_TV_Haar_compare_without_restarts-plot.png', dpi=300)
