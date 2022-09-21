# -*- coding: utf-8 -*-
"""
Compare recoveries of TV-Haar, TV, and Haar minimization.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar, TV,
and Haar analysis operators to recover an image from subsampled Fourier 
measurements.
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

with Image.open("../demo_images/brain_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
lam = 5.            # TV-Haar weight
eta = 1e-4          # noise level
sample_rate = 0.20  # sample rate
outer_iters = 15    # num of restarts + 1
r = 1/4             # decay factor
zeta = 1e-9         # CS error parameter
delta = 0.055       # rNSP parameter

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

m_exact = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().cuda()

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m_exact)
print('Target sample rate:', sample_rate)
print('Actual sample rate:', m_exact/(N*N))


### generate functions for measurement and weight operators

subsampled_ft = lambda x, mode: operators.fourier_2d(x,mode,N,mask_t,use_gpu=True)*(N/math.sqrt(m))

# define TV-Haar operator

# compute maximum Haar wavelet resolution
nlevmax = 0

while N % 2**(nlevmax+1) == 0:
    nlevmax += 1

assert nlevmax > 0

op_tv_haar = lambda x, mode: operators.tv_haar_2d(x,mode,N,lam,nlevmax)
op_tv = lambda x, mode: operators.discrete_gradient_2d(x,mode,N,N)
op_haar = lambda x, mode: operators.discrete_haar_wavelet_2d(x,mode,N,nlevmax)

L_tv_haar = math.sqrt(1+8*lam)
L_tv = 2.*math.sqrt(2.)
L_haar = 1.

M_tv_haar = 3*N*N
M_tv = 2*N*N
M_haar = N*N

op_params = {
    "tv-haar" : (op_tv_haar, L_tv_haar, M_tv_haar), 
    "tv" : (op_tv, L_tv, M_tv), 
    "haar" : (op_haar, L_haar, M_haar),
}


### compute normalizing constant for orthonormal rows of A

e1 = (torch.arange(m_exact) == 0).float().cuda()
c = torch.linalg.norm(subsampled_ft(e1,0), 2)**2
c = c.cpu()


### define the inverse problem

X_flat_t = torch.from_numpy(np.reshape(X,N*N))

noise = torch.randn(m_exact) + 1j*torch.rand(m_exact)
noise = eta * noise / torch.linalg.norm(noise,2)

y = subsampled_ft(X_flat_t,1) + noise


### reconstruct image using restarted NESTA for each analysis operator

# fix the inner iterations for each analysis operator
# note that the largest Lipschitz constnat is L_tv_haar and smallest M is N*N
inner_iters = math.ceil(2*L_tv_haar/(r*N*delta))
print('Inner iterations:', inner_iters)

# compute mu
mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

norm_fro_X = np.linalg.norm(X,'fro')
 
for op_name in op_params:
    
    ### compute restarted NESTA solution

    opW, L_W, M = op_params[op_name]
    
    z0 = torch.zeros(N*N,dtype=y.dtype)
        
    y = y.cuda()
    z0 = z0.cuda()

    X_rec_t, _ = nn.restarted_nesta_wqcbp(
            y, z0,
            subsampled_ft, opW, 
            c, L_W,
            inner_iters, outer_iters,
            eta, mu, False)


    ### extract restart values

    X_rec_t = X_rec_t.cpu()
    X_rec = np.reshape(X_rec_t.numpy(),(N,N))
    
    print("Relative error of final iterate for %s: %.5e" % (op_name, np.linalg.norm(X-X_rec,'fro')/norm_fro_X))

    ### save reconstructed image

    im_rec = np.clip(np.abs(X_rec)*255,0,255).astype('uint8')

    Image.fromarray(im_rec).save(
        'NESTA_TV_Haar_compare_analysis_ops-%s_recon.png' % op_name)

    im_err = np.clip(np.abs(X_rec-X)*255,0,255).astype('uint8')

    Image.fromarray(im_err).save(
        'NESTA_TV_Haar_compare_analysis_ops-%s_error.png' % op_name)
