# -*- coding: utf-8 -*-
"""
Generate plots of exponential decay in the image reconstruction error when
using restarted NESTA.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar analysis 
operator to recover an image from subsampled Fourier measurements.
"""
import math
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


### load image

with Image.open("../images/GPLU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
eta = [1e0, 1e-1, 1e-2, 1e-3, 1e-4]  # noise level
sample_rate = 0.15  # sample rate
outer_iters = 15    # num of restarts + 1
r = 1/4             # decay factor
zeta = 1e-9         # CS error parameter
delta = 1.25e-3     # rNSP parameter
lam = 2.5           # TV-Haar parameter

# inferred parameters (mu and inner_iters are defined later)
eps0 = np.linalg.norm(X,'fro')

N, _ = X.shape          # image size (assumed to be N by N)
m = sample_rate*N*N     # expected number of measurements


### generate sampling mask

var_hist = n_sp.inverse_square_law_hist_2d(N,1)
var_probs = n_sp.bernoulli_sampling_probs_2d(var_hist,N,m/2)
var_mask = n_sp.generate_sampling_mask_from_probs(var_probs)

num_var_samples = np.sum(var_mask)
uni_mask_cond = np.random.rand(N*N-num_var_samples) <= (m/2)/(N*N-num_var_samples)

uni_mask = np.zeros((N,N), dtype=bool)
uni_mask[~var_mask] = uni_mask_cond

# logical OR the two masks
mask = uni_mask | var_mask
assert np.sum(mask) == np.sum(var_mask)+np.sum(uni_mask)

Image.fromarray(mask).save('restarts-mask.png')

m_exact = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().cuda()

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m_exact)
print('Target sample rate:', sample_rate)
print('Actual sample rate:', m_exact/(N*N))


### generate functions for measurement and weight operators

A = lambda x, mode: n_op.fourier_2d(x,mode,N,mask_t,use_gpu=True)*(N/math.sqrt(m))

# compute maximum Haar wavelet resolution
nlevmax = 0

while N % 2**(nlevmax+1) == 0:
    nlevmax += 1

assert nlevmax > 0

W = lambda x, mode: n_op.tv_haar_2d(x,mode,N,lam,nlevmax)
L_W = math.sqrt(1+8*lam)
M = 3*N*N # output dimension of W


### compute normalizing constant for orthonormal rows of A

e1 = (torch.arange(m_exact) == 0).float().cuda()
c_A = torch.linalg.norm(A(e1,0), 2)**2
c_A = c_A.cpu()


### reconstruct image using restarted NESTA for each eta value

# create variables that are only need to be created once
X_vec_t = torch.from_numpy(np.reshape(X,N*N))

norm_fro_X = np.linalg.norm(X,'fro')
print('Frobenius norm of X:', norm_fro_X)

inner_iters = math.ceil(2*L_W/(r*math.sqrt(M)*delta)) - 1
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

    y = A(X_vec_t,1) + e
    
    
    ### compute restarted NESTA solution
    
    z0 = torch.zeros(N*N,dtype=y.dtype)
        
    y = y.cuda()
    z0 = z0.cuda()

    _, iterates = n_nn.restarted_nesta_wqcbp(
        y, z0, A, W, c_A, L_W, 
        inner_iters, outer_iters, noise_level, mu, True)


    ### extract restart values

    final_its = [torch.reshape(its[-1],(N,N)) for its in iterates]

    rel_errs = list()

    for X_final in final_its:
        X_final = X_final.cpu().numpy()
        rel_errs.append(np.linalg.norm(X-X_final,'fro')/norm_fro_X)

    rel_errs_dict[noise_level] = rel_errs


### plots

sns.set(context='paper', style='whitegrid', font='sans', font_scale=1.4, rc={'text.usetex' : True})

for noise_level in eta:
    end_idx = len(rel_errs_dict[noise_level])+1
    plt.semilogy(
        range(1,end_idx), 
        rel_errs_dict[noise_level], 
        label='$\\eta = 10^{%d}$' % math.log10(noise_level),
        marker='o',
        markersize=4,
        linewidth=2)

plt.xlabel('Restart')
plt.ylabel('Relative error')
plt.legend(loc='lower left')
plt.savefig('restarts-plot.pdf', bbox_inches='tight', dpi=300)
