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
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image


### load image

with Image.open("../images/GPLU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# grid parameters
eta = 10**(np.arange(1,-8,-1,dtype=float))  # noise level
zeta = 10**(np.arange(1,-8,-1,dtype=float)) # CS error parameter

# fixed parameters
sample_rate = 0.25  # sample rate
outer_iters = 15    # num of restarts + 1
r = 1/4             # decay factor
delta = 0.05        # rNSP parameter
lam = 2.5           # TV-Haar parameter

# inferred parameters (mu and inner_iters are defined later)

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

Image.fromarray(mask).save('eta_zeta_tuning-mask.png')

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


### compute normalizing constant for orthonormal rows of A

e1 = (torch.arange(m_exact) == 0).float().cuda()
c_A = torch.linalg.norm(A(e1,0), 2)**2
c_A = c_A.cpu()


### reconstruct image using restarted NESTA for each eta value

# create variables that are only need to be created once
X_vec_t = torch.from_numpy(np.reshape(X,N*N))
y = A(X_vec_t,1)

norm_fro_X = np.linalg.norm(X,'fro')
print('Frobenius norm of X:', norm_fro_X)

inner_iters = math.ceil(2*L_W/(r*math.sqrt(N)*delta))
print('Inner iterations:', inner_iters)

eta_grid, zeta_grid = np.meshgrid(eta, zeta, indexing='ij')

errs = np.zeros(eta_grid.shape, dtype=float)

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

        X_rec_t, _ = n_nn.restarted_nesta_wqcbp(
                y, z0,A, W, c_A, L_W,
                inner_iters, outer_iters,
                eta_val, mu, False)

        X_rec = X_rec_t.cpu().numpy()
        X_rec = np.reshape(X_rec, (N,N))

        errs[i,j] = np.linalg.norm(X-X_rec,'fro')

### plots
sns.set(context='paper', style='whitegrid')
xticklabels = np.log10(zeta).astype(int)
yticklabels = np.log10(eta).astype(int)
sns.heatmap(
    errs, 
    xticklabels=xticklabels, yticklabels=yticklabels, 
    norm=LogNorm(), cmap='viridis')
plt.yticks(rotation=0)
plt.xlabel('$\\log_{10}(\\zeta)$')
plt.ylabel('$\\log_{10}(\\eta)$')
plt.savefig('eta_zeta_tuning-plot.png', bbox_inches='tight', dpi=300)
