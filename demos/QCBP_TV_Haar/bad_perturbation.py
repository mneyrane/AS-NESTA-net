# -*- coding: utf-8 -*-
"""
Run stability experiments for restarted NESTA.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar analysis 
operator to recover an image from subsampled Fourier measurements.
"""
import math
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import nestanet.stability as n_st
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


### fix the RNG seed for debugging
#torch.manual_seed(0)
#np.random.seed(0)


### load image

with Image.open("../images/GPLU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### fixed parameters
eta = 1e-2         # noise level
sample_rate = 0.25 # sample rate
outer_iters = 10   # num of restarts + 1
r = 1/4            # decay factor
zeta = 1e-9        # CS error parameter
delta = 0.1        # rNSP parameter
lam = 2.5          # TV-Haar parameter

pga_num_iters = 10
pga_lr = 3.0
stab_eta = 1*eta


# inferred parameters 
# (some of these are defined early since they we will define the
#  reconstruction map via an anonymous function)
eps0 = np.linalg.norm(X,'fro')

N, _ = X.shape           # image size (assumed to be N by N)
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

Image.fromarray(mask).save('bad_perturbation-mask.png')

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


### perform stability experiments on several images

mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

inner_iters = math.ceil(2*L_W/(r*math.sqrt(N)*delta))
print('Inner iterations:', inner_iters)

# define reconstruction map
z0 = torch.zeros(N*N,dtype=torch.complex128)
z0 = z0.cuda()

def R(y):
    rec, _ = n_nn.restarted_nesta_wqcbp(
        y, z0, A, W, c_A, L_W,
        inner_iters, outer_iters,
        eta, mu, False)
    return rec


### compute measurements

X_vec_t = torch.from_numpy(np.reshape(X,N*N))
X_vec_t = X_vec_t.cuda()

y = A(X_vec_t,1)
y = y.cuda()


### reconstruct image using restarted NESTA 

X_rec_t = R(y)

### save reconstructions

X_rec_t = X_rec_t.cpu()
X_rec = np.reshape(X_rec_t.numpy(),(N,N))

im_rec = np.clip(np.abs(X_rec)*255,0,255).astype('uint8')
Image.fromarray(im_rec).save('bad_perturbation-im_rec.png')

print("Relative error:", np.linalg.norm(X-X_rec,'fro')/np.linalg.norm(X,'fro'))

### compute worst-case perturbation

adv_pert_t, X_pert_t, X_pert_rec_t = n_st.adv_perturbation(
    X_vec_t, A, R, c_A=c_A, eta=stab_eta, 
    lr=pga_lr, num_iters=pga_num_iters, use_gpu=True)

adv_pert_t = adv_pert_t.cpu()
X_pert_t = X_pert_t.cpu()
X_pert_rec_t = X_pert_rec_t.cpu()

adv_pert = np.reshape(adv_pert_t.numpy(),(N,N))
X_pert = np.reshape(X_pert_t.numpy(),(N,N))
X_pert_rec = np.reshape(X_pert_rec_t.numpy(),(N,N))

print("Pert size:", np.linalg.norm(adv_pert,'fro'))
print("Pert reconstruction error:", np.linalg.norm(X_rec-X_pert_rec,'fro'))

sns.set(context='paper', style='whitegrid')

# show adversarial perturbation rescaled
plt.figure()
plt.imshow(np.abs(adv_pert), interpolation='none')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.savefig('bad_perturbation-adv_pert.png', bbox_inches='tight', dpi=300)

# show absolute difference of truth and perturbed reconstruction 
plt.figure()
plt.imshow(np.abs(X_rec-X_pert_rec), interpolation='none')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.savefig('bad_perturbation-adv_abs_err.png', bbox_inches='tight', dpi=300)

# save perturbed and reconstruction of perturbed image
im_pert = np.clip(np.abs(X_pert)*255,0,255).astype('uint8')
Image.fromarray(im_pert).save('bad_perturbation-im_pert.png')   

im_pert_rec = np.clip(np.abs(X_pert_rec)*255,0,255).astype('uint8')
Image.fromarray(im_pert_rec).save('bad_perturbation-im_pert_rec.png')
