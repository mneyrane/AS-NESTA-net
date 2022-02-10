"""
Run stability experiments for restarted NESTA.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar analysis 
operator to recover an image from subsampled Fourier measurements.

-- Maksym Neyra-Nesterenko
-- mneyrane@sfu.ca
"""
import math
import os
import operators
import sampling
import stability
import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

### fix the RNG seed for debugging
#torch.manual_seed(0)
#np.random.seed(0)


### load image

with Image.open("../demo_images/brain_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### grid parameters
eta = 10**(np.arange(-3,2,1,dtype=float))       # noise level
pert_factor = 10**(np.arange(0,4,dtype=float))  # perturbation size multiplier
pga_lr = 0.1*eta

### parameters
sample_rate = 0.15      # sample rate
outer_iters = 15        # num of restarts + 1
r = 1/4                 # decay factor
zeta = 1e-12            # CS error parameter
delta = 0.05            # rNSP parameter
pga_num_trials = 2      # projected GA trials
pga_num_iters = 10     # projected GA number of iterations

fname_prefix = "NESTA_TV_Haar_stability"

# inferred parameters 
# (some of these are defined early since they we will define the
#  reconstruction map via an anonymous function)
eps0 = np.linalg.norm(X,'fro')

N, _ = X.shape              # image size (assumed to be N by N)
m = (sample_rate/2)*N*N     # expected number of measurements

mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

sqrt_beta_tv_haar = 2.0

inner_iters = math.ceil(4*sqrt_beta_tv_haar/(r*math.sqrt(N)*delta))
print('Inner iterations:', inner_iters)


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

Image.fromarray(mask).save(fname_prefix + '-mask.png')

m_exact = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().cuda()

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m_exact)
print('Target sample rate:', sample_rate)
print('Actual sample rate:', m_exact/(N*N))


### generate functions for measurement and weight operators

subsampled_ft = lambda x, mode: operators.fourier_2d(
        x,mode,N,mask_t,use_gpu=True)*(N/math.sqrt(m))

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


### compute normalizing constant for orthonormal rows of A

e1 = (torch.arange(m_exact) == 0).float().cuda()
c = torch.linalg.norm(subsampled_ft(e1,0), 2)**2
c = c.cpu()

### compute measurements

X_flat_t = torch.from_numpy(np.reshape(X,N*N))
X_flat_t = X_flat_t.cuda()

y = subsampled_ft(X_flat_t,1)
y = y.cuda()

# define initial guess
z0 = torch.zeros(N*N,dtype=torch.complex128)
z0 = z0.cuda()


### ~~ perform stability experiment ~~
num_exps = len(eta)*len(pert_factor)

exps_label_pad_len = len(str(num_exps))
trial_pad_len = len(str(pga_num_trials))

exps_counter = 0

for i in range(len(eta)):
    for j in range(len(pert_factor)):
        print('Stability experiment (i,j) =', (i,j))
        exps_label = str(exps_counter).zfill(exps_label_pad_len)

        # define reconstruction map
        def recon_map(y):
            rec, _ = nn.restarted_nesta_wqcbp(
                y, z0,
                subsampled_ft, tv_haar, 
                c, sqrt_beta_tv_haar,
                inner_iters, outer_iters,
                eta[i], mu, False)
            return rec

        
        ### compute worst-case perturbation
                 
        for t in range(pga_num_trials):
            print('Trial %d' % t)

            adv_noise_t = stability.adv_perturbation(
                X_flat_t, subsampled_ft, recon_map, 
                eta=pert_factor[j]*eta[i], 
                lr=pert_factor[j]*pga_lr[i], 
                num_iters=pga_num_iters,
                use_gpu=True)

            adv_noise_t = adv_noise_t.cpu()
            adv_noise = adv_noise_t.numpy()
            
            trial_str = 'trial_' + str(t).zfill(trial_pad_len)
            
            np.save(fname_prefix + ('-%s-adv_noise-%s.npy' % (exps_label, trial_str)), adv_noise)

        exps_counter += 1
