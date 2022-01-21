"""
Generate plots of exponential decay in the image reconstruction error when
using restarted NESTA.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar analysis 
operator to recover an image from subsampled Fourier measurements.

NOTE:

In this code, the measurements sample each frequency at most once to be
compatible with NESTA. This results in a measurement matrix with
orthonormal rows (up to a normalizing factor).

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
noise_std = 1e-4   # - percentage of Gaussian noise
sample_rate = 0.265  # - target sample rate
outer_iters = 15   # - number of restarts
r = 1/3
zeta = 1e-12
delta = 5e-2

# inferred parameters (mu and inner_iters are defined later)
eps0 = np.linalg.norm(X,'fro')

N, _ = X.shape      # - image size (assumed to be N by N)
m_target = (sample_rate/2)*N*N


### generate sampling mask

var_hist = sampling.inverse_square_law_hist_2d(N,1)
uni_hist = sampling.uniform_hist_2d(N)

var_probs = sampling.bernoulli_sampling_probs_2d(var_hist,N,m_target)
uni_probs = sampling.bernoulli_sampling_probs_2d(uni_hist,N,m_target)

var_mask = sampling.generate_sampling_mask_from_probs(var_probs)
uni_mask = sampling.generate_sampling_mask_from_probs(uni_probs)

# logical OR the two masks
mask = uni_mask | var_mask

Image.fromarray(mask).save('NESTA_TV_Haar_restarts_mask.png')

m = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().cuda()

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m)
print('Target sample rate:', sample_rate)
print('Actual sample rate:', m/(N*N))


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

print("Number of levels in Haar wavelet decomposition:", nlevmax)

tv_haar = lambda x, mode: custom_op_tv_haar(x,mode,N,nlevmax)

sqrt_beta_tv_haar = 2.0


### define the inverse problem

noise = noise_std*(torch.randn(m) + 1j*torch.rand(m))
X_flat_t = torch.from_numpy(np.reshape(X,N*N))

y = subsampled_ft(X_flat_t,1) + noise

eta = np.linalg.norm(noise)

e1 = (torch.arange(len(y)) == 0).float().cuda()
c = torch.linalg.norm(subsampled_ft(e1,0), 2)**2
c = c.cpu()

print('eta:', eta)


### reconstruct image using restarted NESTA 

z0 = torch.zeros(N*N,dtype=y.dtype)
    
inner_iters = math.ceil(sqrt_beta_tv_haar*(4+r)/(r*math.sqrt(N)*delta))
    
print('Inner iterations:', inner_iters)
    
mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

y = y.cuda()
z0 = z0.cuda()

X_rec_t, iterates = nn.restarted_nesta_wqcbp(
        y, z0,
        subsampled_ft, tv_haar, 
        c, sqrt_beta_tv_haar,
        inner_iters, outer_iters,
        eta, mu, True)


### extract restart values

final_its = [torch.reshape(its[-1],(N,N)) for its in iterates]

norm_fro_X = np.linalg.norm(X,'fro')

rel_errs = []

for X_final in final_its:
    X_final = X_final.cpu().numpy()
    rel_errs.append(np.linalg.norm(X-X_final,'fro')/norm_fro_X)


### save reconstructed image

X_rec_t = X_rec_t.cpu()
X_rec = np.reshape(np.abs(X_rec_t.numpy()),(N,N))
im_rec = np.clip((X_rec*255),0,255).astype('uint8')

Image.fromarray(im_rec).save('NESTA_TV_Haar_restarts_recon.png')


### plots

sns.set(context='paper', style='white')

plt.semilogy(range(1,len(rel_errs)+1), rel_errs, '* r')
plt.xlabel('Restart')
plt.ylabel('Relative error')
plt.savefig('NESTA_TV_Haar_restarts_plot.png')
