"""
Show reconstruction and stability of restarted NESTA.

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
import os
import operators
import sampling
import stability
import nn
import torch
import numpy as np
from PIL import Image
from glob import glob

### fix the RNG seed for debugging
#torch.manual_seed(0)
#np.random.seed(0)

### fixed parameters
N = 512
noise_std = 5e-2    # - percentage of Gaussian noise
sample_rate = 0.105 # - target sample rate
outer_iters = 6  # - number of restarts
r = 1/8
zeta = 1e-12
delta = 0.5

# inferred parameters

mu = []
eps = float(N)
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

sqrt_beta_tv_haar = 2.0 # Lipschitz constant

inner_iters = math.ceil(sqrt_beta_tv_haar*(4+r)/(r*math.sqrt(N)*delta))

m_target = (sample_rate/2)*N*N

eta = 2*noise_std*math.sqrt(2*m_target)

print('Inner iterations:', inner_iters)
print('eta:', eta)

### generate sampling mask
var_hist = sampling.inverse_square_law_hist_2d(N,1)
uni_hist = sampling.uniform_hist_2d(N)

var_probs = sampling.bernoulli_sampling_probs_2d(var_hist,N,m_target)
uni_probs = sampling.bernoulli_sampling_probs_2d(uni_hist,N,m_target)

var_mask = sampling.generate_sampling_mask_from_probs(var_probs)
uni_mask = sampling.generate_sampling_mask_from_probs(uni_probs)

# logical OR the two masks
mask = uni_mask | var_mask

Image.fromarray(mask).save('NESTA_TV_Haar_rec_stab_mask.png')

m = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().cuda()

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m)
print('Target sample rate:', sample_rate)
print('Actual sample rate:', m/(N*N))

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

print("Number of levels in Haar wavelet decomposition:", nlevmax)

tv_haar = lambda x, mode: custom_op_tv_haar(x,mode,N,nlevmax)



### perform experiments on several images

# define reconstruction map
z0 = torch.zeros(N*N,dtype=torch.complex128)
z0 = z0.cuda()

def recon_map(y):
    rec, _ = nn.restarted_nesta_wqcbp(
        y, z0,
        subsampled_ft, tv_haar, 
        c, sqrt_beta_tv_haar,
        inner_iters, outer_iters,
        eta, mu, False)
    return rec

image_files = glob("../demo_images/*%d.png" % N)

for im_path in image_files:
    im_name, _ = os.path.splitext(os.path.basename(im_path))
    
    print("EXPERIMENT WITH %s" % im_path)

    ### load image
    with Image.open(im_path) as im:
        X = np.asarray(im).astype(float) / 255

    ### define the inverse problem
    noise = noise_std*(torch.randn(m) + 1j*torch.rand(m))
    noise = noise.cuda()
    
    X_flat_t = torch.from_numpy(np.reshape(X,N*N))
    X_flat_t = X_flat_t.cuda()
    
    y = subsampled_ft(X_flat_t,1) + noise
    y = y.cuda()


    e1 = (torch.arange(len(y)) == 0).float().cuda()
    c = torch.linalg.norm(subsampled_ft(e1,0), 2)**2
    c = c.cpu()
    
    print('actual eta:', torch.linalg.norm(noise,2))

    ### reconstruct image using restarted NESTA 

    X_rec_t  = recon_map(y)

    ### save reconstructions

    X_rec_t = X_rec_t.cpu()
    X_rec = np.reshape(np.abs(X_rec_t.numpy()),(N,N))
    im_rec = np.clip((X_rec*255),0,254).astype('uint8')

    Image.fromarray(im_rec).save(
        'NESTA_TV_Haar_rec_stab_recon_%s.png' % im_name)
    
    """
    ### compute worst-case perturbation

    eta_factors = [4, 8, 16]
    
    for scalar in eta_factors:
        adv_noise, X_adv_rec_t = stability.projected_adversarial_perturbation(
            X_flat_t, subsampled_ft, recon_map, 
            noise_std/2, scalar*eta, 100, use_gpu=True)
        
        X_adv_rec_t = X_adv_rec_t.cpu()
        X_adv_rec = np.reshape(np.abs(X_adv_rec_t.numpy()),(N,N))
        im_adv_rec = np.clip((X_adv_rec*255),0,255).astype('uint8')
        Image.fromarray(im_adv_rec).save(
            'NESTA_TV_Haar_rec_stab_advrecon_%d_%s.png' % (scalar, im_name))
    """
