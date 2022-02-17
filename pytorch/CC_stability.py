"""
Run stability experiments for restarted NESTA.

The setup is to solve a smoothed analysis QCBP problem with a TV-Haar analysis 
operator to recover an image from subsampled Fourier measurements.

-- Maksym Neyra-Nesterenko
-- mneyrane@sfu.ca
"""
import argparse
import os
import re
import math
from datetime import datetime

import operators
import stability
import nn

import torch
import numpy as np
from PIL import Image


# parse command-line args
parser = argparse.ArgumentParser(description="Stability experiment")
parser.add_argument('--im-path', type=str, action='store')
parser.add_argument('--mask-path', type=str, action='store')
parser.add_argument('--eta', type=float, action='store')
parser.add_argument('--eta-p-mult', type=float, action='store')
args = parser.parse_args()

### load data

with Image.open(args.im_path) as im:
    X = np.asarray(im).astype(float) / 255

with Image.open(args.mask_path) as im_mask:
    mask = np.asarray(im_mask)

# change to a new directory to save results in
dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(dir_name)
os.chdir(dir_name)


# global parameters

eta = args.eta                      # noise level
eta_pert = args.eta_p_mult * eta    # perturbation noise multiplier

outer_iters = 11    # num of restarts + 1
r = 1/5            # decay factor
zeta = 1e-9        # CS error parameter
delta = 0.05      # rNSP parameter

pga_num_iters = 5   # gradient ascent iterations
pga_lr = 1.0        # gradient ascent step size


# inferred parameters

sp_match = re.search("tv_haar_mask_([0-9][0-9]).+", args.mask_path)
sample_rate = float(sp_match.group(1))/100.
print('Sample rate:', sample_rate)

eps0 = np.linalg.norm(X,'fro')
N, _ = X.shape          # image size (assumed to be N by N)
m = sample_rate*N*N     # expected number of measurements

# smoothing parameters
mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

# gradient Lipschitz constant of smoothed TV-Haar
lam = 0.25
L_W = math.sqrt(1+8*lam)

# inner iterations
inner_iters = math.ceil(2*L_W/(r*math.sqrt(N)*delta))
print('Inner iterations:', inner_iters)

# generate subsampled Fourier matrix
mask_t = (torch.from_numpy(mask)).bool().cuda()

A = lambda x, mode: operators.fourier_2d(
    x,mode,N,mask_t,use_gpu=True)*(N/math.sqrt(m))

# compute maximum Haar wavelet resolution
nlevmax = 0

while N % 2**(nlevmax+1) == 0:
    nlevmax += 1

assert nlevmax > 0

# define TV-Haar analysis operator
W = lambda x, mode: operators.tv_haar_2d(x,mode,N,lam,nlevmax)

# compute normalizing constant for orthonormal rows of A
m_exact = np.sum(mask)
e1 = (torch.arange(m_exact) == 0).float().cuda()
c_A = torch.linalg.norm(A(e1,0), 2)**2
c_A = c_A.cpu()

# define reconstruction map
z0 = torch.zeros(N*N,dtype=torch.complex128)
z0 = z0.cuda()

def R(y):
    rec, _ = nn.restarted_nesta_wqcbp(
        y, z0, A, W, c_A, L_W,
        inner_iters, outer_iters,
        eta, mu, False)
    return rec

# compute measurements
X_vec_t = torch.from_numpy(np.reshape(X,N*N))
X_vec_t = X_vec_t.cuda()

y = A(X_vec_t,1)
y = y.cuda()

# reconstruct image using restarted NESTA 
X_rec_t = R(y)

# save reconstructions
X_rec_t = X_rec_t.cpu()
X_rec = np.reshape(X_rec_t.numpy(),(N,N))
np.save('im_rec.npy', X_rec)

# compute worst-case perturbation
adv_pert_t, X_pert_t, X_pert_rec_t = stability.adv_perturbation(
    X_vec_t, A, R, c_A=c_A, eta=eta_pert, 
    lr=pga_lr, num_iters=pga_num_iters, use_gpu=True)

adv_pert_t = adv_pert_t.cpu()
X_pert_t = X_pert_t.cpu()
X_pert_rec_t = X_pert_rec_t.cpu()

adv_pert = np.reshape(adv_pert_t.numpy(),(N,N))
X_pert = np.reshape(X_pert_t.numpy(),(N,N))
X_pert_rec = np.reshape(X_pert_rec_t.numpy(),(N,N))

# save perturbed and reconstruction of perturbed image
np.save('adv_pert.npy', adv_pert)
np.save('im_pert_rec.npy', X_pert_rec)

with open('params.txt','w') as params_fd:
    print("eta:", eta, file=params_fd)
    print("eta pert:", eta_pert, file=params_fd)
    print("pert size:", np.linalg.norm(adv_pert,'fro'), file=params_fd)
    print("pert rec error:", np.linalg.norm(X_rec-X_pert_rec,'fro'), file=params_fd)
