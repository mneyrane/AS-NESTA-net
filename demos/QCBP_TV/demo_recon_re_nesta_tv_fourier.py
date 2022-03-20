"""
Demo for reconstructing an image from subsampled Fourier measurements.
This is done by using restarted NESTA to solve an analysis QCBP problem
where the analysis operator is the discrete gradient (or TV) operator.

In this code, the measurements sample each frequency at most once to be
compatible with NESTA. This results in a measurement matrix with
orthonormal rows (up to a normalizing factor).

-- Maksym Neyra-Nesterenko
-- mneyrane@sfu.ca
"""
import math
import torch
import numpy as np
import operators as op
import nn
from PIL import Image

# fix the RNG seed for debugging
torch.manual_seed(0)
np.random.seed(0)

# load image
with Image.open("../demo_images/phantom_brain_512.png") as im:
    X = np.asarray(im).astype(float) / 255

# parameters
eta = 1e-1          # - noise level
N, _ = X.shape      # - image size (assumed to be N by N)
inner_iters = 5     # - number of NESTA iterations
outer_iters = 8     # - number of restarts
mu0 = 1e-1          # - initial smoothing parameter
delta = 1e-12       # - ?
r = 0.5             # - smoothing decay factor

# generate sampling mask
# - i.e. a zero-one matrix with same shape as the image
# - can also load in a sampling mask from data or an image

# ...
mask = torch.zeros((N,N), dtype=bool)
for i in range(-N//2+1,N//2+1):
    for j in range(-N//2+1,N//2+1):
        if (max(abs(i),abs(j)) <= 0.25*(N/2)):
            mask[i+N//2,j+N//2] = True

mask = mask | (torch.rand((N,N)) < 0.1)

n_mask = mask.numpy()

(Image.fromarray(n_mask)).save('demo_restarted_nesta_tv_reconstruction-mask.png')

m = torch.sum(mask)

print('Image size (number of pixels):', N*N)
print('Number of measurements:', m)
print('Sample rate:', m/(N*N))

# generate functions for measurement and weight operators
subsampled_ft = lambda x, mode: op.fourier_2d(x,mode,N,mask)*(N/math.sqrt(m))
discrete_grad = lambda x, mode: op.discrete_gradient_2d(x,mode,N,N)
L_grad = 2.0

# define the inverse problem
noise = (eta/math.sqrt(m))*torch.randn(m)
T = torch.from_numpy(np.reshape(X.transpose(),N*N))
y = subsampled_ft(T,1) + noise

# compute the restarted NESTA solution
z0 = torch.zeros(N*N,dtype=y.dtype)

T_rec, iterates = nn.restarted_nesta_wqcbp(
        y, z0,
        subsampled_ft, discrete_grad, L_grad,
        inner_iters, outer_iters,
        eta, mu0, delta, r, False)

X_rec = np.transpose(np.reshape(np.real(T_rec.numpy()),(N,N)))
X_rec = np.clip((X_rec*255),0,255).astype('uint8')

Image.fromarray(X_rec).save('demo_restarted_nesta_tv_reconstruction-reconstructed-image.png')

