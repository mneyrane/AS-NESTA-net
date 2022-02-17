"""
-- Maksym Neyra-Nesterenko
-- mneyrane@sfu.ca
"""
import math
import numpy as np
import torch
import torch.fft as _fft

def huber_fn_gradient(x, mu):
    """ Huber function gradient """
    y = torch.zeros_like(x)
    
    with torch.no_grad():
        mask = torch.abs(x) <= mu
    
    y[mask] = x[mask]/mu
    y[~mask] = x[~mask] / torch.abs(x[~mask])
    return y

def discrete_gradient_2d(x,mode,N1,N2):
    """2-D anisotropic discrete gradient (TV) operator 

    Periodic boundary conditions are used. The parameters N1 and N2
    are the height and width of the image x.

    For the forward operator, x is expected to a vectorized matrix X
    in row-major order (C indexing). The output is a stacking of 
    vectorized vertical and horizontal gradients.

    From the definition of the adjoint, if you have two N1xN2 matrices 
    X1, X2, then the adjoint computes the vectorization of

        grad_vert_adjoint(X1) + grad_hori_adjoint(X2).

    Thus, the input x for the adjoint should be a stacking of vectorizations
    of X1, X2, in row-major order. That is:

        x = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=2)
    """
   
    assert len(x.shape) == 1, 'Input x is not a vector'

    if mode == True:
        
        x = x.reshape((N1,N2))
        
        y1, y2 = torch.zeros_like(x), torch.zeros_like(x)
        
        # vertical (axis) gradient
        y1 = torch.roll(x,-1,0) - x

        # horizontal (axis) gradient
        y2 = torch.roll(x,-1,1) - x

        y = torch.stack([y1,y2], dim=2)
        
        return y.reshape(-1)

    else:
       
        x = x.reshape((N1,N2,2))
        
        # vertical (axis) gradient transpose
        y = torch.roll(x[:,:,0],1,0) - x[:,:,0] 
        
        # horizontal (axis) gradient transpose
        y = y + torch.roll(x[:,:,1],1,1) - x[:,:,1]
        
        return y.reshape(-1)

def fourier_2d(x, mode, N, mask, use_gpu=False):
    """ 2-D discrete Fourier transform 

    Normalized so that the forward and inverse mappings are unitary.

    The parameter 'mask' is assumed to be a NxN boolean matrix, used as a 
    sampling mask for Fourier frequencies.
    """

    assert len(x.shape) == 1, 'Input x is not a vector'
    
    if mode == True:
        z = _fft.fftshift(_fft.fft2(x.reshape((N,N)), norm='ortho'))
        y = z[mask]
        return y.reshape(-1)
    
    else: # adjoint
        z = torch.zeros(N*N,dtype=x.dtype)
        if use_gpu:
            z = z.cuda()
        
        mask = mask.reshape(-1)

        z[mask] = x
        z = z.reshape((N,N))
        y = _fft.ifft2(_fft.fftshift(z), norm='ortho')
        return y.reshape(-1)

def discrete_haar_wavelet_2d(x, mode, N, levels=1):
    """ 2-D discrete Haar wavelet transform  

    Boundary conditions are resolved with periodic padding.

    The filters are normalized so that the forward and inverse mappings are
    unitary.
    """
    assert N % 2 == 0
    c = 0.5
 
    X = x.reshape((N,N))
    Y = torch.zeros_like(X)
    
    if mode == 1: # decomposition

        Np = N

        for level in range(levels):
            Nt = Np
            Np = Nt//2
            
            # approximation coeffs
            ll = c*(X[::2,::2]+X[::2,1::2]+X[1::2,::2]+X[1::2,1::2])
            # horizontal details
            lh = c*(X[::2,::2]+X[::2,1::2]-X[1::2,::2]-X[1::2,1::2])
            # vertical details
            hl = c*(X[::2,::2]-X[::2,1::2]+X[1::2,::2]-X[1::2,1::2])
            # diagonal details
            hh = c*(X[::2,::2]-X[::2,1::2]-X[1::2,::2]+X[1::2,1::2])
            
            Y[Np:Nt,Np:Nt] = hh
            Y[Np:Nt,:Np] = hl
            Y[:Np,Np:Nt] = lh
            X = ll

        Y[:Np,:Np] = X

        return Y.reshape(-1)

    else: # reconstruction

        Np = N//(2**levels)
        Nt = 2*Np

        for level in range(levels):
            ll = X[:Np,:Np]
            lh = X[:Np,Np:Nt]
            hl = X[Np:Nt,:Np]
            hh = X[Np:Nt,Np:Nt]

            Y[:Nt:2,:Nt:2] = c*(ll + lh + hl + hh)
            Y[:Nt:2,1:Nt:2] = c*(ll + lh - hl - hh)
            Y[1:Nt:2,:Nt:2] = c*(ll - lh + hl - hh)
            Y[1:Nt:2,1:Nt:2] = c*(ll - lh - hl + hh)
            
            X[:Nt,:Nt] = Y[:Nt,:Nt]

            Np = Nt
            Nt = 2*Np

        return Y.reshape(-1)

def tv_haar_2d(x, mode, N, lam, levels):
    """ 2-D TV-Haar transform.
    
    This combines `discrete_gradient_2d` and `discrete_haar_wavelet_2d`
    into one operation, where the discrete gradient is weighted by the
    square root of `lam`.
    """
    if mode == 1:
        dgrad_x = discrete_gradient_2d(x,1,N,N) * math.sqrt(lam)
        wave_x = discrete_haar_wavelet_2d(x,1,N,levels)
        y = torch.cat([dgrad_x, wave_x])
        return y
    else: # adjoint
        x1 = x[:2*N*N]
        x2 = x[2*N*N:]
        y1 = discrete_gradient_2d(x1,0,N,N) * math.sqrt(lam)
        y2 = discrete_haar_wavelet_2d(x2,0,N,levels)
        y = y1+y2
        return y
