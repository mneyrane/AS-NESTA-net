import torch
import torch.fft as FT

""" Huber function gradient """
def huber_fn_gradient(x, mu):
    y = torch.zeros_like(x)
    mask = torch.abs(x) <= mu
    y[mask] = x[mask]/mu
    y[~mask] = x[~mask] / torch.abs(x[~mask])
    return y

""" 2-D discrete gradient (TV) operator 

Endpoint boundary condition is periodic, so that grad xn = x1 - xn.
"""
def discrete_gradient_2d(x,mode,N1,N2):
    
    assert len(x.shape) == 1, 'Input x is not a vector'

    if mode == True:
        
        x = x.reshape((N1,N2))
        
        y1, y2 = torch.zeros_like(x), torch.zeros_like(x)
        
        # vertical (axis) gradient
        y1 = torch.roll(x,-1,-1) - x

        # horizontal (axis) gradient
        y2 = torch.roll(x,-1,-2) - x

        y = torch.stack([y1,y2], dim=-3)
        
        return y.reshape(-1)

    else:
       
        y = torch.zeros((N2,N1), dtype=x.dtype)

        x = x.reshape((2,N2,N1))
        
        # vertical (axis) gradient transpose
        y = y + torch.roll(x[0,:,:],1,-1) - x[0,:,:] 
        
        # horizontal (axis) gradient transpose
        y = y + torch.roll(x[1,:,:],1,-2) - x[1,:,:]
        
        return y.reshape(-1)

""" 2-D discrete Fourier transform 

Normalized so that the forward and inverse mappings are unitary.
"""
def fourier_2d(x, mode, N, mask):
    assert len(x.shape) == 1, 'Input x is not a vector'
    
    if mode == True:
        z = FT.fftshift(FT.fft2(x.reshape((N,N)), norm='ortho'),(-2,-1))
        y = z[mask]
        return y.reshape(-1)
    
    else: # adjoint
        z = torch.zeros(N*N,dtype=x.dtype)
        mask = mask.reshape(-1)
        z[mask] = x
        z = z.reshape((N,N))
        y = FT.ifft2(FT.fftshift(z,(-2,-1)), norm='ortho')
        return y.reshape(-1)

""" 2-D discrete Haar wavelet transform  

Boundary conditions are resolved with periodic padding:

    ... xn | x1 x2 ... xn | x1 x2 ...

The filters are normalized so that the forward and inverse mappings are
unitary.
"""
def discrete_haar_wavelet(x, mode, N, levels=1):
    assert N % 2 == 0
    c = 0.5
 
    X = x.reshape((N,N))
    Y = torch.zeros_like(X)
    
    if mode == True: # decomposition
        Np = N
        for level in range(levels):
            Np = Np//2

            ll = c*(X[::2,::2]+X[::2,1::2]+X[1::2,::2]+X[1::2,1::2]) # approx
            lh = c*(X[::2,::2]+X[::2,1::2]-X[1::2,::2]-X[1::2,1::2]) # horizontal 
            hl = c*(X[::2,::2]-X[::2,1::2]+X[1::2,::2]-X[1::2,1::2]) # vertical
            hh = c*(X[::2,::2]-X[::2,1::2]-X[1::2,::2]+X[1::2,1::2]) # diagonal

            Y[Np:,Np:] = hh
            Y[Np:,:Np] = hl
            Y[:Np,Np:] = lh
            X = ll

        Y[:Np,:Np] = ll

        return Y.reshape(-1)

    else: # reconstruction
        Np = N//(2**levels)
        Nt = 2*Np
        for level in range(levels):
            ll = X[:Np,:Np]
            lh = X[:Np,Np:Nt]
            hl = X[Np:Nt,:Np]
            hh = X[Np:Nt,Np:Nt]

            print('SHAPE:', ll.shape)

            Y[:Nt:2,:Nt:2] = c*(ll + lh + hl + hh)
            Y[:Nt:2,1:Nt:2] = c*(ll + lh - hl - hh)
            Y[1:Nt:2,:Nt:2] = c*(ll - lh + hl - hh)
            Y[1:Nt:2,1:Nt:2] = c*(ll - lh - hl + hh)
            
            Np = Nt
            Nt = 2*Np

        return Y.reshape(-1)

if __name__ == '__main__':
    X = torch.arange(16)
    mask = torch.rand((4,4)) < 10
    print('mask:', mask)
    FX = fourier_2d(X,4,mask)
    print(FX)
    FFX = fourier_2d(FX,4,mask,adjoint=True)
    print(FFX)
