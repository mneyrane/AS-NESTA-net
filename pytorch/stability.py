import torch

def adversarial_perturbation(x, p, opA, opR, lam, num_iters, lr, m):
    """
    Implemented Algorithm 19.1 from Sec. 19.4 of Adcock & Hansen.

    Note that beta == 1 here.
    """
    y = opA(x,1)
   
    # or torch.rand
    r = torch.rand(x.shape, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([r], lr=lr, momentum=m)

    # squared-norm function for complex tensors
    sq_norm = lambda z : torch.dot(z,torch.conj(z))

    for i in range(num_iters):
        optimizer.zero_grad()
        R = opR(y+opA(r,1))
        obj = 0.5*(lam*sq_norm(r) - sq_norm(R-p))
        obj.backward()
        print(r.grad)
        optimizer.step()
        print('Step', i+1, 'objective value:', obj)

    return r.detach(), R.detach()

def projected_adversarial_perturbation(x, opF, opR, noise_std, eta, num_iters, use_gpu=False):
    """
    TO DO ...
    """
    y = opF(x,1)
    N = x.shape[0]
    m = y.shape[0]

    if use_gpu:
        e = noise_std * torch.randn(2*m, dtype=torch.float64, device=0)
    else:
        e = noise_std * torch.randn(2*m, dtype=torch.float64)

    e.requires_grad_()
    
    optimizer = torch.optim.Adam([e], lr=4e-2)
    
    # squared-norm function for complex tensors
    sq_norm = lambda z : torch.dot(z,torch.conj(z))

    for i in range(num_iters):
        optimizer.zero_grad()
        ee = e[:m] + 1j*e[m:]
        rec = opR(y+ee)
        obj = -0.5*sq_norm(rec-x)/N
        obj.backward()
        optimizer.step()

        with torch.no_grad():
            e_len = torch.linalg.norm(e,ord=2)
            if e_len > eta:
                e = eta*(e/e_len)
                break
            
            print('Step', i+1, 'norm of e:', min(eta, float(e_len)))
    
    ee = e[:m] + 1j*e[m:]
    rec_final = opR(y+ee)
    return e.detach(), rec_final.detach()
