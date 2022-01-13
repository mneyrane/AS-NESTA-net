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
