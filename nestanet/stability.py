import math
import torch

def adv_perturbation(x, opA, opR, c_A, eta, lr, num_iters, use_gpu=False):
    """
    Computes an adversarial perturbation for a (differentiable) reconstruction
    method for recovering a vector x given measurements y = A @ x + e. Here A 
    is assumed to be a fat matrix satisfying A @ A^* = c*I for some constant 
    c > 0. This means A has full row rank and its Moore-Penrose pseudoinverse 
    is A^' = A^*/c.

    Inspired by [1] and [2], this function implements projected gradient 
    ascent to solve the non-convex optimization problem

        max ||R(y + e) - R(y)||_2   s.t.   ||e||_2 <= eta.

    Due to the non-convexity, gradient ascent is run several times with
    different randomly initialized perturbations e. The e that yielding
    largest objective value is selected and the vector z = A^' @ e is
    returned. z is a perturbation in the signal space (where x lives)
    for which e = A @ z.

    Args:
        x (torch.Tensor) : ground truth vector
        opA (function) : measurement operator A
        opR (function) : reconstruction map
        c_A (float) : constant c for which A @ A^* = c*I
        eta (float) : constraint for e
        lr (float) : learning rate for GA
        num_iters (int) : number of iterations of GA
        use_gpu (bool) : use CUDA for calculations

    Returns:
        pert (torch.Tensor) : perturbation z
        x_pert (torch.Tensor) : x + z, perturbed x
        x_pert_rec (torch.Tensor) : R(A(x + z))

    References:
        [1] Ch. 19.4. "Compressive Imaging: Structure, Sampling, Learning"
            Adcock, et al. ISBN:9781108421614.
        [2] Sec. 3.4. "Solving Inverse Problems With Deep Neural Networks --
            Robustness Included?" Genzel, et al. arXiv:2011.04268.
    """
    y = opA(x,1)
    x_rec = opR(y)
    
    N = x.shape[0]
    m = y.shape[0]

    # squared-norm function for complex tensors
    sq_norm = lambda z : torch.vdot(z,z)
    obj_fn = lambda e : -0.5*sq_norm(opR(y+e)-x_rec)
    
    best_obj_val = -float('Inf')
    obj_val = None
    best_e = None
    
    if use_gpu:
        noise = torch.randn(2*m, dtype=torch.float64, device=0)
    else:
        noise = torch.randn(2*m, dtype=torch.float64)

    noise = (eta/math.sqrt(m))*noise/torch.linalg.norm(noise,ord=2)
    
    e = noise[:m] + 1j*noise[m:]
    e.requires_grad_()

    optimizer = torch.optim.SGD([e], lr=lr)
    
    for i in range(num_iters):
        # descent step
        optimizer.zero_grad()
        obj = obj_fn(e)
        obj.backward()
        optimizer.step()

        with torch.no_grad():
            # projection
            e_len = torch.linalg.norm(e,ord=2)
            if e_len > eta:
                e.multiply_(eta/e_len)
            
            obj_val = -torch.real(obj_fn(e))

            obj_val = obj_val.cpu()
            print(
                'Step %d -- norm(e): %.5e -- obj val: %.5e' %
                (i+1, min(eta, float(e_len)), float(obj_val))
            )
            
            if obj_val > best_obj_val:
                best_obj_val = obj_val
                best_e = e.detach().clone()

    pert = opA(best_e,0)/c_A
    x_pert = x + pert
    x_pert_rec = opR(y+best_e)

    return pert, x_pert, x_pert_rec
