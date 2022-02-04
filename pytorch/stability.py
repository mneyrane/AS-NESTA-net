import torch

def adversarial_perturbation(x, opA, opR, c_A, eta, num_trials, iters_per_trial, lr, use_gpu=False):
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
        num_trials (int) : number of GA trials
        iters_per_trial (int) : max number of GA iterations
        lr (float) : learning rate for Adam optimizer
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
    Ry = opR(y)

    N = x.shape[0]
    m = y.shape[0]

    # squared-norm function for complex tensors
    sq_norm = lambda z : torch.vdot(z,z)
    obj_fn = lambda p : -0.5*sq_norm(opR(y+p)-Ry)
    
    max_err = -float('Inf')
    best_e = None

    for t in range(num_trials):

        if use_gpu:
            noise = torch.randn(2*m, dtype=torch.float64, device=0)
        else:
            noise = torch.randn(2*m, dtype=torch.float64)

        noise = (eta/m)*noise/torch.linalg.norm(noise,ord=2)
        
        e = noise[:m] + 1j*noise[m:]
        e.requires_grad_()
        
        optimizer = torch.optim.SGD([e], lr=lr)
        #scheduler = ...
        
        for i in range(iters_per_trial):
            # descent step
            optimizer.zero_grad()
            obj = obj_fn(e)
            obj.backward()
            optimizer.step()

            with torch.no_grad():
                # projection
                e_len = torch.linalg.norm(e,ord=2)
                if e_len > eta:
                    e = eta*(e/e_len)

                obj_val = -torch.real(obj_fn(e))
                obj_val = obj_val.cpu()
                print(
                    'Step %d -- norm(e): %.5e -- obj val: %.5e' % 
                    (i+1, min(eta, float(e_len)), float(obj_val))
                )

            # for autograd
            e.requires_grad_() 
        
        with torch.no_grad():
            z = opA(e,0)/c_A
            
            #torch.testing.assert_close(opA(z,1),e)
            #torch.testing.assert_close(opA(x+z,1),y+e) 
            
            Ry_e = opR(opA(x+z,1))
            
            err = torch.linalg.norm(Ry_e-Ry,ord=2)
            
            if err > max_err:
                print("Found noise with higher error at trial", t)
                max_err = err
                best_e = e.detach()

    print("Maximum absolute error produced:", max_err)
    pert = opA(best_e,0)/c_A
    x_pert = x + pert
    x_pert_rec = opR(opA(x_pert,1))

    return pert, x_pert, x_pert_rec
