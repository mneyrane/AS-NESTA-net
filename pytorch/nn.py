import operators as op
import torch
from torch.nn.functional import relu
from torch.linalg import norm
from torch.utils.checkpoint import checkpoint

def nesta_wqcbp(y, z0, opA, opW, c_A, L_W, num_iters, eta, mu, store_hist):

    z = z0
    q_v = z0

    all_iterates = []

    for n in range(0,num_iters+1):
        # -----------
        # compute x_n
        # -----------
        grad = opW(z,1)
        grad = op.huber_fn_gradient(grad, mu)
        grad = mu/(L_W*L_W)*opW(grad,0)
        q = z-grad

        dy = y-opA(q,1)
        lam = relu(norm(dy,2)/eta - 1)

        x = lam/((lam+1)*c_A)*opA(dy,0) + q

        if store_hist:
            all_iterates.append(x)

        # -----------
        # compute v_n
        # -----------
        alpha = (n+1)/2
        q_v = q_v-alpha*grad
        q = q_v

        dy = y-opA(q,1)
        lam = relu(norm(dy,2)/eta - 1)

        v = lam/((lam+1)*c_A)*opA(dy,0) + q

        # ---------------
        # compute z_{n+1}
        # ---------------
        tau = 2/(n+3)
        z = tau*v+(1-tau)*x
        
    return x, all_iterates

def restarted_nesta_wqcbp(y, z0, opA, opW, c_A, L_W, in_iters, re_iters, eta, mu_seq, store_hist):
    
    z = z0
    all_iterates = []

    assert len(mu_seq) == re_iters

    for k in range(re_iters):
        mu = mu_seq[k]
        z, inner_its_list = checkpoint(
            nesta_wqcbp, 
            y, z, opA, opW, c_A, L_W, in_iters, eta, mu, store_hist)

        if store_hist:
            all_iterates.append(inner_its_list)

    return z, all_iterates

def nesta_bernoulli_wqcbp(y1, y2, z0, opB, opW, c_B, L_W, num_iters, eta, mu, store_hist):

    z = z0
    q_v = z0

    all_iterates = []

    # these quantities only need to be calculated once
    # ------------------------------------------------
    y_sum = y1+y2
    d_noise = 2*eta*eta-torch.real(torch.vdot(y1-y2,y1-y2))
    # ------------------------------------------------

    for n in range(0,num_iters+1):
        # -----------
        # compute x_n
        # -----------
        grad = opW(z,1)
        grad = op.huber_fn_gradient(grad, mu)
        grad = mu/(L_W*L_W)*opW(grad,0)
        q = z-grad

        dy = y_sum-2*opB(q,1)
        lam = relu(0.5*(1-torch.sqrt(d_noise/torch.real(torch.vdot(dy,dy)))))

        x = (lam/c_B)*opB(dy,0) + q

        if store_hist:
            all_iterates.append(x)

        # -----------
        # compute v_n
        # -----------
        alpha = (n+1)/2
        q_v = q_v-alpha*grad
        q = q_v

        dy = y_sum-2*opB(q,1)
        lam = relu(0.5*(1-torch.sqrt(d_noise/torch.real(torch.vdot(dy,dy)))))

        v = (lam/c_B)*opB(dy,0) + q

        # ---------------
        # compute z_{n+1}
        # ---------------
        tau = 2/(n+3)
        z = tau*v+(1-tau)*x
        
    return x, all_iterates

def restarted_nesta_bernoulli_wqcbp(y1, y2, z0, opB, opW, c_B, L_W, in_iters, re_iters, eta, mu_seq, store_hist):

    z = z0
    all_iterates = []

    assert len(mu_seq) == re_iters

    for k in range(re_iters):
        mu = mu_seq[k]
        z, inner_its_list = checkpoint(
            nesta_bernoulli_wqcbp,
            y1, y2, z, opB, opW, c_B, L_W, in_iters, eta, mu, store_hist)

        if store_hist:
            all_iterates.append(inner_its_list)

    return z, all_iterates
