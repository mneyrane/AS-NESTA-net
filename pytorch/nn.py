import operators as op
import torch
from torch.nn.functional import relu
from torch.linalg import norm

def nesta_wqcbp(y, z0, opA, opW, L_W, num_iters, eta, mu, store_hist):

    # could make calculation of c optional or an input
    e1 = (torch.arange(len(y)) == 0).type(y.dtype)
    c = norm(opA(e1,0),2)**2

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

        x = lam/((lam+1)*c)*opA(dy,0) + q

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

        v = lam/((lam+1)*c)*opA(dy,0) + q

        # ---------------
        # compute z_{n+1}
        # ---------------
        tau = 2/(n+3)
        z = tau*v+(1-tau)*x
        
    return x, all_iterates

def restarted_nesta_wqcbp(y, z0, opA, opW, L_W, in_iters, re_iters, eta, mu0, delta, r, store_hist):

    mu = mu0
    z = z0
    all_iterates = []

    for k in range(re_iters):
        mu = r*(mu+delta)
        z, inner_iterates = nesta_wqcbp(y, z, opA, opW, L_W, in_iters, eta, mu, store_hist)

        if store_hist:
            all_iterates[k] = cell_in_iterates

    return z, all_iterates
