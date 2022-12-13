"""
Helper functions for constructing sampling masks.

Note: everything here uses NumPy arrays.
"""
import numpy as np
from scipy.optimize import bisect as _bisect

def bernoulli_sampling_probs_2d(hist,N,m):
    """
    Computes the Bernoulli model probabilities from a NxN histogram 
    representing the target (possibly unnormalized) distribution,
    where if one performs a single Bernoulli trial for each probability,
    the expected number of successes is m.

    Returns probs, which can be used to produce a sample mask using

    > np.random.binomial(1,probs)

    or using the `generate_sampling_mask_from_probs` function below.
    """
    assert hist.dtype == np.float64

    if np.all(m*hist <= 1.0): # avoid computation for trivial case
        return m*hist

    C = _bisect(_constraint, 0.0, 1.0, args=(m, hist), xtol=1e-12, maxiter=1000)

    terms = m*C*hist
    probs = np.where(terms < 1, terms, 1)

    return probs

def generate_sampling_mask_from_probs(probs):
    result = np.random.binomial(1,probs)
    return result.astype(bool)

def uniform_hist_2d(N):
    return np.ones((N,N), dtype=float)/(N*N)

def inverse_square_law_hist_2d(N,alpha):
    i = np.arange(-N//2+1,N//2+1)
    
    ii, jj = np.meshgrid(i, i, indexing='ij')
 
    hist = np.float_power(np.maximum(1, ii**2 + jj**2), -alpha)

    return hist.astype(float)

def optimal_hist_2d(N):
    """
    Computes theoretically-optimal NxN histogram for variable density 
    sampling.
    """
    i = np.arange(-N//2+1,N//2+1)
    
    ii, jj = np.meshgrid(i, i, indexing='ij')
    abs_ii, abs_jj = np.abs(ii), np.abs(jj)
    
    bar_vals = np.maximum(1, np.maximum(abs_ii, abs_jj))
    
    hist = (bar_vals.astype(float))**(-2)

    return hist

def stacked_scheme_2d(mask1, mask2, N1, N2):
    """
    Compute the stacked scheme mask from two sampling masks of size N1xN2.

    Returns the stacked mask, the submasks used to correctly define the
    measurements y1 and y2 (referred to as u1_mask, u2_mask, respectively)
    and the correct permutations of the measurements (perm1 and perm2).
    """
    idxs_1 = np.where(mask1)
    idxs_2 = np.where(mask2)

    idxs_1_flat = np.ravel_multi_index(idxs_1, (N1,N2))
    idxs_2_flat = np.ravel_multi_index(idxs_1, (N1,N2))

    omega_2_diff_1 = np.setdiff1d(idxs_2_flat, idxs_1_flat, assume_unique=True)
    omega_1 = np.concatenate((idxs_1_flat, omega_2_diff_1))
    u2_mask = np.isin(idxs_2_flat, omega_2_diff_1)

    omega_1_diff_2 = np.setdiff1d(idxs_1_flat, idxs_2_flat, assume_unique=True)
    omega_2 = np.concatenate((idxs_2_flat, omega_1_diff_2))
    u1_mask = np.isin(idxs_1_flat, omega_1_diff_2)

    o1, perm1 = np.sort(omega_1), np.argsort(omega_1)
    o2, perm2 = np.sort(omega_2), np.argsort(omega_2)

    assert np.all(o1 == o2) == True

    mask = np.zeros(N1*N2, dtype=bool)
    mask[o1] = True
    mask = mask.reshape((N1,N2))

    return mask, u1_mask, u2_mask, perm1, perm2
    
#def gaussian_multilevel_sampling(N):
#    raise NotImplementedError('Gaussian multilevel sampling not available')

def _constraint(t, m, hist):
    terms = t*m*hist
    probs = np.where(terms < 1, terms, 1)
    return np.sum(probs)-m
