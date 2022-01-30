"""
Helper functions for constructing sampling masks.

Note: everything here uses NumPy arrays.

-- Maksym Neyra-Nesterenko
-- mneyrane@sfu.ca
"""
import numpy as np
from scipy.optimize import bisect

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

    C = bisect(_constraint, 0.0, 1.0, args=(m, hist), xtol=1e-12, maxiter=1000)

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
    
def gaussian_multilevel_sampling(N):
    raise NotImplementedError('Gaussian multilevel sampling not available')

def _constraint(t, m, hist):
    terms = t*m*hist
    probs = np.where(terms < 1, terms, 1)
    return np.sum(probs)-m
