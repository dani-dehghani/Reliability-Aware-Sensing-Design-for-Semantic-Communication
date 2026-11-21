import numpy as np
from scipy.special import ndtr

from src.sensing import covariance, sample, state_means, variances


def sign_extractor(y):
    
    y = np.atleast_2d(y)
    return 2 * (y[:, 0] > 0).astype(int) + (y[:, 1] > 0).astype(int)


def analytic_S(alpha, a_r, a_v, sigma0_sq, eps):
    
    var_r, var_v = variances(alpha, sigma0_sq, eps)
    mu = state_means(a_r, a_v)
    q_r = ndtr(mu[:, 0] / np.sqrt(var_r))
    q_v = ndtr(mu[:, 1] / np.sqrt(var_v))
    return np.stack([(1 - q_r) * (1 - q_v),
                     (1 - q_r) * q_v,
                     q_r * (1 - q_v),
                     q_r * q_v], axis=1)


def empirical_S(alpha, a_r, a_v, sigma0_sq, eps, n, rng):
    
    mu = state_means(a_r, a_v)
    Sigma = covariance(alpha, sigma0_sq, eps)
    S = np.zeros((mu.shape[0], 4))
    for x in range(mu.shape[0]):
        z = sign_extractor(sample(x, n, mu, Sigma, rng))
        S[x] = np.bincount(z, minlength=4) / n
    return S
