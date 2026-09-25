import numpy as np
from scipy.special import ndtr

STATES = np.array([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], dtype=float)


def uniform_prior(C=4):
    return np.full(C, 1.0 / C)


def state_means(a_r, a_v):
    # mu_x = [a_r*Xr, a_v*Xv], shape (4, 2)
    return STATES * np.array([a_r, a_v])


def _check_alpha(alpha):
    a = np.asarray(alpha, dtype=float)
    if np.any(a < 0.0) or np.any(a > 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    return a


def variances(alpha, sigma0_sq, eps):
    # sigma_r^2 and sigma_v^2 as a function of the sensing allocation alpha
    a = _check_alpha(alpha)
    return sigma0_sq / (eps + a), sigma0_sq / (eps + 1.0 - a)


def covariance(alpha, sigma0_sq, eps):
    var_r, var_v = variances(alpha, sigma0_sq, eps)
    return np.diag([float(var_r), float(var_v)])


def likelihood(y, x, mu, Sigma):
    # Gaussian density f(y | x), y can be a single point or a batch (n, d)
    y = np.atleast_2d(y)
    diff = y - mu[x]
    d = mu.shape[1]
    sol = np.linalg.solve(Sigma, diff.T).T
    quad = np.einsum("ij,ij->i", diff, sol)
    norm = (2.0 * np.pi) ** (d / 2.0) * np.sqrt(np.linalg.det(Sigma))
    return np.exp(-0.5 * quad) / norm


def sample(x, n, mu, Sigma, rng):
    L = np.linalg.cholesky(Sigma)
    return mu[x] + rng.standard_normal((n, mu.shape[1])) @ L.T


def bit_reliability(alpha, a_r, a_v, sigma0_sq, eps):
    # P(sign of each component is correct); used only to sanity-check parameters,
    # not the actual token distribution (see semantic.py for that)
    var_r, var_v = variances(alpha, sigma0_sq, eps)
    return ndtr(a_r / np.sqrt(var_r)), ndtr(a_v / np.sqrt(var_v))
