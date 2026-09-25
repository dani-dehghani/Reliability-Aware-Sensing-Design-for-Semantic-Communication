
import numpy as np


def _check_prior(pi, C):
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (C,):
        raise ValueError(f"prior must have shape ({C},)")
    if np.any(pi <= 0) or not np.isclose(pi.sum(), 1.0):
        raise ValueError("prior must be strictly positive and sum to 1")
    return pi


def end_to_end(S, W):
    # A = S W, the law of total probability over the token
    S, W = np.asarray(S, dtype=float), np.asarray(W, dtype=float)
    if S.shape[1] != W.shape[0]:
        raise ValueError("token alphabet of S and W do not match")
    return S @ W


def evidence(pi, A):
    A = np.asarray(A, dtype=float)
    pi = _check_prior(pi, A.shape[0])
    return pi @ A


def posterior(pi, A):
    # column m is rho_m = D_pi a_m / p_m
    A = np.asarray(A, dtype=float)
    pi = _check_prior(pi, A.shape[0])
    p = pi @ A
    if np.any(p <= 0):
        raise ValueError("a received symbol has zero evidence; its posterior is undefined")
    return (pi[:, None] * A) / p[None, :]


def hard_decision(rho):
    return np.argmax(rho, axis=0)
