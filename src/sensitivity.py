import numpy as np

_H2 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def uninformative_experiment(C):
    return np.full((C, C), 1.0 / C)


def H_matrix(S):
    C = S.shape[0]
    return S - uninformative_experiment(C)


def Gs_matrix(S):
    H = H_matrix(S)
    return H.T @ H


def Gdelta_matrix(dW):
    return dW @ dW.T


def L_mis_trace(S, dW):
    C = S.shape[0]
    H = H_matrix(S)
    return float(np.sum((H @ dW) ** 2) / C)


def _hadamard_basis(C):
    if C != 4:
        raise NotImplementedError("shared basis here is specific to the C=4 two-bit case")
    return np.kron(_H2, _H2)


def L_mis_modes(S, dW, atol=1e-8):
    C = S.shape[0]
    H, V = H_matrix(S), _hadamard_basis(C)
    D_s, D_d = V.T @ H @ V, V.T @ dW @ V
    off_s = D_s - np.diag(np.diag(D_s))
    off_d = D_d - np.diag(np.diag(D_d))
    if max(np.abs(off_s).max(), np.abs(off_d).max()) > atol:
        raise ValueError("H_u and Delta_W are not diagonal in the shared Hadamard basis here")
    lam_s, lam_d = np.diag(D_s), np.diag(D_d)
    return float(np.sum(lam_s ** 2 * lam_d ** 2) / C)


def worst_case_spectral(S, delta):
    C = S.shape[0]
    Gs = Gs_matrix(S)
    lam_max = np.linalg.eigvalsh(Gs).max()
    return float(delta / C * lam_max)


def GW_matrix(W):
    C = W.shape[0]
    K = W - uninformative_experiment(C)
    return K @ K.T


def R_task_trace(S, W):
    C = S.shape[0]
    Gs = Gs_matrix(S)
    Gw = GW_matrix(W)
    return float(np.trace(Gs @ Gw) / C)