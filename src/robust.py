import numpy as np

from src.posterior import end_to_end, posterior
from src.metrics import L_mis_direct


def project_row_to_simplex(v):
    # nearest point on {w >= 0, sum(w) = 1} to v, standard sort-based projection
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def project_channel(W):
    return np.array([project_row_to_simplex(row) for row in W])


def l_mis_of_candidate(S, W_hat, W_candidate, pi):
    A_true = end_to_end(S, W_candidate)
    A_hat = end_to_end(S, W_hat)
    rho_true = posterior(pi, A_true)
    rho_hat = posterior(pi, A_hat)
    return L_mis_direct(pi, rho_true, rho_hat, A_true)


def _clip_to_radius(W_hat, W_candidate, epsilon):
    diff = W_candidate - W_hat
    norm = np.linalg.norm(diff)
    if norm > epsilon and norm > 0:
        diff = diff * (epsilon / norm)
        return project_channel(W_hat + diff)
    return W_candidate


def random_search(S, W_hat, pi, epsilon, n_trials, rng):
    C = W_hat.shape[0]
    best = 0.0
    for _ in range(n_trials):
        delta = rng.normal(size=(C, C))
        delta -= delta.mean(axis=1, keepdims=True)  # keep row sums at zero
        norm = np.linalg.norm(delta)
        if norm < 1e-12:
            continue
        delta *= epsilon / norm
        candidate = project_channel(W_hat + delta)
        candidate = _clip_to_radius(W_hat, candidate, epsilon)
        loss = l_mis_of_candidate(S, W_hat, candidate, pi)
        best = max(best, loss)
    return best


def gradient_search(S, W_hat, pi, epsilon, n_steps, n_restarts, lr, rng):
    C = W_hat.shape[0]
    best = 0.0
    h = 1e-5
    for _ in range(n_restarts):
        delta0 = rng.normal(size=(C, C)) * 0.01
        delta0 -= delta0.mean(axis=1, keepdims=True)
        W = project_channel(W_hat + delta0)
        W = _clip_to_radius(W_hat, W, epsilon)

        for _ in range(n_steps):
            base = l_mis_of_candidate(S, W_hat, W, pi)
            grad = np.zeros((C, C))
            for i in range(C):
                for j in range(C):
                    Wp = W.copy()
                    Wp[i, j] += h
                    grad[i, j] = (l_mis_of_candidate(S, W_hat, Wp, pi) - base) / h
            W = project_channel(W + lr * grad)
            W = _clip_to_radius(W_hat, W, epsilon)

        best = max(best, l_mis_of_candidate(S, W_hat, W, pi))
    return best


def L_real_wc(S, W_hat, pi, epsilon, rng, n_trials=1500, n_steps=40, n_restarts=4, lr=0.05):
    """eq. 67: sup of L_mis over the Frobenius-ball of valid channels around
    W_hat. Uses two independent searches (random + gradient ascent) and
    keeps the larger, per 3.2.3 -- neither one alone is trusted."""
    r_random = random_search(S, W_hat, pi, epsilon, n_trials, rng)
    r_grad = gradient_search(S, W_hat, pi, epsilon, n_steps, n_restarts, lr, rng)
    return float(max(r_random, r_grad)), float(r_random), float(r_grad)
