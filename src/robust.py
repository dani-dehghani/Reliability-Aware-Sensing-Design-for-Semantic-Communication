import numpy as np

from src.posterior import end_to_end, posterior
from src.metrics import L_mis_direct


def project_row_to_simplex(v):
    # nearest point on {w >= 0, sum(w) = 1} to v, standard sort-based 
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


def _random_valid_perturbation(W_hat, epsilon, scale, rng):
    C = W_hat.shape[0]
    delta = rng.normal(size=(C, C))
    delta -= delta.mean(axis=1, keepdims=True)  # zero row sums (eq. 26)
    norm = np.linalg.norm(delta)
    if norm < 1e-12:
        return W_hat
    delta *= scale * epsilon / norm
    return _clip_to_radius(W_hat, project_channel(W_hat + delta), epsilon)


def random_search(S, W_hat, pi, epsilon, n_trials, rng, return_candidate=False):
    best, best_W = 0.0, W_hat
    for _ in range(n_trials):
        candidate = _random_valid_perturbation(W_hat, epsilon, 1.0, rng)
        loss = l_mis_of_candidate(S, W_hat, candidate, pi)
        if loss > best:
            best, best_W = loss, candidate
    return (best, best_W) if return_candidate else best


def _finite_difference_gradient(S, W_hat, W, pi, h=1e-6):
    C = W.shape[0]
    base = l_mis_of_candidate(S, W_hat, W, pi)
    grad = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            Wp = W.copy()
            Wp[i, j] += h
            grad[i, j] = (l_mis_of_candidate(S, W_hat, Wp, pi) - base) / h
    return grad - grad.mean(axis=1, keepdims=True)  # feasible directions have zero row sum


def gradient_search(S, W_hat, pi, epsilon, n_steps, n_restarts, step_frac, rng, warm_start=None,
                    return_candidate=False):
    starts = [] if warm_start is None else [warm_start]
    for _ in range(n_restarts):
        starts.append(_random_valid_perturbation(W_hat, epsilon, 0.5, rng))
    best, best_W = 0.0, W_hat
    for W in starts:
        for _ in range(n_steps):
            g = _finite_difference_gradient(S, W_hat, W, pi)
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-14:
                break
            # step of fixed length step_frac*epsilon along the normalized gradient.
            # A raw lr*grad step scales with L_mis (~1e-3) and never reaches the radius.
            W = _clip_to_radius(W_hat, project_channel(W + step_frac * epsilon * g / gnorm), epsilon)
        loss = l_mis_of_candidate(S, W_hat, W, pi)
        if loss > best:
            best, best_W = loss, W
    return (best, best_W) if return_candidate else best


def L_real_wc(S, W_hat, pi, epsilon, rng, n_trials=1500, n_steps=60, n_restarts=4, step_frac=0.1):
    """eq. 67: sup of L_mis over the Frobenius ball of valid channels around W_hat.
    Random search and projected gradient ascent (warm-started from the random
    winner); the larger is reported, per 3.2.3. Returns (max, random, gradient)."""
    r_random, W_random = random_search(S, W_hat, pi, epsilon, n_trials, rng, return_candidate=True)
    r_grad = gradient_search(S, W_hat, pi, epsilon, n_steps, n_restarts, step_frac, rng, warm_start=W_random)
    return float(max(r_random, r_grad)), float(r_random), float(r_grad)