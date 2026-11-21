import numpy as np

from src.posterior import end_to_end, posterior
from src.metrics import L_mis_direct

CHANNEL_SETS = ("row", "doubly")


def _check_set(channel_set):
    if channel_set not in CHANNEL_SETS:
        raise ValueError(f"channel_set must be one of {CHANNEL_SETS}, got {channel_set!r}")


def project_row_to_simplex(v):
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


def _project_affine_doubly(X):
    n = X.shape[0]
    r = X.sum(axis=1) - 1.0
    c = X.sum(axis=0) - 1.0
    s = X.sum() - n
    return X - r[:, None] / n - c[None, :] / n + s / n ** 2


def project_doubly_stochastic(W, tol=1e-12, max_iter=5000):
    
    X = np.asarray(W, dtype=float).copy()
    if X.shape[0] != X.shape[1]:
        raise ValueError("doubly-stochastic projection needs a square matrix")
    corr = np.zeros_like(X)
    for _ in range(max_iter):
        Y = _project_affine_doubly(X)
        X_new = np.maximum(Y + corr, 0.0)
        corr = Y + corr - X_new
        if np.abs(X_new - X).max() < tol:
            X = X_new
            break
        X = X_new
    X = _project_affine_doubly(X)
    return np.maximum(X, 0.0) if X.min() > -1e-9 else X


def project_to_set(W, channel_set):
    _check_set(channel_set)
    return project_channel(W) if channel_set == "row" else project_doubly_stochastic(W)


def feasible_direction(D, channel_set):
    # remove components that would break the row (and column) sum constraints
    _check_set(channel_set)
    D = D - D.mean(axis=1, keepdims=True)
    if channel_set == "doubly":
        D = D - D.mean(axis=0, keepdims=True)
    return D


def is_valid_channel(W, channel_set, atol=1e-9):
    _check_set(channel_set)
    ok = W.min() >= -atol and np.allclose(W.sum(axis=1), 1.0, atol=atol)
    if channel_set == "doubly":
        ok = ok and np.allclose(W.sum(axis=0), 1.0, atol=atol)
    return bool(ok)


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
        return W_hat + diff * (epsilon / norm)
    return W_candidate


def _random_valid_perturbation(W_hat, epsilon, scale, rng, channel_set="row"):
    delta = feasible_direction(rng.normal(size=W_hat.shape), channel_set)
    norm = np.linalg.norm(delta)
    if norm < 1e-12:
        return W_hat
    delta *= scale * epsilon / norm
    cand = W_hat + delta
    if cand.min() < 0:
        cand = project_to_set(cand, channel_set)
    return _clip_to_radius(W_hat, cand, epsilon)


def random_search(S, W_hat, pi, epsilon, n_trials, rng, return_candidate=False, channel_set="row"):
    best, best_W = 0.0, W_hat
    for _ in range(n_trials):
        candidate = _random_valid_perturbation(W_hat, epsilon, 1.0, rng, channel_set)
        loss = l_mis_of_candidate(S, W_hat, candidate, pi)
        if loss > best:
            best, best_W = loss, candidate
    return (best, best_W) if return_candidate else best


def _finite_difference_gradient(S, W_hat, W, pi, h=1e-6, channel_set="row"):
    K, L = W.shape
    base = l_mis_of_candidate(S, W_hat, W, pi)
    grad = np.zeros((K, L))
    for i in range(K):
        for j in range(L):
            Wp = W.copy()
            Wp[i, j] += h
            grad[i, j] = (l_mis_of_candidate(S, W_hat, Wp, pi) - base) / h
    return feasible_direction(grad, channel_set)


def gradient_search(S, W_hat, pi, epsilon, n_steps, n_restarts, step_frac, rng, warm_start=None,
                    return_candidate=False, channel_set="row"):
    starts = [] if warm_start is None else [warm_start]
    for _ in range(n_restarts):
        starts.append(_random_valid_perturbation(W_hat, epsilon, 0.5, rng, channel_set))
    best, best_W = 0.0, W_hat
    for W in starts:
        for _ in range(n_steps):
            g = _finite_difference_gradient(S, W_hat, W, pi, channel_set=channel_set)
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-14:
                break
            W = _clip_to_radius(W_hat, project_to_set(W + step_frac * epsilon * g / gnorm, channel_set), epsilon)
        loss = l_mis_of_candidate(S, W_hat, W, pi)
        if loss > best:
            best, best_W = loss, W
    return (best, best_W) if return_candidate else best


def L_real_wc(S, W_hat, pi, epsilon, rng, n_trials=1500, n_steps=60, n_restarts=4, step_frac=0.1,
              channel_set="row", return_candidate=False):
    
    _check_set(channel_set)
    r_random, W_random = random_search(S, W_hat, pi, epsilon, n_trials, rng, return_candidate=True,
                                       channel_set=channel_set)
    r_grad, W_grad = gradient_search(S, W_hat, pi, epsilon, n_steps, n_restarts, step_frac, rng,
                                     warm_start=W_random, return_candidate=True, channel_set=channel_set)
    best, W_best = (r_grad, W_grad) if r_grad >= r_random else (r_random, W_random)
    out = (float(best), float(r_random), float(r_grad))
    return (*out, W_best) if return_candidate else out
