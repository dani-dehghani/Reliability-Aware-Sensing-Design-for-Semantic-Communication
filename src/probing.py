import numpy as np

from src.channel import bsc
from src.metrics import L_mis_direct, bayes_risk
from src.posterior import end_to_end, posterior
from src.sensitivity import Gs_matrix, Q_blocks




def token_dependent_channel(flip_pairs):
    
    if len(flip_pairs) != 4:
        raise ValueError("the two-bit benchmark has exactly 4 tokens")
    W = np.zeros((4, 4))
    for z, (p_r, p_v) in enumerate(flip_pairs):
        W[z] = np.kron(bsc(p_r), bsc(p_v))[z]
    return W


def floor_rows(W, floor):
    
    W = np.maximum(np.asarray(W, dtype=float), floor)
    return W / W.sum(axis=1, keepdims=True)




def probe_channel(W, n_per_token, rng):
    W = np.asarray(W, dtype=float)
    n_per_token = np.asarray(n_per_token, dtype=int)
    K, L = W.shape
    if n_per_token.shape != (K,):
        raise ValueError(f"n_per_token must have shape ({K},)")
    if np.any(n_per_token < 1):
        raise ValueError("every token needs at least one probe, otherwise its row is undefined")
    counts = np.array([rng.multinomial(n_per_token[z], W[z]) for z in range(K)], dtype=float)
    return counts / n_per_token[:, None]




def round_preserving_total(weights, N, n_min):
    
    weights = np.asarray(weights, dtype=float)
    K = len(weights)
    if N < K * n_min:
        raise ValueError(f"budget N={N} cannot give every one of {K} tokens n_min={n_min} probes")
    if weights.sum() <= 0:
        weights = np.ones(K)
    spare = N - K * n_min
    ideal = spare * weights / weights.sum()
    n = np.floor(ideal).astype(int)
    order = np.argsort(-(ideal - n), kind="stable")
    n[order[:spare - n.sum()]] += 1
    return n + n_min


def allocate_uniform(N, K, n_min=1):
    return round_preserving_total(np.ones(K), N, n_min)


def token_frequencies(S, pi):
    return np.asarray(pi) @ np.asarray(S)


def allocate_frequency(N, S, pi, n_min=1):
    return round_preserving_total(token_frequencies(S, pi), N, n_min)


def sensitivity_weights(S, W_rows):
    # a_z = [G_s]_zz (1 - ||w_z||^2)
    W_rows = np.asarray(W_rows, dtype=float)
    return np.diag(Gs_matrix(S)) * (1.0 - np.sum(W_rows ** 2, axis=1))


def allocate_sensitivity(N, S, W_rows, n_min=1):
    # n_z proportional to sqrt(a_z)
    a = sensitivity_weights(S, W_rows)
    return round_preserving_total(np.sqrt(np.maximum(a, 0.0)), N, n_min)


def allocate_two_stage(N, S, W, rng, pilot_fraction=0.2, n_min=1):
    
    K = S.shape[1]
    N_pilot = max(K * n_min, int(round(pilot_fraction * N)))
    N_main = N - N_pilot
    if N_main < 0:
        raise ValueError("pilot share exceeds the budget")

    n_pilot = allocate_uniform(N_pilot, K, n_min)
    W_pilot = probe_channel(W, n_pilot, rng)
    if N_main == 0:
        return W_pilot, n_pilot

    
    W_smooth = (W_pilot * n_pilot[:, None] + 1.0) / (n_pilot[:, None] + W.shape[1])
    a = sensitivity_weights(S, W_smooth)
    n_main = round_preserving_total(np.sqrt(np.maximum(a, 0.0)), N_main, 0)

    main_counts = np.zeros_like(W_pilot)
    for z in range(K):
        if n_main[z] > 0:
            main_counts[z] = rng.multinomial(n_main[z], W[z])
    n_total = n_pilot + n_main
    W_tilde = (W_pilot * n_pilot[:, None] + main_counts) / n_total[:, None]
    return W_tilde, n_total




def sensitivity_weights_general(S, pi, W):
    # a_z = sum_m [Q_m]_zz w_zm (1 - w_zm)
    W = np.asarray(W, dtype=float)
    blocks = Q_blocks(S, pi, W)
    return sum(np.diag(blocks[m]) * W[:, m] * (1.0 - W[:, m]) for m in range(W.shape[1]))


def allocate_sensitivity_general(N, S, pi, W_rows, n_min=1):
    a = sensitivity_weights_general(S, pi, W_rows)
    return round_preserving_total(np.sqrt(np.maximum(a, 0.0)), N, n_min)




def expected_L_est_approx(S, W, n_per_token):
    # balanced case
    a = sensitivity_weights(S, W)
    return float(np.sum(a / np.asarray(n_per_token, dtype=float)) / S.shape[0])


def expected_L_est_general(S, pi, W, n_per_token):
    return float(np.sum(sensitivity_weights_general(S, pi, W) / np.asarray(n_per_token, dtype=float)))


def required_probes(S, pi, W, target, allocation="sensitivity"):
    
    a = np.maximum(sensitivity_weights_general(S, pi, W), 0.0)
    if allocation == "uniform":
        return float(len(a) * a.sum() / target)
    if allocation == "sensitivity":
        return float(np.sum(np.sqrt(a)) ** 2 / target)
    raise ValueError("allocation must be 'uniform' or 'sensitivity'")


def analytic_probe_saving(a):
    
    a = np.asarray(a, dtype=float)
    return float(1.0 - np.sum(np.sqrt(a)) ** 2 / (len(a) * np.sum(a)))




def l_mis_between(S, pi, W_true, W_model):
    A_true = end_to_end(S, W_true)
    rho_true = posterior(pi, A_true)
    rho_model = posterior(pi, end_to_end(S, W_model))
    return L_mis_direct(pi, rho_true, rho_model, A_true)


def estimation_error_metrics(S, pi, W, W_hat, W_tilde, floor=1e-6):
    
    E_W = float(np.linalg.norm(W_tilde - W) / np.linalg.norm(W))
    W_model = floor_rows(W_tilde, floor)
    L_true = l_mis_between(S, pi, W, W_hat)
    L_est = l_mis_between(S, pi, W_model, W_hat)
    E_L = float(abs(L_est - L_true) / (L_true + 1e-12))
    L_adopt = l_mis_between(S, pi, W, W_model)
    return {"E_W": E_W, "E_L": E_L, "L_mis_true": L_true, "L_mis_est": L_est, "L_adopt": L_adopt}


def total_brier_with_estimated_channel(S, pi, W, W_tilde, floor=1e-6):
    
    A_true = end_to_end(S, W)
    rho_true = posterior(pi, A_true)
    return bayes_risk(pi, rho_true, A_true) + l_mis_between(S, pi, W, floor_rows(W_tilde, floor))




def sensing_budget_params(beta, sigma_ref_sq, eps_s, N_min, N_max):
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    sigma0_sq = sigma_ref_sq / (eps_s + beta)
    N_probe = int(N_min + np.floor((1.0 - beta) * N_max))
    return sigma0_sq, N_probe