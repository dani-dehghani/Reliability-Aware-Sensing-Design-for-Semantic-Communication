import numpy as np

from src.channel import actual_channel, nominal_channel
from src.metrics import L_mis_direct, R_task_direct, hard_accuracy
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensitivity import Gs_matrix


def alpha_sweep(cfg, receiver_modes=("range_optimistic", "velocity_optimistic")):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    pi = np.full(cfg["C"], 1.0 / cfg["C"])
    W = actual_channel(p_r, p_v)
    alphas = np.linspace(0.0, 1.0, cfg["n_alpha"])

    accuracy = np.zeros_like(alphas)
    l_mis = {mode: np.zeros_like(alphas) for mode in receiver_modes}
    for i, a in enumerate(alphas):
        S = analytic_S(a, *args)
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        accuracy[i] = hard_accuracy(pi, rho, A)
        for mode in receiver_modes:
            W_hat = nominal_channel(p_r, p_v, mode, p_low)
            A_hat = end_to_end(S, W_hat)
            rho_hat = posterior(pi, A_hat)
            l_mis[mode][i] = L_mis_direct(pi, rho, rho_hat, A)

    out = {"alpha": alphas, "accuracy": accuracy}
    out.update({f"L_mis_{mode}": arr for mode, arr in l_mis.items()})
    return out


def select_control_pair(alphas, accuracy, l_mis, acc_tol=0.01):
    n = len(alphas)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            if abs(accuracy[i] - accuracy[j]) > acc_tol:
                continue
            gap = abs(l_mis[i] - l_mis[j])
            if best is None or gap > best["l_mis_gap"]:
                lo, hi = (i, j) if l_mis[i] < l_mis[j] else (j, i)
                best = {
                    "alpha_low_loss": float(alphas[lo]), "alpha_high_loss": float(alphas[hi]),
                    "accuracy_low_loss": float(accuracy[lo]), "accuracy_high_loss": float(accuracy[hi]),
                    "L_mis_low": float(l_mis[lo]), "L_mis_high": float(l_mis[hi]),
                    "l_mis_gap": float(gap),
                    "l_mis_ratio": float(l_mis[hi] / l_mis[lo]) if l_mis[lo] > 0 else float("inf"),
                }
    if best is None:
        raise ValueError(f"no pair found with accuracy difference <= {acc_tol}")
    return best


def build_control_table(pair, receiver_label="range-optimistic receiver"):
    return [
        ("Sensing design", f"alpha = {pair['alpha_low_loss']:.2f}", f"alpha = {pair['alpha_high_loss']:.2f}"),
        ("Hard accuracy (matched)", f"{pair['accuracy_low_loss']:.4f}", f"{pair['accuracy_high_loss']:.4f}"),
        (f"L_mis ({receiver_label})", f"{pair['L_mis_low']:.6f}", f"{pair['L_mis_high']:.6f}"),
        ("Relative vulnerability", "1.00x (reference)", f"{pair['l_mis_ratio']:.2f}x"),
    ]


def format_markdown_table(rows, headers=("Metric", "Design 1", "Design 2")):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def task_vs_mismatch_sweep(cfg):
    # same sweep as alpha_sweep, plus R_task and the G_s spectrum
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    pi = np.full(cfg["C"], 1.0 / cfg["C"])
    W = actual_channel(p_r, p_v)
    alphas = np.linspace(0.0, 1.0, cfg["n_alpha"])

    accuracy = np.zeros_like(alphas)
    R_task = np.zeros_like(alphas)
    tr_Gs = np.zeros_like(alphas)
    lam_max_Gs = np.zeros_like(alphas)
    l_mis = {"range_optimistic": np.zeros_like(alphas), "velocity_optimistic": np.zeros_like(alphas)}

    for i, a in enumerate(alphas):
        S = analytic_S(a, *args)
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        accuracy[i] = hard_accuracy(pi, rho, A)
        R_task[i] = R_task_direct(pi, rho, A)
        Gs = Gs_matrix(S)
        tr_Gs[i] = np.trace(Gs)
        lam_max_Gs[i] = np.linalg.eigvalsh(Gs).max()
        for mode in l_mis:
            W_hat = nominal_channel(p_r, p_v, mode, p_low)
            A_hat = end_to_end(S, W_hat)
            rho_hat = posterior(pi, A_hat)
            l_mis[mode][i] = L_mis_direct(pi, rho, rho_hat, A)

    out = {"alpha": alphas, "accuracy": accuracy, "R_task": R_task, "tr_Gs": tr_Gs, "lam_max_Gs": lam_max_Gs}
    out.update({f"L_mis_{mode}": arr for mode, arr in l_mis.items()})
    return out


def argmax_with_ties(values, rtol=1e-9):
    
    v = np.asarray(values, dtype=float)
    best = v.max()
    tol = rtol * max(abs(best), 1e-300)
    ties = np.flatnonzero(v >= best - tol)
    return int(ties[0]), [int(i) for i in ties]


def pareto_front_mask(R_task, L_mis):
   
    R, L = np.asarray(R_task), np.asarray(L_mis)
    mask = np.ones(len(R), dtype=bool)
    for i in range(len(R)):
        dominated = (R >= R[i]) & (L <= L[i]) & ((R > R[i]) | (L < L[i]))
        mask[i] = not dominated.any()
    return mask


def mixed_l_mis(l_mis_range, l_mis_velocity, q):
    
    return q * l_mis_range + (1.0 - q) * l_mis_velocity


def pareto_selection(sweep, lambda_grid, q=0.5):
    # which alpha each design rule picks on the same sweep
    alpha = sweep["alpha"]
    R_task = sweep["R_task"]
    l_mix = mixed_l_mis(sweep["L_mis_range_optimistic"], sweep["L_mis_velocity_optimistic"], q)
    v_prob = R_task - l_mix
    j_lambda = {f"{lam:g}": R_task - lam * l_mix for lam in lambda_grid}

    picks = {
        "max_accuracy": argmax_with_ties(sweep["accuracy"]),
        "max_R_task": argmax_with_ties(R_task),
        "min_L_mis": argmax_with_ties(-l_mix),
        "max_V_prob": argmax_with_ties(v_prob),
    }
    for lam in lambda_grid:
        picks[f"max_J_lambda_{lam:g}"] = argmax_with_ties(j_lambda[f"{lam:g}"])

    return {
        "q": q,
        "L_mis_mixed": l_mix,
        "V_prob": v_prob,
        "J_lambda": j_lambda,
        "pareto_front": pareto_front_mask(R_task, l_mix),
        "picks": {
            name: {
                "alpha": float(alpha[i]),
                "alpha_ties": [float(alpha[k]) for k in ties],
                "accuracy": float(sweep["accuracy"][i]),
                "R_task": float(R_task[i]),
                "L_mis": float(l_mix[i]),
            }
            for name, (i, ties) in picks.items()
        },
    }


def alpha_star_map(sweep, lambda_grid, q_grid):
    
    R_task, alpha = sweep["R_task"], sweep["alpha"]
    rows = []
    for q in q_grid:
        l_mix = mixed_l_mis(sweep["L_mis_range_optimistic"], sweep["L_mis_velocity_optimistic"], q)
        for lam in lambda_grid:
            i_star, ties = argmax_with_ties(R_task - lam * l_mix)
            rows.append({"q": float(q), "lambda": float(lam), "alpha_star": float(alpha[i_star]),
                         "alpha_star_ties": [float(alpha[k]) for k in ties], "tie": len(ties) > 1,
                         "accuracy": float(sweep["accuracy"][i_star]),
                         "R_task": float(R_task[i_star]), "L_mis": float(l_mix[i_star])})
    return rows