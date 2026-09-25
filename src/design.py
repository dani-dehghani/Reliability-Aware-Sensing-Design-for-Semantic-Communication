
import numpy as np

from src.channel import actual_channel, nominal_channel
from src.metrics import L_mis_direct, hard_accuracy
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S


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
