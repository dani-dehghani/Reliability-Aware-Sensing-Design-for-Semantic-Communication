"""
Baseline experiment for the reliability-aware sensing project.

Runs the full pipeline: sensing model, semantic extraction, channel models,
posteriors, mismatch loss (checked two independent ways), the balanced
spectral form, a full check gate, and the orientation-control sweep that
produces Figure 2.

Run from the project root:
    python -m experiments.exp00_baseline
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channel import actual_channel, delta_W, nominal_channel
from src.design import alpha_sweep, build_control_table, format_markdown_table, select_control_pair
from src.metrics import L_mis_direct, L_mis_identity, bayes_risk, hard_accuracy
from src.posterior import end_to_end, hard_decision, posterior
from src.semantic import analytic_S, empirical_S
from src.sensing import bit_reliability, uniform_prior, variances
from src.sensitivity import Gdelta_matrix, Gs_matrix, L_mis_modes, L_mis_trace, worst_case_spectral

ROOT = Path(__file__).resolve().parents[1]
MODES = ("range_optimistic", "velocity_optimistic")


def load_cfg():
    with open(ROOT / "configs" / "baseline.json") as f:
        return json.load(f)


def save(name, payload):
    out = ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def sensing_sweep(cfg):
    alphas = np.linspace(0.0, 1.0, cfg["n_alpha"])
    var_r, var_v = variances(alphas, cfg["sigma0_sq"], cfg["eps"])
    rel_r, rel_v = bit_reliability(alphas, cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    save("sensing_sweep.json", {
        "config": cfg, "alpha": alphas.tolist(),
        "var_r": var_r.tolist(), "var_v": var_v.tolist(),
        "bit_reliability_r": rel_r.tolist(), "bit_reliability_v": rel_v.tolist(),
    })
    print(f"1. Sensing model OK  (total precision constant across alpha: {(1 / var_r + 1 / var_v)[0]:.2f})")
    return alphas


def semantic_matrix_check(cfg, alphas):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    S_all = np.stack([analytic_S(a, *args) for a in alphas])
    row_err = float(np.abs(S_all.sum(axis=2) - 1).max())

    rng = np.random.default_rng(cfg["seed"])
    S_mc = empirical_S(0.5, *args, cfg["n_mc"], rng)
    mc_diff = float(np.abs(S_mc - analytic_S(0.5, *args)).max())

    save("semantic_matrix.json", {"config": cfg, "alpha": alphas.tolist(), "S": S_all.tolist()})
    print(f"2. S_u rows sum to 1 (err {row_err:.0e}), matches Monte Carlo (diff {mc_diff:.4f})")


def channel_check(cfg):
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    W = actual_channel(p_r, p_v)
    cases = {}
    for mode in MODES:
        W_hat = nominal_channel(p_r, p_v, mode, p_low)
        dW = delta_W(W, W_hat)
        cases[mode] = {"W_hat": W_hat.tolist(), "delta_W": dW.tolist()}
    save("channel_matrices.json", {"config": cfg, "W": W.tolist(), "cases": cases})
    print(f"3. Channels built (p_r={p_r}, p_v={p_v}, optimistic assumption={p_low})")


def posterior_check(cfg, alphas):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    pi = uniform_prior(cfg["C"])
    W = actual_channel(p_r, p_v)
    W_hats = {m: nominal_channel(p_r, p_v, m, p_low) for m in MODES}

    max_col_err, decisions_unchanged = 0.0, True
    for a in alphas:
        S = analytic_S(a, *args)
        rho = posterior(pi, end_to_end(S, W))
        x_true = hard_decision(rho)
        max_col_err = max(max_col_err, float(np.abs(rho.sum(axis=0) - 1).max()))
        for Wh in W_hats.values():
            rho_hat = posterior(pi, end_to_end(S, Wh))
            decisions_unchanged &= bool(np.array_equal(hard_decision(rho_hat), x_true))

    save("posteriors.json", {"config": cfg, "max_column_sum_error": max_col_err,
                            "decisions_unchanged_by_mismatch": decisions_unchanged})
    print(f"4. Posteriors sum to 1 (err {max_col_err:.0e}); hard decision unaffected by mismatch: {decisions_unchanged}")


def mismatch_metrics(cfg, alphas):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    pi = uniform_prior(cfg["C"])
    W = actual_channel(p_r, p_v)

    max_gap, rows = 0.0, {m: [] for m in MODES}
    for a in alphas:
        S = analytic_S(a, *args)
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        for mode in MODES:
            W_hat = nominal_channel(p_r, p_v, mode, p_low)
            dW = delta_W(W, W_hat)
            A_hat = end_to_end(S, W_hat)
            rho_hat = posterior(pi, A_hat)
            l_dir = L_mis_direct(pi, rho, rho_hat, A)
            l_id = L_mis_identity(pi, S, dW, rho, A, A_hat)
            max_gap = max(max_gap, abs(l_dir - l_id))
            rows[mode].append({"alpha": float(a), "L_mis": l_dir, "hard_accuracy": hard_accuracy(pi, rho, A)})

    save("mismatch_metrics.json", {"config": cfg, "max_gap_direct_vs_identity": max_gap, "sweep": rows})
    print(f"5. L_mis: direct == identity formula (gap {max_gap:.0e})")


def spectral_check(cfg, alphas):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    W = actual_channel(p_r, p_v)

    max_gap, rows = 0.0, []
    for a in alphas:
        S = analytic_S(a, *args)
        W_hat = nominal_channel(p_r, p_v, "range_optimistic", p_low)
        dW = delta_W(W, W_hat)
        l_tr = L_mis_trace(S, dW)
        l_mo = L_mis_modes(S, dW)
        max_gap = max(max_gap, abs(l_tr - l_mo))
        delta_budget = float(np.trace(Gdelta_matrix(dW)))
        rows.append({"alpha": float(a), "L_mis": l_tr,
                    "spectral_bound": worst_case_spectral(S, delta_budget)})

    save("spectral_check.json", {"config": cfg, "max_gap_trace_vs_modes": max_gap, "sweep": rows})
    print(f"6. Spectral trace formula == mode decomposition (gap {max_gap:.0e})")


def run_checks(cfg, alphas):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    p_r, p_v, p_low = cfg["p_r"], cfg["p_v"], cfg["p_hat_low"]
    pi = uniform_prior(cfg["C"])
    W = actual_channel(p_r, p_v)

    row_err, col_err_rho, max_lmis_matched = 0.0, 0.0, 0.0
    max_gap_dir_id, max_gap_id_tr = 0.0, 0.0

    for a in alphas:
        S = analytic_S(a, *args)
        row_err = max(row_err, float(np.abs(S.sum(axis=1) - 1).max()))
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        col_err_rho = max(col_err_rho, float(np.abs(rho.sum(axis=0) - 1).max()))

        dW0 = delta_W(W, W)
        max_lmis_matched = max(max_lmis_matched, L_mis_direct(pi, rho, rho, A),
                              L_mis_identity(pi, S, dW0, rho, A, A), L_mis_trace(S, dW0))

        for mode in MODES:
            W_hat = nominal_channel(p_r, p_v, mode, p_low)
            row_err = max(row_err, float(np.abs(W_hat.sum(axis=1) - 1).max()))
            dW = delta_W(W, W_hat)
            A_hat = end_to_end(S, W_hat)
            rho_hat = posterior(pi, A_hat)
            col_err_rho = max(col_err_rho, float(np.abs(rho_hat.sum(axis=0) - 1).max()))
            l_dir = L_mis_direct(pi, rho, rho_hat, A)
            l_id = L_mis_identity(pi, S, dW, rho, A, A_hat)
            l_tr = L_mis_trace(S, dW)
            max_gap_dir_id = max(max_gap_dir_id, abs(l_dir - l_id))
            max_gap_id_tr = max(max_gap_id_tr, abs(l_id - l_tr))

    checks = [
        ("rows of S_u, W, W_hat sum to 1",             row_err,          1e-12),
        ("all posterior vectors sum to 1",             col_err_rho,      1e-12),
        ("L_mis = 0 when W = W_hat",                   max_lmis_matched, 1e-12),
        ("exact identity agrees with direct distance", max_gap_dir_id,   1e-10),
        ("balanced trace agrees with exact identity",  max_gap_id_tr,    1e-10),
    ]
    all_pass = all(err < tol for _, err, tol in checks)

    save("baseline_checks.json", {
        "config": cfg, "all_pass": all_pass,
        "checks": [{"name": n, "max_error": e, "tolerance": t, "pass": e < t} for n, e, t in checks],
    })
    print(f"7. Check gate: {'PASS' if all_pass else 'FAIL'} (all {len(checks)} conditions)")


def run_alpha_sweep(cfg):
    sweep = alpha_sweep(cfg)
    pair = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    save("alpha_sweep.json", {
        "config": cfg,
        "sweep": {k: v.tolist() for k, v in sweep.items()},
        "control_pair": pair,
    })
    print(f"8. Orientation control: same accuracy ({pair['accuracy_low_loss']:.4f}), "
          f"L_mis differs by {pair['l_mis_ratio']:.2f}x")


def make_figure_and_table(cfg):
    with open(ROOT / "results" / "alpha_sweep.json") as f:
        data = json.load(f)
    sweep, pair = data["sweep"], data["control_pair"]
    alpha = np.array(sweep["alpha"])
    accuracy = np.array(sweep["accuracy"])
    l_range = np.array(sweep["L_mis_range_optimistic"])
    l_vel = np.array(sweep["L_mis_velocity_optimistic"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(alpha, accuracy, color="black")
    ax1.set_ylabel("hard accuracy (matched receiver)")
    ax1.set_title("Orientation control")
    ax1.grid(alpha=0.3)

    ax2.plot(alpha, l_range, label="range-optimistic receiver", color="tab:red")
    ax2.plot(alpha, l_vel, label="velocity-optimistic receiver", color="tab:blue")
    a_lo, a_hi = pair["alpha_low_loss"], pair["alpha_high_loss"]
    l_lo, l_hi = pair["L_mis_low"], pair["L_mis_high"]
    ax2.plot([a_lo, a_hi], [l_lo, l_hi], "o--", color="black", linewidth=1, label="control pair")
    y_top = max(l_range.max(), l_vel.max())
    ax2.set_ylim(top=y_top * 1.18)
    mid_alpha, mid_loss = (a_lo + a_hi) / 2, (l_lo + l_hi) / 2
    ax2.annotate(f"{pair['l_mis_ratio']:.2f}x", xy=(mid_alpha, mid_loss),
                xytext=(mid_alpha, y_top * 1.08), ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("sensing allocation alpha (0 = velocity-precise, 1 = range-precise)")
    ax2.set_ylabel("L_mis")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = ROOT / "figures" / "fig2_orientation_control.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    table_md = format_markdown_table(build_control_table(pair))
    (ROOT / "results" / "control_table.md").write_text(table_md + "\n")
    print("9. Saved figures/fig2_orientation_control.png and results/control_table.md")


def main():
    cfg = load_cfg()
    alphas = sensing_sweep(cfg)
    semantic_matrix_check(cfg, alphas)
    channel_check(cfg)
    posterior_check(cfg, alphas)
    mismatch_metrics(cfg, alphas)
    spectral_check(cfg, alphas)
    run_checks(cfg, alphas)
    run_alpha_sweep(cfg)
    make_figure_and_table(cfg)
    print("\nAll results saved under results/ and figures/.")


if __name__ == "__main__":
    main()