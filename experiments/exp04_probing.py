import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channel import actual_channel
from src.metrics import bayes_risk
from src.posterior import end_to_end, posterior
from src.probing import (
    allocate_frequency, allocate_sensitivity, allocate_sensitivity_general,
    allocate_two_stage, allocate_uniform, analytic_probe_saving,
    estimation_error_metrics, expected_L_est_approx, expected_L_est_general,
    floor_rows, l_mis_between, probe_channel, sensing_budget_params,
    sensitivity_weights, sensitivity_weights_general, token_dependent_channel,
    total_brier_with_estimated_channel,
)
from src.semantic import analytic_S

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "baseline.json"))
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
BASE_SEED = CFG["seed"]

N_MIN_PER_TOKEN = 2
PILOT_FRACTION = 0.2
PROB_FLOOR = 1e-6
N_GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
N_REP = 300
N_REP_SAVING = 2000
ALPHA_PROBE = 0.5

# token-dependent deployment channel: flip prob p_z = 0.10 + h * d_z; h = 0 is the baseline channel
HETERO_D = np.array([-0.08, -0.04, +0.04, +0.08])
H_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.2]
H_MAIN = 1.0

# receiver calibrated on the baseline channel
W_NOMINAL = actual_channel(CFG["p_r"], CFG["p_v"])

REGIMES = {
    "A_uniform_prior": np.full(4, 0.25),
    "B_nonuniform_prior": np.array([0.40, 0.30, 0.20, 0.10]),
}

SIGMA_REF_SQ, EPS_S = 1.0, 0.1
N_PROBE_MIN, N_PROBE_MAX = 8, 2000
N_MAX_SWEEP = [20, 50, 200, 2000, 20000]
# finer grid near beta = 1, where the optimum lies
BETA_GRID = np.round(np.concatenate([np.linspace(0.0, 0.8, 9), np.arange(0.82, 1.0001, 0.02)]), 2)
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
N_SELECT_SEEDS, N_EVAL_SEEDS = 40, 40


def deployment_channel(h):
    p = 0.10 + h * HETERO_D
    return token_dependent_channel([(pz, pz) for pz in p])


def save(name, payload):
    out = ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)




def part1_convergence():
    W = deployment_channel(H_MAIN)
    rng = np.random.default_rng([BASE_SEED, 1])
    rows = []
    for N in N_GRID:
        n = allocate_uniform(N, 4, N_MIN_PER_TOKEN)
        errs = [np.linalg.norm(probe_channel(W, n, rng) - W) / np.linalg.norm(W) for _ in range(N_REP)]
        rows.append({"N_probe": N, "E_W_mean": float(np.mean(errs)), "E_W_std": float(np.std(errs))})
    slope = float(np.polyfit(np.log(N_GRID), np.log([r["E_W_mean"] for r in rows]), 1)[0])
    print(f"1. Convergence: E_W falls from {rows[0]['E_W_mean']:.3f} (N={N_GRID[0]}) to "
          f"{rows[-1]['E_W_mean']:.4f} (N={N_GRID[-1]}); log-log slope {slope:.3f} (expected -0.5)")
    save("ext3_convergence.json", {"seed": [BASE_SEED, 1], "rows": rows, "loglog_slope": slope})




def allocations_for(N, S, pi, W, rng):
    out = {}
    for name, n in (
        ("uniform", allocate_uniform(N, 4, N_MIN_PER_TOKEN)),
        ("frequency", allocate_frequency(N, S, pi, N_MIN_PER_TOKEN)),
        ("sens_balanced_oracle", allocate_sensitivity(N, S, W, N_MIN_PER_TOKEN)),
        ("sens_general_oracle", allocate_sensitivity_general(N, S, pi, W, N_MIN_PER_TOKEN)),
    ):
        out[name] = (n, probe_channel(W, n, rng))
    W_ts, n_ts = allocate_two_stage(N, S, W, rng, PILOT_FRACTION, N_MIN_PER_TOKEN)
    out["sens_two_stage"] = (n_ts, W_ts)
    return out


def part2_budget_scaling():
    S = analytic_S(ALPHA_PROBE, *ARGS)
    W = deployment_channel(H_MAIN)
    metrics = ("E_W", "E_L", "L_adopt")
    results = {}
    for r_idx, (regime, pi) in enumerate(REGIMES.items()):
        rng = np.random.default_rng([BASE_SEED, 2, r_idx])
        rows = []
        for N in N_GRID:
            acc = {}
            for _ in range(N_REP):
                for name, (n, W_tilde) in allocations_for(N, S, pi, W, rng).items():
                    m = estimation_error_metrics(S, pi, W, W_NOMINAL, W_tilde, PROB_FLOOR)
                    d = acc.setdefault(name, {"n": n, **{k: [] for k in metrics}})
                    for k in metrics:
                        d[k].append(m[k])
            row = {"N_probe": N}
            for name, d in acc.items():
                row[name] = {
                    "n_per_token": [int(v) for v in d["n"]],
                    **{f"{k}_mean": float(np.mean(d[k])) for k in metrics},
                    **{f"{k}_std": float(np.std(d[k])) for k in metrics},
                    "pred_eq77": expected_L_est_approx(S, W, d["n"]),
                    "pred_general": expected_L_est_general(S, pi, W, d["n"]),
                }
            rows.append(row)
        results[regime] = {
            "prior": pi.tolist(),
            "a_balanced_eq78": sensitivity_weights(S, W).tolist(),
            "a_general": sensitivity_weights_general(S, pi, W).tolist(),
            "L_mis_true_nominal_receiver": l_mis_between(S, pi, W, W_NOMINAL),
            "rows": rows,
        }
        mid = rows[N_GRID.index(512)]
        print(f"2. [{regime}] L_adopt at N=512: uniform {mid['uniform']['L_adopt_mean']:.3e}, "
              f"sens. balanced {mid['sens_balanced_oracle']['L_adopt_mean']:.3e}, "
              f"sens. general {mid['sens_general_oracle']['L_adopt_mean']:.3e}, "
              f"two-stage {mid['sens_two_stage']['L_adopt_mean']:.3e}")
        r_bal = mid["uniform"]["L_adopt_mean"] / mid["uniform"]["pred_eq77"]
        r_gen = mid["uniform"]["L_adopt_mean"] / mid["uniform"]["pred_general"]
        print(f"   MC / predicted loss (uniform, N=512): balanced {r_bal:.3f}, general prior {r_gen:.3f}")
    save("ext3_budget_scaling.json", {
        "seed_base": BASE_SEED, "N_grid": N_GRID, "n_rep": N_REP, "alpha": ALPHA_PROBE,
        "h": H_MAIN, "W_deploy": deployment_channel(H_MAIN).tolist(), "W_nominal": W_NOMINAL.tolist(),
        "n_min_per_token": N_MIN_PER_TOKEN, "pilot_fraction": PILOT_FRACTION, "prob_floor": PROB_FLOOR,
        "regimes": results,
    })
    return results




def part3_probe_saving():
    S = analytic_S(ALPHA_PROBE, *ARGS)
    pi = REGIMES["A_uniform_prior"]
    rng = np.random.default_rng([BASE_SEED, 3])
    N_check = 512
    rows = []
    for h in H_GRID:
        W = deployment_channel(h)
        a = sensitivity_weights(S, W)
        n_u = allocate_uniform(N_check, 4, N_MIN_PER_TOKEN)
        n_s = allocate_sensitivity(N_check, S, W, N_MIN_PER_TOKEN)
        l_u = [l_mis_between(S, pi, W, floor_rows(probe_channel(W, n_u, rng), PROB_FLOOR)) for _ in range(N_REP_SAVING)]
        l_s = [l_mis_between(S, pi, W, floor_rows(probe_channel(W, n_s, rng), PROB_FLOOR)) for _ in range(N_REP_SAVING)]
        # loss ~ 1/N, so the loss ratio equals the probe-count ratio
        rows.append({
            "h": h, "flip_probs": (0.10 + h * HETERO_D).tolist(), "a_z": a.tolist(),
            "saving_analytic": analytic_probe_saving(a),
            "saving_mc": float(1.0 - np.mean(l_s) / np.mean(l_u)),
            "L_uniform_mc": float(np.mean(l_u)), "L_sens_mc": float(np.mean(l_s)),
            "n_uniform": n_u.tolist(), "n_sens": n_s.tolist(),
        })
    print("3. Probe saving of sensitivity probing over uniform probing:")
    for r in rows:
        print(f"   h={r['h']:.2f}  analytic {100 * r['saving_analytic']:5.1f}%   MC {100 * r['saving_mc']:5.1f}%")
    save("ext3_probe_saving.json", {"seed": [BASE_SEED, 3], "N_check": N_check, "n_rep": N_REP_SAVING, "rows": rows})
    return rows




def final_loss(alpha, beta, pi, W, rng, N_max):
    sigma0_sq, N_probe = sensing_budget_params(beta, SIGMA_REF_SQ, EPS_S, N_PROBE_MIN, N_max)
    S = analytic_S(alpha, CFG["a_r"], CFG["a_v"], sigma0_sq, CFG["eps"])
    W_tilde, _ = allocate_two_stage(N_probe, S, W, rng, PILOT_FRACTION, N_MIN_PER_TOKEN)
    return total_brier_with_estimated_channel(S, pi, W, W_tilde, PROB_FLOOR)


def joint_grid(N_max, verbose):
    pi = REGIMES["A_uniform_prior"]
    W = deployment_channel(H_MAIN)
    grid = np.zeros((len(BETA_GRID), len(ALPHA_GRID)))
    rows = []
    for i, beta in enumerate(BETA_GRID):
        sigma0_sq, N_probe = sensing_budget_params(beta, SIGMA_REF_SQ, EPS_S, N_PROBE_MIN, N_max)
        for j, alpha in enumerate(ALPHA_GRID):
            rng = np.random.default_rng([BASE_SEED, 4, i, j, 0])
            grid[i, j] = np.mean([final_loss(alpha, beta, pi, W, rng, N_max) for _ in range(N_SELECT_SEEDS)])
        j_star = int(np.argmin(grid[i]))
        alpha_star = float(ALPHA_GRID[j_star])

        
        rng_eval = np.random.default_rng([BASE_SEED, 4, i, j_star, 1])
        eval_losses = [final_loss(alpha_star, beta, pi, W, rng_eval, N_max) for _ in range(N_EVAL_SEEDS)]

        bayes = []
        for alpha in ALPHA_GRID:
            S = analytic_S(alpha, CFG["a_r"], CFG["a_v"], sigma0_sq, CFG["eps"])
            A = end_to_end(S, W)
            bayes.append(bayes_risk(pi, posterior(pi, A), A))
        rows.append({
            "beta": float(beta), "sigma0_sq": sigma0_sq, "N_probe": N_probe,
            "alpha_star": alpha_star,
            "loss_selected_seeds": float(grid[i, j_star]),
            "loss_eval_mean": float(np.mean(eval_losses)), "loss_eval_std": float(np.std(eval_losses)),
            "alpha_star_known_channel": float(ALPHA_GRID[int(np.argmin(bayes))]),
            "bayes_risk_at_alpha_star": float(bayes[j_star]),
            "probing_penalty_at_alpha_star": float(np.mean(eval_losses) - bayes[j_star]),
        })
    best = min(rows, key=lambda r: r["loss_eval_mean"])
    if verbose:
        print(f"4. Joint budget (N_max={N_max}): beta*={best['beta']:.2f} (sigma0^2={best['sigma0_sq']:.2f}, "
              f"N_probe={best['N_probe']}), alpha*={best['alpha_star']:.2f}, "
              f"Brier {best['loss_eval_mean']:.4f} +/- {best['loss_eval_std']:.4f}")
        print("   beta  N_probe  alpha*  final loss   Bayes risk   probing penalty")
        for r in rows:
            print(f"   {r['beta']:.2f}  {r['N_probe']:5d}    {r['alpha_star']:.2f}    {r['loss_eval_mean']:.4f}"
                  f"      {r['bayes_risk_at_alpha_star']:.4f}       {r['probing_penalty_at_alpha_star']:.5f}")
    return grid, rows, best


def part4_joint_budget():
    grid, rows, _ = joint_grid(N_PROBE_MAX, verbose=True)
    sweep = []
    for N_max in N_MAX_SWEEP:
        _, _, best = joint_grid(N_max, verbose=False)
        sweep.append({"N_max": N_max, "beta_star": best["beta"], "alpha_star": best["alpha_star"],
                      "N_probe_at_optimum": best["N_probe"], "loss_eval_mean": best["loss_eval_mean"]})
    print("   N_max -> beta*: " + ", ".join(f"{s['N_max']}->{s['beta_star']:.2f}" for s in sweep))
    save("ext3_joint_budget.json", {
        "seed_base": BASE_SEED, "beta_grid": BETA_GRID.tolist(), "alpha_grid": ALPHA_GRID.tolist(),
        "sigma_ref_sq": SIGMA_REF_SQ, "eps_s": EPS_S, "N_probe_min": N_PROBE_MIN, "N_probe_max": N_PROBE_MAX,
        "n_select_seeds": N_SELECT_SEEDS, "n_eval_seeds": N_EVAL_SEEDS,
        "loss_grid_selection_seeds": grid.tolist(), "rows": rows, "N_max_sweep": sweep,
    })
    return grid, rows, sweep




STYLE = {
    "uniform": ("k", "-", "uniform probing"),
    "frequency": ("tab:gray", "--", "frequency P(Z=z)"),
    "sens_balanced_oracle": ("tab:red", "-", "sensitivity weights (balanced), oracle"),
    "sens_general_oracle": ("tab:blue", "-", "sensitivity weights (general prior), oracle"),
    "sens_two_stage": ("tab:red", ":", "sensitivity, two-stage (20% pilot)"),
}
TITLES = {
    "E_W": "channel estimation error E_W",
    "E_L": "vulnerability estimation error E_L",
    "L_adopt": "excess loss with estimated channel",
}


def fig7(results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for r_idx, (regime, res) in enumerate(results.items()):
        N = [row["N_probe"] for row in res["rows"]]
        for c_idx, metric in enumerate(TITLES):
            ax = axes[r_idx, c_idx]
            for name, (color, ls, label) in STYLE.items():
                mean = np.array([row[name][f"{metric}_mean"] for row in res["rows"]])
                sem = np.array([row[name][f"{metric}_std"] for row in res["rows"]]) / np.sqrt(N_REP)
                ax.plot(N, mean, color=color, ls=ls, marker="o", ms=3, label=label)
                ax.fill_between(N, mean - sem, mean + sem, color=color, alpha=0.15)
            if metric == "L_adopt":
                ax.plot(N, [row["uniform"]["pred_eq77"] for row in res["rows"]], color="tab:red",
                        ls="-.", lw=1, label="predicted loss, balanced (uniform)")
                ax.plot(N, [row["uniform"]["pred_general"] for row in res["rows"]], color="tab:blue",
                        ls="-.", lw=1, label="predicted loss, general prior (uniform)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_xlabel("number of probes N")
            ax.set_title(TITLES[metric], fontsize=10)
        axes[r_idx, 0].set_ylabel(regime.replace("_", " "), fontsize=10)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Probe budget scaling (alpha = 0.5, h = 1)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / "figures" / "fig7_probe_budget_scaling.png", dpi=150)
    plt.close(fig)


def fig7b(saving_rows, results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    h = [r["h"] for r in saving_rows]
    ax.plot(h, [100 * r["saving_analytic"] for r in saving_rows], "k-o", ms=4, label="analytic")
    ax.plot(h, [100 * r["saving_mc"] for r in saving_rows], "r--s", ms=4, label="Monte Carlo (N = 512)")
    ax.set_xlabel("channel heterogeneity h")
    ax.set_ylabel("probes saved vs uniform (%)")
    ax.set_title("Probe saving of sensitivity probing", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    resB = results["B_nonuniform_prior"]
    x = np.arange(4)
    a_bal = np.array(resB["a_balanced_eq78"]) / 4.0
    ag = np.array(resB["a_general"])
    freqs = np.array(resB["prior"]) @ analytic_S(ALPHA_PROBE, *ARGS)
    ax.bar(x - 0.18, a_bal / a_bal.sum(), 0.36, color="tab:red", label="sensitivity weights (balanced)")
    ax.bar(x + 0.18, ag / ag.sum(), 0.36, color="tab:blue", label="sensitivity weights (general prior)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"token {z}\nP(Z)={p:.2f}" for z, p in enumerate(freqs)], fontsize=8)
    ax.set_ylabel("normalized weight a_z")
    ax.set_title("Sensitivity weights, non-uniform prior", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig7b_probe_saving.png", dpi=150)
    plt.close(fig)


def fig8(grid, rows, sweep):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax = axes[0]
    im = ax.pcolormesh(ALPHA_GRID, BETA_GRID, grid, shading="nearest", cmap="viridis_r")  # beta grid is non-uniform
    ax.plot([r["alpha_star"] for r in rows], [r["beta"] for r in rows], "w-o", ms=4, label="alpha* (estimated W)")
    ax.plot([r["alpha_star_known_channel"] for r in rows], [r["beta"] for r in rows], "w--x", ms=4, label="alpha* (known W)")
    ax.set_xlabel("sensing allocation alpha")
    ax.set_ylabel("sensing share of budget beta")
    ax.set_title("Expected Brier loss", fontsize=10)
    fig.colorbar(im, ax=ax)
    ax.legend(fontsize=7, loc="center right")

    ax = axes[1]
    b = [r["beta"] for r in rows]
    ax.errorbar(b, [r["loss_eval_mean"] for r in rows], yerr=[r["loss_eval_std"] for r in rows],
                fmt="k-o", ms=4, capsize=3, label="loss at alpha* (estimated channel)")
    ax.plot(b, [r["bayes_risk_at_alpha_star"] for r in rows], "b--", label="Bayes risk at alpha* (known channel)")
    ax.set_xlabel("sensing share of budget beta")
    ax.set_ylabel("expected Brier loss")
    ax.set_title("Loss vs budget split", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.semilogx([s["N_max"] for s in sweep], [s["beta_star"] for s in sweep], "k-o", ms=4)
    ax.set_xlabel("maximum probe budget N_max")
    ax.set_ylabel("optimal sensing share beta*")
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Optimal split vs probe budget", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig8_joint_sensing_probing_map.png", dpi=150)
    plt.close(fig)


def summary_table(results, saving_rows, joint_rows):
    N = 512
    lines = ["| Allocation rule | regime A: L_adopt at N=512 | regime B: L_adopt at N=512 |", "|---|---|---|"]
    for name, (_, _, label) in STYLE.items():
        a = next(r for r in results["A_uniform_prior"]["rows"] if r["N_probe"] == N)[name]["L_adopt_mean"]
        b = next(r for r in results["B_nonuniform_prior"]["rows"] if r["N_probe"] == N)[name]["L_adopt_mean"]
        lines.append(f"| {label} | {a:.3e} | {b:.3e} |")
    lines += ["", "| h | probes saved (analytic) | probes saved (MC) |", "|---|---|---|"]
    for r in saving_rows:
        lines.append(f"| {r['h']:.2f} | {100 * r['saving_analytic']:.1f}% | {100 * r['saving_mc']:.1f}% |")
    best = min(joint_rows, key=lambda r: r["loss_eval_mean"])
    lines += ["", f"Joint budget optimum: beta* = {best['beta']:.2f}, alpha* = {best['alpha_star']:.2f}, "
                  f"N_probe = {best['N_probe']}, final Brier = {best['loss_eval_mean']:.4f} +/- {best['loss_eval_std']:.4f}"]
    (ROOT / "results" / "ext3_summary_table.md").write_text("\n".join(lines))


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    part1_convergence()
    results = part2_budget_scaling()
    saving_rows = part3_probe_saving()
    grid, joint_rows, sweep = part4_joint_budget()
    fig7(results)
    fig7b(saving_rows, results)
    fig8(grid, joint_rows, sweep)
    summary_table(results, saving_rows, joint_rows)
    print("\n5. Saved figures fig7, fig7b, fig8 and results/ext3_*.json, results/ext3_summary_table.md")


if __name__ == "__main__":
    main()