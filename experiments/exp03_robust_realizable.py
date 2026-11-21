import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# one BLAS thread per worker (must be set before numpy is imported)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.channel import nominal_channel
from src.robust import L_real_wc, is_valid_channel, project_channel, project_doubly_stochastic
from src.semantic import analytic_S
from src.sensitivity import worst_case_spectral

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "baseline.json"))
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
BASE_SEED = CFG["seed"]
PI = np.full(CFG["C"], 1.0 / CFG["C"])

EPS_GRID = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25]
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
N_SEEDS = 3
SEARCH = dict(n_trials=1500, n_steps=60, n_restarts=4, step_frac=0.1)
REGIMES = {
    "near_boundary": dict(p_r=0.10, p_v=0.10, p_low=0.02),
    "interior": dict(p_r=0.30, p_v=0.30, p_low=0.10),
}
SETS = ("doubly", "row")
ALPHAS_PLOTTED = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS_RANKING = (0.03, 0.25)
TIE_RTOL = 1e-9  # L_spec is symmetric in alpha, so ties are kept


def save(name, payload):
    out = ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def nominal(regime):
    ch = REGIMES[regime]
    return nominal_channel(ch["p_r"], ch["p_v"], "range_optimistic", ch["p_low"])




def check_projection_validity(n=200):
    rng = np.random.default_rng([BASE_SEED, 0])
    ok = {"row": 0, "doubly": 0}
    for _ in range(n):
        W = rng.normal(0.25, 0.3, size=(4, 4))
        ok["row"] += is_valid_channel(project_channel(W), "row")
        ok["doubly"] += is_valid_channel(project_doubly_stochastic(W), "doubly")
    print(f"1. Projection validity: row-stochastic {ok['row']}/{n}, doubly-stochastic {ok['doubly']}/{n}")
    for regime in REGIMES:
        assert is_valid_channel(nominal(regime), "doubly"), "nominal must lie in both sets"
    return ok




def _cell(task):
    r_idx, regime, s_idx, cset, a_idx, alpha, e_idx, eps, seed_idx = task
    rng = np.random.default_rng([BASE_SEED, 2, r_idx, s_idx, a_idx, e_idx, seed_idx])
    S = analytic_S(alpha, *ARGS)
    W_hat = nominal(regime)
    best, l_rand, l_grad, W_star = L_real_wc(S, W_hat, PI, eps, rng, channel_set=cset,
                                             return_candidate=True, **SEARCH)
    valid = is_valid_channel(W_star, cset) and np.linalg.norm(W_star - W_hat) <= eps + 1e-9
    return task, best, l_rand, l_grad, bool(valid)


def run_sweep():
    tasks = [(r_idx, regime, s_idx, cset, a_idx, float(alpha), e_idx, eps, k)
             for r_idx, regime in enumerate(REGIMES)
             for s_idx, cset in enumerate(SETS)
             for a_idx, alpha in enumerate(ALPHAS)
             for e_idx, eps in enumerate(EPS_GRID)
             for k in range(N_SEEDS)]
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(_cell, tasks, chunksize=4))

    cells = {}
    for (_, regime, _, cset, _, alpha, _, eps, _), best, l_rand, l_grad, valid in out:
        c = cells.setdefault((regime, cset, alpha, eps), {"best": [], "rand": [], "grad": [], "valid": True})
        c["best"].append(best)
        c["rand"].append(l_rand)
        c["grad"].append(l_grad)
        c["valid"] &= valid

    rows = []
    for (regime, cset, alpha, eps), c in cells.items():
        l_real = float(max(c["best"]))
        l_spec = worst_case_spectral(analytic_S(alpha, *ARGS), eps ** 2)
        rows.append({
            "regime": regime, "channel_set": cset, "alpha": alpha, "epsilon": eps,
            "L_real_wc": l_real,
            "L_real_seeds": [float(v) for v in c["best"]],
            "L_real_seed_min": float(min(c["best"])),
            "L_real_seed_mean": float(np.mean(c["best"])), "L_real_seed_std": float(np.std(c["best"])),
            "L_real_random_best": float(max(c["rand"])), "L_real_grad_best": float(max(c["grad"])),
            "L_spec_wc": l_spec,
            "ratio_real_over_spec": l_real / l_spec,
            "gap_fraction": (l_spec - l_real) / l_real,
            "spec_is_upper_bound": bool(l_spec >= l_real * (1 - 1e-6)),
            "all_candidates_valid": c["valid"],
        })
    rows.sort(key=lambda r: (r["regime"], r["channel_set"], r["epsilon"], r["alpha"]))
    save("ext2_sweep.json", {
        "seed_base": BASE_SEED, "n_seeds": N_SEEDS, "search": SEARCH,
        "eps_grid": EPS_GRID, "alphas": ALPHAS.tolist(), "regimes": REGIMES,
        "channel_sets": list(SETS), "prior": PI.tolist(), "nominal_mode": "range_optimistic",
        "rows": rows,
    })
    return rows


def subset(rows, regime, cset, eps=None, alpha=None):
    out = [r for r in rows if r["regime"] == regime and r["channel_set"] == cset
           and (eps is None or r["epsilon"] == eps) and (alpha is None or r["alpha"] == alpha)]
    return sorted(out, key=lambda r: (r["epsilon"], r["alpha"]))




def summarize_value(rows):
    print("2. Value tightness, L_real / L_spec (1 = tight, >1 = bound violated):")
    summary = []
    for regime in REGIMES:
        for cset in SETS:
            sub = subset(rows, regime, cset)
            ratios = np.array([r["ratio_real_over_spec"] for r in sub])
            n_viol = sum(not r["spec_is_upper_bound"] for r in sub)
            rel_spread = max(r["L_real_seed_std"] / r["L_real_wc"] for r in sub)
            summary.append({"regime": regime, "channel_set": cset, "ratio_min": float(ratios.min()),
                            "ratio_median": float(np.median(ratios)), "ratio_max": float(ratios.max()),
                            "n_bound_violations": n_viol, "n_cells": len(sub),
                            "max_seed_rel_std": float(rel_spread),
                            "all_candidates_valid": all(r["all_candidates_valid"] for r in sub)})
            print(f"   {regime:13s} {cset:6s}: min {ratios.min():.3f}, median {np.median(ratios):.3f}, "
                  f"max {ratios.max():.3f}; violations {n_viol}/{len(sub)}; "
                  f"seed spread <= {100 * rel_spread:.1f}%")
    return summary




def _argmin_set(values, alphas):
    v = np.asarray(values)
    return sorted(float(a) for a, x in zip(alphas, v) if x <= v.min() * (1 + TIE_RTOL))


def design_agreement(rows):
    """Compare the safest alpha chosen by L_spec and by L_real, and the regret of the spectral pick."""
    results = []
    for regime in REGIMES:
        for cset in SETS:
            for eps in EPS_GRID:
                sub = subset(rows, regime, cset, eps=eps)
                alphas = [r["alpha"] for r in sub]
                real = np.array([r["L_real_wc"] for r in sub])
                spec = np.array([r["L_spec_wc"] for r in sub])
                pick_real, pick_spec = _argmin_set(real, alphas), _argmin_set(spec, alphas)
                # if the spectral pick is tied, take the worst of the tied alphas
                real_at_spec = max(real[alphas.index(a)] for a in pick_spec)
                rho = spearmanr(real, spec).statistic
                results.append({
                    "regime": regime, "channel_set": cset, "epsilon": eps,
                    "alpha_safest_real": pick_real, "alpha_safest_spec": pick_spec,
                    "spec_has_tie": len(pick_spec) > 1,
                    "agree": bool(set(pick_real) & set(pick_spec)),
                    "spearman_rho": float(rho),
                    "regret_of_spec_pick": float(real_at_spec / real.min() - 1.0),
                })
    print("3. Design agreement (safest alpha, regret of the spectral pick):")
    for regime in REGIMES:
        for cset in SETS:
            sub = [r for r in results if r["regime"] == regime and r["channel_set"] == cset]
            n_agree = sum(r["agree"] for r in sub)
            rhos = [r["spearman_rho"] for r in sub]
            print(f"   {regime:13s} {cset:6s}: agree {n_agree}/{len(sub)}, Spearman {min(rhos):.3f}..{max(rhos):.3f}, "
                  f"max regret {100 * max(r['regret_of_spec_pick'] for r in sub):.1f}%, "
                  f"real picks {sorted({a for r in sub for a in r['alpha_safest_real']})}, "
                  f"spec picks {sorted({a for r in sub for a in r['alpha_safest_spec']})}")
    save("ext2_design_agreement.json", results)
    return results




def make_figures(rows, agreement):
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ALPHAS_PLOTTED)))
    titles = {"doubly": "doubly-stochastic set", "row": "row-stochastic set"}

    # fig 6: L_real / L_spec vs radius
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for i, cset in enumerate(SETS):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            for c, alpha in zip(colors, ALPHAS_PLOTTED):
                sub = subset(rows, regime, cset, alpha=alpha)
                eps = [r["epsilon"] for r in sub]
                ratio = np.array([r["ratio_real_over_spec"] for r in sub])
                # point = best seed, bar = down to the weakest seed
                lower = np.array([(r["L_real_wc"] - r["L_real_seed_min"]) / r["L_spec_wc"] for r in sub])
                ax.errorbar(eps, ratio, yerr=[lower, np.zeros_like(lower)], color=c, marker="o", ms=3,
                            capsize=2, label=f"alpha = {alpha:.2f}")
            ax.axhline(1.0, color="k", ls="--", lw=1, label="L_real = L_spec")
            ax.set_title(f"{regime.replace('_', ' ')} nominal, {titles[cset]}", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("L_real / L_spec")
            if i == 1:
                ax.set_xlabel("radius epsilon")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Realizable worst case vs spectral bound (best of {N_SEEDS} seeds, bar to weakest seed)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig6_realizable_robust_mismatch.png", dpi=150)
    plt.close(fig)

    # fig 6b: worst-case loss across alpha
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    ls = {EPS_RANKING[0]: "-", EPS_RANKING[1]: "--"}
    for i, cset in enumerate(SETS):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            for eps in EPS_RANKING:
                sub = subset(rows, regime, cset, eps=eps)
                a = [r["alpha"] for r in sub]
                real = np.array([r["L_real_wc"] for r in sub])
                spec = np.array([r["L_spec_wc"] for r in sub])
                ax.plot(a, real / real.min(), color="tab:red", ls=ls[eps], marker="o", ms=3,
                        label=f"L_real, eps = {eps}")
                ax.plot(a, spec / spec.min(), color="tab:blue", ls=ls[eps], label=f"L_spec, eps = {eps}")
                ag = next(r for r in agreement if r["regime"] == regime and r["channel_set"] == cset
                          and r["epsilon"] == eps)
                for x in ag["alpha_safest_real"]:
                    ax.plot(x, 1.0, "v", color="tab:red", ms=9, mfc="none", mew=1.5)
                ax.text(0.5, 0.97 - 0.07 * EPS_RANKING.index(eps),
                        f"eps = {eps}: Spearman {ag['spearman_rho']:.3f}, regret {100 * ag['regret_of_spec_pick']:.1f}%",
                        transform=ax.transAxes, ha="center", va="top", fontsize=7)
            ax.set_title(f"{regime.replace('_', ' ')} nominal, {titles[cset]}", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("worst-case loss / minimum")
            if i == 1:
                ax.set_xlabel("sensing allocation alpha")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Robust design: spectral bound vs realizable worst case (triangles: realizable optimum)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig6b_design_agreement.png", dpi=150)
    plt.close(fig)
    print("4. Saved figures/fig6_realizable_robust_mismatch.png and figures/fig6b_design_agreement.png")


def summary_table(value_summary, agreement):
    lines = ["| Nominal regime | Adversary set | L_real/L_spec (min / median / max) | Bound violations | "
             "Safest-alpha agreement | Spearman (min) | Max regret of spectral pick |",
             "|---|---|---|---|---|---|---|"]
    for v in value_summary:
        sub = [r for r in agreement if r["regime"] == v["regime"] and r["channel_set"] == v["channel_set"]]
        lines.append(
            f"| {v['regime'].replace('_', ' ')} | {v['channel_set']}-stochastic | "
            f"{v['ratio_min']:.3f} / {v['ratio_median']:.3f} / {v['ratio_max']:.3f} | "
            f"{v['n_bound_violations']}/{v['n_cells']} | {sum(r['agree'] for r in sub)}/{len(sub)} | "
            f"{min(r['spearman_rho'] for r in sub):.3f} | {100 * max(r['regret_of_spec_pick'] for r in sub):.1f}% |")
    (ROOT / "results" / "ext2_summary_table.md").write_text("\n".join(lines) + "\n")
    print("5. Saved results/ext2_summary_table.md")


def main():
    if "--figures-only" in sys.argv:
        # rebuild figures and tables from the saved sweep
        with open(ROOT / "results" / "ext2_sweep.json") as f:
            rows = json.load(f)["rows"]
    else:
        check_projection_validity()
        rows = run_sweep()
    value_summary = summarize_value(rows)
    save("ext2_value_summary.json", value_summary)
    agreement = design_agreement(rows)
    make_figures(rows, agreement)
    summary_table(value_summary, agreement)
    print("\nAll results saved under results/ and figures/.")


if __name__ == "__main__":
    main()
