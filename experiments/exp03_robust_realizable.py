
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channel import nominal_channel
from src.robust import L_real_wc, project_channel
from src.semantic import analytic_S
from src.sensitivity import worst_case_spectral

ROOT = Path(__file__).resolve().parents[1]
EPS_GRID = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25]
ALPHAS = [0.0, 0.5, 1.0]
REGIMES = {
    "near_boundary": dict(p_r=0.10, p_v=0.10, p_low=0.02),   # same channel as the baseline
    "interior": dict(p_r=0.30, p_v=0.30, p_low=0.10),
}
ARGS = (1.0, 1.0, 1.0, 0.1)  # a_r, a_v, sigma0_sq, eps -- same as baseline.json


def save(name, payload):
    out = ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def check_projection_validity():
    # Section 5.2 item 8: projected adversarial channels must stay nonnegative
    # and row-stochastic
    rng = np.random.default_rng(2026)
    bad = 0
    for _ in range(200):
        W = rng.normal(size=(4, 4))
        P = project_channel(W)
        if P.min() < -1e-10 or np.abs(P.sum(axis=1) - 1).max() > 1e-10:
            bad += 1
    print(f"1. Projection validity: {200 - bad}/200 random projections stayed nonnegative and row-stochastic")


def run_sweep():
    rng = np.random.default_rng(2026)
    rows = []
    for regime_name, ch in REGIMES.items():
        W_hat = nominal_channel(ch["p_r"], ch["p_v"], "range_optimistic", ch["p_low"])
        for alpha in ALPHAS:
            S = analytic_S(alpha, *ARGS)
            for eps in EPS_GRID:
                l_real, l_rand, l_grad = L_real_wc(S, W_hat, np.full(4, 0.25), eps, rng)
                l_spec = worst_case_spectral(S, eps ** 2)
                gap = (l_spec - l_real) / l_real if l_real > 0 else float("nan")
                rows.append({
                    "regime": regime_name, "alpha": alpha, "epsilon": eps,
                    "L_real_wc": l_real, "L_real_random": l_rand, "L_real_grad": l_grad,
                    "L_spec_wc": l_spec, "gap_fraction": gap,
                    "spec_is_upper_bound": bool(l_spec >= l_real - 1e-9),
                })
    save("ext2_sweep.json", {"eps_grid": EPS_GRID, "alphas": ALPHAS, "regimes": REGIMES, "rows": rows})
    return rows


def summarize(rows):
    n_violated = sum(1 for r in rows if not r["spec_is_upper_bound"])
    print(f"2. Sweep done: {len(rows)} (regime, alpha, epsilon) combinations")
    print(f"3. Spectral bound holds (L_spec >= L_real) in {len(rows) - n_violated}/{len(rows)} cases")
    if n_violated:
        print(f"   {n_violated} cases violate it -- expected: eq. 70 only holds under the 'balanced")
        print("   assumptions' (doubly stochastic W), but the channel-realizable set (eq. 65) only")
        print("   requires row-stochastic, which is strictly larger.")


def design_agreement(rows):
    # for each (regime, epsilon), do L_real_wc and L_spec_wc pick the same alpha?
    results = []
    for regime_name in REGIMES:
        for eps in EPS_GRID:
            subset = [r for r in rows if r["regime"] == regime_name and r["epsilon"] == eps]
            best_real = max(subset, key=lambda r: r["L_real_wc"])
            best_spec = max(subset, key=lambda r: r["L_spec_wc"])
            results.append({
                "regime": regime_name, "epsilon": eps,
                "alpha_real": best_real["alpha"], "alpha_spec": best_spec["alpha"],
                "agree": best_real["alpha"] == best_spec["alpha"],
            })
    n_agree = sum(1 for r in results if r["agree"])
    print(f"4. Design agreement: the two criteria pick the same alpha in {n_agree}/{len(results)} cases")
    save("ext2_design_agreement.json", results)


def make_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {0.0: "tab:blue", 0.5: "tab:green", 1.0: "tab:red"}
    for ax, regime_name in zip(axes, REGIMES):
        for alpha in ALPHAS:
            subset = sorted([r for r in rows if r["regime"] == regime_name and r["alpha"] == alpha],
                             key=lambda r: r["epsilon"])
            eps = [r["epsilon"] for r in subset]
            real = [r["L_real_wc"] for r in subset]
            spec = [r["L_spec_wc"] for r in subset]
            c = colors[alpha]
            ax.plot(eps, spec, "--", color=c, label=f"spec, a={alpha:.1f}")
            ax.plot(eps, real, "-", marker="o", markersize=3, color=c, label=f"real, a={alpha:.1f}")
        ax.set_title(regime_name.replace("_", " "))
        ax.set_xlabel("epsilon (Frobenius radius)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("worst-case L_mis")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Spectral bound (dashed) vs. channel-realizable worst case (solid)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig6_realizable_robust_mismatch.png", dpi=150)
    plt.close(fig)
    print("5. Saved figures/fig6_realizable_robust_mismatch.png")


def main():
    check_projection_validity()
    rows = run_sweep()
    summarize(rows)
    design_agreement(rows)
    make_figure(rows)
    print("\nAll results saved under results/ and figures/.")


if __name__ == "__main__":
    main()