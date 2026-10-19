
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channel import actual_channel
from src.design import alpha_star_map, mixed_l_mis, pareto_selection, task_vs_mismatch_sweep
from src.metrics import R_task_direct, bayes_risk, prior_risk
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensitivity import R_task_trace

ROOT = Path(__file__).resolve().parents[1]
LAMBDA_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]
Q_GRID = np.linspace(0.0, 1.0, 11)
Q_CASES = {"velocity_oriented": 0.0, "mixed_uncertain": 0.5, "range_oriented": 1.0}


def load_cfg():
    with open(ROOT / "configs" / "baseline.json") as f:
        return json.load(f)


def save(name, payload):
    out = ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def check_r_task(cfg):
    args = (cfg["a_r"], cfg["a_v"], cfg["sigma0_sq"], cfg["eps"])
    pi = np.full(cfg["C"], 1.0 / cfg["C"])
    W = actual_channel(cfg["p_r"], cfg["p_v"])
    alphas = np.linspace(0.0, 1.0, cfg["n_alpha"])

    max_gap = 0.0
    for a in alphas:
        S = analytic_S(a, *args)
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        gap = abs(R_task_direct(pi, rho, A) - R_task_trace(S, W))
        max_gap = max(max_gap, gap)

    print(f"1. R_task: direct (prior risk - Bayes risk) == trace formula (gap {max_gap:.0e})")
    return max_gap


def build_sweep(cfg):
    sweep = task_vs_mismatch_sweep(cfg)
    save("ext1_sweep.json", {"config": cfg, "sweep": {k: v.tolist() for k, v in sweep.items()}})
    print("2. Sweep built: accuracy, R_task, L_mis (range/velocity) over 101 alphas")
    return sweep


def run_pareto(sweep):
    cases = {name: pareto_selection(sweep, LAMBDA_GRID, q=q) for name, q in Q_CASES.items()}
    save("ext1_pareto.json", {
        "lambda_grid": LAMBDA_GRID,
        "cases": {
            name: {
                "q": c["q"],
                "picks": c["picks"],
            }
            for name, c in cases.items()
        },
    })
    directional = cases["range_oriented"]["picks"]
    print("3. Pareto selection (q=1.0, range-oriented mismatch -- the case with a real tradeoff):")
    for rule in ("max_accuracy", "max_R_task", "min_L_mis", "max_V_prob"):
        p = directional[rule]
        print(f"   {rule:14s} alpha={p['alpha']:.2f}  R_task={p['R_task']:.4f}  L_mis={p['L_mis']:.4f}")
    return cases


def run_allocation_map(sweep):
    rows = alpha_star_map(sweep, LAMBDA_GRID, Q_GRID)
    save("ext1_alpha_star_map.json", {"lambda_grid": LAMBDA_GRID, "q_grid": Q_GRID.tolist(), "rows": rows})
    print("4. alpha* computed over the full (q, lambda) grid")
    return rows


def make_figures(sweep, pareto_cases, alpha_star_rows):
    alpha = sweep["alpha"]
    R_task = sweep["R_task"]
    # q=0.5 is a degenerate case here: by the model's own range/velocity
    # symmetry, R_task and the mixed L_mis both peak at the same alpha, so
    # there is no real tradeoff to show. q=1.0 (a pure directional mismatch,
    # the realistic case where the receiver only misjudges one physical
    # factor) is where the Pareto tradeoff the guideline describes actually
    # appears -- L_mis grows monotonically with alpha while R_task peaks at
    # alpha=0.5, so alpha > 0.5 is strictly dominated.
    mixed = pareto_cases["range_oriented"]
    l_mix = mixed["L_mis_mixed"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(l_mix, R_task, color="#888", lw=1, zorder=1)
    ax.scatter(l_mix, R_task, c=alpha, cmap="viridis", s=18, zorder=2)
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("sensing allocation alpha")

    markers = {"max_accuracy": "s", "max_R_task": "^", "min_L_mis": "v", "max_V_prob": "D"}
    for rule, marker in markers.items():
        p = mixed["picks"][rule]
        ax.scatter([p["L_mis"]], [p["R_task"]], marker=marker, s=90, edgecolor="black",
                   facecolor="none", linewidth=1.6, label=rule.replace("_", " "))

    ax.set_xlabel("L_mis (range-oriented mismatch, q=1.0)")
    ax.set_ylabel("R_task")
    ax.set_title("Information-reliability Pareto frontier")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig3_pareto_frontier.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lam in LAMBDA_GRID:
        qs = sorted({row["q"] for row in alpha_star_rows})
        a_star = [next(r["alpha_star"] for r in alpha_star_rows if r["q"] == q and r["lambda"] == lam) for q in qs]
        ax.plot(qs, a_star, marker="o", markersize=3, label=f"lambda={lam:g}")
    ax.set_xlabel("q (probability mismatch is range-oriented)")
    ax.set_ylabel("alpha* (selected sensing allocation)")
    ax.set_title("Optimal sensing allocation vs. mismatch-direction belief")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig4_alpha_star_map.png", dpi=150)
    plt.close(fig)

    print("5. Saved figures/fig3_pareto_frontier.png and figures/fig4_alpha_star_map.png")


def make_summary_table(pareto_cases):
    directional = pareto_cases["range_oriented"]["picks"]
    rules = ["max_accuracy", "max_R_task", "min_L_mis", "max_V_prob",
             "max_J_lambda_1", "max_J_lambda_5"]
    lines = ["| Criterion (q=1.0, range-oriented mismatch) | alpha | Hard accuracy | R_task | L_mis |",
             "|---|---|---|---|---|"]
    for rule in rules:
        if rule not in directional:
            continue
        p = directional[rule]
        lines.append(f"| {rule.replace('_', ' ')} | {p['alpha']:.2f} | {p['accuracy']:.4f} | "
                     f"{p['R_task']:.4f} | {p['L_mis']:.4f} |")
    table = "\n".join(lines)
    (ROOT / "results" / "ext1_summary_table.md").write_text(table + "\n")
    print("6. Saved results/ext1_summary_table.md")


def main():
    cfg = load_cfg()
    check_r_task(cfg)
    sweep = build_sweep(cfg)
    pareto_cases = run_pareto(sweep)
    alpha_star_rows = run_allocation_map(sweep)
    make_figures(sweep, pareto_cases, alpha_star_rows)
    make_summary_table(pareto_cases)
    print("\nAll results saved under results/ and figures/.")


if __name__ == "__main__":
    main()