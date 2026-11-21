import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channel import actual_channel
from src.design import alpha_star_map, pareto_selection, task_vs_mismatch_sweep
from src.metrics import R_task_direct
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensitivity import R_task_trace

ROOT = Path(__file__).resolve().parents[1]
# L_mis is ~50x smaller than R_task, so the trade-off shows up around lambda ~ 10
LAMBDA_RULES = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
LAMBDA_MAP = np.concatenate([[0.0], np.logspace(-1, 3, 49)])
LAMBDA_CURVES_Q = [1.0, 0.75, 0.6, 0.5]
Q_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)  # rounded so q values are exact dict keys
Q_CASES = {"velocity_oriented": 0.0, "mixed_uncertain": 0.5, "range_oriented": 1.0}
LAMBDA_HIGHLIGHT = (5.0, 10.0, 20.0)


def load_cfg():
    with open(ROOT / "configs" / "baseline.json") as f:
        return json.load(f)


def save(name, payload):
    out = ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)


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
    save("ext1_sweep.json", {"config": cfg, "sweep": sweep})
    print(f"2. Sweep built over {cfg['n_alpha']} alphas: accuracy, R_task, L_mis (range/velocity), tr(G_s), "
          f"lambda_max(G_s)")
    return sweep


def run_pareto(sweep):
    cases = {name: pareto_selection(sweep, LAMBDA_RULES, q=q) for name, q in Q_CASES.items()}
    save("ext1_pareto.json", {"lambda_grid": LAMBDA_RULES, "alpha": sweep["alpha"], "cases": cases})
    print("3. Selected alpha per rule ('|' = exact tie):")
    for name, c in cases.items():
        picks = ", ".join(f"{rule.replace('max_', '').replace('min_', 'min ')}="
                          f"{'|'.join(f'{a:.2f}' for a in p['alpha_ties'])}"
                          for rule, p in c["picks"].items())
        print(f"   q={c['q']:.1f} ({name}): {picks}")
    return cases


def run_allocation_map(sweep):
    rows = alpha_star_map(sweep, LAMBDA_MAP, Q_GRID)
    save("ext1_alpha_star_map.json", {"lambda_grid": LAMBDA_MAP, "q_grid": Q_GRID, "rows": rows})
    tie_qs = sorted({r["q"] for r in rows if r["tie"]})
    print(f"4. alpha* over the (q, lambda) grid: {len(Q_GRID)} x {len(LAMBDA_MAP)} cells, "
          f"{sum(r['tie'] for r in rows)} exact ties, at q in {tie_qs}")
    q1 = [r for r in rows if r["q"] == 1.0]
    interior = [r for r in q1 if 0.0 < r["alpha_star"] < 0.5]
    if interior:
        print(f"   q=1: alpha* moves continuously 0.50 -> {min(r['alpha_star'] for r in interior):.2f} "
              f"for lambda up to {max(r['lambda'] for r in interior):.1f}, then jumps to "
              f"{min(r['alpha_star'] for r in q1):.2f}")
    return rows


def make_figures(sweep, pareto_cases, alpha_star_rows):
    alpha = sweep["alpha"]
    R_task = sweep["R_task"]
    # q = 0.5 is symmetric about alpha = 0.5, so the trade-off is shown for q = 1
    directional = pareto_cases["range_oriented"]
    l_dir = directional["L_mis_mixed"]
    front = np.asarray(directional["pareto_front"])

    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(l_dir, R_task, color="#bbb", lw=1, zorder=1)
    sc = ax.scatter(l_dir, R_task, c=alpha, cmap="viridis", s=16, zorder=2)
    ax.plot(l_dir[front], R_task[front], color="k", lw=2.2, zorder=3, label="Pareto front (alpha <= 0.5)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("sensing allocation alpha")

    markers = {"max_accuracy": ("s", 150, "max accuracy"), "max_R_task": ("^", 110, "max R_task"),
               "max_V_prob": ("D", 70, "max V_prob"), "min_L_mis": ("v", 110, "min L_mis")}
    for rule, (marker, size, name) in markers.items():
        p = directional["picks"][rule]
        ax.scatter([p["L_mis"]], [p["R_task"]], marker=marker, s=size, edgecolor="black", facecolor="none",
                   linewidth=1.5, zorder=4, label=f"{name} (alpha = {p['alpha']:.2f})")
    for lam, color in zip(LAMBDA_HIGHLIGHT, ("tab:orange", "tab:red", "tab:purple")):
        p = directional["picks"][f"max_J_lambda_{lam:g}"]
        ax.scatter([p["L_mis"]], [p["R_task"]], marker="*", s=180, color=color, edgecolor="k", lw=0.6,
                   zorder=5, label=f"max J, lambda = {lam:g} (alpha = {p['alpha']:.2f})")

    ax.set_xlabel("mismatch loss L_mis (range-oriented mismatch, q = 1)")
    ax.set_ylabel("task informativeness R_task")
    ax.set_title("Information vs reliability trade-off")
    ax.legend(fontsize=7, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig3_pareto_frontier.png", dpi=150)
    plt.close(fig)

    
    lams = sorted({r["lambda"] for r in alpha_star_rows})
    qs = sorted({r["q"] for r in alpha_star_rows})
    lookup = {(r["q"], r["lambda"]): r for r in alpha_star_rows}
    lam_pos = [l for l in lams if l > 0]  # log axis

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for q in LAMBDA_CURVES_Q:
        a_star = [lookup[(q, l)]["alpha_star"] for l in lam_pos]
        ties = [lookup[(q, l)]["tie"] for l in lam_pos]
        line, = ax1.semilogx(lam_pos, a_star, marker="o", ms=3, label=f"q = {q:.2f}")
        tie_l = [l for l, t in zip(lam_pos, ties) if t]
        if tie_l:  # show the other member of an exact tie
            mirror = [lookup[(q, l)]["alpha_star_ties"][-1] for l in tie_l]
            ax1.semilogx(tie_l, mirror, ls="none", marker="o", ms=3, mfc="none", color=line.get_color(),
                         label=f"q = {q:.2f}, tied optimum")
    ax1.axhline(0.5, color="k", lw=0.8, ls=":")
    ax1.set_xlabel("reliability weight lambda (log scale)")
    ax1.set_ylabel("alpha* = argmax J_lambda")
    ax1.set_title("alpha* vs lambda", fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3, which="both")

    grid = np.array([[lookup[(q, l)]["alpha_star"] for q in qs] for l in lam_pos])
    im = ax2.pcolormesh(qs, lam_pos, grid, shading="nearest", cmap="coolwarm", vmin=0, vmax=1)
    ax2.set_yscale("log")
    tq = [q for q in qs for l in lam_pos if lookup[(q, l)]["tie"]]
    tl = [l for q in qs for l in lam_pos if lookup[(q, l)]["tie"]]
    if tq:
        ax2.plot(tq, tl, "kx", ms=4, label="exact tie: alpha* = 0 or 1")
        ax2.legend(fontsize=7, loc="lower right")
    ax2.set_xlabel("q (probability the mismatch is range-oriented)")
    ax2.set_ylabel("reliability weight lambda (log scale)")
    ax2.set_title("alpha* over (q, lambda)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("alpha* (0 = velocity-precise, 1 = range-precise)")
    fig.suptitle("Optimal sensing allocation alpha*")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig4_alpha_star_map.png", dpi=150)
    plt.close(fig)

    print("5. Saved figures/fig3_pareto_frontier.png and figures/fig4_alpha_star_map.png")


def make_summary_table(pareto_cases):
    directional = pareto_cases["range_oriented"]["picks"]
    ref = directional["max_R_task"]
    rules = ["max_accuracy", "max_R_task", "max_V_prob"] + [f"max_J_lambda_{l:g}" for l in LAMBDA_RULES[1:]] \
        + ["min_L_mis"]
    lines = ["| Criterion (q = 1, range-oriented mismatch) | alpha | Hard accuracy | R_task | L_mis | "
             "L_mis vs info-only design |",
             "|---|---|---|---|---|---|"]
    for rule in rules:
        p = directional[rule]
        a = " / ".join(f"{x:.2f}" for x in p["alpha_ties"])
        lines.append(f"| {rule.replace('_', ' ')} | {a} | {p['accuracy']:.4f} | {p['R_task']:.4f} | "
                     f"{p['L_mis']:.5f} | {100 * (p['L_mis'] / ref['L_mis'] - 1):+.0f}% |")
    lines += ["", "lambda = 1 coincides with max V_prob, since V_prob = R_task - L_mis is J_lambda at lambda = 1."]
    (ROOT / "results" / "ext1_summary_table.md").write_text("\n".join(lines) + "\n")
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
