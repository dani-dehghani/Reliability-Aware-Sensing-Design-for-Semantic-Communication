import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.channel import actual_channel, nominal_channel
from src.metrics import L_mis_direct, L_mis_identity
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensitivity import L_mis_quadratic, L_mis_trace, Q_matrix

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "baseline.json"))
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
BASE_SEED = CFG["seed"]

T_GRID = np.logspace(-3, 0, 25)  # mismatch fraction t
ALPHAS_EPS_SWEEP = (0.0, 0.5, 1.0)
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
N_RANDOM_TARGETS = 5

T_OPERATING = 0.8


def asym_bsc(p01, p10):
    # binary asymmetric channel: P(1|0) = p01, P(0|1) = p10
    return np.array([[1.0 - p01, p01], [p10, 1.0 - p10]])


def random_channel(rng, K=4, diag_conc=8.0):
    # Dirichlet rows concentrated on the diagonal
    conc = np.ones((K, K)) + (diag_conc - 1.0) * np.eye(K)
    return np.array([rng.dirichlet(c) for c in conc])


def build_cases():
    rng = np.random.default_rng([BASE_SEED, 5])
    priors = {
        "uniform": np.full(4, 0.25),
        "mild": np.array([0.40, 0.30, 0.20, 0.10]),
        "skewed": np.array([0.70, 0.10, 0.10, 0.10]),
        "dirichlet": rng.dirichlet(np.ones(4)),
    }
    channels = {
        "symmetric_kron": {
            "W": actual_channel(CFG["p_r"], CFG["p_v"]),
            "range_target": nominal_channel(CFG["p_r"], CFG["p_v"], "range_optimistic", 0.0),
        },
        "asymmetric_kron": {
            "W": np.kron(asym_bsc(0.15, 0.05), asym_bsc(0.05, 0.15)),
            "range_target": np.kron(np.eye(2), asym_bsc(0.05, 0.15)),
        },
        "random_dirichlet": {"W": random_channel(rng), "range_target": None},
    }
    targets = {}
    for ch_name, ch in channels.items():
        t = {"optimistic": np.eye(4)}  # receiver believes the channel is error-free
        if ch["range_target"] is not None:
            t["range_optimistic"] = ch["range_target"]
        for k in range(N_RANDOM_TARGETS):
            t[f"random_{k}"] = random_channel(rng)
        targets[ch_name] = t
    return priors, channels, targets


def losses(S, pi, W, W_hat):
    A, A_hat = end_to_end(S, W), end_to_end(S, W_hat)
    rho, rho_hat = posterior(pi, A), posterior(pi, A_hat)
    dW = W_hat - W
    exact = L_mis_direct(pi, rho, rho_hat, A)
    return {
        "exact": exact,
        "identity_gap": abs(exact - L_mis_identity(pi, S, dW, rho, A, A_hat)),
        "quad": L_mis_quadratic(S, pi, W, dW),
        "balanced": L_mis_trace(S, dW),
    }


def is_doubly(M, atol=1e-12):
    return bool(np.allclose(M.sum(axis=0), 1.0, atol=atol) and np.allclose(M.sum(axis=1), 1.0, atol=atol))


def is_balanced(S, pi, W, W_hat):
    # uniform prior and doubly-stochastic S, W, W_hat: both approximations are exact here
    return bool(np.allclose(pi, 1.0 / len(pi)) and is_doubly(S) and is_doubly(W) and is_doubly(W_hat))


def rel_err(approx, exact):
    return abs(approx - exact) / exact if exact > 0 else 0.0


def checks(priors, channels):
    rng = np.random.default_rng([BASE_SEED, 6])
    S = analytic_S(0.3, *ARGS)
    W = channels["symmetric_kron"]["W"]
    # balanced case: quad == exact == trace
    W_hat = 0.7 * W + 0.3 * np.eye(4)
    l = losses(S, priors["uniform"], W, W_hat)
    bal_gap = max(rel_err(l["quad"], l["exact"]), rel_err(l["balanced"], l["exact"]))
    # Q_u is symmetric PSD and the vec form equals the blockwise sum
    max_asym, min_eig, vec_gap = 0.0, np.inf, 0.0
    for pi in priors.values():
        for ch in channels.values():
            Q = Q_matrix(S, pi, ch["W"])
            max_asym = max(max_asym, np.abs(Q - Q.T).max())
            min_eig = min(min_eig, np.linalg.eigvalsh(Q).min())
            dW = 0.05 * (random_channel(rng) - ch["W"])
            v = dW.flatten(order="F")
            vec_gap = max(vec_gap, abs(v @ Q @ v - L_mis_quadratic(S, pi, ch["W"], dW)))
    print(f"1. Checks: balanced case quad == exact == trace (rel gap {bal_gap:.0e}); "
          f"Q_u symmetric (err {max_asym:.0e}), PSD (min eig {min_eig:.1e}); vec form == blockwise (gap {vec_gap:.0e})")
    return {"balanced_rel_gap": bal_gap, "Q_max_asymmetry": max_asym, "Q_min_eig": min_eig, "vec_gap": vec_gap}


def eps_sweep(priors, channels, targets):
    rows = []
    for pr_name, pi in priors.items():
        for ch_name, ch in channels.items():
            W = ch["W"]
            for tg_name, W_t in targets[ch_name].items():
                D = W_t - W
                for alpha in ALPHAS_EPS_SWEEP:
                    S = analytic_S(alpha, *ARGS)
                    for t in T_GRID:
                        l = losses(S, pi, W, W + t * D)
                        rows.append({
                            "balanced": is_balanced(S, pi, W, W + t * D),
                            "prior": pr_name, "channel": ch_name, "mismatch": tg_name,
                            "family": tg_name.split("_")[0] if tg_name.startswith("random") else tg_name,
                            "alpha": alpha, "t": float(t), "eps": float(t * np.linalg.norm(D)),
                            "L_exact": l["exact"], "L_quad": l["quad"], "L_balanced": l["balanced"],
                            "rel_err_quad": rel_err(l["quad"], l["exact"]),
                            "rel_err_balanced": rel_err(l["balanced"], l["exact"]),
                            "identity_gap": l["identity_gap"],
                        })
    # log-log slope of the quadratic rel. error vs eps at small eps (O(eps) predicts 1)
    slopes = []
    # dict.fromkeys (not set) keeps a fixed order so the saved slopes are reproducible
    for key in dict.fromkeys((r["prior"], r["channel"], r["mismatch"], r["alpha"]) for r in rows if not r["balanced"]):
        sub = sorted((r for r in rows if (r["prior"], r["channel"], r["mismatch"], r["alpha"]) == key),
                     key=lambda r: r["eps"])
        pts = [(r["eps"], r["rel_err_quad"]) for r in sub[:10] if r["rel_err_quad"] > 1e-12]
        if len(pts) >= 5:
            slopes.append(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
    bal = [r for r in rows if r["balanced"]]
    print(f"2. eps sweep: {len(rows)} evaluations; exact identity matches direct to "
          f"{max(r['identity_gap'] for r in rows):.0e}")
    print(f"   {len(bal)} balanced evaluations are exact (max rel. error: quadratic "
          f"{max(r['rel_err_quad'] for r in bal):.0e}, balanced trace {max(r['rel_err_balanced'] for r in bal):.0e}); "
          f"excluded below")
    print(f"   quadratic rel. error vs eps: log-log slope median {np.median(slopes):.2f} "
          f"(range {min(slopes):.2f}..{max(slopes):.2f}; expected 1.00)")
    return rows, slopes


def summarize_eps(rows):
    
    out = []
    for pr_name in dict.fromkeys(r["prior"] for r in rows):
        for eps_target in (0.01, 0.05, 0.1, 0.2):
            sub = [r for r in rows if r["prior"] == pr_name and not r["balanced"]]
            # nearest grid point per curve
            best = {}
            for r in sub:
                key = (r["channel"], r["mismatch"], r["alpha"])
                if key not in best or abs(r["eps"] - eps_target) < abs(best[key]["eps"] - eps_target):
                    best[key] = r
            pts = [r for r in best.values() if abs(np.log(r["eps"] / eps_target)) < 0.35]
            if not pts:
                continue
            out.append({"prior": pr_name, "eps": eps_target, "n_curves": len(pts),
                        "rel_err_quad_median": float(np.median([r["rel_err_quad"] for r in pts])),
                        "rel_err_quad_max": float(max(r["rel_err_quad"] for r in pts)),
                        "rel_err_balanced_median": float(np.median([r["rel_err_balanced"] for r in pts])),
                        "rel_err_balanced_max": float(max(r["rel_err_balanced"] for r in pts))})
    print("3. Relative error, non-balanced cases (median / max):")
    print("   prior       eps    quadratic              balanced trace")
    for s in out:
        print(f"   {s['prior']:10s}  {s['eps']:.2f}   {100 * s['rel_err_quad_median']:6.2f}% / {100 * s['rel_err_quad_max']:6.2f}%"
              f"      {100 * s['rel_err_balanced_median']:7.1f}% / {100 * s['rel_err_balanced_max']:7.1f}%")
    return out


def alpha_sweep(priors, channels, targets):
    rows = []
    for pr_name, pi in priors.items():
        for ch_name, ch in channels.items():
            W = ch["W"]
            for tg_name in ("optimistic", "range_optimistic"):
                if tg_name not in targets[ch_name]:
                    continue
                W_hat = W + T_OPERATING * (targets[ch_name][tg_name] - W)
                ex, qu, ba = [], [], []
                for alpha in ALPHA_GRID:
                    l = losses(analytic_S(alpha, *ARGS), pi, W, W_hat)
                    ex.append(l["exact"]), qu.append(l["quad"]), ba.append(l["balanced"])
                ex, qu, ba = map(np.array, (ex, qu, ba))
                rows.append({
                    "prior": pr_name, "channel": ch_name, "mismatch": tg_name, "t": T_OPERATING,
                    "eps": float(np.linalg.norm(W_hat - W)), "alpha": ALPHA_GRID.tolist(),
                    "L_exact": ex.tolist(), "L_quad": qu.tolist(), "L_balanced": ba.tolist(),
                    "rel_err_quad_max": float(np.max(np.abs(qu - ex) / ex)),
                    "rel_err_balanced_max": float(np.max(np.abs(ba - ex) / ex)),
                    "spearman_quad": float(spearmanr(ex, qu).statistic),
                    "spearman_balanced": float(spearmanr(ex, ba).statistic),
                    "argmin_exact": float(ALPHA_GRID[np.argmin(ex)]),
                    "argmin_quad": float(ALPHA_GRID[np.argmin(qu)]),
                    "argmin_balanced": float(ALPHA_GRID[np.argmin(ba)]),
                })
    agree_q = sum(r["argmin_quad"] == r["argmin_exact"] for r in rows)
    agree_b = sum(r["argmin_balanced"] == r["argmin_exact"] for r in rows)
    print(f"4. Design check at t={T_OPERATING} ({len(rows)} cases, {len(ALPHA_GRID)} alphas):")
    print(f"   safest alpha matches exact: quadratic {agree_q}/{len(rows)}, balanced {agree_b}/{len(rows)}; "
          f"Spearman min: quadratic {min(r['spearman_quad'] for r in rows):.3f}, "
          f"balanced {min(r['spearman_balanced'] for r in rows):.3f}")
    return rows


PRIOR_COLORS = {"uniform": "k", "mild": "tab:blue", "skewed": "tab:red", "dirichlet": "tab:green"}


def make_figure(eps_rows, alpha_rows):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    ax = axes[0, 0]
    for pr, c in PRIOR_COLORS.items():
        sub = [r for r in eps_rows if r["prior"] == pr and r["L_exact"] > 1e-14]
        ax.loglog([r["L_exact"] for r in sub], [r["L_quad"] for r in sub], ".", ms=2, color=c, alpha=0.5,
                  label=f"prior: {pr}")
    lim = [min(r["L_exact"] for r in eps_rows if r["L_exact"] > 1e-14), max(r["L_exact"] for r in eps_rows)]
    ax.loglog(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("exact L_mis")
    ax.set_ylabel("quadratic approximation of L_mis")
    ax.set_title("(a) Exact vs quadratic approximation", fontsize=10)
    ax.legend(fontsize=7, markerscale=4)
    ax.grid(alpha=0.3, which="both")

    for ax, key, title in ((axes[0, 1], "rel_err_quad", "(b) Quadratic approximation error"),
                           (axes[1, 0], "rel_err_balanced", "(c) Balanced trace formula error")):
        bins = np.logspace(np.log10(min(r["eps"] for r in eps_rows)), np.log10(max(r["eps"] for r in eps_rows)), 18)
        mid = np.sqrt(bins[:-1] * bins[1:])
        for pr, c in PRIOR_COLORS.items():
            sub = [r for r in eps_rows if r["prior"] == pr and not r["balanced"]]
            med, lo, hi = [], [], []
            for a, b in zip(bins[:-1], bins[1:]):
                v = [max(r[key], 1e-16) for r in sub if a <= r["eps"] < b]
                med.append(np.median(v) if v else np.nan)
                lo.append(np.percentile(v, 10) if v else np.nan)
                hi.append(np.percentile(v, 90) if v else np.nan)
            ax.loglog(mid, med, "-o", ms=3, color=c, label=f"prior: {pr}")
            ax.fill_between(mid, lo, hi, color=c, alpha=0.12)
        if key == "rel_err_quad":
            ref = np.array([1e-3, 0.5])
            ax.loglog(ref, ref * 0.5, "k:", lw=1, label="slope 1 reference")
        ax.text(0.98, 0.02, "median, 10-90% band; exact balanced cases not shown", transform=ax.transAxes,
                fontsize=7, va="bottom", ha="right", bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_xlabel("mismatch size eps = ||W_hat - W||_F")
        ax.set_ylabel("relative error |approx - exact| / exact")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    ax = axes[1, 1]
    show = [("skewed", "asymmetric_kron", "range_optimistic"), ("uniform", "symmetric_kron", "range_optimistic")]
    for (pr, ch, mm), ls in zip(show, ("-", "--")):
        r = next(x for x in alpha_rows if (x["prior"], x["channel"], x["mismatch"]) == (pr, ch, mm))
        case = f"{pr} prior, {ch.replace('_', ' ')}"
        ax.plot(r["alpha"], r["L_exact"], ls, color="k", lw=2, label=f"exact  [{case}]")
        ax.plot(r["alpha"], r["L_quad"], ls, color="tab:orange", marker="o", ms=3, label=f"quadratic  [{case}]")
        ax.plot(r["alpha"], r["L_balanced"], ls, color="tab:purple", marker="x", ms=4,
                label=f"balanced trace  [{case}]")
    ax.text(0.98, 0.05, "dashed (balanced case): all three coincide", transform=ax.transAxes, ha="right", fontsize=7)
    ax.set_xlabel("sensing allocation alpha")
    ax.set_ylabel("mismatch loss L_mis")
    ax.set_title(f"(d) L_mis vs alpha, range-optimistic receiver (t = {T_OPERATING})", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("General-model validation: non-uniform priors and asymmetric channels")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig5_general_model_validation.png", dpi=150)
    plt.close(fig)
    print("5. Saved figures/fig5_general_model_validation.png")


def summary_table(eps_summary, alpha_rows):
    lines = ["| Prior | eps | Quadratic rel. error (median / max) | Balanced trace rel. error (median / max) |",
             "|---|---|---|---|"]
    for s in eps_summary:
        lines.append(f"| {s['prior']} | {s['eps']:.2f} | {100 * s['rel_err_quad_median']:.2f}% / "
                     f"{100 * s['rel_err_quad_max']:.2f}% | {100 * s['rel_err_balanced_median']:.1f}% / "
                     f"{100 * s['rel_err_balanced_max']:.1f}% |")
    lines += ["", f"Operating mismatch t = {T_OPERATING}, sweep over {len(ALPHA_GRID)} alphas:", "",
              "| Prior | Channel | Mismatch | eps | Quadratic max rel. err | Balanced max rel. err | "
              "alpha* exact / quadratic / balanced |", "|---|---|---|---|---|---|---|"]
    for r in alpha_rows:
        lines.append(f"| {r['prior']} | {r['channel']} | {r['mismatch']} | {r['eps']:.3f} | "
                     f"{100 * r['rel_err_quad_max']:.1f}% | {100 * r['rel_err_balanced_max']:.1f}% | "
                     f"{r['argmin_exact']:.2f} / {r['argmin_quad']:.2f} / {r['argmin_balanced']:.2f} |")
    (ROOT / "results" / "general_model_table.md").write_text("\n".join(lines) + "\n")
    print("6. Saved results/general_model_table.md")


def save(name, payload):
    with open(ROOT / "results" / name, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)


def main():
    priors, channels, targets = build_cases()
    check = checks(priors, channels)
    eps_rows, slopes = eps_sweep(priors, channels, targets)
    eps_summary = summarize_eps(eps_rows)
    alpha_rows = alpha_sweep(priors, channels, targets)
    save("general_model.json", {
        "seed": [BASE_SEED, 5], "priors": priors, "channels": {k: v["W"] for k, v in channels.items()},
        "targets": targets, "t_grid": T_GRID, "t_operating": T_OPERATING, "checks": check,
        "loglog_slopes": slopes, "eps_sweep": eps_rows, "eps_summary": eps_summary, "alpha_sweep": alpha_rows,
    })
    make_figure(eps_rows, alpha_rows)
    summary_table(eps_summary, alpha_rows)
    print("\nAll results saved under results/ and figures/.")


if __name__ == "__main__":
    main()
