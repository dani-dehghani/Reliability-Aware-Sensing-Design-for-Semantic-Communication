import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from experiments import exp02_general_model as ex2
from experiments import exp04_probing as ex4
from src.channel import actual_channel, nominal_channel
from src.metrics import L_log, L_mis_direct, R_task_direct, info_log
from src.posterior import end_to_end, posterior
from src.probing import floor_rows
from src.semantic import analytic_S

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "baseline.json"))
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
PI_U = np.full(CFG["C"], 1.0 / CFG["C"])
MODES = ("range_optimistic", "velocity_optimistic")

ALPHAS = np.round(np.linspace(0.0, 1.0, 101), 2)
# normalized weight: J = Info/Info_max - lt * L/L(alpha=0.5), comparable across both scores
LT_GRID = np.concatenate([[0.0], np.logspace(-2.5, 1, 36)])
Q_CURVES = (1.0, 0.75)
# no posterior floor needed in parts A and C; part B uses exp04's channel floor
POSTERIOR_FLOOR = None
FAMILIES = {"range_optimistic": "tab:blue", "optimistic": "tab:red", "random": "tab:gray"}
FAMILY_LABELS = {"range_optimistic": "range-optimistic",
                 "optimistic": "overconfident (W_hat -> I)", "random": "random direction"}


def save(name, payload):
    with open(ROOT / "results" / name, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)


def both_losses(S, pi, W, W_hat):
    A = end_to_end(S, W)
    rho, rho_hat = posterior(pi, A), posterior(pi, end_to_end(S, W_hat))
    return (L_mis_direct(pi, rho, rho_hat, A), L_log(pi, rho, rho_hat, A, POSTERIOR_FLOOR),
            float(rho_hat.min()))


def worst_kl_term(S, pi, W, W_hat):
    """Received symbol with the largest p_m * KL term, with its true and receiver posteriors."""
    A = end_to_end(S, W)
    rho, rho_hat = posterior(pi, A), posterior(pi, end_to_end(S, W_hat))
    p = pi @ A
    terms = [p[m] * np.sum(rho[:, m] * np.log(rho[:, m] / rho_hat[:, m])) for m in range(len(p))]
    m = int(np.argmax(terms))
    return {"m": m, "p_m": float(p[m]), "share_of_L_log": float(terms[m] / sum(terms)),
            "rho_true": rho[:, m].round(4).tolist(), "rho_hat": rho_hat[:, m].round(4).tolist()}




def part_a():
    W = actual_channel(CFG["p_r"], CFG["p_v"])
    R_task, I_log = [], []
    brier = {m: [] for m in MODES}
    logl = {m: [] for m in MODES}
    min_rho_hat = 1.0
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        A = end_to_end(S, W)
        rho = posterior(PI_U, A)
        R_task.append(R_task_direct(PI_U, rho, A))
        I_log.append(info_log(PI_U, rho, A))
        for mode in MODES:
            b, l, mn = both_losses(S, PI_U, W, nominal_channel(CFG["p_r"], CFG["p_v"], mode, CFG["p_hat_low"]))
            brier[mode].append(b), logl[mode].append(l)
            min_rho_hat = min(min_rho_hat, mn)
    R_task, I_log = np.array(R_task), np.array(I_log)
    brier = {k: np.array(v) for k, v in brier.items()}
    logl = {k: np.array(v) for k, v in logl.items()}
    i_mid = int(np.argmin(np.abs(ALPHAS - 0.5)))

    per_mode = {}
    for mode in MODES:
        b, l = brier[mode], logl[mode]
        per_mode[mode] = {"spearman": float(spearmanr(b, l).statistic),
                          "argmin_brier": float(ALPHAS[b.argmin()]), "argmin_log": float(ALPHAS[l.argmin()]),
                          "argmax_brier": float(ALPHAS[b.argmax()]), "argmax_log": float(ALPHAS[l.argmax()]),
                          "ratio_log_over_brier_min": float((l / b).min()),
                          "ratio_log_over_brier_max": float((l / b).max())}

    
    curves = {}
    for q in Q_CURVES:
        lb = q * brier["range_optimistic"] + (1 - q) * brier["velocity_optimistic"]
        ll = q * logl["range_optimistic"] + (1 - q) * logl["velocity_optimistic"]
        rows = []
        for lt in LT_GRID:
            jb = R_task / R_task.max() - lt * lb / lb[i_mid]
            jl = I_log / I_log.max() - lt * ll / ll[i_mid]
            ib, il = int(jb.argmax()), int(jl.argmax())
            rows.append({"lt": float(lt), "alpha_star_brier": float(ALPHAS[ib]), "alpha_star_log": float(ALPHAS[il]),
                         "accuracy_cost_proxy_R_task": float(R_task[ib]), "R_task_at_log_pick": float(R_task[il])})
        curves[f"q={q:g}"] = rows

    print("A. Extension 1 sweep (101 alphas, uniform prior):")
    print(f"   smallest receiver posterior entry {min_rho_hat:.3f}")
    print(f"   R_task vs I(X;M): argmax {ALPHAS[R_task.argmax()]:.2f} / "
          f"{ALPHAS[I_log.argmax()]:.2f}, Spearman {spearmanr(R_task, I_log).statistic:.4f}")
    for mode, d in per_mode.items():
        print(f"   {mode:20s}: Spearman {d['spearman']:.4f}; least vulnerable alpha "
              f"{d['argmin_brier']:.2f} / {d['argmin_log']:.2f}; L_log/L_Brier {d['ratio_log_over_brier_min']:.2f}"
              f"..{d['ratio_log_over_brier_max']:.2f}")
    r = curves["q=1"]
    first_diff = next((x for x in r if abs(x["alpha_star_brier"] - x["alpha_star_log"]) > 0.051), None)
    if first_diff:
        print(f"   J-rule (q=1) picks first differ by >0.05 at weight {first_diff['lt']:.3f}: "
              f"Brier alpha*={first_diff['alpha_star_brier']:.2f}, log alpha*={first_diff['alpha_star_log']:.2f}")
    # posteriors where L_log / L_Brier is largest
    mode = "range_optimistic"
    i_hi = int(np.argmax(logl[mode] / brier[mode]))
    inspect = worst_kl_term(analytic_S(ALPHAS[i_hi], *ARGS), PI_U, W,
                            nominal_channel(CFG["p_r"], CFG["p_v"], mode, CFG["p_hat_low"]))
    print(f"   largest L_log/L_Brier at alpha={ALPHAS[i_hi]:.2f}: symbol m={inspect['m']} gives "
          f"{100 * inspect['share_of_L_log']:.0f}% of L_log; rho={inspect['rho_true']}, rho_hat={inspect['rho_hat']}")
    return {"alpha": ALPHAS, "R_task": R_task, "I_log": I_log,
            "L_brier": brier, "L_log": logl, "per_mode": per_mode, "j_curves": curves,
            "min_rho_hat": min_rho_hat, "inspect_max_ratio": {"alpha": float(ALPHAS[i_hi]), **inspect}}




def part_b():
    """Same loop and random streams as exp04, so the channel estimates are identical."""
    S = analytic_S(ex4.ALPHA_PROBE, *ex4.ARGS)
    W = ex4.deployment_channel(ex4.H_MAIN)
    results = {}
    worst_inf = 0
    for r_idx, (regime, pi) in enumerate(ex4.REGIMES.items()):
        rng = np.random.default_rng([ex4.BASE_SEED, 2, r_idx])
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        rows = []
        for N in ex4.N_GRID:
            acc = {}
            for _ in range(ex4.N_REP):
                for name, (n, W_tilde) in ex4.allocations_for(N, S, pi, W, rng).items():
                    rho_hat = posterior(pi, end_to_end(S, floor_rows(W_tilde, ex4.PROB_FLOOR)))
                    lb = L_mis_direct(pi, rho, rho_hat, A)
                    ll = L_log(pi, rho, rho_hat, A)
                    worst_inf += not np.isfinite(ll)
                    d = acc.setdefault(name, {"brier": [], "log": []})
                    d["brier"].append(lb), d["log"].append(ll)
            row = {"N_probe": N}
            for name, d in acc.items():
                row[name] = {"brier_mean": float(np.mean(d["brier"])), "log_mean": float(np.mean(d["log"])),
                             "brier_std": float(np.std(d["brier"])), "log_std": float(np.std(d["log"])),
                             "log_p99_over_mean": float(np.percentile(d["log"], 99) / np.mean(d["log"]))}
            rows.append(row)
        results[regime] = rows

   
    gap = None
    ref_path = ROOT / "results" / "ext3_budget_scaling.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
        gap = max(abs(row[name]["brier_mean"] - ref_row[name]["L_adopt_mean"]) / ref_row[name]["L_adopt_mean"]
                  for regime in results for row, ref_row in zip(results[regime], ref["regimes"][regime]["rows"])
                  for name in ex4.STYLE)

    rules = list(ex4.STYLE)

    def tied_best(row, score):
        # rules within 2 SE of the best one
        se = {k: row[k][f"{score}_std"] / np.sqrt(ex4.N_REP) for k in rules}
        best = min(rules, key=lambda k: row[k][f"{score}_mean"])
        return sorted(k for k in rules if row[k][f"{score}_mean"] - row[best][f"{score}_mean"]
                      <= 2.0 * np.hypot(se[k], se[best]))

    def pooled_ratio(rows, name, score):
        # loss relative to uniform probing, averaged over N >= 128
        r = [row[name][f"{score}_mean"] / row["uniform"][f"{score}_mean"] for row in rows if row["N_probe"] >= 128]
        return float(np.mean(r)), float(np.std(r) / np.sqrt(len(r)))

    summary = {}
    print(f"B. Extension 3 probing ({ex4.N_REP} reps, alpha={ex4.ALPHA_PROBE}, h={ex4.H_MAIN}, "
          f"channel floor {ex4.PROB_FLOOR:g}; infinite KL values: {worst_inf})")
    if gap is not None:
        print(f"   Brier values match ext3_budget_scaling.json (max rel. diff {gap:.0e})")
    for regime, rows in results.items():
        per_n = []
        for row in rows:
            order_b = sorted(rules, key=lambda k: row[k]["brier_mean"])
            order_l = sorted(rules, key=lambda k: row[k]["log_mean"])
            per_n.append({"N_probe": row["N_probe"],
                          "spearman_rules": float(spearmanr([row[k]["brier_mean"] for k in rules],
                                                            [row[k]["log_mean"] for k in rules]).statistic),
                          "best_brier": order_b[0], "best_log": order_l[0],
                          "tied_best_brier": tied_best(row, "brier"), "tied_best_log": tied_best(row, "log"),
                          "saving_sens_general_brier": 1 - row["sens_general_oracle"]["brier_mean"] / row["uniform"]["brier_mean"],
                          "saving_sens_general_log": 1 - row["sens_general_oracle"]["log_mean"] / row["uniform"]["log_mean"],
                          "saving_two_stage_brier": 1 - row["sens_two_stage"]["brier_mean"] / row["uniform"]["brier_mean"],
                          "saving_two_stage_log": 1 - row["sens_two_stage"]["log_mean"] / row["uniform"]["log_mean"]})
        summary[regime] = {"per_N": per_n,
                           "pooled_ratio_to_uniform": {k: {"brier": pooled_ratio(rows, k, "brier"),
                                                           "log": pooled_ratio(rows, k, "log")}
                                                       for k in rules if k != "uniform"}}
        big = [p for p in per_n if p["N_probe"] >= 128]
        n_agree_stat = sum(bool(set(p["tied_best_brier"]) & set(p["tied_best_log"])) for p in big)
        print(f"   [{regime}] N>=128: rule-ranking Spearman {min(p['spearman_rules'] for p in big):.2f}.."
              f"{max(p['spearman_rules'] for p in big):.2f}; best rule agrees "
              f"{sum(p['best_brier'] == p['best_log'] for p in big)}/{len(big)} (within 2 SE: "
              f"{n_agree_stat}/{len(big)})")
        pr = summary[regime]["pooled_ratio_to_uniform"]
        print("      loss / uniform, N>=128 (Brier | log): " + "; ".join(
            f"{k}: {pr[k]['brier'][0]:.3f} | {pr[k]['log'][0]:.3f}" for k in pr))
        mid = next(p for p in per_n if p["N_probe"] == 512)
        print(f"      saving at N=512: general-prior sensitivity {100 * mid['saving_sens_general_brier']:+.1f}% (Brier) "
              f"/ {100 * mid['saving_sens_general_log']:+.1f}% (log); two-stage {100 * mid['saving_two_stage_brier']:+.1f}% "
              f"/ {100 * mid['saving_two_stage_log']:+.1f}%")
    return {"rows": results, "summary": summary, "brier_reproduction_rel_gap": gap, "n_infinite_kl": worst_inf}




def part_c():
    priors, channels, targets = ex2.build_cases()
    alphas = ex2.ALPHA_GRID
    cases = []
    for pr_name, pi in priors.items():
        for ch_name, ch in channels.items():
            W = ch["W"]
            for tg_name, W_t in targets[ch_name].items():
                W_hat = W + ex2.T_OPERATING * (W_t - W)
                b, l, mn = zip(*(both_losses(analytic_S(a, *ARGS), pi, W, W_hat) for a in alphas))
                b, l = np.array(b), np.array(l)
                ib, il = int(b.argmin()), int(l.argmin())
                case = {"prior": pr_name, "channel": ch_name, "mismatch": tg_name,
                        "family": "random" if tg_name.startswith("random") else tg_name,
                        "spearman": float(spearmanr(b, l).statistic),
                        "argmin_brier": float(alphas[ib]), "argmin_log": float(alphas[il]),
                        "log_regret_of_brier_pick": float(l[ib] / l[il] - 1),
                        "brier_regret_of_log_pick": float(b[il] / b[ib] - 1),
                        "min_rho_hat": float(min(mn)), "L_brier": b, "L_log": l}
                if ib != il or case["spearman"] < 0.9:
                    S_dis = analytic_S(alphas[il], *ARGS)
                    case["inspect"] = worst_kl_term(S_dis, pi, W, W_hat)
                cases.append(case)
    n = len(cases)
    agree = sum(c["argmin_brier"] == c["argmin_log"] for c in cases)
    rhos = np.array([c["spearman"] for c in cases])
    print(f"C. General priors / asymmetric channels: {n} cases, t={ex2.T_OPERATING}, {len(alphas)} alphas")
    print(f"   smallest receiver posterior entry {min(c['min_rho_hat'] for c in cases):.2e}")
    print(f"   least-vulnerable alpha agrees in {agree}/{n} cases; Spearman median {np.median(rhos):.3f}, "
          f"min {rhos.min():.3f}; {sum(rhos < 0.9)} cases below 0.9")
    for fam in FAMILIES:
        sub = [c for c in cases if c["family"] == fam]
        print(f"   {fam:17s}: picks agree {sum(c['argmin_brier'] == c['argmin_log'] for c in sub)}/{len(sub)}, "
              f"Spearman min {min(c['spearman'] for c in sub):+.3f}; max regret: Brier pick under log "
              f"{100 * max(c['log_regret_of_brier_pick'] for c in sub):.1f}%, log pick under Brier "
              f"{100 * max(c['brier_regret_of_log_pick'] for c in sub):.1f}%")
    dis = [c for c in cases if "inspect" in c]
    for c in sorted(dis, key=lambda c: -c["log_regret_of_brier_pick"])[:3]:
        i = c["inspect"]
        print(f"   disagreement {c['prior']}/{c['channel']}/{c['mismatch']}: alpha* Brier {c['argmin_brier']:.2f} vs "
              f"log {c['argmin_log']:.2f}; largest KL term m={i['m']} ({100 * i['share_of_L_log']:.0f}%): "
              f"rho={i['rho_true']} rho_hat={i['rho_hat']}")
    return cases


# ---- figure 9 and table ----

def make_figure(a, b, c):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    ax = axes[0, 0]
    i_mid = int(np.argmin(np.abs(a["alpha"] - 0.5)))
    for mode, color in zip(MODES, ("tab:red", "tab:blue")):
        lb, ll = a["L_brier"][mode], a["L_log"][mode]
        ax.plot(a["alpha"], lb / lb[i_mid], "-", color=color, label=f"Brier, {mode.replace('_', ' ')}")
        ax.plot(a["alpha"], ll / ll[i_mid], "--", color=color, label=f"log, {mode.replace('_', ' ')}")
    ax.set_xlabel("sensing allocation alpha")
    ax.set_ylabel("loss / loss at alpha = 0.5")
    ax.set_title("(a) Mismatch loss vs alpha", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for q, color in zip(Q_CURVES, ("k", "tab:gray")):
        rows = a["j_curves"][f"q={q:g}"]
        lt = [r["lt"] for r in rows if r["lt"] > 0]
        ax.semilogx(lt, [r["alpha_star_brier"] for r in rows if r["lt"] > 0], "-o", ms=3, color=color,
                    label=f"Brier, q = {q:g}")
        ax.semilogx(lt, [r["alpha_star_log"] for r in rows if r["lt"] > 0], "--s", ms=3, color=color, mfc="none",
                    label=f"log, q = {q:g}")
    ax.set_xlabel("normalized trade-off weight lambda")
    ax.set_ylabel("optimal alpha*")
    ax.set_title("(b) Design rule under both scores", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 2]
    jit = np.random.default_rng(0)
    for fam, color in FAMILIES.items():
        sub = [x for x in c if x["family"] == fam]
        ax.scatter([x["argmin_brier"] + jit.uniform(-0.02, 0.02) for x in sub],
                   [x["argmin_log"] + jit.uniform(-0.02, 0.02) for x in sub], s=30, facecolor="none",
                   edgecolor=color, lw=1.3, label=f"{FAMILY_LABELS[fam]} ({len(sub)})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="same design")
    ax.set_xlabel("least vulnerable alpha, Brier")
    ax.set_ylabel("least vulnerable alpha, log")
    ax.set_title("(c) General model: selected design (jittered)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.06))
    ax.grid(alpha=0.3)

    rules = [k for k in ex4.STYLE if k != "uniform"]
    short = {"frequency": "frequency", "sens_balanced_oracle": "sens. balanced\n(oracle)",
             "sens_general_oracle": "sens. general\n(oracle)", "sens_two_stage": "sens. two-stage"}
    for ax, regime, tag in zip(axes[1, :2], ex4.REGIMES, ("d", "e")):
        pr = b["summary"][regime]["pooled_ratio_to_uniform"]
        x = np.arange(len(rules))
        for off, score, color in ((-0.18, "brier", "tab:blue"), (0.18, "log", "tab:orange")):
            ax.bar(x + off, [pr[k][score][0] for k in rules], 0.36, yerr=[pr[k][score][1] for k in rules],
                   color=color, capsize=3, label="Brier" if score == "brier" else "log score")
        ax.axhline(1.0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short[k] for k in rules], fontsize=8)
        lo = min(pr[k][s][0] - pr[k][s][1] for k in rules for s in ("brier", "log"))
        hi = max(pr[k][s][0] + pr[k][s][1] for k in rules for s in ("brier", "log"))
        ax.set_ylim(lo - 0.03, hi + 0.03)
        ax.set_ylabel("loss / uniform probing (N >= 128)")
        ax.set_title(f"({tag}) Probing, {regime.replace('_', ' ')}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 2]
    order = sorted(range(len(c)), key=lambda i: c[i]["spearman"])
    ax.bar(range(len(c)), [c[i]["spearman"] for i in order], color=[FAMILIES[c[i]["family"]] for i in order])
    for fam, color in FAMILIES.items():
        ax.bar([0], [0], color=color, label=FAMILY_LABELS[fam])
    ax.axhline(0.9, color="k", ls=":", lw=1)
    ax.set_xlabel("case (sorted)")
    ax.set_ylabel("Spearman correlation over alpha")
    ax.set_title("(f) General model: ranking agreement", fontsize=10)
    ax.set_ylim(min(0, min(x["spearman"] for x in c)) - 0.05, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Scoring-rule stability: Brier vs log score")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig9_scoring_rule_stability.png", dpi=150)
    plt.close(fig)
    print("D. Saved figures/fig9_scoring_rule_stability.png")


def summary_table(a, b, c):
    lines = ["| Experiment | Ranking agreement (Spearman) | Selected design: Brier vs log | Note |", "|---|---|---|---|"]
    for mode, d in a["per_mode"].items():
        lines.append(f"| Ext. 1 sweep, {mode.replace('_', ' ')} receiver | {d['spearman']:.4f} | least vulnerable "
                     f"alpha {d['argmin_brier']:.2f} vs {d['argmin_log']:.2f} | L_log/L_Brier "
                     f"{d['ratio_log_over_brier_min']:.2f}-{d['ratio_log_over_brier_max']:.2f} |")
    for q, rows in a["j_curves"].items():
        diffs = [abs(r["alpha_star_brier"] - r["alpha_star_log"]) for r in rows]
        lines.append(f"| Ext. 1 J-rule, {q} | - | max alpha* gap {max(diffs):.2f} over the weight grid | "
                     "log moves away from the misjudged mode at smaller weights |")
    for regime, s in b["summary"].items():
        big = [p for p in s["per_N"] if p["N_probe"] >= 128]
        pr = s["pooled_ratio_to_uniform"]
        n_stat = sum(bool(set(p["tied_best_brier"]) & set(p["tied_best_log"])) for p in big)
        lines.append(f"| Ext. 3 probing, {regime.replace('_', ' ')} | {min(p['spearman_rules'] for p in big):.2f}-"
                     f"{max(p['spearman_rules'] for p in big):.2f} (5 rules, N>=128) | best rule agrees within 2 SE at "
                     f"{n_stat}/{len(big)} budgets | loss/uniform, two-stage: {pr['sens_two_stage']['brier'][0]:.3f} "
                     f"(Brier), {pr['sens_two_stage']['log'][0]:.3f} (log); frequency: "
                     f"{pr['frequency']['brier'][0]:.3f} / {pr['frequency']['log'][0]:.3f} |")
    for fam in FAMILIES:
        sub = [x for x in c if x["family"] == fam]
        rhos = [x["spearman"] for x in sub]
        lines.append(f"| General model, {FAMILY_LABELS[fam]} ({len(sub)} cases) | median {np.median(rhos):.3f}, "
                     f"min {min(rhos):+.3f} | least vulnerable alpha agrees "
                     f"{sum(x['argmin_brier'] == x['argmin_log'] for x in sub)}/{len(sub)} | max regret: Brier pick "
                     f"under log {100 * max(x['log_regret_of_brier_pick'] for x in sub):.1f}%, log pick under Brier "
                     f"{100 * max(x['brier_regret_of_log_pick'] for x in sub):.1f}% |")
    (ROOT / "results" / "ext4_summary_table.md").write_text("\n".join(lines) + "\n")
    print("E. Saved results/ext4_summary_table.md")


def main():
    a = part_a()
    b = part_b()
    c = part_c()
    save("ext4_scoring_rules.json", {
        "posterior_floor_parts_A_C": POSTERIOR_FLOOR, "channel_floor_part_B": ex4.PROB_FLOOR,
        "lt_grid": LT_GRID, "part_A_ext1_sweep": a, "part_B_probing": b, "part_C_general": c,
    })
    make_figure(a, b, c)
    summary_table(a, b, c)
    print("\nAll results saved under results/ and figures/.")


if __name__ == "__main__":
    main()
