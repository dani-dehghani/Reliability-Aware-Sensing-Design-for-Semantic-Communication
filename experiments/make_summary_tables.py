import json
import math
from pathlib import Path

import numpy as np

from src.channel import actual_channel, nominal_channel
from src.metrics import L_log, L_mis_direct, R_task_direct, hard_accuracy, info_log
from src.posterior import end_to_end, posterior
from src.probing import (
    allocate_sensitivity_general, floor_rows, l_mis_between, probe_channel, required_probes,
)
from src.semantic import analytic_S

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "baseline.json"))
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
PI = np.full(CFG["C"], 1.0 / CFG["C"])
W = actual_channel(CFG["p_r"], CFG["p_v"])
W_HAT = nominal_channel(CFG["p_r"], CFG["p_v"], "range_optimistic", CFG["p_hat_low"])
ALPHAS = np.round(np.linspace(0.0, 1.0, CFG["n_alpha"]), 2)

TARGET_LOSS = 1e-4         
EPS_ROBUST = 0.10          
LAMBDA_LOG_MATCH = 10.0    
N_MC, N_MIN, PROB_FLOOR = 1000, 1, 1e-6


def load(name):
    path = ROOT / "results" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run the experiment that produces it first")
    return json.load(open(path))


def evaluate(alpha):
    S = analytic_S(alpha, *ARGS)
    A = end_to_end(S, W)
    rho, rho_hat = posterior(PI, A), posterior(PI, end_to_end(S, W_HAT))
    return {"accuracy": hard_accuracy(PI, rho, A), "R_task": R_task_direct(PI, rho, A),
            "L_mis": L_mis_direct(PI, rho, rho_hat, A), "L_log": L_log(PI, rho, rho_hat, A),
            "I_log": info_log(PI, rho, A),
            "N_uniform": required_probes(S, PI, W, TARGET_LOSS, "uniform"),
            "N_sensitivity": required_probes(S, PI, W, TARGET_LOSS, "sensitivity")}


def mc_check(alpha, N, seed):
    
    S = analytic_S(alpha, *ARGS)
    n = allocate_sensitivity_general(N, S, PI, W, N_MIN)
    rng = np.random.default_rng(seed)
    losses = [l_mis_between(S, PI, W, floor_rows(probe_channel(W, n, rng), PROB_FLOOR)) for _ in range(N_MC)]
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(N_MC))


def criteria():
    
    picks = load("ext1_pareto.json")["cases"]["range_oriented"]["picks"]
    rows = [("max hard accuracy", "Ext. 1", picks["max_accuracy"]["alpha"]),
            ("max R_task (information only)", "Ext. 1", picks["max_R_task"]["alpha"]),
            ("max V_prob = R_task - L_mis", "Ext. 1", picks["max_V_prob"]["alpha"])]
    for lam in (5, 10, 20):
        rows.append((f"max J_lambda, lambda = {lam}", "Ext. 1", picks[f"max_J_lambda_{lam}"]["alpha"]))
    rows.append(("min L_mis (reliability only)", "Ext. 1", picks["min_L_mis"]["alpha"]))

    agree = load("ext2_design_agreement.json")
    for cset, label in (("doubly", "doubly-stochastic"), ("row", "row-stochastic")):
        r = next(x for x in agree if x["regime"] == "near_boundary" and x["channel_set"] == cset
                 and x["epsilon"] == EPS_ROBUST)
        if cset == "doubly":
            rows.append((f"min spectral worst case, eps = {EPS_ROBUST}", "Ext. 2", r["alpha_safest_spec"][0]))
        rows.append((f"min realizable worst case, {label}, eps = {EPS_ROBUST}", "Ext. 2", r["alpha_safest_real"][0]))

    
    ev = [evaluate(a) for a in ALPHAS]
    i_mid = int(np.argmin(np.abs(ALPHAS - 0.5)))
    R_max = max(e["R_task"] for e in ev)
    I_max = max(e["I_log"] for e in ev)
    lt = LAMBDA_LOG_MATCH * ev[i_mid]["L_mis"] / R_max
    j_log = [e["I_log"] / I_max - lt * e["L_log"] / ev[i_mid]["L_log"] for e in ev]
    rows.append((f"max I(X;M) - lambda L_log, weight matched to lambda = {LAMBDA_LOG_MATCH:g}", "Ext. 4",
                 float(ALPHAS[int(np.argmax(j_log))])))
    rows.append(("min L_log", "Ext. 4", float(ALPHAS[int(np.argmin([e["L_log"] for e in ev]))])))
    return rows


def main():
    rows = []
    for k, (name, ext, alpha) in enumerate(criteria()):
        e = evaluate(alpha)
        N = math.ceil(e["N_sensitivity"])
        mc_mean, mc_se = mc_check(alpha, N, [CFG["seed"], 8, k])
        rows.append({"criterion": name, "extension": ext, "alpha": alpha, **e,
                     "N_required": N, "mc_loss_at_N": mc_mean, "mc_se": mc_se})

    ref = next(r for r in rows if r["criterion"].startswith("max R_task"))
    print(f"Extension summary (q = 1, range-optimistic receiver; probe target: excess Brier loss <= {TARGET_LOSS:g})")
    hdr = f"{'criterion':62s} {'ext':6s} {'alpha':>5s} {'acc':>7s} {'R_task':>7s} {'L_mis':>8s} {'dL_mis':>6s} {'N_req':>6s} {'MC/target':>9s}"
    print(hdr)
    for r in rows:
        print(f"{r['criterion']:62s} {r['extension']:6s} {r['alpha']:5.2f} {r['accuracy']:7.4f} {r['R_task']:7.4f} "
              f"{r['L_mis']:8.5f} {100 * (r['L_mis'] / ref['L_mis'] - 1):+5.0f}% {r['N_required']:6d} "
              f"{r['mc_loss_at_N'] / TARGET_LOSS:6.2f}+-{r['mc_se'] / TARGET_LOSS:.2f}")
    sav = max(1 - r["N_sensitivity"] / r["N_uniform"] for r in rows)
    print(f"Sensitivity-aware vs uniform probing: at most {100 * sav:.1f}% fewer probes here "
          f"(baseline channel is token-symmetric)")

    lines = [
        f"Setting: Extension 1 baseline (uniform prior, W = B({CFG['p_r']}) kron B({CFG['p_v']})), range-optimistic "
        f"receiver (p_hat_r = {CFG['p_hat_low']}, q = 1). Every row is evaluated in this same setting.",
        f"Required probe budget: known-token probes (sensitivity-aware allocation) so that a receiver "
        f"using the estimated channel has expected excess Brier loss <= {TARGET_LOSS:g}. Predicted by the "
        f"quadratic approximation; the last column is the Monte Carlo mean at that budget ({N_MC} runs) "
        f"divided by the target.", "",
        "| Criterion | Ext. | alpha | Hard accuracy | R_task | L_mis | L_mis vs info-only | L_log | "
        "Required probes | MC loss / target |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['criterion']} | {r['extension']} | {r['alpha']:.2f} | {r['accuracy']:.4f} | "
                     f"{r['R_task']:.4f} | {r['L_mis']:.5f} | {100 * (r['L_mis'] / ref['L_mis'] - 1):+.0f}% | "
                     f"{r['L_log']:.5f} | {r['N_required']:,} | {r['mc_loss_at_N'] / TARGET_LOSS:.2f} +- "
                     f"{r['mc_se'] / TARGET_LOSS:.2f} |")
    lines += ["", f"Sensitivity-aware probing needs at most {100 * sav:.1f}% fewer probes than uniform here because "
                  "the baseline channel is token-symmetric; the saving appears for token-dependent channels (Ext. 3, "
                  "Fig. 7b)."]
    (ROOT / "results" / "extension_summary_table.md").write_text("\n".join(lines) + "\n")
    with open(ROOT / "results" / "extension_summary.json", "w") as f:
        json.dump({"target_loss": TARGET_LOSS, "eps_robust": EPS_ROBUST, "lambda_log_match": LAMBDA_LOG_MATCH,
                   "n_mc": N_MC, "prob_floor": PROB_FLOOR, "rows": rows}, f, indent=2)

    control = (ROOT / "results" / "control_table.md").read_text()
    combined = ["# Summary tables", "",
                "## Table 1: matched-control (similar accuracy, different mismatch vulnerability)", "",
                "From exp00 (range-optimistic receiver).", "", control.strip(), "",
                "## Table 2: extension summary", ""] + lines
    (ROOT / "results" / "summary_tables.md").write_text("\n".join(combined) + "\n")
    print("Saved results/extension_summary_table.md, results/extension_summary.json, results/summary_tables.md")


if __name__ == "__main__":
    main()
