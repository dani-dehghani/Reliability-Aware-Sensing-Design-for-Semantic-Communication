import numpy as np
import pytest

from src.channel import actual_channel
from src.design import (
    alpha_star_map, argmax_with_ties, mixed_l_mis, pareto_front_mask, pareto_selection,
    task_vs_mismatch_sweep,
)
from src.metrics import J_lambda, R_task_direct, V_prob, prior_risk
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensitivity import GW_matrix, Gs_matrix, R_task_trace

CFG = dict(seed=2026, C=4, a_r=1.0, a_v=1.0, sigma0_sq=1.0, eps=0.1,
           p_r=0.10, p_v=0.10, p_hat_low=0.02, n_alpha=21, n_mc=1000)


def test_prior_risk_matches_known_value():
    pi = np.full(4, 0.25)
    assert prior_risk(pi) == pytest.approx(0.75)


def test_r_task_direct_matches_trace_formula():
    args = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
    pi = np.full(CFG["C"], 1.0 / CFG["C"])
    W = actual_channel(CFG["p_r"], CFG["p_v"])
    max_gap = 0.0
    for a in np.linspace(0.0, 1.0, CFG["n_alpha"]):
        S = analytic_S(a, *args)
        A = end_to_end(S, W)
        rho = posterior(pi, A)
        gap = abs(R_task_direct(pi, rho, A) - R_task_trace(S, W))
        max_gap = max(max_gap, gap)
    assert max_gap < 1e-10


def test_r_task_is_zero_for_uninformative_channel():
    
    C = 4
    J = np.full((C, C), 1.0 / C)
    S = analytic_S(0.5, CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
    assert R_task_trace(S, J) == pytest.approx(0.0, abs=1e-12)


def test_gw_matrix_is_symmetric_positive_semidefinite():
    W = actual_channel(CFG["p_r"], CFG["p_v"])
    Gw = GW_matrix(W)
    assert np.allclose(Gw, Gw.T)
    assert np.linalg.eigvalsh(Gw).min() > -1e-12


def test_mixed_l_mis_endpoints_match_pure_directions():
    l_r, l_v = np.array([0.003313]), np.array([0.000520])
    assert mixed_l_mis(l_r, l_v, q=1.0)[0] == pytest.approx(l_r[0])
    assert mixed_l_mis(l_r, l_v, q=0.0)[0] == pytest.approx(l_v[0])


def test_pareto_selection_runs_and_picks_are_within_range():
    sweep = task_vs_mismatch_sweep(CFG)
    result = pareto_selection(sweep, lambda_grid=[0.0, 1.0, 5.0], q=1.0)
    for pick in result["picks"].values():
        assert 0.0 <= pick["alpha"] <= 1.0


def test_j_lambda_zero_equals_r_task():
    assert J_lambda(0.1110, 0.0024, lam=0.0) == pytest.approx(0.1110)


def test_v_prob_equals_j_lambda_at_lambda_one():
    r, l = 0.1110, 0.0024
    assert V_prob(r, l) == pytest.approx(J_lambda(r, l, lam=1.0))


def test_argmax_with_ties_reports_exact_ties_only():
    assert argmax_with_ties([0.1, 0.3, 0.2, 0.3]) == (1, [1, 3])
    assert argmax_with_ties([0.1, 0.3, 0.3 - 1e-6]) == (1, [1])


def test_pareto_front_mask_drops_dominated_points():
    R = np.array([1.0, 2.0, 2.0, 1.5])
    L = np.array([1.0, 2.0, 3.0, 2.5])
    # point 2 is dominated by 1 (same R, more L); point 3 by 1 (less R, more L)
    assert pareto_front_mask(R, L).tolist() == [True, True, False, False]


def test_sweep_trace_and_lambda_max_columns_match_gs():
    sweep = task_vs_mismatch_sweep(CFG)
    args = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
    for i, a in enumerate(sweep["alpha"]):
        Gs = Gs_matrix(analytic_S(a, *args))
        assert sweep["tr_Gs"][i] == pytest.approx(np.trace(Gs))
        assert sweep["lam_max_Gs"][i] == pytest.approx(np.linalg.eigvalsh(Gs).max())
        assert sweep["lam_max_Gs"][i] <= sweep["tr_Gs"][i] + 1e-12


def test_symmetric_mismatch_tie_is_reported_not_broken():
    
    sweep = task_vs_mismatch_sweep(CFG)
    picks = pareto_selection(sweep, lambda_grid=[], q=0.5)["picks"]
    assert picks["min_L_mis"]["alpha_ties"] == [0.0, 1.0]


def test_range_mismatch_moves_alpha_star_toward_velocity_as_lambda_grows():
    sweep = task_vs_mismatch_sweep(CFG)
    rows = alpha_star_map(sweep, [0.0, 5.0, 10.0, 100.0], [1.0])
    a_star = [r["alpha_star"] for r in rows]
    assert a_star[0] == pytest.approx(0.5)
    assert all(x >= y for x, y in zip(a_star, a_star[1:]))
    assert a_star[-1] < 0.5


def test_alpha_star_map_covers_full_grid():
    sweep = task_vs_mismatch_sweep(CFG)
    lambda_grid = [0.0, 1.0]
    q_grid = np.linspace(0.0, 1.0, 5)
    rows = alpha_star_map(sweep, lambda_grid, q_grid)
    assert len(rows) == len(lambda_grid) * len(q_grid)
    assert all(0.0 <= r["alpha_star"] <= 1.0 for r in rows)