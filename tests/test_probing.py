import numpy as np
import pytest

from src.channel import actual_channel
from src.probing import (
    allocate_frequency, allocate_sensitivity, allocate_sensitivity_general, allocate_two_stage,
    allocate_uniform, analytic_probe_saving, expected_L_est_approx, expected_L_est_general,
    floor_rows, l_mis_between, probe_channel, round_preserving_total, sensing_budget_params,
    sensitivity_weights, sensitivity_weights_general, token_dependent_channel,
)
from src.semantic import analytic_S

CFG_ARGS = (1.0, 1.0, 1.0, 0.1)
PI_U = np.full(4, 0.25)
PI_NU = np.array([0.40, 0.30, 0.20, 0.10])
W_SYM = actual_channel(0.10, 0.10)
W_TOK = token_dependent_channel([(0.02, 0.02), (0.06, 0.06), (0.14, 0.14), (0.18, 0.18)])


def _mc_loss(S, pi, W, n, rng, reps=800):
    return float(np.mean([l_mis_between(S, pi, W, floor_rows(probe_channel(W, n, rng), 1e-6)) for _ in range(reps)]))


def test_probe_estimate_is_a_valid_channel():
    rng = np.random.default_rng(0)
    W_tilde = probe_channel(W_TOK, np.array([5, 7, 9, 11]), rng)
    assert W_tilde.shape == W_TOK.shape
    assert W_tilde.min() >= 0.0
    assert np.allclose(W_tilde.sum(axis=1), 1.0)


def test_probe_rejects_unobserved_rows():
    with pytest.raises(ValueError):
        probe_channel(W_TOK, np.array([5, 0, 5, 5]), np.random.default_rng(0))


def test_probe_estimate_converges_to_W():
    rng = np.random.default_rng(1)
    Ns = [16, 128, 1024, 8192]
    errs = []
    for N in Ns:
        n = allocate_uniform(N, 4, 2)
        errs.append(np.mean([np.linalg.norm(probe_channel(W_TOK, n, rng) - W_TOK) for _ in range(200)]))
    assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:]))
    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.07)


def test_rounding_preserves_total_and_minimum():
    for N in (8, 13, 100, 1001):
        n = round_preserving_total(np.array([0.1, 0.2, 0.3, 0.4]), N, 2)
        assert n.sum() == N and n.min() >= 2
    with pytest.raises(ValueError):
        round_preserving_total(np.ones(4), 7, 2)


def test_symmetric_benchmark_makes_all_allocations_coincide():
    S = analytic_S(0.5, *CFG_ARGS)
    N = 200
    assert np.array_equal(allocate_sensitivity(N, S, W_SYM, 2), allocate_uniform(N, 4, 2))
    assert np.array_equal(allocate_frequency(N, S, PI_U, 2), allocate_uniform(N, 4, 2))
    assert analytic_probe_saving(sensitivity_weights(S, W_SYM)) == pytest.approx(0.0, abs=1e-12)


def test_token_dependent_channel_is_valid_and_reduces_to_kronecker():
    assert np.allclose(W_TOK.sum(axis=1), 1.0) and W_TOK.min() >= 0
    W_same = token_dependent_channel([(0.1, 0.1)] * 4)
    assert np.allclose(W_same, W_SYM)


def test_sensitivity_allocation_follows_sqrt_of_weights():
    S = analytic_S(0.5, *CFG_ARGS)
    a = sensitivity_weights(S, W_TOK)
    n = allocate_sensitivity(4000, S, W_TOK, 2)
    assert np.all(np.diff(n) > 0)
    assert np.allclose(n / n.sum(), np.sqrt(a) / np.sqrt(a).sum(), atol=0.01)


def test_two_stage_uses_the_whole_budget_and_returns_valid_estimate():
    S = analytic_S(0.5, *CFG_ARGS)
    for N in (8, 40, 500):
        W_tilde, n = allocate_two_stage(N, S, W_TOK, np.random.default_rng(2), 0.2, 2)
        assert n.sum() == N and n.min() >= 2
        assert np.allclose(W_tilde.sum(axis=1), 1.0) and W_tilde.min() >= 0


def test_balanced_prediction_matches_monte_carlo():
    S = analytic_S(0.5, *CFG_ARGS)
    n = allocate_uniform(256, 4, 2)
    mc = _mc_loss(S, PI_U, W_SYM, n, np.random.default_rng(3))
    assert mc == pytest.approx(expected_L_est_approx(S, W_SYM, n), rel=0.10)


def test_general_weights_reduce_to_balanced_weights():
    S = analytic_S(0.5, *CFG_ARGS)
    a_bal = sensitivity_weights(S, W_SYM) / S.shape[0]
    a_gen = sensitivity_weights_general(S, PI_U, W_SYM)
    assert np.allclose(a_bal, a_gen, rtol=1e-2)


def test_balanced_prediction_is_accurate_with_uniform_prior_for_token_dependent_channel():
    S = analytic_S(0.5, *CFG_ARGS)
    n = allocate_uniform(256, 4, 2)
    mc = _mc_loss(S, PI_U, W_TOK, n, np.random.default_rng(4))
    assert mc == pytest.approx(expected_L_est_approx(S, W_TOK, n), rel=0.10)


def test_balanced_prediction_breaks_under_nonuniform_prior_and_general_weights_fix_it():
    S = analytic_S(0.5, *CFG_ARGS)
    n = allocate_uniform(512, 4, 2)
    mc = _mc_loss(S, PI_NU, W_TOK, n, np.random.default_rng(5), reps=1500)
    assert mc / expected_L_est_approx(S, W_TOK, n) < 0.85
    assert mc == pytest.approx(expected_L_est_general(S, PI_NU, W_TOK, n), rel=0.08)


def test_sensitivity_allocation_beats_uniform_when_its_weights_are_valid():
    S = analytic_S(0.5, *CFG_ARGS)
    rng = np.random.default_rng(6)
    N = 512
    l_u = _mc_loss(S, PI_U, W_TOK, allocate_uniform(N, 4, 2), rng, reps=1500)
    l_s = _mc_loss(S, PI_U, W_TOK, allocate_sensitivity(N, S, W_TOK, 2), rng, reps=1500)
    assert l_s < l_u
    saving = analytic_probe_saving(sensitivity_weights(S, W_TOK))
    assert 0.0 < saving < 0.3
    assert (1.0 - l_s / l_u) == pytest.approx(saving, abs=0.05)


def test_general_allocation_does_not_lose_to_uniform_under_nonuniform_prior():
    S = analytic_S(0.5, *CFG_ARGS)
    N = 1024
    n_u = allocate_uniform(N, 4, 2)
    n_g = allocate_sensitivity_general(N, S, PI_NU, W_TOK, 2)
    assert expected_L_est_general(S, PI_NU, W_TOK, n_g) <= expected_L_est_general(S, PI_NU, W_TOK, n_u) + 1e-12


def test_floor_rows_removes_zeros_and_keeps_rows_stochastic():
    W = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.25] * 4, [0.0, 0.0, 0.0, 1.0]])
    F = floor_rows(W, 1e-6)
    assert F.min() > 0 and np.allclose(F.sum(axis=1), 1.0)
    assert np.allclose(F, W, atol=1e-5)


def test_sensing_budget_params_endpoints():
    s0, N0 = sensing_budget_params(0.0, 1.0, 0.1, 8, 2000)
    s1, N1 = sensing_budget_params(1.0, 1.0, 0.1, 8, 2000)
    assert s0 == pytest.approx(10.0) and N0 == 2008
    assert s1 == pytest.approx(1.0 / 1.1) and N1 == 8
    with pytest.raises(ValueError):
        sensing_budget_params(1.5, 1.0, 0.1, 8, 2000)