import numpy as np
import pytest

from src.channel import nominal_channel
from src.robust import (
    L_real_wc, feasible_direction, is_valid_channel, project_channel,
    project_doubly_stochastic, project_row_to_simplex, project_to_set,
)
from src.semantic import analytic_S
from src.sensitivity import worst_case_spectral

CFG_ARGS = (1.0, 1.0, 1.0, 0.1)


def test_simplex_projection_of_valid_row_is_identity():
    row = np.array([0.7, 0.1, 0.1, 0.1])
    assert np.allclose(project_row_to_simplex(row), row)


def test_simplex_projection_fixes_negative_and_wrong_sum():
    row = np.array([0.9, 0.3, -0.1, -0.05])
    fixed = project_row_to_simplex(row)
    assert fixed.min() >= -1e-12
    assert abs(fixed.sum() - 1.0) < 1e-10


def test_project_channel_rows_stay_valid_for_random_input():
    rng = np.random.default_rng(0)
    for _ in range(100):
        W = rng.normal(size=(4, 4))
        P = project_channel(W)
        assert P.min() >= -1e-10
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10)


def test_l_real_wc_is_zero_at_zero_radius():
    pi = np.full(4, 0.25)
    S = analytic_S(0.5, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)
    rng = np.random.default_rng(1)
    l_real, _, _ = L_real_wc(S, W_hat, pi, epsilon=0.0, rng=rng, n_trials=20, n_steps=5, n_restarts=1)
    assert l_real == pytest.approx(0.0, abs=1e-8)


def test_l_real_wc_increases_with_radius():
    pi = np.full(4, 0.25)
    S = analytic_S(0.5, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)
    rng = np.random.default_rng(2)
    small, _, _ = L_real_wc(S, W_hat, pi, 0.03, rng, n_trials=300, n_steps=15, n_restarts=2)
    large, _, _ = L_real_wc(S, W_hat, pi, 0.20, rng, n_trials=300, n_steps=15, n_restarts=2)
    assert large > small


def test_doubly_projection_is_valid_and_idempotent():
    rng = np.random.default_rng(4)
    for _ in range(100):
        P = project_doubly_stochastic(rng.normal(0.25, 0.4, size=(4, 4)))
        assert is_valid_channel(P, "doubly")
        assert np.allclose(project_doubly_stochastic(P), P, atol=1e-10)


def test_doubly_projection_is_the_nearest_point():
    # compare against a generic constrained solver
    from scipy.optimize import minimize
    rng = np.random.default_rng(5)
    cons = [{"type": "eq", "fun": lambda w: w.reshape(4, 4).sum(1) - 1},
            {"type": "eq", "fun": lambda w: w.reshape(4, 4).sum(0)[:3] - 1}]
    for _ in range(5):
        X = rng.normal(0.25, 0.3, size=(4, 4))
        ref = minimize(lambda w: np.sum((w - X.ravel()) ** 2), np.full(16, 0.25), constraints=cons,
                       bounds=[(0, None)] * 16, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        assert np.sum((project_doubly_stochastic(X) - X) ** 2) <= ref.fun + 1e-9


def test_feasible_direction_has_zero_row_and_column_sums():
    D = np.random.default_rng(6).normal(size=(4, 4))
    assert np.allclose(feasible_direction(D, "row").sum(axis=1), 0.0)
    Dd = feasible_direction(D, "doubly")
    assert np.allclose(Dd.sum(axis=1), 0.0) and np.allclose(Dd.sum(axis=0), 0.0)


def test_unknown_channel_set_is_rejected():
    with pytest.raises(ValueError):
        project_to_set(np.eye(4), "column")


@pytest.mark.parametrize("channel_set", ["row", "doubly"])
def test_search_returns_valid_channel_inside_radius(channel_set):
    pi = np.full(4, 0.25)
    S = analytic_S(0.3, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)  # near-boundary nominal
    eps = 0.2
    _, _, _, W = L_real_wc(S, W_hat, pi, eps, np.random.default_rng(7), n_trials=200, n_steps=15,
                           n_restarts=2, channel_set=channel_set, return_candidate=True)
    assert is_valid_channel(W, channel_set)
    assert np.linalg.norm(W - W_hat) <= eps + 1e-9


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_spectral_bound_holds_for_doubly_stochastic_adversary(alpha):
    # in the doubly-stochastic set the spectral bound is exact, so it cannot be exceeded
    pi = np.full(4, 0.25)
    S = analytic_S(alpha, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)
    eps = 0.08
    l_real, _, _ = L_real_wc(S, W_hat, pi, eps, np.random.default_rng(3), n_trials=300, n_steps=20,
                             n_restarts=2, channel_set="doubly")
    l_spec = worst_case_spectral(S, eps ** 2)
    assert l_real <= l_spec * (1 + 1e-6)
    assert l_real >= 0.8 * l_spec  # and the search gets close to it (not a vacuous pass)


def test_unrestricted_realizable_search_can_exceed_spectral_bound():
    # a row-stochastic (not doubly stochastic) adversary can beat the spectral bound
    rng = np.random.default_rng(1)
    pi = np.full(4, 0.25)
    S = analytic_S(0.5, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)
    l_real, _, _ = L_real_wc(S, W_hat, pi, 0.05, rng, n_trials=1500, n_steps=25, n_restarts=3)
    l_spec = worst_case_spectral(S, 0.05 ** 2)
    assert l_real > l_spec