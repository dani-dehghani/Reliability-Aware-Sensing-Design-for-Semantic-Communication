import numpy as np
import pytest

from src.channel import nominal_channel
from src.robust import L_real_wc, l_mis_of_candidate, project_channel, project_row_to_simplex
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
    # Section 5.2 item 8: projected adversarial channels must stay
    # nonnegative and row-stochastic
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


def test_spectral_bound_holds_for_doubly_stochastic_restricted_search():
    # eq. 70 is explicitly derived under the "balanced assumptions" (S_u, W,
    # W_hat all doubly stochastic). It should hold once the adversarial
    # search is restricted back to (approximately) doubly-stochastic
    # channels -- confirms the mechanism behind the unrestricted violation
    # seen in exp03.
    def sinkhorn(W, n_iter=40):
        W = W.copy()
        for _ in range(n_iter):
            W = W / W.sum(axis=1, keepdims=True)
            W = W / W.sum(axis=0, keepdims=True)
        return W / W.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(3)
    pi = np.full(4, 0.25)
    S = analytic_S(0.5, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)
    eps = 0.08

    best_ds = 0.0
    for _ in range(1500):
        delta = rng.normal(size=(4, 4))
        delta -= delta.mean(axis=1, keepdims=True)
        n = np.linalg.norm(delta)
        if n < 1e-12:
            continue
        delta *= eps / n
        cand = project_channel(W_hat + delta)
        cand = sinkhorn(cand)
        d = cand - W_hat
        n2 = np.linalg.norm(d)
        if n2 > eps:
            d *= eps / n2
            cand = project_channel(W_hat + d)
            cand = sinkhorn(cand)
        best_ds = max(best_ds, l_mis_of_candidate(S, W_hat, cand, pi))

    l_spec = worst_case_spectral(S, eps ** 2)
    assert l_spec >= best_ds - 1e-4  # small slack for Sinkhorn/search imprecision


def test_unrestricted_realizable_search_can_exceed_spectral_bound():
    # the key finding this extension surfaces: once the adversary is only
    # required to stay row-stochastic (eq. 65), not doubly stochastic, the
    # spectral bound (eq. 70) is not guaranteed to hold
    rng = np.random.default_rng(1)
    pi = np.full(4, 0.25)
    S = analytic_S(0.5, *CFG_ARGS)
    W_hat = nominal_channel(0.10, 0.10, "range_optimistic", 0.02)
    l_real, _, _ = L_real_wc(S, W_hat, pi, 0.05, rng, n_trials=1500, n_steps=25, n_restarts=3)
    l_spec = worst_case_spectral(S, 0.05 ** 2)
    assert l_real > l_spec