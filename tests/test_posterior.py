import json
from pathlib import Path

import numpy as np
import pytest

from src.channel import nominal_channel
from src.posterior import end_to_end, evidence, hard_decision, posterior
from src.semantic import analytic_S
from src.sensing import uniform_prior

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
P_R, P_V, P_LOW = CFG["p_r"], CFG["p_v"], CFG["p_hat_low"]
ALPHAS = (0.0, 0.2, 0.5, 0.8, 1.0)
MODES = ("matched", "range_optimistic", "velocity_optimistic")
PI_UNIFORM = uniform_prior(4)
PI_GENERAL = np.array([0.1, 0.2, 0.3, 0.4])


def _A(alpha, mode="matched"):
    S = analytic_S(alpha, *ARGS)
    return end_to_end(S, nominal_channel(P_R, P_V, mode, P_LOW))


def test_end_to_end_matches_manual_sum():
    S = analytic_S(0.3, *ARGS)
    W = nominal_channel(P_R, P_V, "matched", P_LOW)
    A = end_to_end(S, W)
    for x in range(4):
        for m in range(4):
            assert np.isclose(A[x, m], sum(S[x, z] * W[z, m] for z in range(4)), atol=1e-14)


def test_end_to_end_rows_and_columns_sum_to_one():
    for a in ALPHAS:
        for mode in MODES:
            A = _A(a, mode)
            assert np.allclose(A.sum(axis=1), 1.0, atol=1e-14)
            assert np.allclose(A.sum(axis=0), 1.0, atol=1e-14)


def test_evidence_is_flat_in_the_balanced_case():
    for a in ALPHAS:
        assert np.allclose(evidence(PI_UNIFORM, _A(a)), 0.25, atol=1e-14)


def test_evidence_sums_to_one_for_general_prior():
    for a in ALPHAS:
        assert np.isclose(evidence(PI_GENERAL, _A(a)).sum(), 1.0, atol=1e-14)


def test_posteriors_sum_to_one_and_are_nonnegative():
    for pi in (PI_UNIFORM, PI_GENERAL):
        for a in ALPHAS:
            for mode in MODES:
                rho = posterior(pi, _A(a, mode))
                assert rho.shape == (4, 4) and np.all(rho >= 0)
                assert np.allclose(rho.sum(axis=0), 1.0, atol=1e-14)


def test_posterior_matches_bayes_rule_loop():
    A = _A(0.4, "range_optimistic")
    rho = posterior(PI_GENERAL, A)
    for m in range(4):
        denom = sum(PI_GENERAL[j] * A[j, m] for j in range(4))
        for x in range(4):
            assert np.isclose(rho[x, m], PI_GENERAL[x] * A[x, m] / denom, atol=1e-14)


def test_balanced_posterior_equals_likelihood_column():
    for a in ALPHAS:
        A = _A(a)
        assert np.allclose(posterior(PI_UNIFORM, A), A, atol=1e-14)


def test_evidence_weighted_posteriors_return_the_prior():
    for pi in (PI_UNIFORM, PI_GENERAL):
        A = _A(0.7)
        rho, p = posterior(pi, A), evidence(pi, A)
        assert np.allclose((rho * p[None, :]).sum(axis=1), pi, atol=1e-14)


def test_hand_reference_values_at_alpha_half():
    rho = posterior(PI_UNIFORM, _A(0.5, "matched"))
    assert np.isclose(rho[0, 0], 0.5250, atol=5e-4)
    rho_hat = posterior(PI_UNIFORM, _A(0.5, "range_optimistic"))
    assert np.isclose(rho_hat[0, 0], 0.5575, atol=5e-4)


def test_hard_decision_is_unchanged_by_mismatch():
    for a in ALPHAS:
        matched = hard_decision(posterior(PI_UNIFORM, _A(a, "matched")))
        assert list(matched) == [0, 1, 2, 3]
        for mode in ("range_optimistic", "velocity_optimistic"):
            assert list(hard_decision(posterior(PI_UNIFORM, _A(a, mode)))) == list(matched)


def test_mismatched_posterior_differs_but_matched_does_not():
    rho = posterior(PI_UNIFORM, _A(0.5, "matched"))
    assert np.allclose(posterior(PI_UNIFORM, _A(0.5, "matched")), rho, atol=1e-15)
    rho_hat = posterior(PI_UNIFORM, _A(0.5, "range_optimistic"))
    assert np.abs(rho_hat - rho).max() > 1e-2


def test_zero_evidence_is_rejected():
    with pytest.raises(ValueError):
        posterior(np.array([0.5, 0.5]), np.array([[1.0, 0.0], [1.0, 0.0]]))


def test_invalid_prior_is_rejected():
    A = _A(0.5)
    for bad in (np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.3, 0.3, 0.3, 0.3]), np.array([0.5, 0.5])):
        with pytest.raises(ValueError):
            posterior(bad, A)
