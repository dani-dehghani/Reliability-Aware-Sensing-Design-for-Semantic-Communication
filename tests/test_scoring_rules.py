import numpy as np
import pytest

from src.channel import actual_channel, nominal_channel
from src.metrics import L_log, L_mis_direct, floor_probabilities, info_log, kl_divergence
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S

ARGS = (1.0, 1.0, 1.0, 0.1)
PI = np.full(4, 0.25)


def setup(alpha=0.5, mode="range_optimistic"):
    S = analytic_S(alpha, *ARGS)
    W, W_hat = actual_channel(0.1, 0.1), nominal_channel(0.1, 0.1, mode, 0.02)
    A = end_to_end(S, W)
    return posterior(PI, A), posterior(PI, end_to_end(S, W_hat)), A


def test_kl_basic_properties():
    p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-15)
    assert kl_divergence(p, q) > 0
    assert kl_divergence(p, q) == pytest.approx(np.sum(p * np.log(p / q)))
    assert kl_divergence(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == pytest.approx(np.log(2))  # 0 log 0 = 0
    assert kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == float("inf")


def test_l_log_is_zero_when_matched():
    S = analytic_S(0.3, *ARGS)
    A = end_to_end(S, actual_channel(0.1, 0.1))
    rho = posterior(PI, A)
    assert L_log(PI, rho, rho, A) == pytest.approx(0.0, abs=1e-15)


def test_l_log_is_twice_chi_square_brier_for_small_mismatch():
    
    S = analytic_S(0.4, *ARGS)
    W = actual_channel(0.1, 0.1)
    W_hat = 0.999 * W + 0.001 * np.eye(4)
    A = end_to_end(S, W)
    rho, rho_hat = posterior(PI, A), posterior(PI, end_to_end(S, W_hat))
    p = PI @ A
    chi2 = sum(p[m] * 0.5 * np.sum((rho_hat[:, m] - rho[:, m]) ** 2 / rho[:, m]) for m in range(4))
    assert L_log(PI, rho, rho_hat, A) == pytest.approx(chi2, rel=1e-2)


def test_info_log_is_mutual_information():
    S = analytic_S(0.5, *ARGS)
    A = end_to_end(S, actual_channel(0.1, 0.1))
    joint = PI[:, None] * A
    p = joint.sum(axis=0)
    mi = np.sum(joint * np.log(joint / (PI[:, None] * p[None, :])))
    assert info_log(PI, posterior(PI, A), A) == pytest.approx(mi, rel=1e-12)


def test_floor_keeps_columns_normalized_and_only_used_when_asked():
    q = np.array([[0.7, 0.0], [0.3, 1.0]])
    f = floor_probabilities(q, 1e-3)
    assert np.allclose(f.sum(axis=0), 1.0)
    assert f.min() > 0
    assert f[:, 0] == pytest.approx(q[:, 0])  # columns already above the floor are unchanged
    rho, rho_hat, A = setup()
    assert L_log(PI, rho, rho_hat, A) == pytest.approx(L_log(PI, rho, rho_hat, A, floor=1e-12), rel=1e-9)


def test_brier_and_log_rank_directional_sweep_identically():
    alphas = np.linspace(0, 1, 21)
    b, l = [], []
    for a in alphas:
        rho, rho_hat, A = setup(a)
        b.append(L_mis_direct(PI, rho, rho_hat, A))
        l.append(L_log(PI, rho, rho_hat, A))
    assert np.array_equal(np.argsort(b), np.argsort(l))
