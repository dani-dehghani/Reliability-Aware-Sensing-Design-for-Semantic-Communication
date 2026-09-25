import json
from pathlib import Path

import numpy as np
import pytest

from src.channel import actual_channel, delta_W, nominal_channel
from src.metrics import (L_mis_direct, L_mis_identity, bayes_risk, brier_loss,
                         exact_brier_of_mismatched_receiver, hard_accuracy)
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensing import uniform_prior

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
P_R, P_V, P_LOW = CFG["p_r"], CFG["p_v"], CFG["p_hat_low"]
ALPHAS = (0.0, 0.2, 0.5, 0.8, 1.0)
PI = uniform_prior(4)

# hand-checked reference values for the range-optimistic receiver
REFERENCE_LMIS = {0.0: 0.000520, 0.5: 0.002424, 1.0: 0.003313}


def _setup(alpha, mode):
    S = analytic_S(alpha, *ARGS)
    W = actual_channel(P_R, P_V)
    W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
    A, A_hat = end_to_end(S, W), end_to_end(S, W_hat)
    rho, rho_hat = posterior(PI, A), posterior(PI, A_hat)
    return S, delta_W(W, W_hat), A, A_hat, rho, rho_hat


def test_brier_loss_basic_values():
    assert np.isclose(brier_loss(0, [1, 0, 0, 0]), 0.0)
    assert np.isclose(brier_loss(0, [0.25, 0.25, 0.25, 0.25]), 3 * 0.25 ** 2 + 0.75 ** 2)
    assert np.isclose(brier_loss(2, [0, 0, 0, 1]), 2.0)


def test_L_mis_direct_and_identity_agree():
    for a in ALPHAS:
        for mode in ("range_optimistic", "velocity_optimistic"):
            S, dW, A, A_hat, rho, rho_hat = _setup(a, mode)
            l_dir = L_mis_direct(PI, rho, rho_hat, A)
            l_id = L_mis_identity(PI, S, dW, rho, A, A_hat)
            assert np.isclose(l_dir, l_id, atol=1e-12), (a, mode, l_dir, l_id)


def test_L_mis_matches_reference_values():
    for a, expected in REFERENCE_LMIS.items():
        S, dW, A, A_hat, rho, rho_hat = _setup(a, "range_optimistic")
        assert np.isclose(L_mis_direct(PI, rho, rho_hat, A), expected, atol=1e-6)


def test_L_mis_is_zero_when_matched():
    for a in ALPHAS:
        S, dW, A, A_hat, rho, rho_hat = _setup(a, "range_optimistic")
        W = actual_channel(P_R, P_V)
        A_m = end_to_end(S, W)
        rho_m = posterior(PI, A_m)
        dW_zero = delta_W(W, W)
        assert np.isclose(L_mis_direct(PI, rho_m, rho_m, A_m), 0.0, atol=1e-15)
        assert np.isclose(L_mis_identity(PI, S, dW_zero, rho_m, A_m, A_m), 0.0, atol=1e-15)


def test_L_mis_is_nonnegative():
    for a in ALPHAS:
        for mode in ("range_optimistic", "velocity_optimistic"):
            _, _, A, _, rho, rho_hat = _setup(a, mode)
            assert L_mis_direct(PI, rho, rho_hat, A) >= 0.0


def test_brier_decomposition():
    for a in ALPHAS:
        _, _, A, _, rho, rho_hat = _setup(a, "range_optimistic")
        lhs = exact_brier_of_mismatched_receiver(PI, rho, rho_hat, A)
        rhs = bayes_risk(PI, rho, A) + L_mis_direct(PI, rho, rho_hat, A)
        assert np.isclose(lhs, rhs, atol=1e-12)


def test_bayes_risk_matched_equals_exact_brier_of_matched_receiver():
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        A = end_to_end(S, actual_channel(P_R, P_V))
        rho = posterior(PI, A)
        assert np.isclose(exact_brier_of_mismatched_receiver(PI, rho, rho, A),
                          bayes_risk(PI, rho, A), atol=1e-12)


def test_orientation_control_range_optimistic_hurts_more_at_high_alpha():
    l0 = L_mis_direct(PI, *_setup(0.0, "range_optimistic")[4:6], _setup(0.0, "range_optimistic")[2])
    l1 = L_mis_direct(PI, *_setup(1.0, "range_optimistic")[4:6], _setup(1.0, "range_optimistic")[2])
    assert l1 > l0


def test_mirror_symmetry_between_the_two_receiver_modes():
    for a in (0.0, 0.2, 0.5, 0.8, 1.0):
        _, _, A_r, _, rho_r, rho_hat_r = _setup(round(1 - a, 10), "range_optimistic")
        _, _, A_v, _, rho_v, rho_hat_v = _setup(a, "velocity_optimistic")
        l_r = L_mis_direct(PI, rho_r, rho_hat_r, A_r)
        l_v = L_mis_direct(PI, rho_v, rho_hat_v, A_v)
        assert np.isclose(l_r, l_v, atol=1e-12)


def test_hard_accuracy_matches_manual_computation():
    S = analytic_S(0.7, *ARGS)
    A = end_to_end(S, actual_channel(P_R, P_V))
    rho = posterior(PI, A)
    p = PI @ A
    x_hat = np.argmax(rho, axis=0)
    expected = sum(p[m] * rho[x_hat[m], m] for m in range(4))
    assert np.isclose(hard_accuracy(PI, rho, A), expected, atol=1e-14)


def test_hard_accuracy_is_a_probability():
    for a in ALPHAS:
        for mode in ("matched", "range_optimistic", "velocity_optimistic"):
            S = analytic_S(a, *ARGS)
            W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
            A_hat = end_to_end(S, W_hat)
            rho_hat = posterior(PI, A_hat)
            acc = hard_accuracy(PI, rho_hat, A_hat)
            assert 0.0 <= acc <= 1.0


def test_matched_and_mismatched_accuracy_are_equal():
    for a in ALPHAS:
        S, dW, A, A_hat, rho, rho_hat = _setup(a, "range_optimistic")
        acc_matched = hard_accuracy(PI, rho, A)
        p = PI @ A
        x_hat_mismatched = np.argmax(rho_hat, axis=0)
        acc_mismatched_decision = sum(p[m] * rho[x_hat_mismatched[m], m] for m in range(4))
        assert np.isclose(acc_matched, acc_mismatched_decision, atol=1e-12)
