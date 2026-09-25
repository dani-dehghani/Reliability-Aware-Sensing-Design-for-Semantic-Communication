import json
from pathlib import Path

import numpy as np
import pytest

from src.channel import actual_channel, delta_W, nominal_channel
from src.metrics import L_mis_direct, L_mis_identity
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensing import uniform_prior
from src.sensitivity import (Gdelta_matrix, Gs_matrix, H_matrix, L_mis_modes,
                             L_mis_trace, uninformative_experiment, worst_case_spectral)

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
P_R, P_V, P_LOW = CFG["p_r"], CFG["p_v"], CFG["p_hat_low"]
ALPHAS = (0.0, 0.2, 0.5, 0.8, 1.0)
PI = uniform_prior(4)
REFERENCE_LMIS = {0.0: 0.000520, 0.5: 0.002424, 1.0: 0.003313}


def _pieces(alpha, mode):
    S = analytic_S(alpha, *ARGS)
    W = actual_channel(P_R, P_V)
    W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
    dW = delta_W(W, W_hat)
    A, A_hat = end_to_end(S, W), end_to_end(S, W_hat)
    rho, rho_hat = posterior(PI, A), posterior(PI, A_hat)
    return S, dW, A, A_hat, rho, rho_hat


def test_J_is_uninformative():
    J = uninformative_experiment(4)
    assert np.allclose(J, 0.25) and np.allclose(J.sum(axis=1), 1.0) and np.allclose(J @ J, J)


def test_H_rows_and_columns_sum_to_zero():
    for a in ALPHAS:
        H = H_matrix(analytic_S(a, *ARGS))
        assert np.allclose(H.sum(axis=1), 0.0, atol=1e-14)
        assert np.allclose(H.sum(axis=0), 0.0, atol=1e-14)


def test_Gs_and_Gdelta_are_positive_semidefinite():
    for a in ALPHAS:
        S, dW, *_ = _pieces(a, "range_optimistic")
        assert np.all(np.linalg.eigvalsh(Gs_matrix(S)) >= -1e-12)
        assert np.all(np.linalg.eigvalsh(Gdelta_matrix(dW)) >= -1e-12)


def test_trace_formula_agrees_with_exact_identity():
    for a in ALPHAS:
        for mode in ("range_optimistic", "velocity_optimistic"):
            S, dW, A, A_hat, rho, rho_hat = _pieces(a, mode)
            l_id = L_mis_identity(PI, S, dW, rho, A, A_hat)
            l_tr = L_mis_trace(S, dW)
            assert np.isclose(l_id, l_tr, atol=1e-12), (a, mode, l_id, l_tr)


def test_trace_formula_matches_direct_and_reference_values():
    for a, expected in REFERENCE_LMIS.items():
        S, dW, A, A_hat, rho, rho_hat = _pieces(a, "range_optimistic")
        l_dir = L_mis_direct(PI, rho, rho_hat, A)
        l_tr = L_mis_trace(S, dW)
        assert np.isclose(l_tr, expected, atol=1e-6)
        assert np.isclose(l_tr, l_dir, atol=1e-12)


def test_trace_formula_via_explicit_trace():
    for a in ALPHAS:
        S, dW, *_ = _pieces(a, "range_optimistic")
        via_trace = float(np.trace(Gs_matrix(S) @ Gdelta_matrix(dW)) / 4)
        assert np.isclose(via_trace, L_mis_trace(S, dW), atol=1e-12)


def test_L_mis_is_zero_when_matched():
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        W = actual_channel(P_R, P_V)
        dW_zero = delta_W(W, W)
        assert np.isclose(L_mis_trace(S, dW_zero), 0.0, atol=1e-15)


def test_H_and_delta_W_share_the_hadamard_basis():
    for a in ALPHAS:
        S, dW, *_ = _pieces(a, "range_optimistic")
        H = H_matrix(S)
        assert np.allclose(H @ dW, dW @ H, atol=1e-12)


def test_mode_decomposition_matches_trace_formula():
    for a in ALPHAS:
        for mode in ("range_optimistic", "velocity_optimistic"):
            S, dW, *_ = _pieces(a, mode)
            assert np.isclose(L_mis_modes(S, dW), L_mis_trace(S, dW), atol=1e-8)


def test_worst_case_spectral_bounds_the_realized_loss():
    for a in ALPHAS:
        for mode in ("range_optimistic", "velocity_optimistic"):
            S, dW, *_ = _pieces(a, mode)
            delta = float(np.trace(Gdelta_matrix(dW)))
            bound = worst_case_spectral(S, delta)
            assert bound >= L_mis_trace(S, dW) - 1e-12


def test_worst_case_spectral_matches_hand_formula():
    S = analytic_S(0.8, *ARGS)
    Gs = Gs_matrix(S)
    lam_max = np.linalg.eigvalsh(Gs).max()
    assert np.isclose(worst_case_spectral(S, 0.1), 0.1 / 4 * lam_max, atol=1e-14)


def test_spectral_bound_increases_with_delta():
    S = analytic_S(0.8, *ARGS)
    b1, b2 = worst_case_spectral(S, 0.05), worst_case_spectral(S, 0.5)
    assert b2 > b1


def test_spectral_bound_is_symmetric_under_alpha_swap_but_L_mis_is_not():
    # lambda_max(Gs) is the same at alpha and 1-alpha (S(alpha) and S(1-alpha) differ
    # only by a permutation), and Delta_W doesn't depend on alpha at all here, so the
    # eq. (52) bound is identical at both ends even though the true loss is not.
    S0, dW0, *_ = _pieces(0.0, "range_optimistic")
    S1, dW1, *_ = _pieces(1.0, "range_optimistic")
    assert np.allclose(dW0, dW1, atol=1e-14)
    lam0 = np.linalg.eigvalsh(Gs_matrix(S0)).max()
    lam1 = np.linalg.eigvalsh(Gs_matrix(S1)).max()
    assert np.isclose(lam0, lam1, atol=1e-12)
    delta = float(np.trace(Gdelta_matrix(dW0)))
    assert np.isclose(worst_case_spectral(S0, delta), worst_case_spectral(S1, delta), atol=1e-12)
    assert L_mis_trace(S0, dW0) < L_mis_trace(S1, dW1) - 1e-6
