import json
from pathlib import Path

import numpy as np

from src.channel import actual_channel, delta_W, nominal_channel
from src.metrics import L_mis_direct, L_mis_identity
from src.posterior import end_to_end, posterior
from src.semantic import analytic_S
from src.sensing import uniform_prior
from src.sensitivity import L_mis_trace

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
P_R, P_V, P_LOW = CFG["p_r"], CFG["p_v"], CFG["p_hat_low"]
PI = uniform_prior(4)
ALPHAS = np.linspace(0.0, 1.0, CFG["n_alpha"])
MODES = ("range_optimistic", "velocity_optimistic")


def test_every_row_of_S_W_What_sums_to_one():
    W = actual_channel(P_R, P_V)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12)
    for mode in MODES:
        W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
        assert np.allclose(W_hat.sum(axis=1), 1.0, atol=1e-12)
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        assert np.allclose(S.sum(axis=1), 1.0, atol=1e-12)


def test_all_posterior_vectors_sum_to_one():
    W = actual_channel(P_R, P_V)
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        rho = posterior(PI, end_to_end(S, W))
        assert np.allclose(rho.sum(axis=0), 1.0, atol=1e-12)
        for mode in MODES:
            W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
            rho_hat = posterior(PI, end_to_end(S, W_hat))
            assert np.allclose(rho_hat.sum(axis=0), 1.0, atol=1e-12)


def test_L_mis_is_zero_when_W_equals_What():
    W = actual_channel(P_R, P_V)
    dW = delta_W(W, W)
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        A = end_to_end(S, W)
        rho = posterior(PI, A)
        assert L_mis_direct(PI, rho, rho, A) < 1e-14
        assert L_mis_identity(PI, S, dW, rho, A, A) < 1e-14
        assert L_mis_trace(S, dW) < 1e-14


def test_exact_identity_agrees_with_direct_posterior_distance():
    W = actual_channel(P_R, P_V)
    max_gap = 0.0
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        A = end_to_end(S, W)
        rho = posterior(PI, A)
        for mode in MODES:
            W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
            dW = delta_W(W, W_hat)
            A_hat = end_to_end(S, W_hat)
            rho_hat = posterior(PI, A_hat)
            l_dir = L_mis_direct(PI, rho, rho_hat, A)
            l_id = L_mis_identity(PI, S, dW, rho, A, A_hat)
            max_gap = max(max_gap, abs(l_dir - l_id))
    assert max_gap < 1e-10, f"gap too large: {max_gap:.3e}"


def test_balanced_trace_formula_agrees_with_exact_identity():
    W = actual_channel(P_R, P_V)
    max_gap = 0.0
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        A = end_to_end(S, W)
        rho = posterior(PI, A)
        for mode in MODES:
            W_hat = nominal_channel(P_R, P_V, mode, P_LOW)
            dW = delta_W(W, W_hat)
            A_hat = end_to_end(S, W_hat)
            l_id = L_mis_identity(PI, S, dW, rho, A, A_hat)
            l_tr = L_mis_trace(S, dW)
            max_gap = max(max_gap, abs(l_id - l_tr))
    assert max_gap < 1e-10, f"gap too large: {max_gap:.3e}"
