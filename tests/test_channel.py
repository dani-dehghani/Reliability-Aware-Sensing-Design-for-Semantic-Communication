import json
from pathlib import Path

import numpy as np
import pytest

from src.channel import MODES, actual_channel, bsc, delta_W, nominal_channel

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)
P_R, P_V, P_LOW = CFG["p_r"], CFG["p_v"], CFG["p_hat_low"]
PERM = [0, 2, 1, 3]


def test_bsc_basic_properties():
    B = bsc(0.2)
    assert np.allclose(B, [[0.8, 0.2], [0.2, 0.8]])
    assert np.allclose(B, B.T) and np.allclose(B.sum(axis=1), 1.0)
    assert np.allclose(bsc(0.0), np.eye(2))
    assert np.allclose(bsc(0.5), 0.5)


def test_bsc_rejects_invalid_probability():
    for p in (-0.1, 1.5):
        with pytest.raises(ValueError):
            bsc(p)


def test_actual_channel_is_doubly_stochastic():
    W = actual_channel(P_R, P_V)
    assert W.shape == (4, 4) and np.all(W >= 0)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-14)
    assert np.allclose(W.sum(axis=0), 1.0, atol=1e-14)


def test_bit_order():
    assert np.isclose(actual_channel(0.3, 0.0)[0, 2], 0.3)
    assert np.isclose(actual_channel(0.3, 0.0)[0, 1], 0.0)
    assert np.isclose(actual_channel(0.0, 0.4)[0, 1], 0.4)
    assert np.isclose(actual_channel(0.0, 0.4)[0, 2], 0.0)


def test_matches_bitwise_definition():
    W = actual_channel(P_R, P_V)
    for zr in (0, 1):
        for zv in (0, 1):
            for mr in (0, 1):
                for mv in (0, 1):
                    expected = (P_R if mr != zr else 1 - P_R) * (P_V if mv != zv else 1 - P_V)
                    assert np.isclose(W[2 * zr + zv, 2 * mr + mv], expected, atol=1e-14)


def test_matched_receiver_has_zero_mismatch():
    W = actual_channel(P_R, P_V)
    assert np.allclose(delta_W(W, nominal_channel(P_R, P_V, "matched", P_LOW)), 0.0)


def test_mismatch_rows_and_columns_sum_to_zero():
    W = actual_channel(P_R, P_V)
    for mode in MODES:
        dW = delta_W(W, nominal_channel(P_R, P_V, mode, P_LOW))
        assert np.allclose(dW @ np.ones(4), 0.0, atol=1e-14)
        assert np.allclose(np.ones(4) @ dW, 0.0, atol=1e-14)


def test_range_optimistic_uses_smaller_range_flip_only():
    W_hat = nominal_channel(P_R, P_V, "range_optimistic", P_LOW)
    assert np.allclose(W_hat, np.kron(bsc(P_LOW), bsc(P_V)))
    W = actual_channel(P_R, P_V)
    assert np.allclose(delta_W(W, W_hat), np.kron(bsc(P_LOW) - bsc(P_R), bsc(P_V)), atol=1e-14)


def test_velocity_optimistic_uses_smaller_velocity_flip_only():
    W_hat = nominal_channel(P_R, P_V, "velocity_optimistic", P_LOW)
    assert np.allclose(W_hat, np.kron(bsc(P_R), bsc(P_LOW)))


def test_range_and_velocity_cases_mirror_each_other():
    W = actual_channel(0.1, 0.1)
    Wr = nominal_channel(0.1, 0.1, "range_optimistic", 0.02)
    Wv = nominal_channel(0.1, 0.1, "velocity_optimistic", 0.02)
    assert np.allclose(Wv, Wr[np.ix_(PERM, PERM)], atol=1e-14)
    assert np.allclose(W, W[np.ix_(PERM, PERM)], atol=1e-14)


def test_mismatch_eigenvalues_by_hand():
    W = actual_channel(P_R, P_V)
    dW = delta_W(W, nominal_channel(P_R, P_V, "range_optimistic", P_LOW))
    d_r = 2 * (P_R - P_LOW)
    d_rv = (1 - 2 * P_LOW) * (1 - 2 * P_V) - (1 - 2 * P_R) * (1 - 2 * P_V)
    assert np.allclose(np.sort(np.linalg.eigvalsh(dW)), np.sort([0.0, 0.0, d_rv, d_r]), atol=1e-12)


def test_optimistic_mode_requires_smaller_probability():
    with pytest.raises(ValueError):
        nominal_channel(0.1, 0.1, "range_optimistic", 0.2)
    with pytest.raises(ValueError):
        nominal_channel(0.1, 0.1, "velocity_optimistic", 0.1)
    with pytest.raises(ValueError):
        nominal_channel(0.1, 0.1, "not_a_mode", 0.02)
