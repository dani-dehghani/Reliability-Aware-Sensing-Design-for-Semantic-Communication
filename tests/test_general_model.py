import numpy as np
import pytest

from src.channel import actual_channel
from src.metrics import L_mis_direct
from src.posterior import end_to_end, posterior
from src.probing import sensitivity_weights, sensitivity_weights_general
from src.semantic import analytic_S
from src.sensitivity import L_mis_quadratic, L_mis_trace, Q_blocks, Q_matrix

ARGS = (1.0, 1.0, 1.0, 0.1)
SKEWED = np.array([0.7, 0.1, 0.1, 0.1])


def asym_channel():
    a = np.array([[0.85, 0.15], [0.05, 0.95]])
    b = np.array([[0.95, 0.05], [0.15, 0.85]])
    return np.kron(a, b)


def exact(S, pi, W, W_hat):
    A = end_to_end(S, W)
    return L_mis_direct(pi, posterior(pi, A), posterior(pi, end_to_end(S, W_hat)), A)


def test_q_matrix_is_symmetric_psd_and_block_diagonal():
    Q = Q_matrix(analytic_S(0.3, *ARGS), SKEWED, asym_channel())
    assert np.allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > -1e-12
    assert np.allclose(Q[:4, 4:], 0.0)  # blocks for different received symbols m do not couple


def test_vec_form_matches_blockwise_sum():
    S, W = analytic_S(0.7, *ARGS), asym_channel()
    dW = 0.05 * (np.eye(4) - W)
    v = dW.flatten(order="F")
    assert v @ Q_matrix(S, SKEWED, W) @ v == pytest.approx(L_mis_quadratic(S, SKEWED, W, dW), rel=1e-12)


def test_quadratic_is_exact_and_equals_trace_in_balanced_case():
    
    S, W = analytic_S(0.3, *ARGS), actual_channel(0.1, 0.1)
    W_hat = 0.7 * W + 0.3 * np.eye(4)
    pi = np.full(4, 0.25)
    l_exact = exact(S, pi, W, W_hat)
    assert L_mis_quadratic(S, pi, W, W_hat - W) == pytest.approx(l_exact, rel=1e-10)
    assert L_mis_trace(S, W_hat - W) == pytest.approx(l_exact, rel=1e-10)


def test_relative_error_is_first_order_in_mismatch():
    
    S, W = analytic_S(0.5, *ARGS), asym_channel()
    D = np.eye(4) - W
    errs = []
    for t in (0.02, 0.01, 0.005):
        l_ex = exact(S, SKEWED, W, W + t * D)
        errs.append(abs(L_mis_quadratic(S, SKEWED, W, t * D) - l_ex) / l_ex)
    assert 1.7 < errs[0] / errs[1] < 2.3
    assert 1.7 < errs[1] / errs[2] < 2.3
    assert errs[-1] < 0.01


def test_balanced_trace_formula_is_wrong_for_skewed_prior():
    S, W = analytic_S(0.5, *ARGS), actual_channel(0.1, 0.1)
    dW = 0.01 * (np.eye(4) - W)
    l_ex = exact(S, SKEWED, W, W + dW)
    assert abs(L_mis_trace(S, dW) - l_ex) / l_ex > 0.2
    assert abs(L_mis_quadratic(S, SKEWED, W, dW) - l_ex) / l_ex < 0.01


@pytest.mark.parametrize("allocation", ["uniform", "sensitivity"])
def test_required_probes_hits_the_target_under_the_prediction(allocation):
    from src.probing import expected_L_est_general, required_probes, sensitivity_weights_general
    S, W, target = analytic_S(0.3, *ARGS), asym_channel(), 1e-4
    N = required_probes(S, SKEWED, W, target, allocation)
    a = sensitivity_weights_general(S, SKEWED, W)
    n = np.full(4, N / 4) if allocation == "uniform" else N * np.sqrt(a) / np.sqrt(a).sum()
    assert expected_L_est_general(S, SKEWED, W, n) == pytest.approx(target, rel=1e-10)


def test_sensitivity_never_needs_more_probes_than_uniform():
    from src.probing import required_probes
    for alpha in (0.0, 0.3, 0.7):
        S = analytic_S(alpha, *ARGS)
        assert required_probes(S, SKEWED, asym_channel(), 1e-4, "sensitivity") <= \
            required_probes(S, SKEWED, asym_channel(), 1e-4, "uniform") * (1 + 1e-12)


def test_probe_weights_are_diag_of_q_blocks_and_reduce_to_balanced_weights():
    S, W = analytic_S(0.4, *ARGS), asym_channel()
    blocks = Q_blocks(S, SKEWED, W)
    expected = sum(np.diag(blocks[m]) * W[:, m] * (1 - W[:, m]) for m in range(4))
    assert np.allclose(sensitivity_weights_general(S, SKEWED, W), expected)
    
    Wb = actual_channel(0.1, 0.1)
    general = sensitivity_weights_general(S, np.full(4, 0.25), Wb)
    balanced = sensitivity_weights(S, Wb) / 4
    assert np.allclose(general, balanced, rtol=1e-2)
