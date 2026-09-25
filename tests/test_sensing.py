import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src.sensing import (STATES, bit_reliability, covariance, likelihood, sample,
                         state_means, uniform_prior, variances)

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)


def test_prior_is_valid():
    pi = uniform_prior(4)
    assert pi.shape == (4,) and np.all(pi > 0) and np.isclose(pi.sum(), 1.0)


def test_state_map():
    expected = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    assert [tuple(s) for s in STATES.astype(int)] == expected


def test_means():
    mu = state_means(2.0, 3.0)
    assert mu.shape == (4, 2)
    assert np.allclose(mu[3], [2.0, 3.0]) and np.allclose(mu[0], [-2.0, -3.0])


def test_variance_endpoints():
    s0, e = CFG["sigma0_sq"], CFG["eps"]
    vr, vv = variances(1.0, s0, e)
    assert np.isclose(vr, s0 / (e + 1)) and np.isclose(vv, s0 / e)
    vr, vv = variances(0.0, s0, e)
    assert np.isclose(vr, s0 / e) and np.isclose(vv, s0 / (e + 1))
    vr, vv = variances(0.5, s0, e)
    assert np.isclose(vr, vv)


def test_total_precision_is_constant_in_alpha():
    a = np.linspace(0, 1, 11)
    vr, vv = variances(a, CFG["sigma0_sq"], CFG["eps"])
    assert np.allclose(1 / vr + 1 / vv, (1 + 2 * CFG["eps"]) / CFG["sigma0_sq"])


def test_alpha_moves_precision_toward_range():
    a = np.linspace(0, 1, 11)
    vr, vv = variances(a, CFG["sigma0_sq"], CFG["eps"])
    assert np.all(np.diff(vr) < 0) and np.all(np.diff(vv) > 0)


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        variances(1.2, 1.0, 0.1)


def test_covariance_is_positive_definite():
    for a in (0.0, 0.3, 0.5, 1.0):
        assert np.all(np.linalg.eigvalsh(covariance(a, 1.0, 0.1)) > 0)


def test_density_matches_scipy():
    mu = state_means(CFG["a_r"], CFG["a_v"])
    S = covariance(0.7, CFG["sigma0_sq"], CFG["eps"])
    y = np.array([[0.3, -0.8], [1.2, 0.1], [-2.0, 2.0]])
    for x in range(4):
        ref = multivariate_normal(mean=mu[x], cov=S).pdf(y)
        assert np.allclose(likelihood(y, x, mu, S), ref, rtol=1e-12)


def test_monte_carlo_mean_and_cov():
    rng = np.random.default_rng(CFG["seed"])
    mu = state_means(CFG["a_r"], CFG["a_v"])
    S = covariance(0.7, CFG["sigma0_sq"], CFG["eps"])
    for x in range(4):
        y = sample(x, CFG["n_mc"], mu, S, rng)
        assert np.allclose(y.mean(axis=0), mu[x], atol=0.03)
        assert np.allclose(np.cov(y.T), S, rtol=0.03, atol=0.02)


def test_bit_reliability_symmetry():
    r1, v1 = bit_reliability(0.2, 1.0, 1.0, 1.0, 0.1)
    r2, v2 = bit_reliability(0.8, 1.0, 1.0, 1.0, 0.1)
    assert np.isclose(r1, v2) and np.isclose(v1, r2)
