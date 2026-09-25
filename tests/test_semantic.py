import json
from pathlib import Path

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm

from src.semantic import analytic_S, empirical_S, sign_extractor
from src.sensing import covariance, likelihood, state_means

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)
ARGS = (CFG["a_r"], CFG["a_v"], CFG["sigma0_sq"], CFG["eps"])
ALPHAS = (0.0, 0.2, 0.5, 0.8, 1.0)


def test_extractor_token_order():
    y = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    assert list(sign_extractor(y)) == [0, 1, 2, 3]


def test_extractor_accepts_single_vector():
    assert list(sign_extractor(np.array([0.5, -0.5]))) == [2]


def test_rows_sum_to_one_and_nonnegative():
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        assert S.shape == (4, 4) and np.all(S >= 0)
        assert np.allclose(S.sum(axis=1), 1.0, atol=1e-14)


def test_doubly_stochastic_for_symmetric_means():
    for a in ALPHAS:
        for a_r, a_v in ((1.0, 1.0), (2.0, 0.5)):
            S = analytic_S(a, a_r, a_v, CFG["sigma0_sq"], CFG["eps"])
            assert np.allclose(S.sum(axis=0), 1.0, atol=1e-14)


def test_matches_bernoulli_product():
    alpha = 0.3
    vr = CFG["sigma0_sq"] / (CFG["eps"] + alpha)
    vv = CFG["sigma0_sq"] / (CFG["eps"] + 1 - alpha)
    mu = state_means(CFG["a_r"], CFG["a_v"])
    S = analytic_S(alpha, *ARGS)
    for x in range(4):
        pr1 = norm.cdf(mu[x, 0] / np.sqrt(vr))
        pv1 = norm.cdf(mu[x, 1] / np.sqrt(vv))
        for zr in (0, 1):
            for zv in (0, 1):
                expected = (pr1 if zr else 1 - pr1) * (pv1 if zv else 1 - pv1)
                assert np.isclose(S[x, 2 * zr + zv], expected, atol=1e-14)


def test_reference_value_by_hand():
    S = analytic_S(1.0, *ARGS)
    assert np.isclose(S[3, 3], 0.8529 * 0.6241, atol=5e-4)


def test_matches_region_integral():
    alpha, x, L = 0.7, 1, 20.0
    mu = state_means(CFG["a_r"], CFG["a_v"])
    Sigma = covariance(alpha, CFG["sigma0_sq"], CFG["eps"])
    S = analytic_S(alpha, *ARGS)
    for z in range(4):
        zr, zv = divmod(z, 2)
        r_lo, r_hi = (0.0, L) if zr else (-L, 0.0)
        v_lo, v_hi = (0.0, L) if zv else (-L, 0.0)
        val, _ = dblquad(lambda yv, yr: likelihood(np.array([yr, yv]), x, mu, Sigma)[0],
                         r_lo, r_hi, lambda _: v_lo, lambda _: v_hi, epsabs=1e-10, epsrel=1e-10)
        assert abs(val - S[x, z]) < 1e-6


def test_matches_monte_carlo():
    rng = np.random.default_rng(CFG["seed"])
    for a in ALPHAS:
        S = analytic_S(a, *ARGS)
        S_mc = empirical_S(a, *ARGS, CFG["n_mc"], rng)
        assert np.allclose(S_mc.sum(axis=1), 1.0)
        assert np.abs(S - S_mc).max() < 0.005


def test_swapping_range_and_velocity():
    perm = [0, 2, 1, 3]
    for a in (0.0, 0.2, 0.35, 0.5):
        S = analytic_S(a, 1.0, 1.0, CFG["sigma0_sq"], CFG["eps"])
        S_swapped = analytic_S(1.0 - a, 1.0, 1.0, CFG["sigma0_sq"], CFG["eps"])
        assert np.allclose(S_swapped, S[np.ix_(perm, perm)], atol=1e-14)
