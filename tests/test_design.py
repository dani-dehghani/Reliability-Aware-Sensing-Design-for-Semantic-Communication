import json
from pathlib import Path

import numpy as np
import pytest

from src.design import alpha_sweep, build_control_table, format_markdown_table, select_control_pair

with open(Path(__file__).resolve().parents[1] / "configs" / "baseline.json") as f:
    CFG = json.load(f)


def test_sweep_shapes_and_ranges():
    sweep = alpha_sweep(CFG)
    n = CFG["n_alpha"]
    for key in ("alpha", "accuracy", "L_mis_range_optimistic", "L_mis_velocity_optimistic"):
        assert sweep[key].shape == (n,)
    assert np.all((sweep["accuracy"] >= 0.0) & (sweep["accuracy"] <= 1.0))
    assert np.all(sweep["L_mis_range_optimistic"] >= 0.0)
    assert np.all(sweep["L_mis_velocity_optimistic"] >= 0.0)


def test_accuracy_is_symmetric_under_alpha_swap():
    sweep = alpha_sweep(CFG)
    assert np.allclose(sweep["accuracy"], sweep["accuracy"][::-1], atol=1e-12)


def test_L_mis_modes_mirror_each_other():
    sweep = alpha_sweep(CFG)
    assert np.allclose(sweep["L_mis_range_optimistic"], sweep["L_mis_velocity_optimistic"][::-1], atol=1e-12)


def test_select_control_pair_matches_reference_endpoints():
    sweep = alpha_sweep(CFG)
    pair = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    assert np.isclose(pair["alpha_low_loss"], 0.0, atol=1e-9)
    assert np.isclose(pair["alpha_high_loss"], 1.0, atol=1e-9)
    assert np.isclose(pair["accuracy_low_loss"], pair["accuracy_high_loss"], atol=0.005)
    assert np.isclose(pair["L_mis_low"], 0.000520, atol=1e-6)
    assert np.isclose(pair["L_mis_high"], 0.003313, atol=1e-6)


def test_control_pair_accuracy_gap_within_tolerance():
    sweep = alpha_sweep(CFG)
    for tol in (0.001, 0.005, 0.02):
        pair = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=tol)
        assert abs(pair["accuracy_low_loss"] - pair["accuracy_high_loss"]) <= tol + 1e-12


def test_control_pair_has_substantial_loss_ratio():
    sweep = alpha_sweep(CFG)
    pair = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    assert pair["l_mis_ratio"] > 5.0


def test_select_control_pair_raises_if_tolerance_too_tight():
    sweep = alpha_sweep(CFG)
    with pytest.raises(ValueError):
        select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=-1.0)


def test_select_control_pair_is_deterministic():
    sweep = alpha_sweep(CFG)
    p1 = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    p2 = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    assert p1 == p2


def test_build_control_table_reflects_the_pair():
    sweep = alpha_sweep(CFG)
    pair = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    rows = build_control_table(pair)
    assert rows[0] == ("Sensing design", "alpha = 0.00", "alpha = 1.00")
    assert rows[1][1] == rows[1][2] == "0.4688"
    assert rows[2][1] == "0.000520" and rows[2][2] == "0.003313"
    assert rows[3][2] == "6.37x"


def test_format_markdown_table_has_valid_structure():
    sweep = alpha_sweep(CFG)
    pair = select_control_pair(sweep["alpha"], sweep["accuracy"], sweep["L_mis_range_optimistic"], acc_tol=0.005)
    md = format_markdown_table(build_control_table(pair))
    lines = md.split("\n")
    assert lines[0].startswith("| Metric |")
    assert lines[1].startswith("|---")
    assert len(lines) == 2 + 4
