import numpy as np
import pytest

from quant_numerical import (
    ASSIGNMENT_EXPECTED,
    ASSIGNMENT_TARGET,
    ASSIGNMENT_X,
    ASSIGNMENT_Y,
    neville_interpolate,
    write_csv,
)


def test_assignment_result_matches_expected_value():
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    assert result.estimate == pytest.approx(ASSIGNMENT_EXPECTED, abs=1e-9)
    assert result.estimate == pytest.approx(0.5118276664, abs=1e-9)


def test_assignment_result_is_not_hard_coded():
    """The estimate must come from the recurrence, not a stored constant equal to it."""
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    # A different target on the same nodes must produce a different, independently
    # computed answer -- proof the function is actually running the recurrence.
    other = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, 2.0)
    assert other.estimate != pytest.approx(result.estimate)


def test_worked_example_q11_matches_hand_calculation():
    """Q[1,1] from the assignment brief.

    ((1.5-1.0)(0.6200860) - (1.5-1.3)(0.7651977)) / (1.3-1.0)
    """
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    assert result.table[1, 1] == pytest.approx(0.5233448667, abs=1e-9)


def test_final_row_matches_assignment_brief():
    """Q[5,0] through Q[5,5], as given in the assignment brief for the last table row."""
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    expected_last_row = [
        -0.0483838000,
        0.4807698667,
        0.5301984222,
        0.5119069901,
        0.5118430107,
        0.5118276664,
    ]
    np.testing.assert_allclose(result.table[5, :], expected_last_row, atol=1e-9)


def test_generates_fifteen_recursive_calculations_for_six_points():
    """Six points -> 6 leaf assignments (j=0) + 15 recursive entries (j>=1): C(6,2) = 15."""
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    leaf_steps = [s for s in result.steps if s.j == 0]
    recursive_steps = [s for s in result.steps if s.j >= 1]
    assert len(leaf_steps) == 6
    assert len(recursive_steps) == 15
    assert all(step.formula for step in result.steps)


def test_evaluating_at_a_known_node_reproduces_its_value():
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_X[2])
    assert result.estimate == pytest.approx(ASSIGNMENT_Y[2], abs=1e-9)


def test_recovers_a_known_quadratic():
    x = np.array([-2.0, 0.0, 1.0])
    y = 3 * x**2 - 2 * x + 4
    result = neville_interpolate(x, y, 0.5)
    assert result.estimate == pytest.approx(3 * 0.5**2 - 2 * 0.5 + 4, abs=1e-9)


def test_rejects_duplicate_x_values():
    with pytest.raises(ValueError, match="distinct"):
        neville_interpolate([1.0, 1.0, 2.0], [1.0, 2.0, 3.0], 1.5)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same non-zero length"):
        neville_interpolate([1.0, 2.0], [1.0, 2.0, 3.0], 1.5)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="same non-zero length"):
        neville_interpolate([], [], 1.5)


def test_rejects_non_finite_target():
    with pytest.raises(ValueError, match="finite"):
        neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, float("nan"))


def test_csv_export_writes_one_row_per_step(tmp_path):
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    csv_path = write_csv(result, tmp_path / "neville_complete_table.csv")

    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "i,j,value,substitution"
    # header + 6 leaf rows + 15 recursive rows
    assert len(lines) == 1 + 6 + 15

    last_row = lines[-1]
    assert last_row.startswith("5,5,")
    assert "0.5118276664" in last_row
