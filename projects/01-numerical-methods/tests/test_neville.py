import csv

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
from quant_numerical.neville import CSV_HEADER, main


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


def test_assignment_data_survives_a_gui_display_round_trip():
    """Guards a real bug found by inspection: the Gradio apps format ASSIGNMENT_Y for their
    default textbox value with a %g-style specifier before the user ever presses Calculate.
    Python's default `:g` precision is 6 significant figures, which silently rounds
    0.7651977 -> "0.765198" and 0.2818186 -> "0.281819" -- close enough to look right at a
    glance, but the app then parses that *rounded string* back into a float and computes
    from it, so the very first thing a visitor sees no longer reproduces the assignment's own
    f(1.5) = 0.5118276664. `.10g` (used by both app.py files) must round-trip exactly.
    """
    displayed = [f"{v:.10g}" for v in ASSIGNMENT_Y]
    reparsed = [float(text) for text in displayed]
    assert reparsed == list(ASSIGNMENT_Y)

    result = neville_interpolate(ASSIGNMENT_X, reparsed, ASSIGNMENT_TARGET)
    assert result.estimate == pytest.approx(ASSIGNMENT_EXPECTED, abs=1e-9)


def test_csv_export_writes_one_row_per_step(tmp_path):
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    csv_path = write_csv(result, tmp_path / "neville_complete_table.csv")

    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    # header + 6 leaf rows + 15 recursive rows
    assert len(lines) == 1 + 6 + 15

    last_row = lines[-1]
    assert last_row.startswith("5,5,")
    assert "0.5118276664" in last_row


def test_csv_numeric_columns_are_actually_recomputable(tmp_path):
    """The whole point of breaking the substitution into its own numeric columns: a
    spreadsheet formula built from just those columns (not the value column) must reproduce
    it -- confirms the CSV is something to test, not only something to read.
    """
    result = neville_interpolate(ASSIGNMENT_X, ASSIGNMENT_Y, ASSIGNMENT_TARGET)
    csv_path = write_csv(result, tmp_path / "neville_complete_table.csv")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    recursive_rows = [row for row in rows if int(row["j"]) != 0]
    assert len(recursive_rows) == 15
    for row in recursive_rows:
        x_target = float(row["x_target"])
        x_i_minus_j = float(row["x_i_minus_j"])
        q_i_jminus1 = float(row["Q_i_jminus1"])
        x_i = float(row["x_i"])
        q_iminus1_jminus1 = float(row["Q_iminus1_jminus1"])
        recomputed = (
            (x_target - x_i_minus_j) * q_i_jminus1 - (x_target - x_i) * q_iminus1_jminus1
        ) / (x_i - x_i_minus_j)
        assert recomputed == pytest.approx(float(row["value"]), abs=1e-9)

    leaf_rows = [row for row in rows if int(row["j"]) == 0]
    assert all(row["x_target"] == "" for row in leaf_rows)


def test_cli_main_reproduces_assignment_result_and_writes_csv(tmp_path, capsys):
    """quant_numerical/neville.py's own CLI (not nevilles_method.py's wrapper) end to end --
    this is the entry point someone gets if they download only this one file.
    """
    csv_path = tmp_path / "out.csv"
    exit_code = main(["--csv", str(csv_path)])

    assert exit_code == 0
    assert csv_path.exists()
    captured = capsys.readouterr()
    assert "f(1.5) ~= 0.5118276664" in captured.out


def test_cli_main_reports_invalid_input_without_crashing(capsys):
    exit_code = main(["--x", "1.0,1.0", "--y", "1.0,2.0", "--target", "1.5"])
    assert exit_code == 1
    assert "distinct" in capsys.readouterr().err
