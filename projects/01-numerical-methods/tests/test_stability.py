import json

import pytest

from quant_numerical.stability import main, run_stability_study


def test_stability_study_is_reproducible_and_degree_order_independent():
    first = run_stability_study(degrees=(4, 8), grid_size=41, decimal_precision=50)
    reordered = run_stability_study(degrees=(8, 4), grid_size=41, decimal_precision=50)

    assert first == reordered
    assert first["degrees"] == [4, 8]
    assert first["node_designs"] == ["equidistant", "chebyshev-lobatto"]
    assert len(first["rows"]) == 4


def test_error_decomposition_has_nonnegative_finite_metrics():
    study = run_stability_study(
        degrees=(4,),
        node_designs=("equidistant",),
        grid_size=41,
        decimal_precision=50,
    )
    row = study["rows"][0]

    for field in (
        "max_approximation_error",
        "max_floating_point_error",
        "max_total_float64_error",
        "approximation_error_at_total_argmax",
        "floating_point_error_at_total_argmax",
    ):
        assert row[field] >= 0.0
        assert row[field] < float("inf")
    assert -1.0 <= row["total_error_argmax"] <= 1.0


def test_runge_growth_is_approximation_error_not_float64_roundoff():
    study = run_stability_study(
        degrees=(4, 16),
        node_designs=("equidistant",),
        grid_size=401,
        decimal_precision=80,
    )
    degree4, degree16 = study["rows"]

    assert degree16["max_approximation_error"] > degree4["max_approximation_error"]
    assert degree16["max_approximation_error"] > 1_000_000 * degree16[
        "max_floating_point_error"
    ]
    assert degree16["approximation_error_at_total_argmax"] > 1_000_000 * degree16[
        "floating_point_error_at_total_argmax"
    ]


def test_chebyshev_lobatto_nodes_reduce_degree_16_runge_error():
    study = run_stability_study(degrees=(16,), grid_size=401, decimal_precision=80)
    equidistant, chebyshev = study["rows"]

    assert equidistant["node_design"] == "equidistant"
    assert chebyshev["node_design"] == "chebyshev-lobatto"
    assert chebyshev["max_approximation_error"] < equidistant["max_approximation_error"] / 100
    assert chebyshev["max_approximation_error"] > 1_000_000 * chebyshev[
        "max_floating_point_error"
    ]


def test_node_design_order_does_not_change_output():
    first = run_stability_study(
        degrees=(4,),
        node_designs=("equidistant", "chebyshev-lobatto"),
        grid_size=41,
        decimal_precision=50,
    )
    reordered = run_stability_study(
        degrees=(4,),
        node_designs=("chebyshev-lobatto", "equidistant"),
        grid_size=41,
        decimal_precision=50,
    )

    assert first == reordered


@pytest.mark.parametrize("kwargs", [
    {"degrees": ()},
    {"degrees": (0,)},
    {"degrees": (True,)},
    {"degrees": (4, 4)},
    {"node_designs": ()},
    {"node_designs": ("unknown",)},
    {"node_designs": ("equidistant", "equidistant")},
    {"node_designs": (1,)},
    {"grid_size": 2},
    {"grid_size": True},
    {"decimal_precision": 29},
    {"decimal_precision": True},
])
def test_invalid_stability_configuration(kwargs):
    with pytest.raises(ValueError):
        run_stability_study(**kwargs)


def test_cli_writes_provenance_and_valid_json(tmp_path):
    output = tmp_path / "stability.json"
    assert main(["--output", str(output), "--code-version", "test-commit"]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["provenance"]["code_version"] == "test-commit"
    assert set(report["provenance"]["source_sha256"]) == {
        "stability.py",
        "interpolation.py",
    }
