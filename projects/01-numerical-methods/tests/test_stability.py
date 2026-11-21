import json

import pytest

from quant_numerical.stability import (
    main,
    run_grid_refinement_study,
    run_stability_study,
)


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
        "estimated_lebesgue_constant",
        "perturbation_magnitude",
        "observed_output_change",
        "observed_amplification",
        "lebesgue_bound_gap",
    ):
        assert row[field] >= 0.0
        assert row[field] < float("inf")
    assert -1.0 <= row["total_error_argmax"] <= 1.0
    assert -1.0 <= row["lebesgue_argmax"] <= 1.0
    assert row["estimated_lebesgue_constant"] >= 1.0
    assert row["perturbation_magnitude"] == 1e-6


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


def test_linear_interpolation_has_lebesgue_constant_one():
    study = run_stability_study(degrees=(1,), grid_size=41, decimal_precision=50)

    assert [row["estimated_lebesgue_constant"] for row in study["rows"]] == [1.0, 1.0]


def test_chebyshev_lobatto_nodes_reduce_degree_16_error_amplification():
    study = run_stability_study(degrees=(16,), grid_size=401, decimal_precision=80)
    equidistant, chebyshev = study["rows"]

    assert equidistant["estimated_lebesgue_constant"] > 100 * chebyshev[
        "estimated_lebesgue_constant"
    ]


def test_adversarial_perturbation_attains_recorded_lebesgue_bound():
    study = run_stability_study(degrees=(16,), grid_size=401, decimal_precision=80)

    for row in study["rows"]:
        assert row["perturbation_target"] == row["lebesgue_argmax"]
        assert row["observed_amplification"] == pytest.approx(
            row["estimated_lebesgue_constant"], rel=1e-14, abs=1e-14
        )
        assert row["observed_output_change"] == pytest.approx(
            row["perturbation_magnitude"] * row["estimated_lebesgue_constant"],
            rel=1e-14,
            abs=1e-14,
        )
        assert row["lebesgue_bound_gap"] < 1e-70


def test_degree_16_lebesgue_conclusion_is_stable_under_grid_refinement():
    study = run_grid_refinement_study()

    assert study["grid_sizes"] == [401, 801, 1601]
    assert len(study["rows"]) == 6
    assert study["conclusion"] == {
        "all_estimates_stable_within_tolerance": True,
        "equidistant_over_100x_chebyshev_on_every_grid": True,
        "stable": True,
        "scope": (
            "an operational grid-sensitivity check, not a proof of the continuous "
            "Lebesgue constants"
        ),
    }
    for summary in study["summaries"]:
        assert summary["coarse_to_fine_relative_change"] <= 0.005
        assert summary["absolute_argmax_movement"] <= 0.005
    assert all(
        row["equidistant_to_chebyshev_ratio"] > 300
        for row in study["design_ratios"]
    )


def test_grid_refinement_study_is_order_independent():
    first = run_grid_refinement_study(
        grid_sizes=(401, 801),
        node_designs=("equidistant", "chebyshev-lobatto"),
        decimal_precision=50,
    )
    reordered = run_grid_refinement_study(
        grid_sizes=(801, 401),
        node_designs=("chebyshev-lobatto", "equidistant"),
        decimal_precision=50,
    )

    assert first == reordered


@pytest.mark.parametrize("kwargs", [
    {"degree": 0},
    {"degree": True},
    {"grid_sizes": ()},
    {"grid_sizes": (2,)},
    {"grid_sizes": (401, 401)},
    {"grid_sizes": (True,)},
    {"node_designs": ()},
    {"node_designs": ("unknown",)},
    {"node_designs": ("equidistant", "equidistant")},
    {"decimal_precision": 29},
    {"relative_stability_tolerance": 0},
    {"relative_stability_tolerance": float("inf")},
    {"relative_stability_tolerance": True},
])
def test_invalid_grid_refinement_configuration(kwargs):
    with pytest.raises(ValueError):
        run_grid_refinement_study(**kwargs)


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
    refinement_output = tmp_path / "grid-refinement.json"
    assert main([
        "--output",
        str(output),
        "--refinement-output",
        str(refinement_output),
        "--code-version",
        "test-commit",
    ]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["provenance"]["code_version"] == "test-commit"
    assert set(report["provenance"]["source_sha256"]) == {
        "stability.py",
        "interpolation.py",
    }
    refinement = json.loads(refinement_output.read_text(encoding="utf-8"))
    assert refinement["provenance"]["code_version"] == "test-commit"
    assert refinement["conclusion"]["stable"] is True
