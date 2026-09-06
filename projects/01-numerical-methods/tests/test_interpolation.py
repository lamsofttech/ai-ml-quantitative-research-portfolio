import numpy as np
import pytest

from quant_numerical import lagrange_interpolate, lagrange_weights


def test_interpolant_passes_through_nodes():
    x = np.array([-1.0, 0.0, 2.0])
    y = x**2 + 2 * x + 3
    np.testing.assert_allclose(lagrange_interpolate(x, y, x), y)


def test_recovers_quadratic_between_nodes():
    x = np.array([-2.0, 0.0, 1.0])
    y = 3 * x**2 - 2 * x + 4
    evaluation = np.linspace(-2.0, 1.0, 25)
    expected = 3 * evaluation**2 - 2 * evaluation + 4
    np.testing.assert_allclose(lagrange_interpolate(x, y, evaluation), expected)


def test_scalar_evaluation_returns_float():
    value = lagrange_interpolate([0, 1], [1, 3], 0.5)
    assert isinstance(value, float)
    assert value == pytest.approx(2.0)


def test_weights_for_two_nodes():
    np.testing.assert_allclose(lagrange_weights([0, 1]), [-1, 1])


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [([], [], "non-zero"), ([0, 0], [1, 2], "distinct"), ([0], [1, 2], "same non-zero")],
)
def test_rejects_invalid_nodes(x, y, message):
    with pytest.raises(ValueError, match=message):
        lagrange_interpolate(x, y, 0.5)

