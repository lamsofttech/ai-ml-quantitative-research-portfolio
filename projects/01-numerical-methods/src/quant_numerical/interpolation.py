"""Lagrange interpolation with explicit validation and vectorized evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_nodes(x_nodes: ArrayLike, y_nodes: ArrayLike) -> tuple[NDArray, NDArray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be one-dimensional")
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("x_nodes and y_nodes must have the same non-zero length")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nodes must contain only finite values")
    if len(np.unique(x)) != len(x):
        raise ValueError("x_nodes must be distinct")
    return x, y


def lagrange_weights(x_nodes: ArrayLike) -> NDArray:
    """Return first-form barycentric weights for distinct interpolation nodes."""
    x = np.asarray(x_nodes, dtype=float)
    _validated_nodes(x, np.zeros_like(x))
    differences = x[:, None] - x[None, :]
    np.fill_diagonal(differences, 1.0)
    return 1.0 / np.prod(differences, axis=1)


def lagrange_interpolate(x_nodes: ArrayLike, y_nodes: ArrayLike, x_eval: ArrayLike):
    """Evaluate the unique interpolating polynomial using barycentric Lagrange form.

    Values evaluated exactly at a node return that node's observed value. This avoids
    division by zero and preserves the defining interpolation property.
    """
    x, y = _validated_nodes(x_nodes, y_nodes)
    points = np.asarray(x_eval, dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("x_eval must contain only finite values")

    flat = points.reshape(-1)
    delta = flat[:, None] - x[None, :]
    exact = delta == 0.0
    weights = lagrange_weights(x)
    result = np.empty_like(flat)

    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        result[exact_rows] = y[np.argmax(exact[exact_rows], axis=1)]
    if (~exact_rows).any():
        terms = weights / delta[~exact_rows]
        result[~exact_rows] = (terms @ y) / terms.sum(axis=1)

    shaped = result.reshape(points.shape)
    return float(shaped) if shaped.ndim == 0 else shaped

