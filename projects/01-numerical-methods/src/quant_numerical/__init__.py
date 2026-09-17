"""Numerical methods for quantitative research."""

from .interpolation import lagrange_interpolate, lagrange_weights
from .neville import (
    ASSIGNMENT_EXPECTED,
    ASSIGNMENT_TARGET,
    ASSIGNMENT_X,
    ASSIGNMENT_Y,
    NevilleResult,
    NevilleStep,
    format_table,
    neville_interpolate,
    table_grid,
    write_csv,
)

__all__ = [
    "lagrange_interpolate",
    "lagrange_weights",
    "ASSIGNMENT_EXPECTED",
    "ASSIGNMENT_TARGET",
    "ASSIGNMENT_X",
    "ASSIGNMENT_Y",
    "NevilleResult",
    "NevilleStep",
    "format_table",
    "neville_interpolate",
    "table_grid",
    "write_csv",
]

