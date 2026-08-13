"""Shared strict-validation helpers and input type aliases for :mod:`limTOD.tris`.

Nothing here is public API.  The helpers exist so that every public TRIS
entry point coerces its inputs the same way: real scalars are detached into
built-in floats, arrays are copied and marked read-only, and non-finite or
boolean values are rejected at the boundary rather than inside the physics.
"""

import math
import numbers as _numbers
import os as _os
from pathlib import Path
import typing as _typing
from typing import Optional, Tuple, Union

import numpy as np

_PathInput = Union[str, Path, _os.PathLike]
_RealScalarInput = Union[int, float, np.integer, np.floating, np.ndarray]
_IntegerScalarInput = Union[int, np.integer]
_VectorLike = Union[np.ndarray, _typing.Sequence[_RealScalarInput]]
_MatrixLike = Union[np.ndarray, _typing.Sequence[_typing.Sequence[_RealScalarInput]]]
_Header = _typing.List[Tuple[int, str]]
_Rows = _typing.List[Tuple[int, _typing.List[str]]]


def _readonly_finite_array(values: _VectorLike, name: str) -> np.ndarray:
    """Return a one-dimensional immutable finite float array.

    The write flag is advisory: NumPy lets a caller re-enable writing on an
    array that owns its buffer, so this protects against accident, not
    against a determined caller.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional array".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    array = array.copy()
    array.setflags(write=False)
    return array


def _readonly_finite_matrix(values: _MatrixLike, name: str) -> np.ndarray:
    """Return an immutable, finite two-dimensional float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} must be a finite two-dimensional array".format(name)
        ) from error
    if array.ndim != 2 or array.size == 0:
        raise ValueError("{} must be a non-empty two-dimensional array".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    array = array.copy()
    array.setflags(write=False)
    return array


def _coerce_real_scalar(value: object, name: str, *, finite: bool = True) -> float:
    """Return a detached built-in float for one strict real scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a real scalar".format(name))
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("{} must be a real scalar".format(name))
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (_numbers.Real, np.integer, np.floating)
    ):
        raise ValueError("{} must be a real scalar".format(name))
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("{} must be a real scalar".format(name)) from error
    if math.isnan(result) or (finite and not math.isfinite(result)):
        requirement = "finite " if finite else "non-NaN "
        raise ValueError("{} must be a {}real scalar".format(name, requirement))
    return result


def _validate_frequency_metadata(
    nominal: _RealScalarInput,
    effective: _RealScalarInput,
    bandwidth: _RealScalarInput,
) -> Tuple[float, float, float]:
    normalized = []
    for name, value in (
        ("nominal_frequency_mhz", nominal),
        ("effective_frequency_mhz", effective),
        ("bandwidth_mhz", bandwidth),
    ):
        scalar = _coerce_real_scalar(value, name)
        if scalar <= 0:
            raise ValueError("{} must be finite and positive".format(name))
        normalized.append(scalar)
    return normalized[0], normalized[1], normalized[2]


def _validate_finite_scalar(value: _RealScalarInput, name: str) -> float:
    return _coerce_real_scalar(value, name)


def _validate_latitude(value: _RealScalarInput, name: str) -> float:
    latitude = _validate_finite_scalar(value, name)
    if latitude < -90.0 or latitude > 90.0:
        raise ValueError("{} must be in [-90, 90] degrees".format(name))
    return latitude


def _validate_positive_scalar(value: _RealScalarInput, name: str) -> float:
    normalized = _coerce_real_scalar(value, name)
    if normalized <= 0.0:
        raise ValueError("{} must be a finite positive scalar".format(name))
    return normalized


def _validate_nonnegative_scalar(value: _RealScalarInput, name: str) -> float:
    normalized = _coerce_real_scalar(value, name)
    if normalized < 0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return normalized


def _validate_optional_positive_scalar(
    value: Optional[_RealScalarInput], name: str
) -> Optional[float]:
    if value is None:
        return None
    return _validate_positive_scalar(value, name)


def _validate_optional_nonnegative_scalar(
    value: Optional[_RealScalarInput], name: str
) -> Optional[float]:
    if value is None:
        return None
    normalized = _coerce_real_scalar(value, name)
    if normalized < 0.0:
        raise ValueError("{} must be a finite non-negative scalar".format(name))
    return normalized


def _validate_matching_samples(*named: Tuple[str, _VectorLike]) -> None:
    """Check that the named sample arrays are valid and equal in length.

    Each argument is a ``(name, values)`` pair so that a malformed column is
    reported under its own field name rather than a placeholder.
    """
    lengths: _typing.List[int] = []
    for name, array in named:
        lengths.append(_readonly_finite_array(array, name).size)
    if len(set(lengths)) != 1:
        detail = ", ".join(
            "{}={}".format(name, length)
            for (name, _values), length in zip(named, lengths)
        )
        raise ValueError("sample arrays must have the same length: {}".format(detail))


def _validate_unique_coordinates(
    values: _VectorLike,
    name: str,
    source: Optional[_PathInput] = None,
    line_numbers: Optional[_typing.Sequence[int]] = None,
) -> None:
    """Reject repeated coordinates, retaining reader source rows when available."""
    first_indices: _typing.Dict[float, int] = {}
    for index, value in enumerate(values):
        coordinate = float(value)
        if coordinate in first_indices:
            first_index = first_indices[coordinate]
            if source is not None and line_numbers is not None:
                raise ValueError(
                    "{}: duplicate {} {} at line {}; first occurs at line {}".format(
                        source,
                        name,
                        coordinate,
                        line_numbers[index],
                        line_numbers[first_index],
                    )
                )
            raise ValueError("duplicate {} {}".format(name, coordinate))
        first_indices[coordinate] = index


def _source_location(
    source: Optional[_PathInput], line_number: Optional[int] = None
) -> str:
    if source is None:
        return ""
    location = str(source)
    if line_number is not None:
        location = "{}: line {}".format(location, line_number)
    return "{}: ".format(location)
