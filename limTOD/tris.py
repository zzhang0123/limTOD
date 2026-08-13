"""Offline readers for the public TRIS archive text products.

The readers in this module intentionally only parse local ASCII files.  They
preserve the archive's right-ascension labels and retain common zero-level
uncertainties separately from per-sample statistical uncertainties.
"""

from dataclasses import dataclass
import math
import numbers as _numbers
import os as _os
from pathlib import Path
import re
import typing as _typing
from typing import Optional, Tuple, Union

import healpy as hp
import numpy as np

_RA_RE = re.compile(r"(\d{1,2})h(\d{2})m(?:(\d{2})s)?\Z")
_RING_FREQUENCIES = {
    0.6: (600.0, 600.5, 0.3),
    0.82: (820.0, 817.8, 0.3),
}

_PathInput = Union[str, Path, _os.PathLike]
_RealScalarInput = Union[int, float, np.integer, np.floating, np.ndarray]
_IntegerScalarInput = Union[int, np.integer]
_VectorLike = Union[np.ndarray, _typing.Sequence[_RealScalarInput]]
_MatrixLike = Union[np.ndarray, _typing.Sequence[_typing.Sequence[_RealScalarInput]]]
_CommonUncertaintyInput = Union[_RealScalarInput, "AsymmetricUncertainty"]
_StoredCommonUncertainty = Union[float, "AsymmetricUncertainty"]
_Rows = _typing.List[Tuple[int, _typing.List[str]]]


class _BeamFunc(_typing.Protocol):
    def __call__(self, *, freq: _RealScalarInput, nside: int) -> np.ndarray: ...


def _readonly_finite_array(values: _VectorLike, name: str) -> np.ndarray:
    """Return a one-dimensional immutable finite float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional array".format(name))
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


@dataclass(frozen=True, init=False)
class AsymmetricUncertainty:
    """A positive and negative common uncertainty in kelvin."""

    positive_k: float
    negative_k: float

    def __init__(
        self, positive_k: _RealScalarInput, negative_k: _RealScalarInput
    ) -> None:
        object.__setattr__(self, "positive_k", positive_k)
        object.__setattr__(self, "negative_k", negative_k)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name, value in (
            ("positive_k", self.positive_k),
            ("negative_k", self.negative_k),
        ):
            normalized = _coerce_real_scalar(value, name)
            if normalized < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True, init=False)
class TRISRing:
    """One fixed-declination TRIS drift ring."""

    nominal_frequency_mhz: float
    effective_frequency_mhz: float
    bandwidth_mhz: float
    ra_text: Tuple[str, ...]
    ra_deg: np.ndarray
    temperature_k: np.ndarray
    statistical_uncertainty_k: np.ndarray
    zero_level_uncertainty_k: _StoredCommonUncertainty
    declination_label_deg: float = 42.0

    def __init__(
        self,
        nominal_frequency_mhz: _RealScalarInput,
        effective_frequency_mhz: _RealScalarInput,
        bandwidth_mhz: _RealScalarInput,
        ra_text: Tuple[str, ...],
        ra_deg: _VectorLike,
        temperature_k: _VectorLike,
        statistical_uncertainty_k: _VectorLike,
        zero_level_uncertainty_k: _CommonUncertaintyInput,
        declination_label_deg: _RealScalarInput = 42.0,
    ) -> None:
        object.__setattr__(self, "nominal_frequency_mhz", nominal_frequency_mhz)
        object.__setattr__(self, "effective_frequency_mhz", effective_frequency_mhz)
        object.__setattr__(self, "bandwidth_mhz", bandwidth_mhz)
        object.__setattr__(self, "ra_text", ra_text)
        object.__setattr__(self, "ra_deg", ra_deg)
        object.__setattr__(self, "temperature_k", temperature_k)
        object.__setattr__(self, "statistical_uncertainty_k", statistical_uncertainty_k)
        object.__setattr__(self, "zero_level_uncertainty_k", zero_level_uncertainty_k)
        object.__setattr__(self, "declination_label_deg", declination_label_deg)
        self.__post_init__()

    def __post_init__(self) -> None:
        nominal, effective, bandwidth = _validate_frequency_metadata(
            self.nominal_frequency_mhz, self.effective_frequency_mhz, self.bandwidth_mhz
        )
        declination = _validate_latitude(
            self.declination_label_deg, "declination_label_deg"
        )
        _validate_ra_data(self.ra_text, self.ra_deg)
        _validate_matching_samples(
            self.ra_deg, self.temperature_k, self.statistical_uncertainty_k
        )
        object.__setattr__(
            self, "ra_deg", _readonly_finite_array(self.ra_deg, "ra_deg")
        )
        object.__setattr__(
            self,
            "temperature_k",
            _readonly_finite_array(self.temperature_k, "temperature_k"),
        )
        statistical = _readonly_finite_array(
            self.statistical_uncertainty_k, "statistical_uncertainty_k"
        )
        if np.any(statistical < 0):
            raise ValueError("statistical_uncertainty_k must be non-negative")
        object.__setattr__(self, "statistical_uncertainty_k", statistical)
        object.__setattr__(self, "nominal_frequency_mhz", nominal)
        object.__setattr__(self, "effective_frequency_mhz", effective)
        object.__setattr__(self, "bandwidth_mhz", bandwidth)
        object.__setattr__(self, "declination_label_deg", declination)
        object.__setattr__(
            self,
            "zero_level_uncertainty_k",
            _validate_common_uncertainty(self.zero_level_uncertainty_k),
        )


@dataclass(frozen=True, init=False)
class TRISPointSet:
    """The sparse 2.5-GHz TRIS measurements."""

    nominal_frequency_mhz: float
    effective_frequency_mhz: float
    bandwidth_mhz: float
    ra_text: Tuple[str, ...]
    ra_deg: np.ndarray
    temperature_k: np.ndarray
    statistical_uncertainty_k: Optional[np.ndarray]
    zero_level_uncertainty_k: float
    declination_label_deg: float = 42.0

    def __init__(
        self,
        nominal_frequency_mhz: _RealScalarInput,
        effective_frequency_mhz: _RealScalarInput,
        bandwidth_mhz: _RealScalarInput,
        ra_text: Tuple[str, ...],
        ra_deg: _VectorLike,
        temperature_k: _VectorLike,
        statistical_uncertainty_k: Optional[_VectorLike],
        zero_level_uncertainty_k: _RealScalarInput,
        declination_label_deg: _RealScalarInput = 42.0,
    ) -> None:
        object.__setattr__(self, "nominal_frequency_mhz", nominal_frequency_mhz)
        object.__setattr__(self, "effective_frequency_mhz", effective_frequency_mhz)
        object.__setattr__(self, "bandwidth_mhz", bandwidth_mhz)
        object.__setattr__(self, "ra_text", ra_text)
        object.__setattr__(self, "ra_deg", ra_deg)
        object.__setattr__(self, "temperature_k", temperature_k)
        object.__setattr__(self, "statistical_uncertainty_k", statistical_uncertainty_k)
        object.__setattr__(self, "zero_level_uncertainty_k", zero_level_uncertainty_k)
        object.__setattr__(self, "declination_label_deg", declination_label_deg)
        self.__post_init__()

    def __post_init__(self) -> None:
        nominal, effective, bandwidth = _validate_frequency_metadata(
            self.nominal_frequency_mhz, self.effective_frequency_mhz, self.bandwidth_mhz
        )
        declination = _validate_latitude(
            self.declination_label_deg, "declination_label_deg"
        )
        _validate_ra_data(self.ra_text, self.ra_deg)
        _validate_matching_samples(self.ra_deg, self.temperature_k)
        object.__setattr__(
            self, "ra_deg", _readonly_finite_array(self.ra_deg, "ra_deg")
        )
        object.__setattr__(
            self,
            "temperature_k",
            _readonly_finite_array(self.temperature_k, "temperature_k"),
        )
        if self.statistical_uncertainty_k is not None:
            _validate_matching_samples(self.ra_deg, self.statistical_uncertainty_k)
            statistical = _readonly_finite_array(
                self.statistical_uncertainty_k, "statistical_uncertainty_k"
            )
            if np.any(statistical < 0):
                raise ValueError("statistical_uncertainty_k must be non-negative")
            object.__setattr__(self, "statistical_uncertainty_k", statistical)
        object.__setattr__(self, "nominal_frequency_mhz", nominal)
        object.__setattr__(self, "effective_frequency_mhz", effective)
        object.__setattr__(self, "bandwidth_mhz", bandwidth)
        object.__setattr__(self, "declination_label_deg", declination)
        object.__setattr__(
            self,
            "zero_level_uncertainty_k",
            _validate_common_uncertainty(self.zero_level_uncertainty_k),
        )


@dataclass(frozen=True, init=False)
class TRISPrincipalPlaneCuts:
    """Peak-relative TRIS H- and E-plane beam cuts in archive dB units."""

    angle_deg: np.ndarray
    h_plane_db: np.ndarray
    e_plane_db: np.ndarray

    def __init__(
        self,
        angle_deg: _VectorLike,
        h_plane_db: _VectorLike,
        e_plane_db: _VectorLike,
    ) -> None:
        object.__setattr__(self, "angle_deg", angle_deg)
        object.__setattr__(self, "h_plane_db", h_plane_db)
        object.__setattr__(self, "e_plane_db", e_plane_db)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_matching_samples(self.angle_deg, self.h_plane_db, self.e_plane_db)
        _validate_unique_coordinates(self.angle_deg, "angle_deg")
        object.__setattr__(
            self, "angle_deg", _readonly_finite_array(self.angle_deg, "angle_deg")
        )
        object.__setattr__(
            self, "h_plane_db", _readonly_finite_array(self.h_plane_db, "h_plane_db")
        )
        object.__setattr__(
            self, "e_plane_db", _readonly_finite_array(self.e_plane_db, "e_plane_db")
        )

    @property
    def h_plane_relative_power(self) -> np.ndarray:
        """H-plane power relative to the peak, assuming the archive dB cuts are power."""
        return 10.0 ** (self.h_plane_db / 10.0)

    @property
    def e_plane_relative_power(self) -> np.ndarray:
        """E-plane power relative to the peak, assuming the archive dB cuts are power."""
        return 10.0 ** (self.e_plane_db / 10.0)


def _validate_ra_data(ra_text: Tuple[str, ...], ra_deg: _VectorLike) -> None:
    if not isinstance(ra_text, tuple) or not ra_text:
        raise ValueError("ra_text must be a non-empty tuple")
    values = _readonly_finite_array(ra_deg, "ra_deg")
    if len(ra_text) != values.size:
        raise ValueError("ra_text and ra_deg must have the same length")
    if np.any(values < 0) or np.any(values >= 360):
        raise ValueError("ra_deg must be in [0, 360)")
    _validate_unique_coordinates(values, "ra_deg")


def _validate_matching_samples(*arrays: _VectorLike) -> None:
    lengths: _typing.List[int] = []
    for array in arrays:
        lengths.append(_readonly_finite_array(array, "sample array").size)
    if len(set(lengths)) != 1:
        raise ValueError("sample arrays must have the same length")


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


def _validate_common_uncertainty(
    value: _CommonUncertaintyInput,
) -> _StoredCommonUncertainty:
    if isinstance(value, AsymmetricUncertainty):
        return value
    normalized = _coerce_real_scalar(value, "zero_level_uncertainty_k")
    if normalized < 0:
        raise ValueError("zero_level_uncertainty_k must be finite and non-negative")
    return normalized


def _source_location(
    source: Optional[_PathInput], line_number: Optional[int] = None
) -> str:
    if source is None:
        return ""
    location = str(source)
    if line_number is not None:
        location = "{}: line {}".format(location, line_number)
    return "{}: ".format(location)


def parse_tris_ra(
    token: str,
    source: Optional[_PathInput] = None,
    line_number: Optional[int] = None,
) -> float:
    """Convert an archive ``hh hmm`` or ``hh hmm ss`` token to degrees."""
    match = _RA_RE.fullmatch(token)
    if match is None:
        raise ValueError(
            "{}invalid TRIS right-ascension token: {!r}".format(
                _source_location(source, line_number), token
            )
        )
    hour, minute, second = (
        int(value) if value is not None else 0 for value in match.groups()
    )
    if hour >= 24 or minute >= 60 or second >= 60:
        raise ValueError(
            "{}TRIS right ascension is out of range: {!r}".format(
                _source_location(source, line_number), token
            )
        )
    return 15.0 * (hour + minute / 60.0 + second / 3600.0)


def _read_ascii_lines(
    source: _PathInput,
) -> Tuple[Path, _typing.List[str]]:
    path = Path(source)
    with path.open("r", encoding="ascii", newline=None) as handle:
        return path, handle.readlines()


def _header_and_rows(
    source: _PathInput, expected_columns: int
) -> Tuple[Path, _typing.List[str], _Rows]:
    header: _typing.List[str] = []
    rows: _Rows = []
    path, lines = _read_ascii_lines(source)
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header.append(stripped)
            continue
        fields = stripped.split()
        if len(fields) != expected_columns:
            raise ValueError(
                "{}: line {} has {} columns; expected {}".format(
                    path, line_number, len(fields), expected_columns
                )
            )
        rows.append((line_number, fields))
    if not rows:
        raise ValueError("{}: TRIS archive file contains no data rows".format(path))
    return path, header, rows


def _parse_finite_float(
    value: str,
    description: str,
    source: Optional[_PathInput] = None,
    line_number: Optional[int] = None,
) -> float:
    try:
        result = float(value)
    except ValueError:
        raise ValueError(
            "{}{} must be numeric: {!r}".format(
                _source_location(source, line_number), description, value
            )
        )
    if not math.isfinite(result):
        raise ValueError(
            "{}{} must be finite".format(
                _source_location(source, line_number), description
            )
        )
    return result


def _find_ring_metadata(
    source: _PathInput, header: _typing.Sequence[str]
) -> Tuple[float, float, float, _StoredCommonUncertainty]:
    joined = "\n".join(header)
    frequency_match = re.search(
        r"Frequency\s*=\s*([0-9.]+)\s*GHz", joined, re.IGNORECASE
    )
    if frequency_match is None:
        raise ValueError("{}: ring file is missing its frequency header".format(source))
    frequency_ghz = _parse_finite_float(frequency_match.group(1), "frequency")
    try:
        frequency = _RING_FREQUENCIES[frequency_ghz]
    except KeyError:
        raise ValueError(
            "{}: unsupported TRIS ring frequency: {} GHz".format(source, frequency_ghz)
        )
    zero_match = re.search(
        r"Systematic Zero Level Uncertainty\s*=\s*([^\n]+)", joined, re.IGNORECASE
    )
    if zero_match is None:
        raise ValueError(
            "{}: ring file is missing its zero-level uncertainty header".format(source)
        )
    zero_text = zero_match.group(1).strip()
    zero_level: _StoredCommonUncertainty
    asymmetry = re.fullmatch(r"\+?([0-9.]+)K\s*/\s*-([0-9.]+)K(?:\s*.*)?", zero_text)
    if asymmetry is not None:
        zero_level = AsymmetricUncertainty(
            positive_k=_parse_finite_float(
                asymmetry.group(1), "positive zero-level uncertainty"
            ),
            negative_k=_parse_finite_float(
                asymmetry.group(2), "negative zero-level uncertainty"
            ),
        )
    else:
        single = re.match(r"([0-9.]+)K(?:\s*.*)?\Z", zero_text)
        if single is None:
            raise ValueError(
                "{}: invalid zero-level uncertainty header: {!r}".format(
                    source, zero_text
                )
            )
        zero_level = _parse_finite_float(single.group(1), "zero-level uncertainty")
    return frequency + (zero_level,)


def read_tris_ring(source: _PathInput) -> TRISRing:
    """Read a local 600- or 820-MHz TRIS absolute-temperature drift ring."""
    path, header, rows = _header_and_rows(source, expected_columns=3)
    nominal, effective, bandwidth, zero_level = _find_ring_metadata(path, header)
    ra_text = tuple(row[0] for _line_number, row in rows)
    ra_deg = [parse_tris_ra(row[0], path, line_number) for line_number, row in rows]
    _validate_unique_coordinates(
        ra_deg, "ra_deg", path, [line_number for line_number, _row in rows]
    )
    temperature = [
        _parse_finite_float(row[1], "temperature", path, line_number)
        for line_number, row in rows
    ]
    statistical = [
        _parse_finite_float(row[2], "statistical uncertainty", path, line_number)
        for line_number, row in rows
    ]
    for (line_number, _row), value in zip(rows, statistical):
        if value < 0:
            raise ValueError(
                "{}: line {} statistical uncertainty must be non-negative".format(
                    path, line_number
                )
            )
    return TRISRing(
        nominal_frequency_mhz=nominal,
        effective_frequency_mhz=effective,
        bandwidth_mhz=bandwidth,
        ra_text=ra_text,
        ra_deg=np.asarray(ra_deg),
        temperature_k=np.asarray(temperature),
        statistical_uncertainty_k=np.asarray(statistical),
        zero_level_uncertainty_k=zero_level,
        declination_label_deg=42.0,
    )


def read_tris_point_set(source: _PathInput) -> TRISPointSet:
    """Read the local sparse 2.5-GHz TRIS absolute-temperature samples."""
    path, header, rows = _header_and_rows(source, expected_columns=3)
    if not re.search(r"2\.5\s*GHz", "\n".join(header), re.IGNORECASE):
        raise ValueError(
            "{}: point-set file is missing its 2.5-GHz header".format(path)
        )
    ra_text = tuple(row[0] for _line_number, row in rows)
    ra_deg = [parse_tris_ra(row[0], path, line_number) for line_number, row in rows]
    _validate_unique_coordinates(
        ra_deg, "ra_deg", path, [line_number for line_number, _row in rows]
    )
    temperature = [
        _parse_finite_float(row[1], "temperature", path, line_number)
        for line_number, row in rows
    ]
    common_uncertainties = [
        _parse_finite_float(row[2], "zero-level uncertainty", path, line_number)
        for line_number, row in rows
    ]
    for (line_number, _row), value in zip(rows, common_uncertainties):
        if value < 0:
            raise ValueError(
                "{}: line {} zero-level uncertainty must be non-negative".format(
                    path, line_number
                )
            )
    first_line_number = rows[0][0]
    first_value = common_uncertainties[0]
    for (line_number, _row), value in zip(rows[1:], common_uncertainties[1:]):
        if value != first_value:
            raise ValueError(
                "{}: 2.5-GHz zero-level uncertainty must be common; {} at line {} "
                "differs from {} at line {}".format(
                    path, first_value, first_line_number, value, line_number
                )
            )
    return TRISPointSet(
        nominal_frequency_mhz=2500.0,
        effective_frequency_mhz=2427.8,
        bandwidth_mhz=3.0,
        ra_text=ra_text,
        ra_deg=np.asarray(ra_deg),
        temperature_k=np.asarray(temperature),
        statistical_uncertainty_k=None,
        zero_level_uncertainty_k=common_uncertainties[0],
        declination_label_deg=42.0,
    )


def read_tris_beam_cuts(source: _PathInput) -> TRISPrincipalPlaneCuts:
    """Read local TRIS H- and E-principal-plane beam cuts in raw dB units."""
    path, _header, rows = _header_and_rows(source, expected_columns=3)
    angle = [
        _parse_finite_float(row[0], "beam angle", path, line_number)
        for line_number, row in rows
    ]
    _validate_unique_coordinates(
        angle, "angle_deg", path, [line_number for line_number, _row in rows]
    )
    h_cut = [
        _parse_finite_float(row[1], "H-plane dB cut", path, line_number)
        for line_number, row in rows
    ]
    e_cut = [
        _parse_finite_float(row[2], "E-plane dB cut", path, line_number)
        for line_number, row in rows
    ]
    return TRISPrincipalPlaneCuts(
        angle_deg=np.asarray(angle),
        h_plane_db=np.asarray(h_cut),
        e_plane_db=np.asarray(e_cut),
    )


def approximate_tris_gaussian_beam_map(
    *,
    nside: int,
    fwhm_e_deg: _RealScalarInput = 18.0,
    fwhm_h_deg: _RealScalarInput = 23.0,
    normalization: str = "peak",
) -> np.ndarray:
    """Return an approximate scalar TRIS main-lobe HEALPix RING beam map.

    The archive supplies only E- and H-principal-plane cuts, so this is an
    explicitly approximate elliptical Gaussian: its intrinsic E axis is
    ``phi=0/180`` and its H axis is ``phi=90/270``.  This two-dimensional
    beam is an approximation.  ``normalization`` may be ``"peak"`,
    ``"sum"``, or ``"none"``; ``"sum"`` uses limTOD's discrete HEALPix sum.
    """
    normalized_e = _validate_finite_scalar(fwhm_e_deg, "fwhm_e_deg")
    normalized_h = _validate_finite_scalar(fwhm_h_deg, "fwhm_h_deg")
    if normalized_e <= 0.0:
        raise ValueError("fwhm_e_deg must be positive")
    if normalized_h <= 0.0:
        raise ValueError("fwhm_h_deg must be positive")
    if normalization not in ("peak", "sum", "none"):
        raise ValueError('normalization must be "peak", "sum", or "none"')

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    sigma_e = np.deg2rad(normalized_e / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    sigma_h = np.deg2rad(normalized_h / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    e_offset = theta * np.cos(phi)
    h_offset = theta * np.sin(phi)
    beam_map = np.exp(-0.5 * ((e_offset / sigma_e) ** 2 + (h_offset / sigma_h) ** 2))

    if normalization == "peak":
        beam_map /= np.max(beam_map)
    elif normalization == "sum":
        beam_map /= np.sum(beam_map)
    return beam_map


def tris_beam_func(
    *,
    fwhm_e_deg: _RealScalarInput = 18.0,
    fwhm_h_deg: _RealScalarInput = 23.0,
    normalization: str = "peak",
) -> _BeamFunc:
    """Return an achromatic callable for the approximate scalar TRIS beam.

    The returned ``beam_func(*, freq, nside)`` follows limTOD's existing
    keyword-only protocol.  It validates a positive finite MHz frequency but
    deliberately does not use it: the public archive states one common beam,
    and the returned two-dimensional Gaussian is only an approximation.
    """

    normalized_e = _validate_finite_scalar(fwhm_e_deg, "fwhm_e_deg")
    normalized_h = _validate_finite_scalar(fwhm_h_deg, "fwhm_h_deg")
    if normalized_e <= 0.0:
        raise ValueError("fwhm_e_deg must be positive")
    if normalized_h <= 0.0:
        raise ValueError("fwhm_h_deg must be positive")

    def beam_func(*, freq: _RealScalarInput, nside: int) -> np.ndarray:
        normalized_freq = _validate_finite_scalar(freq, "freq")
        if normalized_freq <= 0.0:
            raise ValueError("freq must be finite and positive MHz")
        return approximate_tris_gaussian_beam_map(
            nside=nside,
            fwhm_e_deg=normalized_e,
            fwhm_h_deg=normalized_h,
            normalization=normalization,
        )

    return beam_func


@dataclass(frozen=True, init=False)
class TRISZenithGeometry:
    """Immutable zenith geometry for the approximate TRIS drift-ring bridge."""

    lst_deg: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    selfrot_deg: np.ndarray
    latitude_deg: float

    def __init__(
        self,
        lst_deg: _VectorLike,
        azimuth_deg: _VectorLike,
        elevation_deg: _VectorLike,
        selfrot_deg: _VectorLike,
        latitude_deg: _RealScalarInput,
    ) -> None:
        object.__setattr__(self, "lst_deg", lst_deg)
        object.__setattr__(self, "azimuth_deg", azimuth_deg)
        object.__setattr__(self, "elevation_deg", elevation_deg)
        object.__setattr__(self, "selfrot_deg", selfrot_deg)
        object.__setattr__(self, "latitude_deg", latitude_deg)
        self.__post_init__()

    def __post_init__(self) -> None:
        latitude = _validate_latitude(self.latitude_deg, "latitude_deg")
        arrays = (
            ("lst_deg", self.lst_deg),
            ("azimuth_deg", self.azimuth_deg),
            ("elevation_deg", self.elevation_deg),
            ("selfrot_deg", self.selfrot_deg),
        )
        validated = [
            (name, _readonly_finite_array(array, name)) for name, array in arrays
        ]
        lengths = {array.size for _name, array in validated}
        if len(lengths) != 1:
            raise ValueError("TRIS geometry arrays must have the same length")
        for name, array in validated:
            object.__setattr__(self, name, array)
        object.__setattr__(self, "latitude_deg", latitude)


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


@dataclass(frozen=True, init=False)
class TRISRankDiagnostic:
    """Immutable SVD identifiability diagnostic for a whitened design."""

    singular_values: np.ndarray
    numerical_rank: int
    parameter_count: int
    tolerance: float
    rank_rtol: float
    condition_number: float

    def __init__(
        self,
        singular_values: _VectorLike,
        numerical_rank: _IntegerScalarInput,
        parameter_count: _IntegerScalarInput,
        tolerance: _RealScalarInput,
        rank_rtol: _RealScalarInput,
        condition_number: _RealScalarInput,
    ) -> None:
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "numerical_rank", numerical_rank)
        object.__setattr__(self, "parameter_count", parameter_count)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "condition_number", condition_number)
        self.__post_init__()

    def __post_init__(self) -> None:
        singular_values = _readonly_finite_array(
            self.singular_values, "singular_values"
        )
        if (
            not isinstance(self.numerical_rank, (int, np.integer))
            or isinstance(self.numerical_rank, bool)
            or self.numerical_rank < 0
            or self.numerical_rank > singular_values.size
        ):
            raise ValueError("numerical_rank must be a valid non-negative integer")
        if (
            not isinstance(self.parameter_count, (int, np.integer))
            or isinstance(self.parameter_count, bool)
            or self.parameter_count <= 0
        ):
            raise ValueError("parameter_count must be a positive integer")
        tolerance = _coerce_real_scalar(self.tolerance, "tolerance")
        rank_rtol = _coerce_real_scalar(self.rank_rtol, "rank_rtol")
        condition_number = _coerce_real_scalar(
            self.condition_number, "condition_number", finite=False
        )
        if tolerance < 0.0 or rank_rtol <= 0.0:
            raise ValueError("tolerance must be non-negative and rank_rtol positive")
        if condition_number < 1.0:
            raise ValueError("condition_number must be at least one")
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "numerical_rank", int(self.numerical_rank))
        object.__setattr__(self, "parameter_count", int(self.parameter_count))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "condition_number", condition_number)


@dataclass(frozen=True, init=False)
class TRISLinearFit:
    """Immutable rank-gated generalized least-squares fit to one TRIS ring."""

    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    prediction_k: np.ndarray
    residual_k: np.ndarray
    rank_diagnostic: TRISRankDiagnostic

    def __init__(
        self,
        coefficients: _VectorLike,
        coefficient_covariance: _MatrixLike,
        prediction_k: _VectorLike,
        residual_k: _VectorLike,
        rank_diagnostic: TRISRankDiagnostic,
    ) -> None:
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "coefficient_covariance", coefficient_covariance)
        object.__setattr__(self, "prediction_k", prediction_k)
        object.__setattr__(self, "residual_k", residual_k)
        object.__setattr__(self, "rank_diagnostic", rank_diagnostic)
        self.__post_init__()

    def __post_init__(self) -> None:
        coefficients = _readonly_finite_array(self.coefficients, "coefficients")
        covariance = _readonly_finite_matrix(
            self.coefficient_covariance, "coefficient_covariance"
        )
        if covariance.shape != (coefficients.size, coefficients.size):
            raise ValueError(
                "coefficient_covariance must be square in coefficient count"
            )
        prediction = _readonly_finite_array(self.prediction_k, "prediction_k")
        residual = _readonly_finite_array(self.residual_k, "residual_k")
        if prediction.size != residual.size:
            raise ValueError("prediction_k and residual_k must have the same length")
        if not isinstance(self.rank_diagnostic, TRISRankDiagnostic):
            raise ValueError("rank_diagnostic must be a TRISRankDiagnostic")
        if self.rank_diagnostic.parameter_count != coefficients.size:
            raise ValueError("rank diagnostic and coefficient count must agree")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "coefficient_covariance", covariance)
        object.__setattr__(self, "prediction_k", prediction)
        object.__setattr__(self, "residual_k", residual)

    @property
    def singular_values(self) -> np.ndarray:
        """Whitened-design singular values from the rank diagnostic."""
        return self.rank_diagnostic.singular_values

    @property
    def numerical_rank(self) -> int:
        """Whitened-design numerical rank."""
        return self.rank_diagnostic.numerical_rank

    @property
    def parameter_count(self) -> int:
        """Number of fitted template coefficients."""
        return self.rank_diagnostic.parameter_count

    @property
    def tolerance(self) -> float:
        """SVD rank tolerance."""
        return self.rank_diagnostic.tolerance

    @property
    def rank_rtol(self) -> float:
        """Relative SVD rank tolerance."""
        return self.rank_diagnostic.rank_rtol

    @property
    def condition_number(self) -> float:
        """Whitened-design condition number."""
        return self.rank_diagnostic.condition_number


def build_tris_fourier_design(
    ra_deg: _VectorLike, m_max: int, *, include_constant: bool = True
) -> np.ndarray:
    """Build a finite low-dimensional Fourier design at the supplied TRIS RAs.

    With ``include_constant=True``, columns are ordered as ``[1, cos(alpha),
    sin(alpha), ..., cos(m alpha), sin(m alpha)]`` with ``alpha`` in radians.
    The supplied sampling is used verbatim; no uniform-grid approximation is
    applied.
    """
    if not isinstance(m_max, (int, np.integer)) or isinstance(m_max, bool) or m_max < 0:
        raise ValueError("m_max must be a non-negative integer")
    if not isinstance(include_constant, (bool, np.bool_)):
        raise ValueError("include_constant must be boolean")
    ra = _readonly_finite_array(ra_deg, "ra_deg")
    if np.any(ra < 0.0) or np.any(ra >= 360.0):
        raise ValueError("ra_deg must be in [0, 360)")

    alpha = np.deg2rad(ra)
    columns: _typing.List[np.ndarray] = []
    if include_constant:
        columns.append(np.ones(ra.size))
    for mode in range(1, int(m_max) + 1):
        columns.extend((np.cos(mode * alpha), np.sin(mode * alpha)))
    if not columns:
        return np.empty((ra.size, 0), dtype=float)
    design = np.column_stack(columns)
    design.setflags(write=False)
    return design


def _validate_optional_positive_scalar(
    value: Optional[_RealScalarInput], name: str
) -> Optional[float]:
    if value is None:
        return None
    normalized = _coerce_real_scalar(value, name)
    if normalized <= 0.0:
        raise ValueError("{} must be a finite positive scalar".format(name))
    return normalized


def _validate_optional_nonnegative_scalar(
    value: Optional[_RealScalarInput], name: str
) -> Optional[float]:
    if value is None:
        return None
    normalized = _coerce_real_scalar(value, name)
    if normalized < 0.0:
        raise ValueError("{} must be a finite non-negative scalar".format(name))
    return normalized


def fit_tris_linear_model(
    ring: TRISRing,
    design_matrix: _MatrixLike,
    *,
    uncertainty_floor_k: Optional[_RealScalarInput] = None,
    common_mode_sigma_k: Optional[_RealScalarInput] = None,
    rank_rtol: Optional[_RealScalarInput] = None,
) -> TRISLinearFit:
    """Fit an identifiable caller-supplied template model to one ``TRISRing``.

    The data design is Cholesky-whitened using the per-row statistical errors
    and, only if requested, a symmetric common-mode covariance.  Its SVD rank
    is checked before coefficients are solved, preventing this API from being
    used as a free-pixel mapmaker.  Archive zero-level metadata is deliberately
    not read or converted by this function.
    """
    if not isinstance(ring, TRISRing):
        raise TypeError("fit_tris_linear_model requires a TRISRing")
    design = _readonly_finite_matrix(design_matrix, "design_matrix")
    sample_count, parameter_count = design.shape
    if parameter_count == 0:
        raise ValueError("design_matrix must contain at least one parameter column")
    if sample_count != ring.temperature_k.size:
        raise ValueError("design_matrix row count must match the TRISRing samples")
    if parameter_count >= sample_count:
        raise ValueError(
            "design must be low-dimensional: parameter count={}, sample count={}; "
            "reduce the model before fitting".format(parameter_count, sample_count)
        )

    floor = _validate_optional_positive_scalar(
        uncertainty_floor_k, "uncertainty_floor_k"
    )
    common_mode = _validate_optional_nonnegative_scalar(
        common_mode_sigma_k, "common_mode_sigma_k"
    )
    supplied_rtol = _validate_optional_positive_scalar(rank_rtol, "rank_rtol")
    statistical = ring.statistical_uncertainty_k
    if np.any(statistical == 0.0) and floor is None:
        raise ValueError(
            "zero statistical uncertainty requires a positive uncertainty_floor_k"
        )
    if floor is not None:
        statistical = np.maximum(statistical, floor)

    covariance = np.diag(statistical**2)
    if common_mode is not None:
        covariance = covariance + common_mode**2 * np.ones((sample_count, sample_count))
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("statistical covariance must be positive definite") from error
    whitened_design = np.linalg.solve(cholesky, design)
    whitened_data = np.linalg.solve(cholesky, ring.temperature_k)
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        whitened_design, full_matrices=False
    )
    applied_rtol = (
        np.finfo(float).eps * max(whitened_design.shape)
        if supplied_rtol is None
        else supplied_rtol
    )
    tolerance = applied_rtol * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float("inf")
        if numerical_rank < parameter_count
        else float(singular_values[0] / singular_values[-1])
    )
    diagnostic = TRISRankDiagnostic(
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        parameter_count=parameter_count,
        tolerance=tolerance,
        rank_rtol=applied_rtol,
        condition_number=condition_number,
    )
    if numerical_rank < parameter_count:
        raise ValueError(
            "rank-deficient whitened design: rank={}, parameter count={}; "
            "reduce the model before fitting".format(numerical_rank, parameter_count)
        )

    coefficients = right_vectors_t.T @ (
        (left_vectors.T @ whitened_data) / singular_values
    )
    coefficient_covariance = (
        right_vectors_t.T / (singular_values**2)
    ) @ right_vectors_t
    prediction = design @ coefficients
    return TRISLinearFit(
        coefficients=coefficients,
        coefficient_covariance=coefficient_covariance,
        prediction_k=prediction,
        residual_k=ring.temperature_k - prediction,
        rank_diagnostic=diagnostic,
    )


def tris_zenith_geometry(
    ra_deg: _VectorLike,
    *,
    latitude_deg: _RealScalarInput = 42.0 + 26.0 / 60.0,
    e_plane_east_of_meridian_deg: _RealScalarInput = 7.0,
) -> TRISZenithGeometry:
    """Translate TRIS RA labels to its approximate parked-zenith geometry.

    Supplied RA samples are preserved as LST samples.  The park is azimuth
    zero and elevation 90 degrees, while the E plane lies east of the
    meridian, so limTOD's roll convention uses
    ``selfrot=-e_plane_east_of_meridian_deg``.  The default latitude is the
    measured 42 deg 26 arcmin site latitude; callers may explicitly request
    42 degrees for the rounded archive declination-label approximation.
    """
    latitude = _validate_latitude(latitude_deg, "latitude_deg")
    e_plane_offset = _validate_finite_scalar(
        e_plane_east_of_meridian_deg, "e_plane_east_of_meridian_deg"
    )
    lst_deg = _readonly_finite_array(ra_deg, "ra_deg")
    ntime = lst_deg.size
    return TRISZenithGeometry(
        lst_deg=lst_deg,
        azimuth_deg=np.zeros(ntime),
        elevation_deg=np.full(ntime, 90.0),
        selfrot_deg=np.full(ntime, -e_plane_offset),
        latitude_deg=latitude,
    )
