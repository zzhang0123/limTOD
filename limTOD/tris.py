"""Offline readers for the public TRIS archive text products.

The readers in this module intentionally only parse local ASCII files.  They
preserve the archive's right-ascension labels and retain common zero-level
uncertainties separately from per-sample statistical uncertainties.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Optional, Tuple, Union

import numpy as np

_RA_RE = re.compile(r"(\d{1,2})h(\d{2})m(?:(\d{2})s)?\Z")
_RING_FREQUENCIES = {
    0.6: (600.0, 600.5, 0.3),
    0.82: (820.0, 817.8, 0.3),
}


def _readonly_finite_array(values, name):
    """Return a one-dimensional immutable finite float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional array".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    array = array.copy()
    array.setflags(write=False)
    return array


def _validate_frequency_metadata(nominal, effective, bandwidth):
    for name, value in (
        ("nominal_frequency_mhz", nominal),
        ("effective_frequency_mhz", effective),
        ("bandwidth_mhz", bandwidth),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(name))


@dataclass(frozen=True)
class AsymmetricUncertainty:
    """A positive and negative common uncertainty in kelvin."""

    positive_k: float
    negative_k: float

    def __post_init__(self):
        for name, value in (
            ("positive_k", self.positive_k),
            ("negative_k", self.negative_k),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative".format(name))


@dataclass(frozen=True)
class TRISRing:
    """One fixed-declination TRIS drift ring."""

    nominal_frequency_mhz: float
    effective_frequency_mhz: float
    bandwidth_mhz: float
    ra_text: Tuple[str, ...]
    ra_deg: np.ndarray
    temperature_k: np.ndarray
    statistical_uncertainty_k: np.ndarray
    zero_level_uncertainty_k: Union[float, AsymmetricUncertainty]

    def __post_init__(self):
        _validate_frequency_metadata(
            self.nominal_frequency_mhz, self.effective_frequency_mhz, self.bandwidth_mhz
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
        _validate_common_uncertainty(self.zero_level_uncertainty_k)


@dataclass(frozen=True)
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

    def __post_init__(self):
        _validate_frequency_metadata(
            self.nominal_frequency_mhz, self.effective_frequency_mhz, self.bandwidth_mhz
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
        _validate_common_uncertainty(self.zero_level_uncertainty_k)


@dataclass(frozen=True)
class TRISPrincipalPlaneCuts:
    """Peak-relative TRIS H- and E-plane beam cuts in archive dB units."""

    angle_deg: np.ndarray
    h_plane_db: np.ndarray
    e_plane_db: np.ndarray

    def __post_init__(self):
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
    def h_plane_relative_power(self):
        """H-plane power relative to the peak, assuming the archive dB cuts are power."""
        return 10.0 ** (self.h_plane_db / 10.0)

    @property
    def e_plane_relative_power(self):
        """E-plane power relative to the peak, assuming the archive dB cuts are power."""
        return 10.0 ** (self.e_plane_db / 10.0)


def _validate_ra_data(ra_text, ra_deg):
    if not isinstance(ra_text, tuple) or not ra_text:
        raise ValueError("ra_text must be a non-empty tuple")
    values = _readonly_finite_array(ra_deg, "ra_deg")
    if len(ra_text) != values.size:
        raise ValueError("ra_text and ra_deg must have the same length")
    if np.any(values < 0) or np.any(values >= 360):
        raise ValueError("ra_deg must be in [0, 360)")
    _validate_unique_coordinates(values, "ra_deg")


def _validate_matching_samples(*arrays):
    lengths = []
    for array in arrays:
        lengths.append(_readonly_finite_array(array, "sample array").size)
    if len(set(lengths)) != 1:
        raise ValueError("sample arrays must have the same length")


def _validate_unique_coordinates(values, name, source=None, line_numbers=None):
    """Reject repeated coordinates, retaining reader source rows when available."""
    first_indices = {}
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


def _validate_common_uncertainty(value):
    if isinstance(value, AsymmetricUncertainty):
        return
    if (
        not isinstance(value, (int, float, np.floating))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("zero_level_uncertainty_k must be finite and non-negative")


def _source_location(source, line_number=None):
    if source is None:
        return ""
    location = str(source)
    if line_number is not None:
        location = "{}: line {}".format(location, line_number)
    return "{}: ".format(location)


def parse_tris_ra(token, source=None, line_number=None):
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


def _read_ascii_lines(source):
    path = Path(source)
    with path.open("r", encoding="ascii", newline=None) as handle:
        return path, handle.readlines()


def _header_and_rows(source, expected_columns):
    header = []
    rows = []
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


def _parse_finite_float(value, description, source=None, line_number=None):
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


def _find_ring_metadata(source, header):
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


def read_tris_ring(source):
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
    )


def read_tris_point_set(source):
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
    )


def read_tris_beam_cuts(source):
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
