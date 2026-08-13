"""Offline readers and typed models for the public TRIS archive text products.

The readers only ever parse local ASCII files.  They preserve the archive's
right-ascension labels verbatim and keep the common zero-level uncertainty
separate from the per-sample statistical uncertainties, because those two
quantities enter a likelihood in completely different ways.

Provenance of the non-file metadata
-----------------------------------
The effective frequencies and bandwidths below are **not** stated anywhere in
the four archive text files.  They come from the reference the archive itself
names in every ring header, M. Zannoni et al., *ApJ* 688:12-23 (2008)
(``arXiv:0806.1415``), Sect. 2 and Tab. 1.  The file-name frequencies
(600/820/2500 MHz) are labels, not band centres: each effective frequency lies
outside its own nominal band, which is expected because the label rounds to the
advertised channel while the effective value is the measured band centre.
Verify these against the paper before using them for a spectral index -- a
2 MHz error at 600 MHz shifts a synchrotron index by about 0.03.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import typing as _typing
from typing import Optional, Tuple, Union

import numpy as np

from ._validate import (
    _coerce_real_scalar,
    _Header,
    _PathInput,
    _readonly_finite_array,
    _RealScalarInput,
    _Rows,
    _source_location,
    _validate_frequency_metadata,
    _validate_latitude,
    _validate_matching_samples,
    _validate_nonnegative_scalar,
    _validate_unique_coordinates,
    _VectorLike,
)

_RA_RE = re.compile(r"(\d{1,2})h(\d{2})m(?:(\d{2})s)?\Z")

#: ``header GHz -> (nominal MHz, effective MHz, bandwidth MHz)``.
#: See the module docstring for the provenance of the effective values.
_RING_FREQUENCIES = {
    0.6: (600.0, 600.5, 0.3),
    0.82: (820.0, 817.8, 0.3),
}

_CommonUncertaintyInput = Union[_RealScalarInput, "AsymmetricUncertainty"]
_StoredCommonUncertainty = Union[float, "AsymmetricUncertainty"]


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


def _validate_common_uncertainty(
    value: _CommonUncertaintyInput,
) -> _StoredCommonUncertainty:
    if isinstance(value, AsymmetricUncertainty):
        return value
    normalized = _coerce_real_scalar(value, "zero_level_uncertainty_k")
    if normalized < 0:
        raise ValueError("zero_level_uncertainty_k must be finite and non-negative")
    return normalized


# ``eq=False`` on every array-carrying model below: the generated dataclass
# ``__eq__`` compares field tuples, which raises
# "truth value of an array ... is ambiguous" for ndarray fields, and the
# generated ``__hash__`` raises TypeError while ``isinstance(obj, Hashable)``
# still reports True. Identity comparison and identity hashing are both
# well-defined, so these models use them.
@dataclass(frozen=True, init=False, eq=False)
class TRISRing:
    """One fixed-declination TRIS drift ring.

    ``temperature_k`` is the published absolute sky brightness: a
    Rayleigh-Jeans (antenna) temperature that **includes the CMB monopole**.
    See :func:`limTOD.tris.cmb_monopole_rj_k` and ``docs/tris.md`` for how that
    was established from the data rather than from the archive headers.
    """

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
            ("ra_deg", self.ra_deg),
            ("temperature_k", self.temperature_k),
            ("statistical_uncertainty_k", self.statistical_uncertainty_k),
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


@dataclass(frozen=True, init=False, eq=False)
class TRISPointSet:
    """The sparse 2.5-GHz TRIS measurements.

    Six samples with no per-row statistical column: the archive's third column
    repeats one common zero-level uncertainty, so
    ``statistical_uncertainty_k`` is ``None``.  This is a point set, never a
    ring -- there is no 2.5-GHz drift profile in the public archive.
    """

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
        _validate_matching_samples(
            ("ra_deg", self.ra_deg), ("temperature_k", self.temperature_k)
        )
        object.__setattr__(
            self, "ra_deg", _readonly_finite_array(self.ra_deg, "ra_deg")
        )
        object.__setattr__(
            self,
            "temperature_k",
            _readonly_finite_array(self.temperature_k, "temperature_k"),
        )
        if self.statistical_uncertainty_k is not None:
            _validate_matching_samples(
                ("ra_deg", self.ra_deg),
                ("statistical_uncertainty_k", self.statistical_uncertainty_k),
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
            _validate_nonnegative_scalar(
                self.zero_level_uncertainty_k, "zero_level_uncertainty_k"
            ),
        )


@dataclass(frozen=True, init=False, eq=False)
class TRISPrincipalPlaneCuts:
    """Peak-relative TRIS H- and E-plane beam cuts in archive dB units.

    The archive samples both principal planes from 0 to 176 degrees, i.e. the
    public product is *not* limited to the main lobe: it carries the shoulders
    and far sidelobes: nulls reach -60 dB and the
    anti-boresight response is about -48 dB.  ``H`` and ``E`` name spatial
    principal planes; they are not HH/VV, Stokes, or polarization labels.
    """

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
        _validate_matching_samples(
            ("angle_deg", self.angle_deg),
            ("h_plane_db", self.h_plane_db),
            ("e_plane_db", self.e_plane_db),
        )
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

    def half_power_full_width_deg(self) -> Tuple[float, float]:
        """Return the measured ``(E, H)`` full widths at half power, in degrees.

        These are interpolated from the archive cuts themselves at the exact
        half-power level ``-10*log10(2) = -3.0103 dB``, so they supersede any
        rounded prose figure.  For the public product they are about
        ``19.155`` degrees (E) and ``23.366`` degrees (H); the ring headers'
        "18 degrees wide (E-plane)" is a rounded restatement, 6 per cent
        narrower than this file's own measurement.
        """
        level = -10.0 * math.log10(2.0)
        widths = []
        for name, cut in (
            ("e_plane_db", self.e_plane_db),
            ("h_plane_db", self.h_plane_db),
        ):
            crossing = None
            for index in range(cut.size - 1):
                if cut[index] >= level >= cut[index + 1]:
                    span = cut[index + 1] - cut[index]
                    fraction = 0.0 if span == 0.0 else (level - cut[index]) / span
                    crossing = self.angle_deg[index] + fraction * (
                        self.angle_deg[index + 1] - self.angle_deg[index]
                    )
                    break
            if crossing is None:
                raise ValueError(
                    "{} never crosses the half-power level; the cut cannot "
                    "define a full width at half power".format(name)
                )
            widths.append(2.0 * float(crossing))
        return widths[0], widths[1]


def _validate_ra_data(ra_text: Tuple[str, ...], ra_deg: _VectorLike) -> None:
    if not isinstance(ra_text, tuple) or not ra_text:
        raise ValueError("ra_text must be a non-empty tuple")
    values = _readonly_finite_array(ra_deg, "ra_deg")
    if len(ra_text) != values.size:
        raise ValueError("ra_text and ra_deg must have the same length")
    if np.any(values < 0) or np.any(values >= 360):
        raise ValueError("ra_deg must be in [0, 360)")
    _validate_unique_coordinates(values, "ra_deg")


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
    """Read a local archive file in universal-newline mode.

    ``newline=None`` is load-bearing, not boilerplate.  Two of the four public
    products use a **bare CR** as the record separator inside their data block
    (``TRIS_Beam_Profile.txt`` has 54 of them; ``TRIS_absolute_2500MHz.txt``
    packs its first two records onto one physical line that way), while the
    600/820 rings are pure LF.  Pinning this to ``"\\n"`` makes those two
    products unreadable.  Regression-tested in ``tests/test_tris.py``.
    """
    path = Path(source)
    with path.open("r", encoding="ascii", newline=None) as handle:
        return path, handle.readlines()


def _header_and_rows(
    source: _PathInput, expected_columns: int
) -> Tuple[Path, _Header, _Rows]:
    header: _Header = []
    rows: _Rows = []
    path, lines = _read_ascii_lines(source)
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header.append((line_number, stripped))
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
    source: _PathInput, header: _typing.Sequence[Tuple[int, str]]
) -> Tuple[float, float, float, _StoredCommonUncertainty]:
    frequency_match = None
    frequency_line = None
    for line_number, line in header:
        frequency_match = re.search(r"Frequency\s*=\s*(\S+)\s*GHz", line, re.IGNORECASE)
        if frequency_match is not None:
            frequency_line = line_number
            break
    if frequency_match is None:
        raise ValueError("{}: ring file is missing its frequency header".format(source))
    frequency_ghz = _parse_finite_float(
        frequency_match.group(1), "frequency", source, frequency_line
    )
    try:
        frequency = _RING_FREQUENCIES[frequency_ghz]
    except KeyError:
        raise ValueError(
            "{}: unsupported TRIS ring frequency: {} GHz".format(source, frequency_ghz)
        )
    zero_match = None
    zero_line = None
    for line_number, line in header:
        zero_match = re.search(
            r"Systematic Zero Level Uncertainty\s*=\s*(.+)", line, re.IGNORECASE
        )
        if zero_match is not None:
            zero_line = line_number
            break
    if zero_match is None:
        raise ValueError(
            "{}: ring file is missing its zero-level uncertainty header".format(source)
        )
    zero_text = zero_match.group(1).strip()
    zero_level: _StoredCommonUncertainty
    asymmetry = re.fullmatch(r"\+?(\S+)K\s*/\s*-(\S+)K(?:\s*.*)?", zero_text)
    if asymmetry is not None:
        zero_level = AsymmetricUncertainty(
            positive_k=_parse_finite_float(
                asymmetry.group(1),
                "positive zero-level uncertainty",
                source,
                zero_line,
            ),
            negative_k=_parse_finite_float(
                asymmetry.group(2),
                "negative zero-level uncertainty",
                source,
                zero_line,
            ),
        )
    else:
        single = re.match(r"(\S+)K(?:\s*.*)?\Z", zero_text)
        if single is None:
            raise ValueError(
                "{}invalid zero-level uncertainty header: {!r}".format(
                    _source_location(source, zero_line), zero_text
                )
            )
        zero_level = _parse_finite_float(
            single.group(1), "zero-level uncertainty", source, zero_line
        )
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
    if not re.search(
        r"2\.5\s*GHz", "\n".join(line for _line_number, line in header), re.IGNORECASE
    ):
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
