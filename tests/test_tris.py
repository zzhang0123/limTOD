"""Offline archive-reader contracts for the public TRIS text products."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from limTOD.tris import (
    AsymmetricUncertainty,
    TRISPrincipalPlaneCuts,
    parse_tris_ra,
    read_tris_beam_cuts,
    read_tris_point_set,
    read_tris_ring,
)

RING_600 = """# Frequency = 0.6 GHz
# Systematic Zero Level Uncertainty = 0.066K
# Column 1 = Right Ascension in hh mm
# Column 2 = Sky Brightness Temperature (K)
# Column 3 = Statistical Uncertainty (K)
0h00m 15.145 0.004
1h11m 15.335 0.012
"""

RING_820 = """# Frequency = 0.82 GHz
# Systematic Zero Level Uncertainty = +0.430K/-0.300K
0h00m 7.781 0.004
1h11m 8.020 0.016
"""

POINTS_2500 = """# TRIS Absolute Sky Temperature at 2.5 GHz
# Column 1 = RA in hours, minutes, seconds
# Column 2 = Sky Temperature in K
# Column 3 = Zero Level uncertainty in K
11h26m04s 2.329 0.284
13h42m32s 2.331 0.284
"""

BEAM_CUTS = """# Column 1= angle (degree)
# Column 2= H plane cut (dB)
# Column 3= E plane cut (dB)
0 0.0 0.0
3 -0.175 -0.335
"""


def _write_ascii(tmp_path, name, text, newline="\n"):
    path = tmp_path / name
    path.write_bytes(text.replace("\n", newline).encode("ascii"))
    return path


def _assert_diagnostic(error, path, *line_numbers):
    message = str(error.value)
    assert str(path) in message
    for line_number in line_numbers:
        assert "line {}".format(line_number) in message


@pytest.mark.parametrize(
    ("token", "expected_deg"),
    [("1h02m", 15.5), ("1h02m03s", 15.5125)],
)
def test_parse_tris_ra_accepts_archive_token_formats(token, expected_deg):
    """A parser restricted to minute tokens would discard 2.5-GHz positions."""
    assert parse_tris_ra(token) == pytest.approx(expected_deg)


@pytest.mark.parametrize("token", ["24h00m", "1h60m", "1h02m60s", "01:02"])
def test_parse_tris_ra_rejects_out_of_range_or_unknown_tokens(token):
    """Permissive coordinate normalization would hide malformed archive input."""
    with pytest.raises(ValueError):
        parse_tris_ra(token)


def test_ring_reader_preserves_ra_text_and_supports_crlf(tmp_path):
    """Normalizing labels or assuming LF-only input loses public metadata."""
    ring = read_tris_ring(_write_ascii(tmp_path, "600.txt", RING_600, "\r\n"))

    assert ring.ra_text == ("0h00m", "1h11m")
    np.testing.assert_allclose(ring.ra_deg, [0.0, 17.75])
    np.testing.assert_allclose(ring.temperature_k, [15.145, 15.335])
    np.testing.assert_allclose(ring.statistical_uncertainty_k, [0.004, 0.012])


def test_ring_reader_separates_nominal_effective_and_statistical_uncertainty(tmp_path):
    """Collapsing frequency metadata or zero level into diagonal errors is wrong."""
    ring = read_tris_ring(_write_ascii(tmp_path, "600.txt", RING_600))

    assert ring.nominal_frequency_mhz == 600.0
    assert ring.effective_frequency_mhz == 600.5
    assert ring.bandwidth_mhz == 0.3
    assert ring.zero_level_uncertainty_k == 0.066
    assert ring.statistical_uncertainty_k.shape == (2,)
    with pytest.raises(FrozenInstanceError):
        ring.nominal_frequency_mhz = 1.0


def test_820_ring_keeps_asymmetric_zero_level_separate_from_statistical_errors(
    tmp_path,
):
    """Replacing the published asymmetric common uncertainty with row noise loses meaning."""
    ring = read_tris_ring(_write_ascii(tmp_path, "820.txt", RING_820))

    assert ring.nominal_frequency_mhz == 820.0
    assert ring.effective_frequency_mhz == 817.8
    assert ring.zero_level_uncertainty_k == AsymmetricUncertainty(
        positive_k=0.430, negative_k=0.300
    )
    np.testing.assert_allclose(ring.statistical_uncertainty_k, [0.004, 0.016])


def test_point_reader_keeps_repeated_2500_zero_level_as_one_common_uncertainty(
    tmp_path,
):
    """Duplicating a common 2.5-GHz uncertainty into per-row errors changes covariance."""
    points = read_tris_point_set(_write_ascii(tmp_path, "2500.txt", POINTS_2500))

    assert points.nominal_frequency_mhz == 2500.0
    assert points.effective_frequency_mhz == 2427.8
    assert points.bandwidth_mhz == 3.0
    assert points.ra_text == ("11h26m04s", "13h42m32s")
    np.testing.assert_allclose(points.ra_deg, [171.5166666667, 205.6333333333])
    assert points.statistical_uncertainty_k is None
    assert points.zero_level_uncertainty_k == 0.284


@pytest.mark.parametrize(
    ("reader", "contents"),
    [
        (read_tris_ring, "# Frequency = 0.6 GHz\n0h00m 15.145\n"),
        (read_tris_point_set, "# 2.5 GHz\n11h26m04s 2.329 nope\n"),
        (read_tris_beam_cuts, "0 0.0 nan\n"),
    ],
)
def test_readers_reject_wrong_column_count_or_nonfinite_values(
    tmp_path, reader, contents
):
    """Silently accepting malformed rows would corrupt an offline scientific product."""
    with pytest.raises(ValueError):
        reader(_write_ascii(tmp_path, "bad.txt", contents))


def test_point_reader_rejects_noncommon_repeated_zero_level(tmp_path):
    """Per-row values that disagree cannot represent a single common zero-level error."""
    bad_points = POINTS_2500.replace("2.331 0.284", "2.331 0.285")
    path = _write_ascii(tmp_path, "2500-bad.txt", bad_points)
    with pytest.raises(ValueError, match="common") as error:
        read_tris_point_set(path)
    _assert_diagnostic(error, path, 5, 6)


def test_beam_cuts_keep_raw_db_and_expose_power_relative_to_peak(tmp_path):
    """Treating dB cuts as linear values gives physically nonsensical beam weights."""
    cuts = read_tris_beam_cuts(_write_ascii(tmp_path, "beam.txt", BEAM_CUTS, "\r\n"))

    assert isinstance(cuts, TRISPrincipalPlaneCuts)
    np.testing.assert_allclose(cuts.angle_deg, [0.0, 3.0])
    np.testing.assert_allclose(cuts.h_plane_db, [0.0, -0.175])
    np.testing.assert_allclose(cuts.e_plane_db, [0.0, -0.335])
    np.testing.assert_allclose(cuts.h_plane_relative_power, [1.0, 10 ** (-0.175 / 10)])
    np.testing.assert_allclose(cuts.e_plane_relative_power, [1.0, 10 ** (-0.335 / 10)])


def test_ring_reader_rejects_duplicate_parsed_ra_with_source_rows(tmp_path):
    """Distinct raw tokens at one RA would otherwise make a drift ring ambiguous."""
    duplicate_ring = RING_600.replace("1h11m", "0h00m00s")
    path = _write_ascii(tmp_path, "duplicate-ring.txt", duplicate_ring)

    with pytest.raises(ValueError, match="duplicate") as error:
        read_tris_ring(path)

    _assert_diagnostic(error, path, 6, 7)
    assert "0.0" in str(error.value)


def test_point_reader_rejects_duplicate_parsed_ra_with_source_rows(tmp_path):
    """Sparse point coordinates must remain uniquely identifiable after parsing."""
    duplicate_points = POINTS_2500.replace("13h42m32s", "11h26m04s")
    path = _write_ascii(tmp_path, "duplicate-points.txt", duplicate_points)

    with pytest.raises(ValueError, match="duplicate") as error:
        read_tris_point_set(path)

    _assert_diagnostic(error, path, 5, 6)


def test_beam_reader_rejects_duplicate_angle_with_source_rows(tmp_path):
    """Duplicate angles would make an archive principal-plane cut ill-defined."""
    duplicate_cuts = BEAM_CUTS.replace("3 -0.175 -0.335", "0 -0.175 -0.335")
    path = _write_ascii(tmp_path, "duplicate-beam.txt", duplicate_cuts)

    with pytest.raises(ValueError, match="duplicate") as error:
        read_tris_beam_cuts(path)

    _assert_diagnostic(error, path, 4, 5)


@pytest.mark.parametrize(
    ("reader", "contents", "line_number"),
    [
        (read_tris_ring, "# Frequency = 0.6 GHz\n0h00m 15.145\n", 2),
        (
            read_tris_ring,
            RING_600.replace("1h11m", "1h60m"),
            7,
        ),
        (
            read_tris_point_set,
            POINTS_2500.replace("2.331 0.284", "2.331 nope"),
            6,
        ),
        (read_tris_beam_cuts, BEAM_CUTS.replace("-0.335", "nan"), 5),
    ],
)
def test_reader_row_errors_identify_source_and_line(
    tmp_path, reader, contents, line_number
):
    """A malformed public row must be traceable to its exact source location."""
    path = _write_ascii(tmp_path, "diagnostic.txt", contents)

    with pytest.raises(ValueError) as error:
        reader(path)

    _assert_diagnostic(error, path, line_number)
