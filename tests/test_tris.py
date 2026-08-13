"""Offline archive-reader contracts for the public TRIS text products."""

from dataclasses import FrozenInstanceError

import healpy as hp
import numpy as np
import pytest

from limTOD.tris import (
    AsymmetricUncertainty,
    TRISLinearFit,
    TRISRankDiagnostic,
    TRISPointSet,
    TRISPrincipalPlaneCuts,
    TRISRing,
    TRISZenithGeometry,
    approximate_tris_gaussian_beam_map,
    build_tris_fourier_design,
    fit_tris_linear_model,
    parse_tris_ra,
    read_tris_beam_cuts,
    read_tris_point_set,
    read_tris_ring,
    tris_beam_func,
    tris_zenith_geometry,
)
from limTOD.simulator import generate_TOD_sky, pointing_beam_in_eq_sys

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


def test_star_import_does_not_expose_private_typing_helpers():
    """Star import must expose exactly the binding-spec public TRIS API."""
    namespace = {}
    exec("from limTOD.tris import *", {}, namespace)

    assert set(namespace) == {
        "AsymmetricUncertainty",
        "TRISRing",
        "TRISPointSet",
        "TRISPrincipalPlaneCuts",
        "TRISZenithGeometry",
        "TRISRankDiagnostic",
        "TRISLinearFit",
        "parse_tris_ra",
        "read_tris_ring",
        "read_tris_point_set",
        "read_tris_beam_cuts",
        "approximate_tris_gaussian_beam_map",
        "tris_beam_func",
        "tris_zenith_geometry",
        "build_tris_fourier_design",
        "fit_tris_linear_model",
    }


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
    assert ring.declination_label_deg == 42.0


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


@pytest.mark.parametrize(
    ("contents", "line_number", "description"),
    [
        (RING_600.replace("0.6 GHz", "nope GHz"), 1, "frequency"),
        (RING_600.replace("0.066K", "nopeK"), 2, "zero-level uncertainty"),
    ],
)
def test_ring_header_numeric_errors_identify_source_and_line(
    tmp_path, contents, line_number, description
):
    """Malformed header numbers must retain their archive source location."""
    path = _write_ascii(tmp_path, "bad-header.txt", contents)

    with pytest.raises(ValueError, match=description) as error:
        read_tris_ring(path)

    _assert_diagnostic(error, path, line_number)


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
    assert points.declination_label_deg == 42.0


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


@pytest.mark.parametrize("model", [TRISRing, TRISPointSet])
@pytest.mark.parametrize("declination_label_deg", [float("nan"), float("inf")])
def test_tris_sample_products_reject_nonfinite_declination_labels(
    model, declination_label_deg
):
    """A rounded archive label must still be finite metadata."""
    common = dict(
        nominal_frequency_mhz=600.0,
        effective_frequency_mhz=600.5,
        bandwidth_mhz=0.3,
        ra_text=("0h00m",),
        ra_deg=np.array([0.0]),
        temperature_k=np.array([1.0]),
        declination_label_deg=declination_label_deg,
    )
    if model is TRISRing:
        common.update(
            statistical_uncertainty_k=np.array([0.1]),
            zero_level_uncertainty_k=0.1,
        )
    else:
        common.update(
            statistical_uncertainty_k=None,
            zero_level_uncertainty_k=0.1,
        )

    with pytest.raises(ValueError, match="declination_label_deg"):
        model(**common)


def test_frozen_scalar_metadata_is_detached_from_zero_dimensional_arrays():
    """Accepted scalar arrays must not leave frozen metadata caller-mutable."""
    sources = {
        "nominal_frequency_mhz": np.array(2500.0),
        "effective_frequency_mhz": np.array(2427.8),
        "bandwidth_mhz": np.array(3.0),
        "zero_level_uncertainty_k": np.array(0.284),
        "declination_label_deg": np.array(42.0),
    }
    points = TRISPointSet(
        **sources,
        ra_text=("0h00m",),
        ra_deg=np.array([0.0]),
        temperature_k=np.array([2.3]),
        statistical_uncertainty_k=None,
    )
    positive = np.array(0.43)
    negative = np.array(0.30)
    uncertainty = AsymmetricUncertainty(positive, negative)

    for source in sources.values():
        source[...] = -99.0
    positive[...] = -99.0
    negative[...] = -99.0

    assert points.nominal_frequency_mhz == 2500.0
    assert points.effective_frequency_mhz == 2427.8
    assert points.bandwidth_mhz == 3.0
    assert points.zero_level_uncertainty_k == 0.284
    assert points.declination_label_deg == 42.0
    assert uncertainty == AsymmetricUncertainty(0.43, 0.30)
    assert all(
        isinstance(value, float)
        for value in (
            points.nominal_frequency_mhz,
            points.effective_frequency_mhz,
            points.bandwidth_mhz,
            points.zero_level_uncertainty_k,
            points.declination_label_deg,
            uncertainty.positive_k,
            uncertainty.negative_k,
        )
    )


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(True),
        1.0 + 0.0j,
        "1.0",
        [1.0],
        np.array([1.0]),
        pytest.param(10**10000, id="overflowing-int"),
    ],
)
def test_real_scalar_metadata_rejects_boolean_and_non_scalar_values(bad_value):
    """Coercion must not admit truth values or size-one vectors as physics."""
    with pytest.raises(ValueError, match="positive_k"):
        AsymmetricUncertainty(positive_k=bad_value, negative_k=0.3)


@pytest.mark.parametrize(
    "field",
    [
        "nominal_frequency_mhz",
        "effective_frequency_mhz",
        "bandwidth_mhz",
        "zero_level_uncertainty_k",
        "declination_label_deg",
    ],
)
def test_point_set_rejects_boolean_public_scalar_metadata(field):
    """Every point-set physics scalar must use the strict coercion contract."""
    values = dict(
        nominal_frequency_mhz=2500.0,
        effective_frequency_mhz=2427.8,
        bandwidth_mhz=3.0,
        ra_text=("0h00m",),
        ra_deg=np.array([0.0]),
        temperature_k=np.array([2.3]),
        statistical_uncertainty_k=None,
        zero_level_uncertainty_k=0.284,
        declination_label_deg=42.0,
    )
    values[field] = True

    with pytest.raises(ValueError, match=field):
        TRISPointSet(**values)


def test_point_set_rejects_asymmetric_zero_level_uncertainty():
    """Sparse point metadata is one scalar common uncertainty, never a union."""
    with pytest.raises(ValueError, match="zero_level_uncertainty_k"):
        TRISPointSet(
            nominal_frequency_mhz=2500.0,
            effective_frequency_mhz=2427.8,
            bandwidth_mhz=3.0,
            ra_text=("0h00m",),
            ra_deg=[0.0],
            temperature_k=[2.3],
            statistical_uncertainty_k=None,
            zero_level_uncertainty_k=AsymmetricUncertainty(0.43, 0.30),
        )


def test_approximate_tris_gaussian_beam_map_has_ring_shape_and_normalizations():
    """The explicitly approximate scalar beam follows limTOD's RING convention."""
    nside = 32
    peak = approximate_tris_gaussian_beam_map(nside=nside, normalization="peak")
    summed = approximate_tris_gaussian_beam_map(nside=nside, normalization="sum")
    unnormalized = approximate_tris_gaussian_beam_map(nside=nside, normalization="none")

    assert peak.shape == (hp.nside2npix(nside),)
    assert np.all(np.isfinite(peak))
    assert np.all(peak >= 0.0)
    assert np.max(peak) == pytest.approx(1.0)
    assert np.sum(summed) == pytest.approx(1.0)
    np.testing.assert_allclose(peak, unnormalized / np.max(unnormalized))
    np.testing.assert_allclose(summed, unnormalized / np.sum(unnormalized))


def test_narrow_gaussian_beam_normalizations_remain_finite():
    """Tiny positive FWHM must not underflow before peak or sum normalization."""
    peak = approximate_tris_gaussian_beam_map(
        nside=32, fwhm_e_deg=0.01, fwhm_h_deg=0.01, normalization="peak"
    )
    summed = approximate_tris_gaussian_beam_map(
        nside=32, fwhm_e_deg=0.01, fwhm_h_deg=0.01, normalization="sum"
    )

    assert np.all(np.isfinite(peak))
    assert np.all(np.isfinite(summed))
    assert np.max(peak) == pytest.approx(1.0)
    assert np.sum(summed) == pytest.approx(1.0)


def test_approximate_tris_gaussian_beam_map_uses_e_and_h_principal_axes():
    """The public one-dimensional cuts define an approximate 18/23-degree ellipse."""
    beam = approximate_tris_gaussian_beam_map(nside=256, normalization="peak")
    e_half_power = hp.get_interp_val(beam, np.deg2rad(9.0), 0.0)
    h_half_power = hp.get_interp_val(beam, np.deg2rad(11.5), np.pi / 2.0)
    e_on_h_axis = hp.get_interp_val(beam, np.deg2rad(9.0), np.pi / 2.0)
    h_on_e_axis = hp.get_interp_val(beam, np.deg2rad(11.5), 0.0)

    assert e_half_power == pytest.approx(0.5, abs=0.01)
    assert h_half_power == pytest.approx(0.5, abs=0.01)
    assert e_on_h_axis > 0.5
    assert h_on_e_axis < 0.5


def test_tris_beam_func_is_achromatic_but_validates_mhz_frequency():
    """The archive's one common beam does not justify frequency scaling."""
    beam_func = tris_beam_func(normalization="sum")

    np.testing.assert_array_equal(
        beam_func(freq=600.0, nside=32), beam_func(freq=2500.0, nside=32)
    )
    for invalid_frequency in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="freq"):
            beam_func(freq=invalid_frequency, nside=32)


def test_tris_zenith_geometry_keeps_ra_labels_and_distinguishes_site_from_label():
    """The rounded archive label is distinct from the measured site latitude."""
    geometry = tris_zenith_geometry([0.0, 17.75])
    nominal_label_geometry = tris_zenith_geometry([0.0], latitude_deg=42.0)

    assert isinstance(geometry, TRISZenithGeometry)
    np.testing.assert_array_equal(geometry.lst_deg, [0.0, 17.75])
    np.testing.assert_array_equal(geometry.azimuth_deg, [0.0, 0.0])
    np.testing.assert_array_equal(geometry.elevation_deg, [90.0, 90.0])
    np.testing.assert_array_equal(geometry.selfrot_deg, [-7.0, -7.0])
    assert geometry.latitude_deg == pytest.approx(42.0 + 26.0 / 60.0)
    assert nominal_label_geometry.latitude_deg == 42.0
    with pytest.raises(ValueError):
        geometry.lst_deg[0] = 1.0


def _displaced_marker_map(nside, phi_deg):
    """An asymmetric marker that independently reveals the pointing-chain roll."""
    theta0 = np.deg2rad(10.0)
    sigma = np.deg2rad(1.5)
    target = np.asarray(hp.ang2vec(theta0, np.deg2rad(phi_deg))).ravel()
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    vectors = np.asarray(hp.ang2vec(theta, phi))
    separation = np.arccos(np.clip(vectors @ target, -1.0, 1.0))
    return np.exp(-0.5 * (separation / sigma) ** 2)


def _landing_azimuth_deg(geometry, phi_deg):
    nside = 64
    marker_alm = hp.map2alm(_displaced_marker_map(nside, phi_deg), lmax=3 * nside - 1)
    pointed = pointing_beam_in_eq_sys(
        marker_alm,
        LST_deg=geometry.lst_deg[0],
        lat_deg=geometry.latitude_deg,
        azimuth_deg=geometry.azimuth_deg[0],
        elevation_deg=geometry.elevation_deg[0],
        selfrot_deg=geometry.selfrot_deg[0],
        nside=nside,
        normalize=False,
    )
    vector = np.asarray(hp.pix2vec(nside, int(np.argmax(pointed))))
    # At latitude=0 and LST=0, north/east are the equatorial +z/+y axes.
    return float(np.rad2deg(np.arctan2(vector[1], vector[2])) % 360.0)


def test_tris_zenith_geometry_roll_carries_e_axis_north_to_ne_and_south_to_sw():
    """An asymmetric marker through limTOD's pointing chain pins selfrot=-7."""
    geometry = tris_zenith_geometry([0.0], latitude_deg=0.0)

    north_branch_azimuth = _landing_azimuth_deg(geometry, phi_deg=180.0)
    south_branch_azimuth = _landing_azimuth_deg(geometry, phi_deg=0.0)

    assert north_branch_azimuth == pytest.approx(7.0, abs=1.5)
    assert south_branch_azimuth == pytest.approx(187.0, abs=1.5)


def test_tris_zenith_geometry_zenith_has_ra_equal_to_lst_and_dec_equal_to_latitude():
    """Parking at zenith retains the supplied LST without hidden coordinate shifts."""
    geometry = tris_zenith_geometry([31.0], latitude_deg=42.0 + 26.0 / 60.0)
    nside = 64
    theta, _phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    boresight_alm = hp.map2alm(np.exp(-0.5 * (theta / np.deg2rad(1.5)) ** 2))
    pointed = pointing_beam_in_eq_sys(
        boresight_alm,
        LST_deg=geometry.lst_deg[0],
        lat_deg=geometry.latitude_deg,
        azimuth_deg=geometry.azimuth_deg[0],
        elevation_deg=geometry.elevation_deg[0],
        selfrot_deg=geometry.selfrot_deg[0],
        nside=nside,
        normalize=False,
    )
    theta_peak, phi_peak = hp.pix2ang(nside, int(np.argmax(pointed)))
    assert np.rad2deg(phi_peak) == pytest.approx(geometry.lst_deg[0], abs=1.5)
    assert 90.0 - np.rad2deg(theta_peak) == pytest.approx(
        geometry.latitude_deg, abs=1.5
    )


def _inference_ring(ra_deg, temperature_k, uncertainty_k):
    """Build a ring fixture without giving inference access to zero-level metadata."""
    ra_deg = np.asarray(ra_deg, dtype=float)
    return TRISRing(
        nominal_frequency_mhz=600.0,
        effective_frequency_mhz=600.5,
        bandwidth_mhz=0.3,
        ra_text=tuple("{}h00m".format(index) for index in range(ra_deg.size)),
        ra_deg=ra_deg,
        temperature_k=np.asarray(temperature_k, dtype=float),
        statistical_uncertainty_k=np.asarray(uncertainty_k, dtype=float),
        zero_level_uncertainty_k=AsymmetricUncertainty(0.43, 0.30),
    )


def test_fourier_design_preserves_irregular_ra_and_column_order():
    """Swapping sin/cos or regridding published RA labels changes the fitted model."""
    ra_deg = np.array([0.0, 90.0, 215.0])

    design = build_tris_fourier_design(ra_deg, m_max=2)

    alpha = np.deg2rad(ra_deg)
    expected = np.column_stack(
        (
            np.ones(3),
            np.cos(alpha),
            np.sin(alpha),
            np.cos(2.0 * alpha),
            np.sin(2.0 * alpha),
        )
    )
    np.testing.assert_allclose(design, expected)
    assert design.flags.writeable is False


def test_fourier_design_can_omit_constant_and_validates_arguments():
    """A malformed harmonic order or RA array must not produce a silent design."""
    np.testing.assert_allclose(
        build_tris_fourier_design([0.0, 90.0], m_max=1, include_constant=False),
        [[1.0, 0.0], [0.0, 1.0]],
        atol=1e-15,
    )
    for bad_m_max in (-1, 1.5, True):
        with pytest.raises(ValueError, match="m_max"):
            build_tris_fourier_design([0.0], m_max=bad_m_max)
    with pytest.raises(ValueError, match="ra_deg"):
        build_tris_fourier_design([0.0, np.nan], m_max=1)


def test_linear_fit_recovers_known_coefficients_and_gls_covariance():
    """An incorrect whitening or SVD solve would bias an exactly representable ring."""
    ra_deg = np.arange(8) * 45.0
    design = build_tris_fourier_design(ra_deg, m_max=1)
    coefficients = np.array([10.0, 2.0, -3.0])
    ring = _inference_ring(ra_deg, design @ coefficients, np.full(8, 0.5))

    fit = fit_tris_linear_model(ring, design)

    assert isinstance(fit, TRISLinearFit)
    assert isinstance(fit.rank_diagnostic, TRISRankDiagnostic)
    np.testing.assert_allclose(fit.coefficients, coefficients, atol=1e-12)
    np.testing.assert_allclose(fit.prediction_k, ring.temperature_k, atol=1e-12)
    np.testing.assert_allclose(fit.residual_k, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        fit.coefficient_covariance,
        np.diag([0.5**2 / 8.0, 0.5**2 / 4.0, 0.5**2 / 4.0]),
        atol=1e-12,
    )
    assert fit.rank_diagnostic.numerical_rank == 3
    assert fit.rank_diagnostic.parameter_count == 3
    assert fit.coefficients.flags.writeable is False
    assert fit.rank_diagnostic.singular_values.flags.writeable is False


def test_linear_fit_requires_explicit_floor_for_zero_statistical_error():
    """Treating a zero error as an infinite weight hides an invalid likelihood."""
    ring = _inference_ring([0.0, 90.0], [3.0, 3.0], [0.0, 0.2])
    design = np.ones((2, 1))

    with pytest.raises(ValueError, match="uncertainty_floor_k"):
        fit_tris_linear_model(ring, design)

    fit = fit_tris_linear_model(ring, design, uncertainty_floor_k=0.1)
    assert fit.coefficients[0] == pytest.approx(3.0)
    assert fit.coefficient_covariance[0, 0] == pytest.approx(1.0 / 125.0)


def test_common_mode_covariance_only_expands_constant_mode_when_requested():
    """Automatically reading asymmetric archive metadata would make this unsupported choice."""
    ring = _inference_ring([0.0, 90.0, 180.0, 270.0], [4.0] * 4, [0.1] * 4)
    design = np.ones((4, 1))

    statistical_only = fit_tris_linear_model(ring, design)
    with_common_mode = fit_tris_linear_model(ring, design, common_mode_sigma_k=0.5)

    assert statistical_only.coefficient_covariance[0, 0] == pytest.approx(0.1**2 / 4)
    assert with_common_mode.coefficient_covariance[0, 0] == pytest.approx(
        0.1**2 / 4 + 0.5**2
    )
    with pytest.raises(ValueError, match="common_mode_sigma_k"):
        fit_tris_linear_model(ring, design, common_mode_sigma_k=float("nan"))


def test_correlated_gls_matches_direct_normal_equation_oracle():
    """A general correlated fit must agree beyond an axis-aligned constant case."""
    design = np.array(
        [
            [1.0, 0.2, -1.1],
            [1.0, 1.4, 0.3],
            [1.0, -0.7, 2.2],
            [1.0, 2.1, -0.4],
            [1.0, -1.3, 0.8],
            [1.0, 0.5, 1.7],
        ]
    )
    statistical = np.array([0.12, 0.31, 0.18, 0.44, 0.27, 0.36])
    common_mode = 0.23
    temperature = design @ np.array([2.5, -0.7, 1.2])
    temperature += np.array([0.03, -0.07, 0.11, -0.02, 0.08, -0.05])
    ring = _inference_ring(
        np.array([0.0, 37.0, 83.0, 151.0, 224.0, 319.0]),
        temperature,
        statistical,
    )

    covariance = np.diag(statistical**2) + common_mode**2 * np.ones((6, 6))
    precision = np.linalg.inv(covariance)
    normal_matrix = design.T @ precision @ design
    expected_covariance = np.linalg.inv(normal_matrix)
    expected_coefficients = expected_covariance @ design.T @ precision @ temperature

    fit = fit_tris_linear_model(ring, design, common_mode_sigma_k=common_mode)

    np.testing.assert_allclose(
        fit.coefficients, expected_coefficients, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        fit.coefficient_covariance,
        expected_covariance,
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize(
    ("design", "parameter_count"),
    [
        (np.column_stack((np.ones(3), np.ones(3))), 2),
        (np.eye(3, 12), 12),
    ],
)
def test_linear_fit_rejects_duplicate_or_free_map_shaped_designs_before_solving(
    design, parameter_count
):
    """Rank-deficient template and free-pixel designs are not TRIS measurements."""
    ring = _inference_ring([0.0, 90.0, 180.0], [1.0, 2.0, 3.0], [0.1] * 3)

    with pytest.raises(ValueError, match="reduce the model") as error:
        fit_tris_linear_model(ring, design)

    assert "parameter count={}".format(parameter_count) in str(error.value)


def test_linear_fit_rejects_rank_tolerance_below_machine_safe_floor():
    """Caller tolerance must not make duplicated columns appear identifiable."""
    ring = _inference_ring([0.0, 90.0, 180.0], [1.0, 2.0, 3.0], [0.1] * 3)
    duplicated = np.column_stack((np.ones(3), np.ones(3)))

    with pytest.raises(ValueError, match="rank_rtol.*at least"):
        fit_tris_linear_model(ring, duplicated, rank_rtol=1e-30)


def test_linear_fit_applies_valid_explicit_rank_tolerance():
    """A conservative caller tolerance is retained in the rank diagnostic."""
    ring = _inference_ring([0.0, 90.0, 180.0], [2.0, 4.0, 2.0], [0.2] * 3)
    design = np.column_stack((np.ones(3), [0.0, 1.0, 0.0]))

    fit = fit_tris_linear_model(ring, design, rank_rtol=1e-8)

    assert fit.rank_diagnostic.rank_rtol == 1e-8


def test_linear_fit_rejects_saturated_square_design_before_solving():
    """One free coefficient per ring datum is not a low-dimensional TRIS model."""
    ring = _inference_ring([0.0, 90.0, 180.0], [1.0, 2.0, 3.0], [0.1] * 3)

    with pytest.raises(ValueError, match="low-dimensional.*reduce the model") as error:
        fit_tris_linear_model(ring, np.eye(3))

    assert "parameter count=3" in str(error.value)
    assert "sample count=3" in str(error.value)


def test_reduced_full_rank_template_design_passes_rank_gate():
    """The rank gate must permit a compact identifiable caller-supplied template."""
    ring = _inference_ring([0.0, 90.0, 180.0], [2.0, 4.0, 2.0], [0.2] * 3)
    design = np.column_stack((np.ones(3), [0.0, 1.0, 0.0]))

    fit = fit_tris_linear_model(ring, design)

    np.testing.assert_allclose(fit.coefficients, [2.0, 2.0])
    assert fit.rank_diagnostic.numerical_rank == 2


def test_normalized_tris_beam_recovers_constant_sky_under_positive_rescaling():
    """A normalized beam must retain a constant sky's Kelvin scale after rotation."""
    nside = 8
    sky = np.full(hp.nside2npix(nside), 7.25)
    beam = approximate_tris_gaussian_beam_map(nside=nside, normalization="none")
    pointing = dict(
        LST_deg_list=np.array([0.0, 120.0]),
        lat_deg=42.0,
        azimuth_deg_list=np.array([0.0, 15.0]),
        elevation_deg_list=np.array([90.0, 70.0]),
        selfrot_deg_list=np.array([-7.0, 3.0]),
        normalize_beam=True,
    )

    tod = generate_TOD_sky(beam, sky, **pointing)
    scaled_tod = generate_TOD_sky(4.5 * beam, sky, **pointing)

    np.testing.assert_allclose(tod, 7.25, atol=1e-10)
    np.testing.assert_allclose(scaled_tod, 7.25, atol=1e-10)
