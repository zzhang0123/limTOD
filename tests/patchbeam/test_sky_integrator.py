"""Integration tests for limTOD.patchbeam.sky_integrator."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from limTOD.patchbeam import disc as disc_mod
from limTOD.patchbeam import integrate_sample, integrate_tod, synthetic_gaussian_beam


@pytest.fixture(autouse=True)
def _clear_disc_cache():
    disc_mod.clear_disc_cache()
    yield
    disc_mod.clear_disc_cache()


@pytest.fixture
def beam():
    """A circular Gaussian beam wide enough to be well-sampled at nside=128."""
    freq_MHz = np.array([900.0])
    margin_deg = np.linspace(-6, 6, 121)
    return synthetic_gaussian_beam(freq_MHz=freq_MHz, margin_deg=margin_deg, fwhm_deg=2.5)


def _uniform_sky(nside: int, T: float = 10.0, n_freq: int = 1) -> np.ndarray:
    return np.full((n_freq, hp.nside2npix(nside)), T, dtype=np.float64)


@pytest.mark.integration
def test_uniform_sky_returns_sky_temperature(beam):
    """For a uniform sky T0, the antenna temperature equals T0.

    Requires both: (a) the disc to enclose the full beam support, and
    (b) HEALPix sampling fine enough that the discrete Riemann sum of
    the beam matches its on-grid integral to within a few percent.
    """
    nside = 128
    sky = _uniform_sky(nside, T=5.0)

    sample = integrate_sample(
        lst_deg=180.0,
        az_pointing_deg=180.0,
        el_pointing_deg=45.0,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        sky_freq_indices=np.array([0]),
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
        polarization="HH",
    )
    np.testing.assert_allclose(sample[0], 5.0, rtol=2e-2)


@pytest.mark.integration
def test_point_source_at_pointing(beam):
    """A delta-function source at the beam centre recovers T_src * dOmega_pix / Omega_b * B(offset).

    The offset is the small angle between the pointing direction and
    the centre of the HEALPix pixel containing it, so B(offset) is
    slightly below 1.
    """
    nside = 256
    sky = np.zeros((1, hp.nside2npix(nside)), dtype=np.float64)
    az_p, el_p, lst, lat = 180.0, 60.0, 180.0, -30.7130

    from limTOD.patchbeam.projection import (
        direction_cosines,
        equatorial_to_horizon,
        horizon_to_equatorial,
    )

    ra_p, dec_p = horizon_to_equatorial(az_p, el_p, lst, lat)
    theta = np.deg2rad(90.0 - float(dec_p))
    phi = np.deg2rad(float(ra_p) % 360.0)
    src_pix = hp.ang2pix(nside, theta, phi)
    T_src = 1.0e3
    sky[0, src_pix] = T_src

    sample = integrate_sample(
        lst_deg=lst,
        az_pointing_deg=az_p,
        el_pointing_deg=el_p,
        lat_deg=lat,
        beam=beam,
        sky_maps=sky,
        sky_freq_indices=np.array([0]),
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
    )

    # Compute the actual (l, m) of the source pixel relative to the pointing.
    src_theta, src_phi = hp.pix2ang(nside, src_pix)
    src_ra = np.rad2deg(src_phi)
    src_dec = 90.0 - np.rad2deg(src_theta)
    src_az, src_el = equatorial_to_horizon(src_ra, src_dec, lst, lat)
    l, m = direction_cosines(az_p, el_p, src_az, src_el)  # noqa: E741
    r_deg = np.rad2deg(np.hypot(float(l), float(m)))
    # Power-beam sigma matching synthetic_gaussian_beam's convention.
    sigma_deg = 2.5 / (2.0 * np.sqrt(np.log(2.0)))
    B_at_src = float(np.exp(-(r_deg**2) / sigma_deg**2))

    d_omega_pix = 4.0 * np.pi / hp.nside2npix(nside)
    omega_b = beam.beam_solid_angle("HH")[0]
    expected = T_src * B_at_src * d_omega_pix / omega_b
    np.testing.assert_allclose(sample[0], expected, rtol=1e-2)


@pytest.mark.integration
def test_integrate_tod_shape_and_consistency(beam):
    """Driver returns (n_freq, n_time) and gives ~uniform output for uniform sky."""
    nside = 128
    sky = _uniform_sky(nside, T=2.0)
    ntime = 4
    lst = np.array([0.0, 90.0, 180.0, 270.0])
    az = np.full(ntime, 180.0)
    el = np.full(ntime, 45.0)

    tod = integrate_tod(
        lst_deg_list=lst,
        az_deg_list=az,
        el_deg_list=el,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        freq_MHz=[900.0],
        disc_radius_deg=8.0,
        n_jobs=1,
    )
    assert tod.shape == (1, ntime)
    np.testing.assert_allclose(tod[0], 2.0, rtol=3e-2)


@pytest.mark.integration
def test_zero_sky_gives_zero_tod(beam):
    """A zero sky produces a zero TOD."""
    nside = 64
    sky = np.zeros((1, hp.nside2npix(nside)), dtype=np.float64)
    sample = integrate_sample(
        lst_deg=0.0,
        az_pointing_deg=180.0,
        el_pointing_deg=45.0,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        sky_freq_indices=np.array([0]),
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
    )
    assert sample[0] == 0.0


@pytest.mark.integration
def test_below_horizon_pointing_returns_zero(beam):
    """No sky pixels are above the horizon -> TOD is zero."""
    # Pointing through the floor: el < 0 means even nearby sky pixels are below 0.
    nside = 64
    sky = _uniform_sky(nside, T=5.0)
    sample = integrate_sample(
        lst_deg=0.0,
        az_pointing_deg=180.0,
        el_pointing_deg=-30.0,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        sky_freq_indices=np.array([0]),
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
    )
    np.testing.assert_allclose(sample[0], 0.0, atol=0.0)


@pytest.mark.integration
def test_vv_polarization_path(beam):
    """The VV polarisation path produces a sensible result (parity with HH for
    a symmetric synthetic beam)."""
    nside = 128
    sky = _uniform_sky(nside, T=3.0)
    common = dict(
        lst_deg=180.0,
        az_pointing_deg=180.0,
        el_pointing_deg=45.0,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        sky_freq_indices=np.array([0]),
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
    )
    hh = integrate_sample(**common, polarization="HH")
    vv = integrate_sample(**common, polarization="VV")
    np.testing.assert_allclose(hh, vv, rtol=1e-8)
    np.testing.assert_allclose(hh[0], 3.0, rtol=2e-2)


@pytest.mark.integration
def test_horizontal_mask_wrong_shape_raises(beam):
    nside = 64
    sky = _uniform_sky(nside, T=1.0)
    bad_mask = np.ones(hp.nside2npix(32), dtype=bool)  # wrong nside
    with pytest.raises(ValueError, match="horizontal_mask"):
        integrate_sample(
            lst_deg=0.0,
            az_pointing_deg=180.0,
            el_pointing_deg=45.0,
            lat_deg=-30.7130,
            beam=beam,
            sky_maps=sky,
            sky_freq_indices=np.array([0]),
            beam_freq_indices=np.array([0]),
            disc_radius_deg=8.0,
            horizontal_mask=bad_mask,
        )


@pytest.mark.integration
def test_horizontal_mask_zeroes_contribution(beam):
    """Masking out every disc pixel gives zero TOD."""
    nside = 64
    sky = _uniform_sky(nside, T=10.0)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)  # mask everything
    sample = integrate_sample(
        lst_deg=0.0,
        az_pointing_deg=180.0,
        el_pointing_deg=45.0,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        sky_freq_indices=np.array([0]),
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
        horizontal_mask=mask,
    )
    np.testing.assert_allclose(sample[0], 0.0, atol=0.0)


@pytest.mark.integration
def test_sky_freq_indices_optional(beam):
    """Omitting sky_freq_indices is equivalent to passing arange(n_freq_out)."""
    nside = 64
    sky = _uniform_sky(nside, T=4.0)
    common = dict(
        lst_deg=10.0,
        az_pointing_deg=180.0,
        el_pointing_deg=45.0,
        lat_deg=-30.7130,
        beam=beam,
        sky_maps=sky,
        beam_freq_indices=np.array([0]),
        disc_radius_deg=8.0,
    )
    auto = integrate_sample(**common)
    explicit = integrate_sample(**common, sky_freq_indices=np.array([0]))
    np.testing.assert_allclose(auto, explicit)


@pytest.mark.integration
def test_stokes_modes_other_than_I_raises(beam):
    nside = 64
    sky = _uniform_sky(nside, T=1.0)
    with pytest.raises(NotImplementedError, match="Stokes"):
        integrate_sample(
            lst_deg=0.0,
            az_pointing_deg=180.0,
            el_pointing_deg=45.0,
            lat_deg=-30.7130,
            beam=beam,
            sky_maps=sky,
            beam_freq_indices=np.array([0]),
            disc_radius_deg=8.0,
            stokes_modes=("I", "Q", "U"),
        )
