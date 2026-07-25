"""Cross-check Simeer's (l, m) disc integrator against limTOD's HEALPix-SH path.

For a symmetric Gaussian *power* beam, both pipelines compute the same
beam-weighted sky integral and so should agree to a few percent (limited
by HEALPix discretisation on the limTOD side and bilinear interpolation
on the Simeer side).

These tests require ``limTOD`` to be installed in the same Python
environment as Simeer. ``pytest.importorskip`` skips the whole module
when limTOD is absent.

Boundary-validation methodology (per ``rules/common/boundary-validation.md``):
the parametrisation sweeps the dispatch boundaries where each pipeline's
approximation might break down -- low / high elevation, az wrap, narrow
vs wide beams, near-zenith pointing.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

limtod = pytest.importorskip("limTOD")  # noqa: F841
from limTOD.simulator import generate_TOD_sky  # noqa: E402

from limTOD.simeer import (  # noqa: E402
    disc as disc_mod,
    integrate_tod,
    synthetic_gaussian_beam,
)

MEERKAT_LAT = -30.7130


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _limtod_gaussian_beam(*, fwhm_deg: float, nside: int) -> np.ndarray:
    """Symmetric Gaussian *power* beam on HEALPix, centred at the pole.

    Matches limTOD's ``example_symm_beam_map`` formula exactly, but
    written here so the test does not depend on limTOD's internal helper
    (which is also amplitude-parameterised but consumed as a power beam
    downstream)."""
    sigma_rad = np.deg2rad(fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    theta, _phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    beam = np.exp(-0.5 * (theta / sigma_rad) ** 2)
    beam /= beam.sum()  # limTOD convention: sum-normalised
    return beam


def _simeer_gaussian_beam(*, fwhm_deg: float, margin_deg_n: int = 121):
    """Same power FWHM, on the Simeer (l, m) grid."""
    return synthetic_gaussian_beam(
        freq_MHz=np.array([1000.0]),
        margin_deg=np.linspace(-6, 6, margin_deg_n),
        fwhm_deg=fwhm_deg,
    )


@pytest.fixture(autouse=True)
def _clear_disc_cache():
    disc_mod.clear_disc_cache()
    yield
    disc_mod.clear_disc_cache()


def _run_pair(
    *,
    fwhm_deg: float,
    sky_nside: int,
    sky_map: np.ndarray,
    lst_deg: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
):
    """Return (tod_limtod, tod_simeer) for one configuration."""
    limtod_beam = _limtod_gaussian_beam(fwhm_deg=fwhm_deg, nside=sky_nside)
    simeer_beam = _simeer_gaussian_beam(fwhm_deg=fwhm_deg)

    selfrot = np.zeros_like(lst_deg)

    tod_limtod = generate_TOD_sky(
        limtod_beam,
        sky_map,
        LST_deg_list=lst_deg,
        lat_deg=MEERKAT_LAT,
        azimuth_deg_list=az_deg,
        elevation_deg_list=el_deg,
        selfrot_deg_list=selfrot,
    )

    tod_simeer = integrate_tod(
        lst_deg_list=lst_deg,
        az_deg_list=az_deg,
        el_deg_list=el_deg,
        lat_deg=MEERKAT_LAT,
        beam=simeer_beam,
        sky_maps=sky_map[None, :],
        freq_MHz=[1000.0],
        disc_radius_deg=8.0,
    )
    return tod_limtod, tod_simeer[0]


# ---------------------------------------------------------------------- #
# Uniform-sky tests: each pipeline should return T0; they should agree   #
# ---------------------------------------------------------------------- #
@pytest.mark.integration
@pytest.mark.parametrize(
    "elevation_deg, az_center, fwhm_deg",
    [
        (45.0, 180.0, 2.5),  # baseline (mid-elevation, due south)
        (45.0, 0.0, 2.5),  # az = 0 (north / wrap edge)
        (45.0, 359.0, 2.5),  # other side of az wrap
        (89.0, 180.0, 2.5),  # near zenith
        (15.0, 180.0, 2.5),  # low elevation
        (45.0, 180.0, 1.5),  # narrower beam
        (45.0, 180.0, 4.0),  # wider beam (still fits inside +/-6 deg grid)
    ],
)
def test_uniform_sky(elevation_deg, az_center, fwhm_deg):
    """For a uniform T0 sky, both should return T0; they should agree."""
    sky_nside = 256
    T0 = 7.5
    sky_map = np.full(hp.nside2npix(sky_nside), T0)

    ntime = 5
    lst_deg = np.linspace(0.0, 30.0, ntime)
    az_deg = np.full(ntime, az_center)
    el_deg = np.full(ntime, elevation_deg)

    tod_l, tod_s = _run_pair(
        fwhm_deg=fwhm_deg,
        sky_nside=sky_nside,
        sky_map=sky_map,
        lst_deg=lst_deg,
        az_deg=az_deg,
        el_deg=el_deg,
    )
    # Each side should recover T0 within HEALPix discretisation noise.
    np.testing.assert_allclose(tod_l, T0, rtol=3e-2)
    np.testing.assert_allclose(tod_s, T0, rtol=3e-2)
    # And they should agree with each other.
    np.testing.assert_allclose(tod_s, tod_l, rtol=3e-2)


# ---------------------------------------------------------------------- #
# Gradient-sky test: a smooth gradient picks up real beam-weight detail. #
# ---------------------------------------------------------------------- #
@pytest.mark.integration
def test_gradient_sky_baseline():
    """A smooth Dec-gradient sky exercises real beam weighting, not just T0."""
    sky_nside = 256
    fwhm_deg = 2.5

    theta, _phi = hp.pix2ang(sky_nside, np.arange(hp.nside2npix(sky_nside)))
    sky_map = 10.0 + 5.0 * np.cos(theta)  # gradient with declination

    ntime = 8
    lst_deg = np.linspace(0.0, 60.0, ntime)
    az_deg = np.full(ntime, 180.0)
    el_deg = np.full(ntime, 45.0)

    tod_l, tod_s = _run_pair(
        fwhm_deg=fwhm_deg,
        sky_nside=sky_nside,
        sky_map=sky_map,
        lst_deg=lst_deg,
        az_deg=az_deg,
        el_deg=el_deg,
    )
    np.testing.assert_allclose(tod_s, tod_l, rtol=3e-2)


# ---------------------------------------------------------------------- #
# Az-raster test: varying az at fixed el (the typical MeerKLASS scan).   #
# ---------------------------------------------------------------------- #
@pytest.mark.integration
def test_azimuth_raster_uniform():
    """Az raster at fixed el; uniform sky -> flat TOD on both sides."""
    sky_nside = 256
    fwhm_deg = 2.5
    T0 = 4.0
    sky_map = np.full(hp.nside2npix(sky_nside), T0)

    ntime = 16
    lst_deg = np.full(ntime, 123.0)
    az_deg = np.linspace(170.0, 190.0, ntime)
    el_deg = np.full(ntime, 41.5)

    tod_l, tod_s = _run_pair(
        fwhm_deg=fwhm_deg,
        sky_nside=sky_nside,
        sky_map=sky_map,
        lst_deg=lst_deg,
        az_deg=az_deg,
        el_deg=el_deg,
    )
    np.testing.assert_allclose(tod_l, T0, rtol=3e-2)
    np.testing.assert_allclose(tod_s, T0, rtol=3e-2)
    np.testing.assert_allclose(tod_s, tod_l, rtol=3e-2)


# ---------------------------------------------------------------------- #
# SimeerTODSim end-to-end (Full TOD path inherited from limTOD.TODSim)   #
# ---------------------------------------------------------------------- #
@pytest.mark.integration
def test_simeer_tod_sim_simulate_sky_tod_uniform():
    """SimeerTODSim.simulate_sky_TOD on a uniform sky recovers T0."""
    from limTOD.simeer import SimeerTODSim

    beam = _simeer_gaussian_beam(fwhm_deg=2.5)
    sky_nside = 128
    T0 = 5.0

    def sky_func(*, freq, nside):
        return np.full(hp.nside2npix(nside), T0)

    sim = SimeerTODSim(
        beam=beam,
        sky_func=sky_func,
        sky_nside=sky_nside,
        disc_radius_deg=8.0,
        polarization="HH",
        n_jobs=1,
    )

    ntime = 8
    time_list = np.arange(ntime, dtype=np.float64) * 2.0
    az_list = np.linspace(170.0, 190.0, ntime)

    tod = sim.simulate_sky_TOD(
        freq_list=[1000.0],
        time_list=time_list,
        azimuth_deg_list=az_list,
        elevation_deg=41.5,
    )
    assert tod.shape == (1, ntime)
    np.testing.assert_allclose(tod[0], T0, rtol=3e-2)


@pytest.mark.integration
def test_simeer_tod_sim_generate_TOD_uniform():
    """SimeerTODSim.generate_TOD (Full TOD with gain + noise) end-to-end.

    Verifies the limTOD-inherited assembly works: shape of all returned
    arrays, sky_TOD ~ T0, overall_TOD non-zero and finite.
    """
    from limTOD.simeer import SimeerTODSim

    beam = _simeer_gaussian_beam(fwhm_deg=2.5)
    sky_nside = 64
    T0 = 5.0

    def sky_func(*, freq, nside):
        return np.full(hp.nside2npix(nside), T0)

    sim = SimeerTODSim(
        beam=beam,
        sky_func=sky_func,
        sky_nside=sky_nside,
        disc_radius_deg=8.0,
        polarization="HH",
        n_jobs=1,
    )

    ntime = 16
    time_list = np.arange(ntime, dtype=np.float64) * 2.0
    az_list = np.linspace(170.0, 190.0, ntime)

    overall_tod, sky_tod, gain_noise = sim.generate_TOD(
        freq_list=[1000.0],
        time_list=time_list,
        azimuth_deg_list=az_list,
        elevation_deg=41.5,
    )
    assert overall_tod.shape == (1, ntime)
    assert sky_tod.shape == (1, ntime)
    assert gain_noise.shape == (1, ntime)
    assert np.all(np.isfinite(overall_tod))
    assert np.all(np.isfinite(sky_tod))
    # Sky TOD is the pure beam-weighted sky -> ~ T0.
    np.testing.assert_allclose(sky_tod[0], T0, rtol=3e-2)
    # Overall TOD is sky modulated by gain + 1/f + white noise. It should
    # be close to the sky temperature on average but not identical to it.
    assert np.abs(overall_tod.mean() - T0) < 1.0  # rough order-of-magnitude check
    assert not np.allclose(overall_tod, sky_tod)  # noise should perturb it
