"""Unit tests for simeer.projection."""

from __future__ import annotations

import numpy as np
import pytest

from limTOD.simeer import projection


@pytest.mark.unit
def test_horizon_equatorial_roundtrip():
    """Horizon -> equatorial -> horizon recovers the inputs."""
    rng = np.random.default_rng(0)
    az = rng.uniform(0, 360, size=100)
    el = rng.uniform(10, 80, size=100)  # avoid horizon and zenith
    lst = 123.4
    lat = -30.7130

    ra, dec = projection.horizon_to_equatorial(az, el, lst, lat)
    az_back, el_back = projection.equatorial_to_horizon(ra, dec, lst, lat)

    np.testing.assert_allclose(np.cos(np.deg2rad(az)), np.cos(np.deg2rad(az_back)), atol=1e-10)
    np.testing.assert_allclose(np.sin(np.deg2rad(az)), np.sin(np.deg2rad(az_back)), atol=1e-10)
    np.testing.assert_allclose(el, el_back, atol=1e-10)


@pytest.mark.unit
def test_direction_cosines_zero_at_pointing():
    """Source at the pointing direction gives (l, m) = (0, 0)."""
    l, m = projection.direction_cosines(120.0, 45.0, 120.0, 45.0)  # noqa: E741
    assert abs(float(l)) < 1e-12
    assert abs(float(m)) < 1e-12


@pytest.mark.unit
def test_direction_cosines_small_offsets():
    """For small dAz, dEl the direction cosines reduce to dAz*cos(el), dEl."""
    az_p, el_p = 120.0, 45.0
    daz_deg = 0.001
    del_deg = 0.001
    az_s = az_p + daz_deg
    el_s = el_p + del_deg

    l, m = projection.direction_cosines(az_p, el_p, az_s, el_s)  # noqa: E741

    expected_l = np.deg2rad(daz_deg) * np.cos(np.deg2rad(el_p))
    expected_m = np.deg2rad(del_deg)
    np.testing.assert_allclose(float(l), expected_l, rtol=1e-4)
    np.testing.assert_allclose(float(m), expected_m, rtol=1e-4)


@pytest.mark.unit
def test_direction_cosines_vectorised():
    """Vectorised inputs work and broadcast correctly."""
    az_s = np.array([121.0, 119.0, 120.0])
    el_s = np.array([45.0, 45.0, 46.0])
    l, m = projection.direction_cosines(120.0, 45.0, az_s, el_s)  # noqa: E741
    assert l.shape == (3,)
    assert m.shape == (3,)
    # Source offset to the east (az_s > az_p) -> positive l.
    assert l[0] > 0
    # Source offset to the west -> negative l.
    assert l[1] < 0
    # Source higher than pointing -> positive m.
    assert m[2] > 0


@pytest.mark.unit
def test_pixel_directions_to_az_el_shape():
    import healpy as hp

    nside = 16
    pix = np.arange(hp.nside2npix(nside))
    az, el = projection.pixel_directions_to_az_el(nside, pix, lst_deg=0.0, lat_deg=-30.7)
    assert az.shape == pix.shape
    assert el.shape == pix.shape
    assert np.all((-180 <= el) & (el <= 90))
