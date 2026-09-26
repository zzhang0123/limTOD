"""GDSM_sky_model hands the simulator an EQUATORIAL map.

pygdsm generates GSM16 in Galactic coordinates, while TODSim and the other
engines read a sky map as RA = phi, Dec = 90 - theta. Unrotated, the Galactic
centre was read as RA 0, Dec 0 and a zenith drift at latitude +53 swept the
b = +53 ring instead of the Dec = +53 one. pygdsm is replaced here by a stub
honouring its documented contract (Galactic, RING), so no sky data is
downloaded; the expected position comes from astropy, not from healpy's
Rotator that the implementation uses.
"""

import sys
import types

import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from limTOD.simulator import generate_TOD_sky
from limTOD.sky_model import GDSM_sky_model

NSIDE = 64
_GC = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic").icrs
GC_RA, GC_DEC = _GC.ra.deg, _GC.dec.deg  # 266.40, -28.94


def _galactic_centre_blob(nside=128):
    """1 K floor plus a 3 deg Gaussian at (l, b) = (0, 0), Galactic, RING."""
    vecs = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    ang = np.arccos(np.clip(hp.ang2vec(0.0, 0.0, lonlat=True) @ vecs, -1.0, 1.0))
    return 1.0 + 1000.0 * np.exp(-0.5 * (ang / np.radians(3.0)) ** 2)


class _StubGSM16:
    def generate(self, freq):
        return _galactic_centre_blob()


@pytest.fixture
def stub_pygdsm(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pygdsm", types.SimpleNamespace(GlobalSkyModel16=_StubGSM16)
    )


def _peak_lonlat(m):
    lon, lat = hp.pix2ang(hp.npix2nside(m.size), int(np.argmax(m)), lonlat=True)
    return lon % 360.0, lat


def test_default_map_is_equatorial(stub_pygdsm):
    ra, dec = _peak_lonlat(GDSM_sky_model(freq=70.0, nside=NSIDE))
    sep = SkyCoord(ra * u.deg, dec * u.deg).separation(_GC).deg
    assert sep < 1.5, (ra, dec, sep)  # one nside-64 pixel is 0.9 deg


def test_galactic_map_on_request(stub_pygdsm):
    lon, lat = _peak_lonlat(GDSM_sky_model(freq=70.0, nside=NSIDE, coord="G"))
    assert min(lon, 360.0 - lon) < 1.5 and abs(lat) < 1.5, (lon, lat)


def test_rotation_keeps_the_monopole(stub_pygdsm):
    eq = GDSM_sky_model(freq=70.0, nside=NSIDE)
    gal = GDSM_sky_model(freq=70.0, nside=NSIDE, coord="G")
    np.testing.assert_allclose(eq.mean(), gal.mean(), rtol=1e-6)


def test_zenith_beam_sees_the_galactic_centre_at_its_transit(stub_pygdsm):
    """The frame the simulator points the beam in matches the sky's frame."""
    sky = GDSM_sky_model(freq=70.0, nside=NSIDE)
    theta = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))[0]
    beam = np.exp(-0.5 * (theta / np.radians(2.0)) ** 2)
    lsts = np.array([GC_RA, GC_RA + 180.0])  # GC transit, then 12 h later
    tod = generate_TOD_sky(
        beam, sky, lsts, GC_DEC, np.zeros(2), np.full(2, 90.0), np.zeros(2),
        normalize_beam=True,
    )
    assert tod[0] > 100.0 and tod[1] < 2.0, tod


def test_unknown_coord_is_rejected():
    with pytest.raises(ValueError, match="coord"):
        GDSM_sky_model(freq=70.0, nside=NSIDE, coord="E")
