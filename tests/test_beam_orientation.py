"""Numerical pins of the beam coordinate convention (docs/theory.md).

The phi-orientation of the beam map is invisible to every symmetric-beam
test, so it is pinned here explicitly (boundary-validation policy:
conventions are locked numerically, never trusted on paper — the retired
conventions.pdf figure disagreed with the implementation by 90 degrees).

Method: place a small Gaussian blob at (theta0, phi_b) on the beam map,
push it through the full pointing chain (``pointing_beam_in_eq_sys``),
and compare the landing direction against the exact expectation

    v_expect = cos(theta0) * b_hat + sin(theta0) * t_hat,
    t_hat    = cos(phi_b + psi) * e_el + sin(phi_b + psi) * e_az,

where e_el = d(b_hat)/d(el) is the increasing-elevation tangent and
e_az the increasing-azimuth tangent (docs/theory.md). The site is at
(lat = 0, LST = 0) so the horizontal->equatorial part is trivial:
zenith = (RA 0, Dec 0), North horizon = NCP, East point = (RA 90, Dec 0).
"""

import healpy as hp
import numpy as np
import pytest

from limTOD.simulator import pointing_beam_in_eq_sys

NSIDE = 64
LMAX = 3 * NSIDE - 1
THETA0 = np.deg2rad(10.0)
SIGMA = np.deg2rad(2.0)
# argmax on the HEALPix grid quantizes at the pixel scale (~0.9 deg).
TOL_DEG = 1.5

# Equatorial unit vectors of the lat=0, LST=0 reference directions.
NCP = np.array([0.0, 0.0, 1.0])                  # North horizon point
ZENITH = np.array([1.0, 0.0, 0.0])               # (RA 0, Dec 0)
EAST = np.array([0.0, 1.0, 0.0])                 # (RA 90, Dec 0)


def _blob_map(phi_b_deg: float) -> np.ndarray:
    vec0 = np.asarray(hp.ang2vec(THETA0, np.deg2rad(phi_b_deg))).ravel()
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    vecs = np.asarray(hp.ang2vec(theta, phi))
    ang = np.arccos(np.clip(vecs @ vec0, -1.0, 1.0))
    return np.exp(-0.5 * (ang / SIGMA) ** 2)


def _landing_vec(phi_b_deg: float, az: float, el: float, selfrot: float = 0.0) -> np.ndarray:
    alm = hp.map2alm(_blob_map(phi_b_deg), lmax=LMAX)
    out = pointing_beam_in_eq_sys(
        alm, LST_deg=0.0, lat_deg=0.0, azimuth_deg=az, elevation_deg=el,
        selfrot_deg=selfrot, nside=NSIDE, normalize=False,
    )
    return np.asarray(hp.pix2vec(NSIDE, int(np.argmax(out))))


def _expected_vec(b_hat, e_el, e_az, phi_b_deg: float, psi_deg: float = 0.0):
    ang = np.deg2rad(phi_b_deg + psi_deg)
    t_hat = np.cos(ang) * e_el + np.sin(ang) * e_az
    return np.cos(THETA0) * b_hat + np.sin(THETA0) * t_hat


def _sep_deg(u, v) -> float:
    return float(np.rad2deg(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


@pytest.mark.integration
class TestBeamOrientation:
    """phi=0 -> e_el, phi=90 -> e_az, at two independent pointings."""

    # Pointing az=0 (North horizon): b = NCP, e_el = ZENITH, e_az = EAST.
    @pytest.mark.parametrize("phi_b", [0.0, 90.0, 180.0, 270.0])
    def test_identity_pointing_reads_map_as_equatorial(self, phi_b):
        """lat=0, LST=0, az=0, el=0 is the identity of the chain: the map
        IS the equatorial map (pole -> NCP, phi = RA)."""
        got = _landing_vec(phi_b, az=0.0, el=0.0)
        expected = _expected_vec(NCP, ZENITH, EAST, phi_b)
        assert _sep_deg(got, expected) < TOL_DEG

    # Pointing az=90 (East horizon): b = EAST point, e_el = ZENITH,
    # e_az (increasing azimuth) = the horizon great-circle direction from
    # (RA 90, Dec 0) toward the SCP.
    @pytest.mark.parametrize("phi_b", [0.0, 90.0, 270.0])
    def test_east_pointing_e_el_and_e_az(self, phi_b):
        got = _landing_vec(phi_b, az=90.0, el=0.0)
        e_az = np.array([0.0, 0.0, -1.0])  # toward the SCP
        expected = _expected_vec(EAST, ZENITH, e_az, phi_b)
        assert _sep_deg(got, expected) < TOL_DEG

    def test_selfrot_rotates_e_el_toward_e_az(self):
        """Positive selfrot carries the phi = 0 feature toward phi = +90
        (from e_el toward e_az), by the selfrot angle."""
        got = _landing_vec(0.0, az=0.0, el=0.0, selfrot=30.0)
        expected = _expected_vec(NCP, ZENITH, EAST, 0.0, psi_deg=30.0)
        assert _sep_deg(got, expected) < TOL_DEG
        # And the mirror sense must be wrong by ~2*psi*sin(theta0).
        mirror = _expected_vec(NCP, ZENITH, EAST, 0.0, psi_deg=-30.0)
        assert _sep_deg(got, mirror) > 4.0 * TOL_DEG

    def test_mirror_convention_rejected(self):
        """phi = 90 landing along -e_az (the mirrored convention) must be
        far off — this is the case a symmetric beam can never detect."""
        got = _landing_vec(90.0, az=0.0, el=0.0)
        mirrored = _expected_vec(NCP, ZENITH, -EAST, 90.0)
        assert _sep_deg(got, mirrored) > 10.0 * TOL_DEG
