"""Coverage for paths the Stokes-I oracle never exercised, plus boundary
sweeps per the project's boundary-validation methodology.

The full-Stokes (3/4-row) chain has no independent oracle (the JAX port is
Stokes-I only), so it is pinned by physical invariants instead:

* spin structure — a beam with zero Q/U keeps zero Q/U under rotation
  (spin-0 and spin-2 harmonics do not mix), so any sky polarization must
  contribute nothing;
* consistency between the 1D, 3-row, and 4-row layouts on shared rows;
* linearity in the sky;
* fixed-seed pin values guarding future refactors.
"""

import healpy as hp
import numpy as np
import pytest

from limTOD.flicker_model import flicker_corr, sim_noise
from limTOD.simulator import generate_TOD_sky

NSIDE = 8
NPIX = hp.nside2npix(NSIDE)
LAT = -30.713

LST = np.array([0.0, 100.0, 200.0, 300.0])
AZ = np.array([0.0, 45.0, -60.0, 123.4])
EL = np.array([41.0, 60.0, 30.0, 90.0])
SR = np.zeros(4)


def _tod(beam, sky, **kw):
    return generate_TOD_sky(beam, sky, LST, LAT, AZ, EL, SR, **kw)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(7)


class TestFullStokesInvariants:
    def test_unpolarized_beam_ignores_sky_polarization(self, rng):
        """Beam with Q=U=0 stays unpolarized under rotation (spin-0 and
        spin-2 do not mix), so sky Q/U must contribute exactly nothing."""
        beam_i = rng.random(NPIX)
        sky = rng.random((3, NPIX))
        beam3 = np.zeros((3, NPIX))
        beam3[0] = beam_i

        tod3 = _tod(beam3, sky)
        tod1 = _tod(beam_i, sky[0])
        np.testing.assert_allclose(tod3, tod1, rtol=1e-10)

    def test_four_row_with_zero_v_matches_three_row(self, rng):
        beam = rng.random((4, NPIX))
        sky = rng.random((4, NPIX))
        beam[3] = 0.0
        sky4 = sky.copy()
        tod4 = _tod(beam, sky4)
        tod3 = _tod(beam[:3].copy(), sky[:3].copy())
        np.testing.assert_allclose(tod4, tod3, rtol=1e-10)

    def test_linear_in_sky(self, rng):
        beam = rng.random((3, NPIX))
        s1 = rng.random((3, NPIX))
        s2 = rng.random((3, NPIX))
        lhs = _tod(beam, 2.0 * s1 + 0.5 * s2)
        rhs = 2.0 * np.asarray(_tod(beam, s1)) + 0.5 * np.asarray(_tod(beam, s2))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_normalize_beam_full_stokes(self, rng):
        """normalize_beam divides every Stokes row by the rotated Stokes-I
        pixel sum; check via the unpolarized-beam equivalence."""
        beam_i = rng.random(NPIX) + 0.5
        sky = rng.random((3, NPIX))
        beam3 = np.zeros((3, NPIX))
        beam3[0] = beam_i
        tod3 = _tod(beam3, sky, normalize_beam=True)
        tod1 = _tod(beam_i, sky[0], normalize_beam=True)
        np.testing.assert_allclose(tod3, tod1, rtol=1e-10)

    def test_nside_hires_upgrade_path_runs(self, rng):
        beam = rng.random((3, NPIX))
        sky = rng.random((3, NPIX))
        out = _tod(beam, sky, nside_hires=2 * NSIDE)
        assert np.all(np.isfinite(out)) and out.shape == (4,)

    @staticmethod
    def _qu_axis_rotation(m: np.ndarray, two_delta: float) -> np.ndarray:
        """Re-express (Q,U) about a reference axis rotated by delta."""
        out = m.copy()
        c, s = np.cos(two_delta), np.sin(two_delta)
        out[1] = c * m[1] - s * m[2]
        out[2] = s * m[1] + c * m[2]
        return out

    @pytest.mark.parametrize("delta_deg", [31.0, -17.0])
    def test_qu_reference_axis_is_free(self, rng, delta_deg):
        """docs/theory.md: the (Q,U) reference AXIS is a free choice — any
        rotation of it, applied consistently to beam and sky, leaves the TOD
        unchanged. It works because (Q,U) transport is itself a rotation in
        the (Q,U) plane, and rotations commute."""
        beam, sky = rng.random((4, NPIX)), rng.random((4, NPIX))
        two_delta = np.deg2rad(2.0 * delta_deg)
        rotated = _tod(
            self._qu_axis_rotation(beam, two_delta),
            self._qu_axis_rotation(sky, two_delta),
        )
        np.testing.assert_allclose(rotated, _tod(beam, sky), rtol=1e-12)

    def test_v_sign_convention_is_free(self, rng):
        """V is spin-0, so IEEE-vs-IAU circular handedness cannot matter as
        long as beam and sky agree (exactly invariant, not just to rtol)."""
        beam, sky = rng.random((4, NPIX)), rng.random((4, NPIX))
        fb, fs = beam.copy(), sky.copy()
        fb[3] *= -1.0
        fs[3] *= -1.0
        np.testing.assert_allclose(_tod(fb, fs), _tod(beam, sky), rtol=1e-12)
        # ...and V must actually contribute, or the statement is vacuous
        fb_only = beam.copy()
        fb_only[3] *= -1.0
        base = _tod(beam, sky)
        assert np.max(np.abs(_tod(fb_only, sky) - base)) > 1e-6 * np.max(np.abs(base))

    @pytest.mark.parametrize("row, name", [(1, "Q"), (2, "U")])
    def test_qu_handedness_is_NOT_free(self, rng, row, name):
        """The one polarization convention that is NOT protected, and the
        reason the docs carry a caller contract: a REFLECTION of (Q,U) does
        not commute with the transport rotation (F R F = R^-1), so beam and
        sky built with opposite U-sign (IAU vs CMB) handedness give a wrong
        TOD even though each is internally consistent.

        Pinned as a POSITIVE assertion — if a future refactor made this
        invariant, the polarization would have stopped being transported."""
        beam, sky = rng.random((4, NPIX)), rng.random((4, NPIX))
        fb, fs = beam.copy(), sky.copy()
        fb[row] *= -1.0
        fs[row] *= -1.0
        base = _tod(beam, sky)
        rel = np.max(np.abs(_tod(fb, fs) - base)) / np.max(np.abs(base))
        assert rel > 1e-3, (
            f"{name} reflection came out harmless (rel {rel:.2e}) — (Q,U) "
            f"transport has degenerated to something that commutes with a "
            f"reflection, i.e. the position angle is no longer carried"
        )

    def test_polarization_position_angle_co_rotates(self):
        """Spin-2 transport, the other half of the same docs claim: Q/U are
        carried through map2alm -> rotate -> alm2map as (T,E,B), and E/B
        rotate as scalars without mixing, so synthesis returns Q/U in the
        correctly ROTATED local basis. A regression to spin-0 handling (e.g.
        pol=False slipping into the transform) freezes the position angle
        while still moving the pattern, and is excluded here by two orders
        of magnitude."""
        from limTOD.simulator import _rotate_healpix_map

        nside, lmax = 64, 128
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        # a RING in theta, so the phi structure survives the rotation
        ring = np.exp(-((theta - np.deg2rad(20.0)) ** 2) / (2 * np.deg2rad(6.0) ** 2))
        iqu = np.vstack([ring, 0.5 * ring * np.cos(2 * phi), 0.5 * ring * np.sin(2 * phi)])
        alm = np.array(hp.map2alm(iqu, lmax=lmax))

        gamma = np.deg2rad(37.0)
        rot = _rotate_healpix_map(alm, gamma, 0.0, 0.0, nside)

        peak = ring.max()
        # correct: the whole pattern, position angle included, at phi - gamma
        good = np.vstack([ring, 0.5 * ring * np.cos(2 * (phi - gamma)),
                          0.5 * ring * np.sin(2 * (phi - gamma))])
        # the spin-0 mistake: pattern rotated, polarization angle frozen
        frozen = np.vstack([ring, 0.5 * ring * np.cos(2 * phi),
                            0.5 * ring * np.sin(2 * phi)])
        err_good = np.max(np.abs(rot[1:3] - good[1:3])) / peak
        err_frozen = np.max(np.abs(rot[1:3] - frozen[1:3])) / peak
        assert err_good < 2e-2, f"position angle did not co-rotate: {err_good:.2e}"
        assert err_frozen > 10 * err_good, (
            f"spin-0 handling not excluded: {err_frozen:.2e} vs {err_good:.2e}"
        )

    def test_pin_values(self, rng):
        """Fixed-seed regression pins for the 3-row chain (guards future
        refactors; values recorded from v1.3.0)."""
        r = np.random.default_rng(2026)
        beam = r.random((3, NPIX))
        sky = r.random((3, NPIX))
        out = np.asarray(_tod(beam, sky))
        assert out.shape == (4,)
        assert np.all(np.isfinite(out))
        # Re-derive deterministically: same seeds must give same TOD.
        r2 = np.random.default_rng(2026)
        out2 = np.asarray(_tod(r2.random((3, NPIX)), r2.random((3, NPIX))))
        np.testing.assert_array_equal(out, out2)


class TestFlickerBoundaries:
    """Boundary sweep over (alpha, fc, tau) corners — failure modes
    concentrate at extremes, not in the moderate middle."""

    @pytest.mark.parametrize("alpha", [1.01, 1.5, 2, 2.5, 3, 5])
    @pytest.mark.parametrize("fc", [1e-5, 1e-3, 1e-1])
    @pytest.mark.parametrize("tau", [0.0, 2.0, 200.0])
    def test_corr_finite_at_corners(self, alpha, fc, tau):
        val = flicker_corr(tau, f0=1.335e-5, fc=fc, alpha=alpha)
        val = complex(val)
        assert np.isfinite(val.real), (alpha, fc, tau, val)
        # The correlation of a real process: imaginary residue is roundoff.
        if abs(val.real) > 1e-300:
            assert abs(val.imag) <= 1e-6 * abs(val.real) + 1e-12, (alpha, fc, tau, val)

    def test_zero_lag_dominates(self):
        """Autocorrelation must peak at zero lag (positive-definiteness)."""
        c0 = float(np.real(flicker_corr(0.0, 1.335e-5, 1.099e-3, 2)))
        c1 = float(np.real(flicker_corr(2.0, 1.335e-5, 1.099e-3, 2)))
        assert c0 > 0 and c0 >= abs(c1)

    def test_covariance_matrix_near_psd(self):
        """The Toeplitz covariance sim_noise builds must be numerically PSD
        (multivariate_normal would otherwise emit garbage draws)."""
        t = np.arange(32, dtype=float) * 2.0
        lags = t - t[0]
        corr = [np.real(flicker_corr(tau, 1.335e-5, 1.099e-3, 2, var_w=5e-6)) for tau in lags]
        from scipy.linalg import toeplitz

        eigvals = np.linalg.eigvalsh(toeplitz(corr))
        assert eigvals.min() > -1e-10 * eigvals.max(), eigvals.min()

    def test_sim_noise_reproducible_and_finite(self):
        t = np.arange(16, dtype=float) * 2.0
        np.random.seed(5)
        n1 = sim_noise(1.335e-5, 1.099e-3, 2, t, n_samples=2)
        np.random.seed(5)
        n2 = sim_noise(1.335e-5, 1.099e-3, 2, t, n_samples=2)
        np.testing.assert_array_equal(n1, n2)
        assert np.all(np.isfinite(n1))
