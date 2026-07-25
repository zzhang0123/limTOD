"""Regression tests pinning the v1.3.0 pre-release review fixes
(numpy package): horizontal-mask orientation, zero-sum-beam guards,
flicker-model error handling, short-TOD Wiener filtering, and
global-RNG hygiene."""

import healpy as hp
import numpy as np
import pytest

from limTOD.flicker_model import flicker_corr, sim_noise
from limTOD.HPW_filter import wiener_filter_map
from limTOD.simulator import (
    _beam_weighted_sum,
    _normalize_map,
    generate_TOD_sky,
    pointing_beam_in_eq_sys,
)
from limTOD.sky_model import generate_gaussian_field


class TestHorizontalMaskOrientation:
    """The horizontal-frame mask (pole at zenith) must land at the zenith's
    equatorial position (RA = LST, Dec = latitude) — before the v1.3.0 fix it
    was rotated 90 deg onto the horizon (elevation=0 instead of 90)."""

    def test_mask_peak_lands_at_zenith_radec(self):
        nside = 16
        npix = hp.nside2npix(nside)
        lst_deg, lat_deg = 123.4, -30.713

        # Mask: a 25-degree cap around the zenith (the horizontal pole).
        theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
        mask = (np.degrees(theta_pix) < 25.0).astype(float)

        # Uniform beam: the masked, pointed beam IS the rotated mask.
        beam_alm = hp.map2alm(np.ones(npix), lmax=3 * nside - 1)
        pointed = pointing_beam_in_eq_sys(
            beam_alm, lst_deg, lat_deg, azimuth_deg=0.0, elevation_deg=41.0,
            selfrot_deg=0.0, nside=nside, normalize=False,
            horizontal_mask=mask,
        )

        kept = pointed > 0.5
        assert kept.any(), "mask should keep a nonempty region"
        theta_kept, phi_kept = hp.pix2ang(nside, np.where(kept)[0])
        # Centroid direction of the kept region (unit-vector mean).
        vec = hp.ang2vec(theta_kept, phi_kept).mean(axis=0)
        vec /= np.linalg.norm(vec)
        theta_c, phi_c = hp.vec2ang(vec)
        dec_c = 90.0 - np.degrees(float(theta_c[0]))
        ra_c = np.degrees(float(phi_c[0])) % 360.0

        assert abs(dec_c - lat_deg) < 3.0, f"Dec {dec_c} != lat {lat_deg}"
        d_ra = (ra_c - lst_deg + 180.0) % 360.0 - 180.0
        assert abs(d_ra) < 3.0, f"RA {ra_c} != LST {lst_deg}"


class TestZeroSumBeamGuards:
    def test_normalize_map_raises_on_zero_sum(self):
        with pytest.raises(ValueError, match="pixel sum is zero"):
            _normalize_map(np.zeros(48))

    def test_beam_weighted_sum_1d_zero_beam_raises(self):
        with pytest.raises(ValueError, match="pixel sum is zero"):
            _beam_weighted_sum(np.zeros(48), np.ones(48), normalize=True)

    def test_beam_weighted_sum_stokes_zero_I_raises(self):
        beam = np.zeros((3, 48))
        beam[1:] = 1.0  # nonzero Q/U but zero Stokes I
        with pytest.raises(ValueError, match="pixel sum is zero"):
            _beam_weighted_sum(beam, np.ones((3, 48)), normalize=True)

    def test_negative_sum_scales_all_rows_consistently(self):
        rng = np.random.default_rng(0)
        beam = rng.random((3, 48))
        beam[0] *= -1.0  # negative Stokes-I sum: still a valid normalization
        sky = rng.random((3, 48))
        expected = np.sum((beam / np.sum(beam[0])) * sky)
        got = _beam_weighted_sum(beam, sky, normalize=True)
        np.testing.assert_allclose(got, expected, rtol=1e-12)


class TestFlickerModel:
    def test_alpha_one_rejected(self):
        with pytest.raises(ValueError, match="singular at alpha=1"):
            flicker_corr(0.0, 1e-5, 1e-3, alpha=1.0)
        with pytest.raises(ValueError, match="singular at alpha=1"):
            flicker_corr(2.0, 1e-5, 1e-3, alpha=np.float64(1.0))
        with pytest.raises(ValueError, match="singular at alpha=1"):
            sim_noise(1e-5, 1e-3, 1.0, np.arange(8, dtype=float))

    def test_aux_int_errors_propagate_with_context(self, monkeypatch):
        import limTOD.flicker_model as fm

        def boom(*args, **kwargs):
            raise ValueError("synthetic gammainc failure")

        monkeypatch.setattr(fm, "gammainc", boom)
        with pytest.raises(RuntimeError, match="aux_int failed"):
            fm.aux_int(0.5, 1.0)

    def test_sim_noise_shapes(self):
        t = np.arange(16, dtype=float) * 2.0
        assert sim_noise(1.335e-5, 1.099e-3, 2, t).shape == (1, 16)
        assert sim_noise(1.335e-5, 1.099e-3, 2, t, n_samples=3).shape == (3, 16)


class TestWienerShortTOD:
    def test_short_tod_auto_variance_runs(self):
        """TODs shorter than the 100-sample rolling window used to crash with
        an opaque matmul shape error in the default auto-variance path."""
        rng = np.random.default_rng(0)
        n_time, n_pix = 30, 5
        operator = rng.random((n_time, n_pix))
        tod = operator @ rng.random(n_pix) + 0.01 * rng.standard_normal(n_time)
        est, unc = wiener_filter_map(tod, operator, prior_inv_cov=1e-8)
        assert est.shape == (n_pix,) and unc.shape == (n_pix,)
        assert np.all(np.isfinite(est)) and np.all(np.isfinite(unc))

    def test_wrong_length_noise_variance_raises(self):
        rng = np.random.default_rng(0)
        operator = rng.random((30, 5))
        tod = rng.random(30)
        with pytest.raises(ValueError, match="noise_variance has length"):
            wiener_filter_map(tod, operator, noise_variance=np.ones(11))


class TestGaussianFieldRNG:
    def test_seed_none_is_deterministic_given_global_state(self):
        freqs = np.linspace(900.0, 1000.0, 3)
        np.random.seed(123)
        m1 = generate_gaussian_field(freqs=freqs, nside=8, amp=1.0, seed=None)
        np.random.seed(123)
        m2 = generate_gaussian_field(freqs=freqs, nside=8, amp=1.0, seed=None)
        np.testing.assert_array_equal(m1, m2)

    def test_explicit_seed_reproducible(self):
        freqs = np.linspace(900.0, 1000.0, 3)
        m1 = generate_gaussian_field(freqs=freqs, nside=8, amp=1.0, seed=42)
        m2 = generate_gaussian_field(freqs=freqs, nside=8, amp=1.0, seed=42)
        np.testing.assert_array_equal(m1, m2)


class TestPointingLengthValidation:
    def test_mismatched_pointing_arrays_raise(self):
        nside = 4
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="equal lengths"):
            generate_TOD_sky(
                rng.random(npix), rng.random(npix),
                np.zeros(3), 53.2, np.zeros(2), np.full(3, 90.0), np.zeros(3),
            )
