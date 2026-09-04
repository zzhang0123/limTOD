"""Regression tests pinning the v1.3.0 pre-release review fixes
(numpy package): horizontal-mask orientation, zero-sum-beam guards,
flicker-model error handling, short-TOD Wiener filtering, and
global-RNG hygiene."""

import warnings

import healpy as hp
import numpy as np
import pytest

from limTOD.flicker_model import flicker_corr, sim_noise
from limTOD.HPW_filter import HPW_mapmaking, wiener_filter_map
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


class TestWienerNoiseVarianceRank:
    """``noise_variance`` is a scalar or a 1D per-sample variance — never a
    covariance matrix. The three cases the guard dispatches on are ndim 0, 1
    and 2, so all three are pinned here.

    Passing a full (n_time, n_time) covariance is the natural mistake: the
    argument is called a "noise variance", and the neighbouring GLS_mapmaking
    really does take a full ``noise_inv_cov_group``. It used to slip through
    the length check (len() of a 2D array is its row count, which for a
    time-time covariance is exactly n_time), warn on the off-diagonal zeros in
    ``1.0 / noise_variance``, get silently collapsed to its diagonal by
    np.diag, and finally surface as an opaque matmul core-dimension error.
    """

    @staticmethod
    def _system(n_time=12, n_pix=3):
        rng = np.random.default_rng(0)
        operator = rng.random((n_time, n_pix)) + 0.5
        tod = operator @ np.arange(1.0, n_pix + 1.0)
        return tod, operator

    def test_2d_noise_covariance_raises(self):
        tod, operator = self._system()
        with pytest.raises(ValueError, match="1D per-sample variance"):
            wiener_filter_map(tod, operator, noise_variance=np.eye(12) * 0.01)

    def test_2d_message_points_at_gls_mapmaking(self):
        """The message has to name the alternative, not just the rejection."""
        tod, operator = self._system()
        with pytest.raises(ValueError) as excinfo:
            wiener_filter_map(tod, operator, noise_variance=np.eye(12) * 0.01)
        message = str(excinfo.value)
        assert "(12, 12)" in message  # the shape actually passed
        assert "noise_inv_cov=" in message  # the nearest exit
        assert "GLS_mapmaking" in message
        assert "noise_inv_cov_group" in message

    def test_2d_rejected_before_the_divide(self):
        """The guard must fire ahead of ``1.0 / noise_variance``: no
        divide-by-zero RuntimeWarning on the off-diagonal zeros."""
        tod, operator = self._system()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="1D per-sample variance"):
                wiener_filter_map(tod, operator, noise_variance=np.eye(12) * 0.01)

    def test_0d_array_noise_variance_raises(self):
        """The other side of the boundary: a 0-d array is not a Python scalar,
        so it reached ``len()`` and died on "len() of unsized object"."""
        tod, operator = self._system()
        with pytest.raises(ValueError, match="1D per-sample variance"):
            wiener_filter_map(tod, operator, noise_variance=np.array(0.01))

    def test_scalar_variance_matches_normal_equations_on_noisy_data(self):
        """The weight-invariance caveat below is why this test exists: with
        noise and a prior, the scalar path's N^-1 = I/sigma^2 actually shows
        up in the answer, so a wrong construction (I, or sigma^2 I) fails."""
        rng = np.random.default_rng(21)
        n_time, n_pix, sigma2 = 30, 4, 0.02
        operator = rng.random((n_time, n_pix)) + 0.5
        tod = operator @ np.arange(1.0, n_pix + 1.0) + np.sqrt(sigma2) * rng.standard_normal(n_time)
        s_inv = np.diag(np.full(n_pix, 0.3))
        mu = np.full(n_pix, 0.7)

        est, _ = wiener_filter_map(
            tod, operator, noise_variance=sigma2, prior_inv_cov=np.diag(s_inv).copy(), guess=mu
        )
        lhs = operator.T @ operator / sigma2 + s_inv
        rhs = operator.T @ tod / sigma2 + s_inv @ mu
        np.testing.assert_allclose(est, np.linalg.solve(lhs, rhs), rtol=1e-9)

    def test_masked_noise_variance_raises(self):
        rng = np.random.default_rng(0)
        operator = rng.random((12, 3)) + 0.5
        tod = operator @ np.array([1.0, 2.0, 3.0])
        masked = np.ma.masked_array(np.full(12, 0.01), mask=np.zeros(12, dtype=bool))
        masked.mask[4] = True
        with pytest.raises(ValueError, match="masked array"):
            wiener_filter_map(tod, operator, noise_variance=masked)

    def test_1d_and_scalar_variance_still_accepted(self):
        """Positive control at ndim 1 (and the scalar path): the guard must
        not have narrowed what the function legitimately accepts."""
        tod, operator = self._system()
        est_vec, _ = wiener_filter_map(tod, operator, noise_variance=np.full(12, 0.01))
        est_scalar, _ = wiener_filter_map(tod, operator, noise_variance=0.01)
        np.testing.assert_allclose(est_vec, np.arange(1.0, 4.0), atol=1e-8)
        np.testing.assert_allclose(est_vec, est_scalar, rtol=1e-10)


class TestWienerFullNoiseInvCov:
    """``noise_inv_cov=`` takes a full (n_time, n_time) N^-1, so correlated
    noise can be weighted properly without leaving this function. It is the
    INVERSE covariance (as in GLS_mapmaking's ``noise_inv_cov_group``) — the
    O(n^3) inversion and its conditioning stay with the caller.

    None of these tests pass ``regularization``: the ridge is 1e-12 against
    normal-equation entries of order 1e3 here, i.e. below double-precision
    round-off, so every assertion holds identically with or without it.
    """

    @staticmethod
    def _system(n_time=40, n_pix=4, rho=0.7, sigma2=0.01, seed=7):
        """AR(1) noise: N[i,j] = sigma2 * rho**|i-j|, strongly correlated so
        the off-diagonal weighting genuinely changes the answer."""
        rng = np.random.default_rng(seed)
        operator = rng.random((n_time, n_pix)) + 0.5
        truth = np.arange(1.0, n_pix + 1.0)
        lags = np.abs(np.subtract.outer(np.arange(n_time), np.arange(n_time)))
        cov = sigma2 * rho ** lags
        tod = operator @ truth + np.linalg.cholesky(cov) @ rng.standard_normal(n_time)
        return tod, operator, np.linalg.inv(cov), truth

    def test_matches_normal_equations_with_prior(self):
        """The defining statement: x = (A^T N^-1 A + S^-1)^-1 (A^T N^-1 d +
        S^-1 mu) for a full N^-1. Also pins that supplying noise_inv_cov does
        not trip the mutual-exclusion check via the auto-variance estimator
        (which would otherwise fill noise_variance in behind our back)."""
        tod, operator, n_inv, _ = self._system()
        n_pix = operator.shape[1]
        s_inv_diag = np.full(n_pix, 1e-3)
        mu = np.full(n_pix, 0.5)

        est, unc, post_cov = wiener_filter_map(
            tod, operator, noise_inv_cov=n_inv,
            prior_inv_cov=s_inv_diag, guess=mu, return_full_cov=True,
        )

        s_inv = np.diag(s_inv_diag)
        lhs = operator.T @ n_inv @ operator + s_inv
        rhs = operator.T @ n_inv @ tod + s_inv @ mu
        expected_cov = np.linalg.inv(lhs)
        np.testing.assert_allclose(est, np.linalg.solve(lhs, rhs), rtol=1e-9)
        np.testing.assert_allclose(post_cov, expected_cov, rtol=1e-7)
        np.testing.assert_allclose(unc, np.sqrt(np.diag(expected_cov)), rtol=1e-7)

    def test_matches_whitened_least_squares(self):
        """Independent derivation, not a re-typing of the implementation's own
        normal equations: with N^-1 = L L^T, the GLS estimate is the ordinary
        least-squares fit of (L^T A) x = L^T d."""
        tod, operator, n_inv, _ = self._system()
        est, _ = wiener_filter_map(tod, operator, noise_inv_cov=n_inv)
        lower = np.linalg.cholesky(n_inv)
        expected = np.linalg.lstsq(lower.T @ operator, lower.T @ tod, rcond=None)[0]
        np.testing.assert_allclose(est, expected, rtol=1e-7)

    def test_diagonal_inv_cov_matches_noise_variance(self):
        """The two arguments must agree where they overlap."""
        tod, operator, _, _ = self._system()
        var = np.linspace(0.005, 0.02, tod.size)
        via_variance, _ = wiener_filter_map(tod, operator, noise_variance=var)
        via_inv_cov, _ = wiener_filter_map(tod, operator, noise_inv_cov=np.diag(1.0 / var))
        np.testing.assert_allclose(via_variance, via_inv_cov, rtol=1e-10)

    def test_off_diagonals_are_not_ignored(self):
        """Guards the tests as much as the code: if the correlated part of
        N^-1 happened not to matter for this system, the oracle above would
        pass even against an implementation that silently dropped it."""
        tod, operator, n_inv, _ = self._system()
        full, _ = wiener_filter_map(tod, operator, noise_inv_cov=n_inv)
        diag_only, _ = wiener_filter_map(
            tod, operator, noise_inv_cov=np.diag(np.diag(n_inv))
        )
        assert not np.allclose(full, diag_only, rtol=1e-3)

    def test_zero_weight_samples_are_allowed(self):
        """A flagged sample is expressed as zero weight, which makes N^-1
        positive SEMI-definite — the diagonal-sign check must not reject it.
        Zero weight must also mean exactly 'drop this sample'."""
        n_time = 20
        tod, operator, _, _ = self._system(n_time=n_time, n_pix=3)
        weights = np.full(n_time, 100.0)
        flagged = [3, 11]
        weights[flagged] = 0.0

        est, _ = wiener_filter_map(tod, operator, noise_inv_cov=np.diag(weights))

        kept = np.setdiff1d(np.arange(n_time), flagged)
        expected = np.linalg.lstsq(operator[kept], tod[kept], rcond=None)[0]
        np.testing.assert_allclose(est, expected, rtol=1e-6)

    def test_both_noise_arguments_raises(self):
        tod, operator, n_inv, _ = self._system()
        with pytest.raises(ValueError, match="not both"):
            wiener_filter_map(
                tod, operator, noise_variance=0.01, noise_inv_cov=n_inv
            )

    @pytest.mark.parametrize("bad", ["1d", "too_big", "not_square"])
    def test_wrong_shape_raises(self, bad):
        tod, operator, n_inv, _ = self._system()
        n_time = tod.size
        candidate = {
            "1d": np.ones(n_time),
            "too_big": np.eye(n_time + 1),
            "not_square": np.ones((n_time, n_time - 1)),
        }[bad]
        with pytest.raises(ValueError, match="inverse noise covariance"):
            wiener_filter_map(tod, operator, noise_inv_cov=candidate)

    def test_asymmetric_raises(self):
        """solve(..., assume_a='pos') reads only one triangle, so an
        asymmetric N^-1 would be silently symmetrised into a different
        weighting and return a wrong map instead of an error."""
        tod, operator, n_inv, _ = self._system()
        skewed = n_inv.copy()
        skewed[0, 1] += 0.5 * np.abs(n_inv).max()
        with pytest.raises(ValueError, match="Hermitian"):
            wiener_filter_map(tod, operator, noise_inv_cov=skewed)

    def test_asymmetric_at_roundoff_scale_is_accepted(self):
        """The other side of that boundary. The perturbation must be genuinely
        ASYMMETRIC — an earlier version of this test used
        ``np.eye(n)[::-1]``, the exchange matrix, which is symmetric and so
        probed nothing."""
        tod, operator, n_inv, _ = self._system()
        jittered = n_inv.copy()
        jittered[0, 1] += 1e-12 * np.abs(n_inv).max()
        assert not np.array_equal(jittered, jittered.T)
        est, _ = wiener_filter_map(tod, operator, noise_inv_cov=jittered)
        assert np.all(np.isfinite(est))

    @pytest.mark.parametrize("cond", [1e6, 1e8, 1e10, 1e12])
    def test_ill_conditioned_inverse_is_accepted(self, cond):
        """np.linalg.inv of a well-formed symmetric covariance is asymmetric
        at ~eps*cond relative to max|N^-1| — 1.2e-10 at cond 1e8, 7.9e-7 at
        1e12. A fixed strict tolerance would false-reject a caller who did
        exactly what the docstring tells them to do."""
        n_time, n_pix = 60, 3
        rng = np.random.default_rng(5)
        basis, _ = np.linalg.qr(rng.standard_normal((n_time, n_time)))
        cov = basis @ np.diag(np.geomspace(1.0, 1.0 / cond, n_time)) @ basis.T
        cov = 0.5 * (cov + cov.T)  # exactly symmetric by construction
        n_inv = np.linalg.inv(cov)
        assert not np.array_equal(n_inv, n_inv.T), "inv() should not be exactly symmetric here"

        operator = rng.random((n_time, n_pix)) + 0.5
        tod = operator @ np.arange(1.0, n_pix + 1.0)
        est, _ = wiener_filter_map(tod, operator, noise_inv_cov=n_inv)
        np.testing.assert_allclose(est, np.arange(1.0, n_pix + 1.0), rtol=1e-4)

    def test_subthreshold_asymmetry_is_symmetrised_not_left_to_the_solver(self):
        """Accepted-but-asymmetric input must be replaced by its Hermitian
        part. Left alone, solve(..., assume_a='pos') reads only the UPPER
        triangle, so the answer would depend on which triangle LAPACK happens
        to look at — a solver detail, not a documented choice."""
        tod, operator, n_inv, _ = self._system()
        delta = 1e-4 * np.abs(n_inv).max()  # skew = delta/2, just under the cut
        skewed = n_inv.copy()
        skewed[0, 1] += delta

        got, _ = wiener_filter_map(tod, operator, noise_inv_cov=skewed)
        hermitian, _ = wiener_filter_map(
            tod, operator, noise_inv_cov=0.5 * (skewed + skewed.T)
        )
        upper_mirrored = np.triu(skewed) + np.triu(skewed, 1).T
        via_upper, _ = wiener_filter_map(tod, operator, noise_inv_cov=upper_mirrored)

        np.testing.assert_allclose(got, hermitian, rtol=1e-14)
        # ...and the two really are distinguishable, so the assertion above
        # has teeth rather than passing because nothing differs.
        assert not np.allclose(got, via_upper, rtol=1e-9)

    def test_masked_noise_inv_cov_raises(self):
        """np.asarray drops the mask and uses whatever is underneath it, which
        is routinely finite, positive and symmetric — so every other check
        passes and the fit is quietly reweighted."""
        tod, operator, n_inv, _ = self._system()
        masked = np.ma.masked_array(n_inv, mask=np.zeros(n_inv.shape, dtype=bool))
        masked.mask[3, 3] = True
        with pytest.raises(ValueError, match="masked array"):
            wiener_filter_map(tod, operator, noise_inv_cov=masked)

    def test_masked_array_with_nothing_masked_is_fine(self):
        """Other side of that boundary: the type alone is not the problem."""
        tod, operator, n_inv, _ = self._system()
        unmasked = np.ma.masked_array(n_inv, mask=np.zeros(n_inv.shape, dtype=bool))
        est, _ = wiener_filter_map(tod, operator, noise_inv_cov=unmasked)
        expected, _ = wiener_filter_map(tod, operator, noise_inv_cov=n_inv)
        np.testing.assert_allclose(est, expected, rtol=1e-12)

    def test_non_finite_raises(self):
        """NaN must be caught explicitly: it sails through the symmetry test
        (NaN - NaN is NaN, and every comparison against NaN is False)."""
        tod, operator, n_inv, _ = self._system()
        for bad_value in (np.nan, np.inf):
            spoiled = n_inv.copy()
            spoiled[2, 2] = bad_value
            with pytest.raises(ValueError, match="NaN or inf"):
                wiener_filter_map(tod, operator, noise_inv_cov=spoiled)

    def test_negative_diagonal_raises(self):
        tod, operator, n_inv, _ = self._system()
        flipped = n_inv.copy()
        flipped[4, 4] *= -1.0
        with pytest.raises(ValueError, match="negative diagonal"):
            wiener_filter_map(tod, operator, noise_inv_cov=flipped)


class TestHPWNoiseVariancePerTODList:
    """Same mistake one level up: a per-TOD *list* of full covariances.
    ``HPW_mapmaking._normalize_noise_variance`` already rejected a 2D entry on
    shape, but said only "!= (n,)" — it now points at the map-maker that can
    actually use a covariance."""

    @staticmethod
    def _mapmaker(n1, n2, nside=4):
        rng = np.random.default_rng(3)
        return HPW_mapmaking(
            beam_map=rng.random(hp.nside2npix(nside)) + 0.2,
            LST_deg_list_group=[np.linspace(0.0, 10.0, n1), np.linspace(40.0, 50.0, n2)],
            lat_deg=-30.713,
            azimuth_deg_list_group=[np.linspace(-10.0, 10.0, n1), np.zeros(n2)],
            elevation_deg_list_group=[np.full(n1, 55.0), np.full(n2, 90.0)],
            threshold=0.9,
            nside_target=nside,
        )

    def test_per_tod_list_with_2d_entry_raises(self):
        n1, n2 = 8, 6
        mm = self._mapmaker(n1, n2)
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError) as excinfo:
            mm(
                TOD_group=[rng.random(n1), rng.random(n2)],
                dtime=2.0,
                noise_variance=[np.eye(n1) * 1e-4, np.eye(n2) * 1e-4],
            )
        message = str(excinfo.value)
        assert "noise_variance[0]" in message
        assert f"({n1}, {n1})" in message
        assert "GLS_mapmaking" in message and "noise_inv_cov_group" in message


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
