"""Tests for limTOD.gls_mapmaking (the hydra-tod GLS port).

Three layers:

1. Unit: the noise-covariance builder matches ``sim_noise``'s Toeplitz
   construction exactly; ``iterative_gls`` recovers known parameters.
2. Oracle: ``GLS_mapmaking`` output must equal an independently coded
   IRLS/GLS built in the test from the class's own system operators
   (same style as ``test_hpw_mapmaking_oracle``).
3. End-to-end: on limTOD-simulated TOD with 1/f gain noise, the GLS map
   (exact noise covariance) beats the OLS/uniform-weight map.
"""

import healpy as hp
import numpy as np
import pytest
from scipy.linalg import toeplitz

from limTOD.HPW_filter import HPW_mapmaking
from limTOD.flicker_model import flicker_corr
from limTOD.gls_mapmaking import (
    GLS_mapmaking,
    flicker_noise_cov,
    flicker_noise_inv_cov,
    iterative_gls,
)

NSIDE = 4
LAT = -30.713
REG = 1e-10
FLICKER = (1.335e-5, 1.099e-3, 2)
WVAR = 2.5e-6

N1, N2 = 120, 100


# ---------------------------------------------------------------------- #
# Noise covariance builder                                               #
# ---------------------------------------------------------------------- #
class TestFlickerNoiseCov:
    def test_matches_sim_noise_toeplitz(self):
        t = np.arange(50) * 2.0
        N = flicker_noise_cov(t, FLICKER, WVAR)
        corr = [flicker_corr(tau, *FLICKER, var_w=WVAR) for tau in t - t[0]]
        np.testing.assert_array_equal(N, toeplitz(corr))

    def test_inverse_times_cov_is_identity(self):
        t = np.arange(64) * 2.0
        N = flicker_noise_cov(t, FLICKER, WVAR)
        N_inv = flicker_noise_inv_cov(t, FLICKER, WVAR)
        np.testing.assert_allclose(N_inv @ N, np.eye(64), atol=1e-8)

    def test_white_only(self):
        t = np.arange(10) * 1.0
        N = flicker_noise_cov(t, None, 0.5)
        np.testing.assert_array_equal(N, 0.5 * np.eye(10))

    def test_zero_noise_rejected(self):
        with pytest.raises(ValueError, match="singular"):
            flicker_noise_cov(np.arange(4.0), None, 0.0)

    def test_negative_white_var_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            flicker_noise_cov(np.arange(4.0), FLICKER, -1.0)

    def test_alpha_one_rejected(self):
        # flicker_corr's alpha=1 singularity must propagate with its message
        with pytest.raises(ValueError, match="alpha=1"):
            flicker_noise_cov(np.arange(4.0), (1e-5, 1e-3, 1.0), WVAR)

    def test_empty_time_list_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            flicker_noise_cov(np.array([]), FLICKER, WVAR)


# ---------------------------------------------------------------------- #
# iterative_gls (faithful hydra-tod port)                                #
# ---------------------------------------------------------------------- #
class TestIterativeGLS:
    def _system(self, rng, n_time=80, n_par=6):
        U = rng.random((n_time, n_par)) + 0.1
        p_true = rng.random(n_par) * 5.0 + 1.0
        return U, p_true

    def test_noise_free_exact_recovery(self):
        """d = (U p + mu) exactly => GLS recovers p for ANY valid N."""
        rng = np.random.default_rng(0)
        U, p_true = self._system(rng)
        mu = 2.0
        d = (U @ p_true + mu)
        N_inv = flicker_noise_inv_cov(np.arange(len(d)) * 2.0, FLICKER, WVAR)
        p_hat, Sigma_inv = iterative_gls(d, U, N_inv, mu=mu)
        np.testing.assert_allclose(p_hat, p_true, rtol=1e-8)
        assert Sigma_inv.shape == (len(d), len(d))

    def test_white_noise_recovery(self):
        rng = np.random.default_rng(1)
        U, p_true = self._system(rng, n_time=400)
        mu = 1.0
        sigma = 1e-3
        n = rng.normal(0.0, sigma, size=400)
        d = (U @ p_true + mu) * (1.0 + n)
        N_inv = np.eye(400) / sigma**2
        p_hat, _ = iterative_gls(d, U, N_inv, mu=mu)
        np.testing.assert_allclose(p_hat, p_true, rtol=5e-3)

    def test_sigma_inv_is_evaluated_at_converged_estimate(self):
        rng = np.random.default_rng(2)
        U, p_true = self._system(rng)
        d = U @ p_true + 0.5
        N_inv = np.eye(len(d))
        p_hat, Sigma_inv = iterative_gls(d, U, N_inv, mu=0.5)
        D_inv = 1.0 / (U @ p_hat + 0.5)
        np.testing.assert_allclose(Sigma_inv, N_inv * np.outer(D_inv, D_inv))

    def test_zero_model_raises(self):
        U = np.ones((10, 1))
        d = np.zeros(10)  # OLS init gives p ~= 0 -> model ~ 0 with mu=0
        with pytest.raises(FloatingPointError, match="crossed zero"):
            iterative_gls(d, U, np.eye(10), mu=0.0)


# ---------------------------------------------------------------------- #
# GLS_mapmaking oracle (same well-posed geometry as the HPW oracle)      #
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def geometry():
    rng = np.random.default_rng(11)
    beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
    lst_g = [np.linspace(0.0, 60.0, N1), np.linspace(100.0, 160.0, N2)]
    az_g = [np.linspace(-30.0, 30.0, N1), np.zeros(N2)]
    el_g = [np.full(N1, 55.0), np.full(N2, 90.0)]
    kw = dict(
        beam_map=beam_map,
        LST_deg_list_group=lst_g,
        lat_deg=LAT,
        azimuth_deg_list_group=az_g,
        elevation_deg_list_group=el_g,
        threshold=0.7,
        nside_target=NSIDE,
    )
    mm = GLS_mapmaking(**kw)
    ops = list(mm.Tsys_operators)
    a_full = np.concatenate(ops, axis=0)
    assert np.linalg.matrix_rank(a_full) == a_full.shape[1]
    truth = rng.random(a_full.shape[1]) * 10.0 + 2.0  # strictly positive Tsys
    return mm, kw, ops, truth, rng


def _oracle_irls(d_list, U_list, Ninv_list, mu_list, truth_len, reg=REG,
                 s_inv=None, prior_mean=None, n_iter=60):
    """Independent IRLS implementation (deliberately written differently:
    explicit diag matrices instead of outer-product weighting)."""
    n_par = truth_len
    s_inv = np.zeros((n_par, n_par)) if s_inv is None else s_inv
    prior_mean = np.zeros(n_par) if prior_mean is None else prior_mean
    U_stack = np.vstack(U_list)
    r_stack = np.concatenate([d - mu for d, mu in zip(d_list, mu_list)])
    p = np.linalg.lstsq(U_stack, r_stack, rcond=None)[0]
    for _ in range(n_iter):
        A = s_inv + reg * np.eye(n_par)
        b = s_inv @ prior_mean
        for d, U, Ninv, mu in zip(d_list, U_list, Ninv_list, mu_list):
            Dinv = np.diag(1.0 / (U @ p + mu))
            Sinv = Dinv @ Ninv @ Dinv
            A = A + U.T @ Sinv @ U
            b = b + U.T @ Sinv @ (d - mu)
        p = np.linalg.solve(A, b)
    return p, A


class TestGLSMapmakingOracle:
    def test_multiplicative_matches_independent_irls(self, geometry):
        mm, kw, ops, truth, rng = geometry
        gen = np.random.default_rng(21)
        d_list, Ninv_list = [], []
        for U in ops:
            n_i = U.shape[0]
            t = np.arange(n_i) * 2.0
            N = flicker_noise_cov(t, FLICKER, WVAR)
            n_frac = gen.multivariate_normal(np.zeros(n_i), N)
            d_list.append((U @ truth) * (1.0 + n_frac))
            Ninv_list.append(flicker_noise_inv_cov(t, FLICKER, WVAR))

        est, unc = mm(
            TOD_group=d_list, dtime=2.0,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=REG, tol=1e-13, max_iter=60,
        )
        expected, A = _oracle_irls(
            d_list, ops, Ninv_list, [0.0, 0.0], len(truth)
        )
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(unc), np.sqrt(np.diag(np.linalg.inv(A))), rtol=1e-4
        )

    def test_additive_matches_single_gls_solve(self, geometry):
        mm, kw, ops, truth, rng = geometry
        gen = np.random.default_rng(22)
        sigma2 = 1e-4
        d_list = [U @ truth + gen.normal(0, np.sqrt(sigma2), U.shape[0])
                  for U in ops]
        Ninv = [np.eye(U.shape[0]) / sigma2 for U in ops]
        est, _ = mm(
            TOD_group=d_list, noise_inv_cov_group=Ninv,
            noise_model="additive", regularization=REG,
        )
        a_full = np.vstack(ops)
        d_full = np.concatenate(d_list)
        lhs = a_full.T @ a_full / sigma2 + REG * np.eye(a_full.shape[1])
        rhs = a_full.T @ d_full / sigma2
        np.testing.assert_allclose(
            np.asarray(est), np.linalg.solve(lhs, rhs), rtol=1e-8
        )

    def test_noise_free_multiplicative_recovers_truth(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        est, _ = mm(
            TOD_group=d_list, dtime=2.0,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=1e-14,
        )
        np.testing.assert_allclose(np.asarray(est), truth, rtol=1e-6)

    def test_known_injection_stays_in_model_not_subtracted(self, geometry):
        """mu must enter (U p + mu)(1+n), not d - mu: check against the
        oracle with a large injection, where the two treatments differ."""
        mm, kw, ops, truth, rng = geometry
        gen = np.random.default_rng(23)
        mu_vals = [np.full(U.shape[0], 30.0) for U in ops]
        d_list, Ninv_list = [], []
        for U, mu in zip(ops, mu_vals):
            n_i = U.shape[0]
            t = np.arange(n_i) * 2.0
            N = flicker_noise_cov(t, FLICKER, WVAR)
            n_frac = gen.multivariate_normal(np.zeros(n_i), N)
            d_list.append((U @ truth + mu) * (1.0 + n_frac))
            Ninv_list.append(flicker_noise_inv_cov(t, FLICKER, WVAR))
        est, _ = mm(
            TOD_group=d_list, dtime=2.0, known_injection_group=mu_vals,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=REG, tol=1e-13, max_iter=60,
        )
        expected, _ = _oracle_irls(d_list, ops, Ninv_list, mu_vals, len(truth))
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-6)
        assert np.all(np.isfinite(np.asarray(est)))

    def test_gain_division(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [2.0 * (U @ truth) for U in ops]
        est, _ = mm(
            TOD_group=d_list, dtime=2.0, gain_group=[2.0, 2.0],
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=1e-14,
        )
        np.testing.assert_allclose(np.asarray(est), truth, rtol=1e-6)

    def test_prior_pulls_toward_mean(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        prior_mean = np.full(len(truth), 5.0)
        s_diag = np.full(len(truth), 1e3)
        est, _ = mm(
            TOD_group=d_list, dtime=2.0,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            Tsky_prior_mean=prior_mean, Tsky_prior_inv_cov_diag=s_diag,
            regularization=REG, tol=1e-13, max_iter=60,
        )
        t = np.arange(ops[0].shape[0]) * 2.0
        Ninv_list = [
            flicker_noise_inv_cov(np.arange(U.shape[0]) * 2.0, FLICKER, WVAR)
            for U in ops
        ]
        expected, _ = _oracle_irls(
            d_list, ops, Ninv_list, [0.0, 0.0], len(truth),
            s_inv=np.diag(s_diag), prior_mean=prior_mean,
        )
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-6)

    def test_return_full_cov(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        est, unc, cov = mm(
            TOD_group=d_list, dtime=2.0,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=REG, return_full_cov=True,
        )
        n_par = len(truth)
        assert cov.shape == (n_par, n_par)
        np.testing.assert_allclose(np.asarray(unc), np.sqrt(np.diag(cov)),
                                   rtol=1e-10)

    def test_single_tod_flat_array(self):
        rng = np.random.default_rng(31)
        n = 400
        beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
        mm = GLS_mapmaking(
            beam_map=beam_map,
            LST_deg_list_group=np.linspace(0.0, 120.0, n),
            lat_deg=LAT,
            azimuth_deg_list_group=np.linspace(-25.0, 25.0, n),
            elevation_deg_list_group=np.full(n, 60.0),
            threshold=0.7,
            nside_target=NSIDE,
        )
        a = np.asarray(mm.Tsys_operators)
        assert np.linalg.matrix_rank(a) == a.shape[1], "test system must be well-posed"
        truth = rng.random(a.shape[1]) * 10.0 + 2.0
        d = a @ truth
        est, unc = mm(
            TOD_group=d, dtime=2.0,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=1e-14,
        )
        np.testing.assert_allclose(np.asarray(est), truth, rtol=1e-6)
        assert np.asarray(unc).shape == np.asarray(est).shape


class TestReviewRegressions:
    """Pins for the two pre-merge review findings."""

    def test_uncertainty_matches_estimate_when_not_converged(self, geometry):
        """The returned uncertainties must come from the normal equations
        evaluated AT the returned estimate. With the in-loop A (one
        iteration stale), max_iter=1 non-convergence used to skew the
        reported uncertainty at the ~1e-2 level."""
        mm, kw, ops, truth, rng = geometry
        gen = np.random.default_rng(77)
        d_list, Ninv_list = [], []
        for U in ops:
            n_i = U.shape[0]
            t = np.arange(n_i) * 2.0
            N = flicker_noise_cov(t, FLICKER, WVAR)
            n_frac = gen.multivariate_normal(np.zeros(n_i), N)
            d_list.append((U @ truth) * (1.0 + n_frac))
            Ninv_list.append(flicker_noise_inv_cov(t, FLICKER, WVAR))

        # Force non-convergence: 1 iteration, impossible tolerance.
        est, unc = mm(
            TOD_group=d_list, dtime=2.0,
            gain_noise_params=FLICKER, white_noise_var=WVAR,
            regularization=REG, tol=1e-300, min_iter=1, max_iter=1,
        )
        # Independent rebuild of A at the RETURNED estimate.
        n_par = len(truth)
        A = REG * np.eye(n_par)
        for d, U, Ninv in zip(d_list, ops, Ninv_list):
            Dinv = np.diag(1.0 / (U @ np.asarray(est)))
            Sinv = Dinv @ Ninv @ Dinv
            A = A + U.T @ Sinv @ U
        expected_unc = np.sqrt(np.diag(np.linalg.inv(A)))
        np.testing.assert_allclose(np.asarray(unc), expected_unc, rtol=1e-8)

    def test_additive_parametric_covariance_warns(self, geometry, caplog):
        """additive mode falling back to the fractional-noise parametric
        covariance must say so loudly."""
        import logging

        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        with caplog.at_level(logging.WARNING, logger="limTOD.gls_mapmaking"):
            mm(TOD_group=d_list, dtime=2.0, noise_model="additive",
               regularization=REG)
        assert any("FRACTIONAL" in r.message for r in caplog.records)

    def test_additive_with_explicit_cov_does_not_warn(self, geometry, caplog):
        import logging

        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        Ninv = [np.eye(U.shape[0]) for U in ops]
        with caplog.at_level(logging.WARNING, logger="limTOD.gls_mapmaking"):
            mm(TOD_group=d_list, noise_inv_cov_group=Ninv,
               noise_model="additive", regularization=REG)
        assert not any("FRACTIONAL" in r.message for r in caplog.records)


class TestGLSErrorPaths:
    def test_bad_noise_model(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        with pytest.raises(ValueError, match="noise_model"):
            mm(TOD_group=d_list, dtime=2.0, noise_model="banana")

    def test_missing_noise_spec(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        with pytest.raises(ValueError, match="dtime"):
            mm(TOD_group=d_list)

    def test_wrong_tod_count(self, geometry):
        mm, kw, ops, truth, rng = geometry
        with pytest.raises(ValueError, match="TODs"):
            mm(TOD_group=[ops[0] @ truth], dtime=2.0)

    def test_wrong_noise_cov_shape(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [U @ truth for U in ops]
        bad = [np.eye(3), np.eye(3)]
        with pytest.raises(ValueError, match="shape"):
            mm(TOD_group=d_list, noise_inv_cov_group=bad)

    def test_tod_operator_length_mismatch(self, geometry):
        mm, kw, ops, truth, rng = geometry
        d_list = [(U @ truth)[:-3] for U in ops]
        with pytest.raises(ValueError, match="samples"):
            mm(TOD_group=d_list, dtime=2.0)


# ---------------------------------------------------------------------- #
# End-to-end: GLS with the exact covariance beats uniform weighting      #
# ---------------------------------------------------------------------- #
class TestGLSvsOLSEndToEnd:
    def test_gls_beats_uniform_weights_under_1f_noise(self):
        """Same geometry, same 1/f-contaminated TOD: the GLS map (exact
        Toeplitz N) must have a smaller error than the additive
        uniform-weight (OLS-like) map. Fixed seed => deterministic.

        The red-noise knee is placed INSIDE the chunk (several correlation
        lengths per observation) — ultra-red noise whose period exceeds
        the chunk is degenerate with the sky's overall scale and neither
        estimator can do anything about it."""
        rng = np.random.default_rng(42)
        n = 400
        dt = 2.0
        beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
        kw = dict(
            beam_map=beam_map,
            LST_deg_list_group=np.linspace(0.0, 120.0, n),
            lat_deg=LAT,
            azimuth_deg_list_group=np.linspace(-25.0, 25.0, n),
            elevation_deg_list_group=np.full(n, 55.0),
            threshold=0.7,
            nside_target=NSIDE,
        )
        mm = GLS_mapmaking(**kw)
        a = np.asarray(mm.Tsys_operators)
        assert np.linalg.matrix_rank(a) == a.shape[1], "test system must be well-posed"
        truth = rng.random(a.shape[1]) * 10.0 + 5.0

        t = np.arange(n) * dt
        # Knee at ~1/8 of the chunk; rms(n_g) ~ 1% >> white 0.16%.
        fc = 2.0 * np.pi * 8.0 / (n * dt)
        f0 = np.sqrt(1e-4 * np.pi * fc)
        red_flicker = (f0, fc, 2)
        N = flicker_noise_cov(t, red_flicker, WVAR)
        n_frac = rng.multivariate_normal(np.zeros(n), N)
        d = (a @ truth) * (1.0 + n_frac)

        est_gls, _ = mm(
            TOD_group=d, time_list_group=[t],
            gain_noise_params=red_flicker, white_noise_var=WVAR,
            regularization=1e-8, tol=1e-13, max_iter=60,
        )
        est_uni, _ = mm(
            TOD_group=d, noise_inv_cov_group=[np.eye(n)],
            noise_model="additive", regularization=1e-8,
        )
        err_gls = np.linalg.norm(np.asarray(est_gls) - truth)
        err_uni = np.linalg.norm(np.asarray(est_uni) - truth)
        # Measured at this seed: 1115 vs 4587 (4.1x) — demand at least 2x.
        assert err_gls < 0.5 * err_uni, (err_gls, err_uni)
        # Pixel-space norms are inflated by the geometry's weak modes
        # (cond ~ 3e5); in the well-measured (TOD-projected) metric the
        # GLS reconstruction must be accurate in absolute terms.
        tod_rel = (
            np.linalg.norm(a @ (np.asarray(est_gls) - truth))
            / np.linalg.norm(a @ truth)
        )
        assert tod_rel < 0.02, tod_rel

    def test_matches_hpw_on_white_noise_additive(self):
        """With pure white noise both estimators solve the same normal
        equations: GLS(additive, N=sigma^2 I) == HPW(unfiltered)."""
        rng = np.random.default_rng(7)
        n = 400
        beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
        kw = dict(
            beam_map=beam_map,
            LST_deg_list_group=np.linspace(0.0, 120.0, n),
            lat_deg=LAT,
            azimuth_deg_list_group=np.linspace(-20.0, 20.0, n),
            elevation_deg_list_group=np.full(n, 60.0),
            threshold=0.7,
            nside_target=NSIDE,
        )
        gls = GLS_mapmaking(**kw)
        hpw = HPW_mapmaking(**kw)
        a = np.asarray(gls.Tsys_operators)
        assert np.linalg.matrix_rank(a) == a.shape[1], "test system must be well-posed"
        truth = rng.random(a.shape[1]) * 10.0
        sigma2 = 1e-4
        d = a @ truth + rng.normal(0, np.sqrt(sigma2), n)

        est_g, _ = gls(
            TOD_group=d, noise_inv_cov_group=[np.eye(n) / sigma2],
            noise_model="additive", regularization=1e-8,
        )
        est_h, _ = hpw(
            TOD_group=d, dtime=2.0, noise_variance=sigma2,
            regularization=1e-8,
        )
        # rtol 1e-5: HPW solves with assume_a='pos' (LAPACK posv), the GLS
        # with assume_a='sym' (sysv) — same equations, different
        # factorisation round-off.
        np.testing.assert_allclose(np.asarray(est_g), np.asarray(est_h),
                                   rtol=1e-5)
