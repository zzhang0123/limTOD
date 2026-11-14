"""Tests for GLS_mapmaking.posterior_samples (map sampler through hydra-tod).

The draws come from hydra-tod's ``params_space_oper_and_data`` and
``sample_p_v2``; the checks below use only limTOD's own map-maker as the
oracle.

1. Distribution: draws whitened by ``__call__``'s posterior covariance
   (``np.linalg.inv`` of limTOD's normal equations, a code path independent
   of hydra-tod) are standard normal, with and without a prior, for both
   noise models.
2. Approximation: the fixed-Sigma Gaussian that hydra-tod samples matches
   the exact multiplicative posterior's mode and curvature at limTOD noise
   levels (needs no hydra-tod).
3. API: seeding, shapes, other-Tsys groups, polarisation, error paths.
"""

import os
import subprocess
import sys

import healpy as hp
import numpy as np
import pytest

from limTOD.gls_mapmaking import (
    GLS_mapmaking,
    flicker_noise_cov,
    flicker_noise_inv_cov,
)
from limTOD.gls_sampler import draw_gls_posterior, hydra_map_sampler

try:
    import hydra_tod.linear_sampler  # noqa: F401

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

needs_hydra = pytest.mark.skipif(
    not HAS_HYDRA, reason="hydra-tod (with mpi4py) is not importable"
)

NSIDE = 4
LAT = -30.713
FLICKER = (1.335e-5, 1.099e-3, 2)
WVAR = 2.5e-6
N1, N2 = 120, 100
N_DRAWS = 4000


def _scan_kwargs(beam_map, n1=N1, n2=N2):
    return dict(
        beam_map=beam_map,
        LST_deg_list_group=[np.linspace(0.0, 60.0, n1), np.linspace(100.0, 160.0, n2)],
        lat_deg=LAT,
        azimuth_deg_list_group=[np.linspace(-30.0, 30.0, n1), np.zeros(n2)],
        elevation_deg_list_group=[np.full(n1, 55.0), np.full(n2, 90.0)],
        threshold=0.7,
        nside_target=NSIDE,
    )


def _multiplicative_tod(ops, params, seed):
    gen = np.random.default_rng(seed)
    d_list = []
    for U in ops:
        n_i = U.shape[0]
        N = flicker_noise_cov(np.arange(n_i) * 2.0, FLICKER, WVAR)
        d_list.append((U @ params) * (1.0 + gen.multivariate_normal(np.zeros(n_i), N)))
    return d_list


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(11)
    beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
    mm = GLS_mapmaking(**_scan_kwargs(beam_map))
    ops = list(mm.Tsys_operators)
    truth = rng.random(ops[0].shape[1]) * 10.0 + 2.0
    d_list = _multiplicative_tod(ops, truth, seed=21)
    noise_kw = dict(TOD_group=d_list, dtime=2.0,
                    gain_noise_params=FLICKER, white_noise_var=WVAR)
    return mm, ops, truth, d_list, noise_kw


@pytest.fixture(scope="module")
def prior_kw(setup):
    """A prior with 1/5 of the data's precision: dropping the prior's own
    perturbation from a draw shows up clearly in the whitened statistics."""
    mm, ops, truth, d_list, noise_kw = setup
    _, unc = mm(**noise_kw)
    return dict(
        Tsky_prior_mean=truth + 0.5,
        Tsky_prior_inv_cov_diag=1.0 / (2.0 * np.asarray(unc)) ** 2,
    )


def _assert_standard_normal(draws, mean, cov):
    """Whiten with ``cov``'s eigenbasis and check N(0, I) to MC precision."""
    lam, V = np.linalg.eigh(cov)
    W = (draws - mean) @ V / np.sqrt(lam)
    n, k = W.shape
    # overall scale: E|w|^2 = k, relative MC error sqrt(2 / (k n)) ~ 2e-3
    assert abs((W ** 2).sum(axis=1).mean() / k - 1.0) < 0.02
    assert np.abs(W.mean(axis=0)).max() * np.sqrt(n) < 4.5
    C = np.cov(W.T) - np.eye(k)
    assert np.abs(C).max() < 5.0 * np.sqrt(2.0 / n)
    # distributed errors: ||C - I||_F of exact N(0, I) draws is
    # sqrt(k(k+1)/n) with std 0.016 (k=163, n=4000, simulated); a 3% error
    # in every direction's variance adds ~0.1
    assert np.linalg.norm(C) < np.sqrt(k * (k + 1) / n) + 0.1


# ---------------------------------------------------------------------- #
# Distribution of the draws                                              #
# ---------------------------------------------------------------------- #
@needs_hydra
class TestPosteriorDistribution:
    def test_multiplicative_with_prior(self, setup, prior_kw):
        mm, ops, truth, d_list, noise_kw = setup
        est, _, cov = mm(**noise_kw, **prior_kw, return_full_cov=True)
        s = mm.posterior_samples(N_DRAWS, seed=1, **noise_kw, **prior_kw)
        _assert_standard_normal(s.params, np.asarray(est), cov)

    def test_multiplicative_without_prior(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        est, _, cov = mm(**noise_kw, return_full_cov=True)
        s = mm.posterior_samples(N_DRAWS, seed=2, **noise_kw)
        _assert_standard_normal(s.params, np.asarray(est), cov)

    def test_additive_exact_posterior(self, setup, prior_kw):
        mm, ops, truth, d_list, noise_kw = setup
        gen = np.random.default_rng(22)
        sigma2 = 1e-2
        d_add = [U @ truth + gen.normal(0.0, np.sqrt(sigma2), U.shape[0]) for U in ops]
        kw = dict(TOD_group=d_add, noise_model="additive",
                  noise_inv_cov_group=[np.eye(U.shape[0]) / sigma2 for U in ops])
        est, _, cov = mm(**kw, **prior_kw, return_full_cov=True)
        s = mm.posterior_samples(N_DRAWS, seed=3, **kw, **prior_kw)
        _assert_standard_normal(s.params, np.asarray(est), cov)

    @pytest.mark.parametrize("whole_tod", [False, True])
    def test_flagged_samples_semidefinite_noise_inv(self, setup, prior_kw, whole_tod):
        """Zeroed rows/columns of N^-1 (flagged samples, or a whole flagged
        TOD) make it only semi-definite; __call__ handles that, so the
        sampler must too."""
        mm, ops, truth, d_list, noise_kw = setup
        gen = np.random.default_rng(24)
        sigma2 = 1e-2
        d_add = [U @ truth + gen.normal(0.0, np.sqrt(sigma2), U.shape[0]) for U in ops]
        Ninv = [np.eye(U.shape[0]) / sigma2 for U in ops]
        if whole_tod:
            Ninv[1][:] = 0.0
        else:
            flagged = np.arange(30, 45)
            Ninv[0][flagged, :] = 0.0
            Ninv[0][:, flagged] = 0.0
        kw = dict(TOD_group=d_add, noise_model="additive", noise_inv_cov_group=Ninv,
                  **prior_kw)
        est, _, cov = mm(**kw, return_full_cov=True)
        s = mm.posterior_samples(N_DRAWS, seed=6, **kw)
        _assert_standard_normal(s.params, np.asarray(est), cov)

    def test_draws_centred_on_given_estimate(self, setup, prior_kw):
        """The draws are centred on p_hat itself, not on one more IRLS step
        from it: the two differ whenever the IRLS stopped short of its fixed
        point. A p_hat 3 sigma off the fixed point makes that visible."""
        mm, ops, truth, d_list, noise_kw = setup
        system = mm._solve_system(**noise_kw, **prior_kw)
        std = np.sqrt(np.diag(np.linalg.inv(system.A)))
        p_off = system.p + 3.0 * std
        draws, _ = draw_gls_posterior(
            hydra_map_sampler(),
            d_list=system.d_list, ops=system.ops, Ninv_list=system.Ninv_list,
            mu_list=system.mu_list, p_hat=p_off, S_inv=system.S_inv,
            noise_model=system.noise_model,
            n_draws=N_DRAWS, seed=7,
        )
        z = (draws.mean(axis=0) - p_off) / std * np.sqrt(N_DRAWS)
        assert np.abs(z).max() < 4.5

    def test_mean_and_precision_match_map_maker(self, setup, prior_kw):
        mm, ops, truth, d_list, noise_kw = setup
        est, _, cov = mm(**noise_kw, **prior_kw, return_full_cov=True)
        s = mm.posterior_samples(1, seed=4, **noise_kw, **prior_kw)
        np.testing.assert_allclose(s.mean, np.asarray(est), rtol=1e-12)
        np.testing.assert_allclose(s.precision @ cov, np.eye(len(est)), atol=1e-8)


# ---------------------------------------------------------------------- #
# Fixed-Sigma Gaussian vs the exact multiplicative posterior             #
# ---------------------------------------------------------------------- #
def _exact_grad_hess(p, d_list, ops, Ninv_list):
    """Gradient and Hessian of -log L for d = (U p)(1 + n), Cov[n] = N:
    -log L = 1/2 e^T N^-1 e + sum log m + const,  m = U p,  e = d/m - 1."""
    k = len(p)
    g, H = np.zeros(k), np.zeros((k, k))
    for d, U, Ninv in zip(d_list, ops, Ninv_list):
        m = U @ p
        e = d / m - 1.0
        J = (d / m ** 2)[:, None] * U
        Ne = Ninv @ e
        g += -J.T @ Ne + U.T @ (1.0 / m)
        H += (J.T @ Ninv @ J
              + U.T @ ((2.0 * d / m ** 3 * Ne)[:, None] * U)
              - U.T @ ((1.0 / m ** 2)[:, None] * U))
    return g, H


class TestFixedSigmaApproximation:
    def test_matches_exact_multiplicative_posterior(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        Ninv_list = [flicker_noise_inv_cov(np.arange(U.shape[0]) * 2.0, FLICKER, WVAR)
                     for U in ops]
        est, _, cov = mm(**noise_kw, return_full_cov=True)
        p_gls = np.asarray(est)

        # oracle check: analytic Hessian against central differences
        g0, H0 = _exact_grad_hess(p_gls, d_list, ops, Ninv_list)
        h = 1e-6 * np.abs(p_gls)
        cols = []
        for j in range(len(p_gls)):
            step = np.zeros_like(p_gls)
            step[j] = h[j]
            cols.append((_exact_grad_hess(p_gls + step, d_list, ops, Ninv_list)[0]
                         - _exact_grad_hess(p_gls - step, d_list, ops, Ninv_list)[0])
                        / (2.0 * h[j]))
        np.testing.assert_allclose(np.column_stack(cols), H0,
                                   atol=1e-6 * np.abs(H0).max())

        p = p_gls.copy()
        for _ in range(8):  # Newton on the exact posterior from the GLS point
            g, H = _exact_grad_hess(p, d_list, ops, Ninv_list)
            p = p - np.linalg.solve(H, g)
        g, H = _exact_grad_hess(p, d_list, ops, Ninv_list)
        assert np.abs(g).max() < 1e-6 * np.abs(g0).max() + 1e-8

        # this realisation: 0.004 std and 0.6%; over 20 realisations at most
        # 0.004 std and 0.9%. The bounds leave 5-10x headroom.
        assert (np.abs(p - p_gls) / np.sqrt(np.diag(cov))).max() < 0.05
        ratio = np.linalg.eigvals(cov @ H).real
        assert ratio.min() > 0.95 and ratio.max() < 1.05


# ---------------------------------------------------------------------- #
# API                                                                    #
# ---------------------------------------------------------------------- #
@needs_hydra
class TestSamplerAPI:
    def test_integer_seed_is_reproducible_and_isolated(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        a = mm.posterior_samples(5, seed=7, **noise_kw)
        after = np.random.get_state()[1]
        b = mm.posterior_samples(5, seed=7, **noise_kw)
        c = mm.posterior_samples(5, seed=8, **noise_kw)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(a.params, b.params)
        assert not np.allclose(a.params, c.params)

    def test_first_import_keeps_global_rng_state(self):
        """Importing hydra-tod pulls in jax, whose import calls np.random;
        the import must leave the caller's global stream where it was.
        Run in a fresh process, since this one has imported hydra-tod."""
        code = (
            "import numpy as np\n"
            "from limTOD.gls_sampler import hydra_map_sampler\n"
            "np.random.seed(5)\n"
            "expected = np.random.rand(3)\n"
            "np.random.seed(5)\n"
            "hydra_map_sampler()\n"
            "assert np.array_equal(np.random.rand(3), expected)\n"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        run = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]

    def test_generator_seed_is_reproducible(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        a = mm.posterior_samples(3, seed=np.random.default_rng(9), **noise_kw)
        b = mm.posterior_samples(3, seed=np.random.default_rng(9), **noise_kw)
        np.testing.assert_array_equal(a.params, b.params)

    def test_seed_none_draws_from_global_stream(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        np.random.seed(5)
        a = mm.posterior_samples(3, **noise_kw)
        np.random.seed(5)
        b = mm.posterior_samples(3, **noise_kw)
        np.testing.assert_array_equal(a.params, b.params)

    def test_shapes_single_pol_no_others(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        s = mm.posterior_samples(1, seed=0, **noise_kw)
        n_par = ops[0].shape[1]
        assert s.sky.shape == (1, mm.num_pixels)
        assert s.params.shape == (1, n_par)
        assert s.mean.shape == (n_par,)
        assert s.precision.shape == (n_par, n_par)
        assert s.others is None
        np.testing.assert_array_equal(s.sky, s.params[:, : mm.nsky_params])

    def test_other_tsys_groups(self, setup):
        mm0, ops0, truth, d_list, noise_kw = setup
        rng = np.random.default_rng(31)
        beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
        others = [np.column_stack([np.ones(n), np.linspace(0.0, 1.0, n)])
                  for n in (N1, N2)]
        mm = GLS_mapmaking(**_scan_kwargs(beam_map), Tsys_others_operator_group=others)
        ops = list(mm.Tsys_operators)
        params = np.concatenate([rng.random(mm.nsky_params) * 10.0 + 2.0,
                                 [20.0, 1.0], [25.0, -1.0]])
        d = _multiplicative_tod(ops, params, seed=32)
        s = mm.posterior_samples(6, seed=0, TOD_group=d, dtime=2.0,
                                 gain_noise_params=FLICKER, white_noise_var=WVAR)
        assert len(s.others) == 2
        n_sky = mm.nsky_params
        for i, block in enumerate(s.others):
            assert block.shape == (6, 2)
            np.testing.assert_array_equal(block, s.params[:, n_sky + 2 * i: n_sky + 2 * i + 2])

    def test_polarised_sky_shape(self):
        rng = np.random.default_rng(41)
        beam_map = rng.random((3, hp.nside2npix(NSIDE))) * 0.1
        beam_map[0] += 0.2
        mm = GLS_mapmaking(**_scan_kwargs(beam_map, n1=40, n2=40))
        ops = list(mm.Tsys_operators)
        d = [U @ np.ones(U.shape[1]) for U in ops]
        s = mm.posterior_samples(
            4, seed=0, TOD_group=d, noise_model="additive",
            noise_inv_cov_group=[np.eye(U.shape[0]) for U in ops],
            Tsky_prior_mean=np.zeros(mm.nsky_params),
            Tsky_prior_inv_cov_diag=np.ones(mm.nsky_params),
        )
        assert s.sky.shape == (4, 3, mm.num_pixels)


# ---------------------------------------------------------------------- #
# Error paths                                                            #
# ---------------------------------------------------------------------- #
class TestSamplerErrors:
    @pytest.mark.parametrize("bad", [0, -3, 2.5, True])
    def test_bad_n_draws(self, setup, bad):
        mm, ops, truth, d_list, noise_kw = setup
        with pytest.raises(ValueError, match="n_draws"):
            mm.posterior_samples(bad, **noise_kw)

    def test_return_full_cov_rejected(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        with pytest.raises(TypeError, match="return_full_cov"):
            mm.posterior_samples(2, return_full_cov=True, **noise_kw)

    @pytest.mark.parametrize("bad, err", [("a", TypeError), (1.5, TypeError),
                                          (True, TypeError), (-1, ValueError)])
    def test_bad_seed(self, setup, bad, err):
        mm, ops, truth, d_list, noise_kw = setup
        with pytest.raises(err, match="seed"):
            mm.posterior_samples(2, seed=bad, **noise_kw)

    @needs_hydra
    def test_indefinite_noise_inv_rejected(self, setup):
        mm, ops, truth, d_list, noise_kw = setup
        Ninv = [np.eye(U.shape[0]) for U in ops]
        Ninv[1][0, 0] = -1.0
        d_add = [U @ truth for U in ops]
        with pytest.raises(ValueError, match="semi-definite"):
            mm.posterior_samples(2, TOD_group=d_add, noise_model="additive",
                                 noise_inv_cov_group=Ninv)

    def test_missing_hydra_names_the_install(self, setup, monkeypatch):
        mm, ops, truth, d_list, noise_kw = setup
        monkeypatch.setitem(sys.modules, "hydra_tod.map_sampler", None)
        with pytest.raises(ImportError, match="hydra-tod"):
            mm.posterior_samples(2, **noise_kw)

    @needs_hydra
    def test_singular_full_prior_rejected(self):
        rng = np.random.default_rng(51)
        beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
        others = [np.column_stack([np.ones(n), np.linspace(0.0, 1.0, n)])
                  for n in (N1, N2)]
        mm = GLS_mapmaking(**_scan_kwargs(beam_map), Tsys_others_operator_group=others)
        ops = list(mm.Tsys_operators)
        params = np.concatenate([rng.random(mm.nsky_params) * 10.0 + 2.0,
                                 [20.0, 1.0], [25.0, -1.0]])
        d = _multiplicative_tod(ops, params, seed=52)
        full = np.array([[1.0, 0.2], [0.2, 1.0]])
        with pytest.raises(ValueError, match="positive definite"):
            mm.posterior_samples(
                2, TOD_group=d, dtime=2.0,
                gain_noise_params=FLICKER, white_noise_var=WVAR,
                Tsys_other_prior_mean_group=[np.zeros(2), np.zeros(2)],
                Tsys_other_prior_inv_cov_group=[full, full],
            )
