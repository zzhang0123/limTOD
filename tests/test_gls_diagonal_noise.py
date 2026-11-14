"""Diagonal (white) noise in GLS_mapmaking and its posterior sampler.

A ``noise_inv_cov_group`` entry may be the ``(n_time,)`` diagonal of
``N^-1`` instead of the dense ``(n_time, n_time)`` matrix, and
``gain_noise_params=None`` builds that diagonal. No ``n_time x n_time``
array is then formed, in the solve or in the draws.

TestDiagonalMatchesDense compares the two representations of the same
noise. The diagonal and dense paths must give the same estimate and
covariance. With the same seed and no flagged sample, they must also give
the same draws: both use the same standard normals, and the Cholesky
factor of a positive diagonal matrix is its element-wise square root.
"""

import tracemalloc

import healpy as hp
import numpy as np
import pytest

from limTOD.gls_mapmaking import GLS_mapmaking

try:
    import hydra_tod.linear_sampler  # noqa: F401

    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

needs_hydra = pytest.mark.skipif(
    not HAS_HYDRA, reason="hydra-tod (with mpi4py) is not importable"
)

NSIDE = 4
N1, N2 = 120, 100
WVAR = 2.5e-6


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(11)
    mm = GLS_mapmaking(
        beam_map=rng.random(hp.nside2npix(NSIDE)) + 0.2,
        LST_deg_list_group=[np.linspace(0.0, 60.0, N1), np.linspace(100.0, 160.0, N2)],
        lat_deg=-30.713,
        azimuth_deg_list_group=[np.linspace(-30.0, 30.0, N1), np.zeros(N2)],
        elevation_deg_list_group=[np.full(N1, 55.0), np.full(N2, 90.0)],
        threshold=0.7,
        nside_target=NSIDE,
    )
    ops = list(mm.Tsys_operators)
    sky = rng.random(mm.nsky_params) * 10.0 + 2.0
    mu = [np.full(U.shape[0], 20.0) for U in ops]  # a known receiver temperature
    gen = np.random.default_rng(3)
    # per-sample inverse variances; w has flagged (zero-weight) samples
    w_full = [gen.uniform(0.5, 2.0, U.shape[0]) / WVAR for U in ops]
    w = [x.copy() for x in w_full]
    w[0][30:45] = 0.0
    d_mul = [(U @ sky + m) * (1.0 + gen.normal(0.0, 1.0, U.shape[0]) / np.sqrt(wi))
             for U, m, wi in zip(ops, mu, w_full)]
    d_add = [U @ sky + m + gen.normal(0.0, 0.1, U.shape[0]) for U, m in zip(ops, mu)]
    prior = dict(Tsky_prior_mean=sky + 0.5,
                 Tsky_prior_inv_cov_diag=np.full(mm.nsky_params, 1e-2))
    return mm, ops, sky, mu, (w, w_full), d_mul, d_add, prior


def _models(setup, flagged=True):
    """(label, kwargs-with-diagonal, kwargs-with-dense) for both noise models.

    ``flagged=False`` keeps every weight positive, so the dense path's
    square root is the Cholesky factor diag(sqrt(w)) rather than an eigh
    factor whose columns come out permuted."""
    mm, ops, sky, mu, (w_flag, w_full), d_mul, d_add, prior = setup
    w = w_flag if flagged else w_full
    w_add = [np.full(U.shape[0], 100.0) for U in ops]
    if flagged:
        w_add[1][:10] = 0.0
    return [
        ("multiplicative",
         dict(TOD_group=d_mul, known_injection_group=mu, noise_inv_cov_group=w),
         dict(TOD_group=d_mul, known_injection_group=mu,
              noise_inv_cov_group=[np.diag(x) for x in w])),
        ("additive",
         dict(TOD_group=d_add, known_injection_group=mu, noise_model="additive",
              noise_inv_cov_group=w_add),
         dict(TOD_group=d_add, known_injection_group=mu, noise_model="additive",
              noise_inv_cov_group=[np.diag(x) for x in w_add])),
    ]


def _assert_same_up_to_roundoff(x, y, cov):
    """Assert |x - y| <= 10 eps cond(cov) max|y|, the round-off that a solve
    with this conditioning can amplify. A fixed rtol of 1e-9 is too tight:
    without a prior the geometry's A has cond ~ 9e6.

    Measured for the two paths: A identical; p differs by 1.9e-10 max|p|;
    eps cond(A) = 2e-9.
    """
    bound = 10.0 * np.finfo(float).eps * np.linalg.cond(cov) * np.abs(y).max()
    np.testing.assert_allclose(x, y, rtol=0.0, atol=bound)


# ---------------------------------------------------------------------- #
# Diagonal vs dense: the same white noise, two representations          #
# ---------------------------------------------------------------------- #
class TestDiagonalMatchesDense:
    @pytest.mark.parametrize("model", [0, 1], ids=["multiplicative", "additive"])
    @pytest.mark.parametrize("with_prior", [False, True])
    def test_estimate_and_covariance(self, setup, model, with_prior):
        prior = setup[-1] if with_prior else {}
        _, diag_kw, dense_kw = _models(setup)[model]
        mm = setup[0]
        est_d, _, cov_d = mm(**diag_kw, **prior, return_full_cov=True)
        est_n, _, cov_n = mm(**dense_kw, **prior, return_full_cov=True)
        _assert_same_up_to_roundoff(np.asarray(est_d), np.asarray(est_n), cov_n)
        _assert_same_up_to_roundoff(cov_d, cov_n, cov_n)

    def test_gain_noise_params_none_matches_dense_white(self, setup):
        mm, ops, sky, mu, _, d_mul, d_add, prior = setup
        est_d, _, cov_d = mm(TOD_group=d_mul, known_injection_group=mu, dtime=2.0,
                             gain_noise_params=None, white_noise_var=WVAR,
                             return_full_cov=True)
        est_n, _, cov_n = mm(TOD_group=d_mul, known_injection_group=mu,
                             noise_inv_cov_group=[np.eye(U.shape[0]) / WVAR for U in ops],
                             return_full_cov=True)
        _assert_same_up_to_roundoff(np.asarray(est_d), np.asarray(est_n), cov_n)
        _assert_same_up_to_roundoff(cov_d, cov_n, cov_n)

    @needs_hydra
    @pytest.mark.parametrize("model", [0, 1], ids=["multiplicative", "additive"])
    def test_same_seed_same_draws(self, setup, model):
        """The diagonal and dense sampler paths give the same draws for one
        seed, to round-off."""
        mm, prior = setup[0], setup[-1]
        _, diag_kw, dense_kw = _models(setup, flagged=False)[model]
        s_d = mm.posterior_samples(50, seed=4, **diag_kw, **prior)
        s_n = mm.posterior_samples(50, seed=4, **dense_kw, **prior)
        _assert_same_up_to_roundoff(s_d.params, s_n.params, np.linalg.inv(s_n.precision))

    @needs_hydra
    def test_draws_follow_posterior(self, setup):
        """Independent of the dense path: whitened draws are N(0, I)."""
        mm, prior = setup[0], setup[-1]
        _, diag_kw, _ = _models(setup)[0]
        est, _, cov = mm(**diag_kw, **prior, return_full_cov=True)
        s = mm.posterior_samples(4000, seed=5, **diag_kw, **prior)
        lam, V = np.linalg.eigh(cov)
        W = (s.params - np.asarray(est)) @ V / np.sqrt(lam)
        n, k = W.shape
        assert abs((W ** 2).sum(axis=1).mean() / k - 1.0) < 0.02
        assert np.abs(np.cov(W.T) - np.eye(k)).max() < 5.0 * np.sqrt(2.0 / n)

    def test_fully_flagged_tod(self, setup):
        mm, ops, sky, mu, (w, _), d_mul, d_add, prior = setup
        w0 = [w[0], np.zeros(N2)]
        est_d, _, cov_d = mm(TOD_group=d_mul, known_injection_group=mu,
                             noise_inv_cov_group=w0, **prior, return_full_cov=True)
        est_n, _, cov_n = mm(TOD_group=d_mul, known_injection_group=mu,
                             noise_inv_cov_group=[np.diag(x) for x in w0], **prior,
                             return_full_cov=True)
        _assert_same_up_to_roundoff(np.asarray(est_d), np.asarray(est_n), cov_n)
        _assert_same_up_to_roundoff(cov_d, cov_n, cov_n)


# ---------------------------------------------------------------------- #
# Validation of diagonal entries                                         #
# ---------------------------------------------------------------------- #
class TestDiagonalValidation:
    def test_wrong_length(self, setup):
        mm, ops, sky, mu, (w, _), d_mul, *_ = setup
        with pytest.raises(ValueError, match=r"noise_inv_cov_group\[1\]"):
            mm(TOD_group=d_mul, noise_inv_cov_group=[w[0], w[1][:-1]])

    def test_negative_weight(self, setup):
        mm, ops, sky, mu, (w, _), d_mul, *_ = setup
        bad = w[1].copy()
        bad[3] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            mm(TOD_group=d_mul, noise_inv_cov_group=[w[0], bad])

    def test_non_finite_weight(self, setup):
        mm, ops, sky, mu, (w, _), d_mul, *_ = setup
        bad = w[1].copy()
        bad[3] = np.inf
        with pytest.raises(ValueError, match="finite"):
            mm(TOD_group=d_mul, noise_inv_cov_group=[w[0], bad])

    def test_white_only_needs_positive_variance(self, setup):
        mm, ops, sky, mu, (w, _), d_mul, *_ = setup
        with pytest.raises(ValueError, match="white_noise_var"):
            mm(TOD_group=d_mul, dtime=2.0, gain_noise_params=None, white_noise_var=0.0)


# ---------------------------------------------------------------------- #
# Memory of long white-noise TODs                                       #
# ---------------------------------------------------------------------- #
N_LONG, K_LONG = 8000, 20  # a dense N^-1 would be 512 MB per TOD


def _synthetic_mapmaker(ops):
    """GLS_mapmaking over given operators, skipping the beam-geometry build
    (~7 ms per sample), so that long TODs can be tested quickly."""
    mm = object.__new__(GLS_mapmaking)
    mm.num_tods = len(ops)
    mm.Tsys_operators = list(ops)
    mm.npol = 1
    mm.num_pixels = mm.nsky_params = ops[0].shape[1]
    mm.Tsys_others = False
    mm.n_params_others = 0
    return mm


@pytest.fixture(scope="module")
def long_white():
    gen = np.random.default_rng(0)
    ops = [gen.random((N_LONG, K_LONG)) + 0.1 for _ in range(2)]
    truth = gen.random(K_LONG) * 10.0 + 2.0
    d = [(U @ truth) * (1.0 + gen.normal(0.0, np.sqrt(WVAR), N_LONG)) for U in ops]
    kw = dict(TOD_group=d, dtime=2.0, gain_noise_params=None, white_noise_var=WVAR)
    return _synthetic_mapmaker(ops), ops, d, kw


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        out = fn()
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


class TestLongWhiteTODs:
    BUDGET = N_LONG * N_LONG * 8 / 10  # a tenth of one dense matrix

    def test_map_maker_memory_and_answer(self, long_white):
        mm, ops, d, kw = long_white
        (est, unc), peak = _peak_bytes(lambda: mm(**kw))
        assert peak < self.BUDGET, f"peak {peak / 1e6:.0f} MB"
        # oracle: IRLS written out with explicit per-sample weights
        p = np.linalg.lstsq(np.vstack(ops), np.concatenate(d), rcond=None)[0]
        for _ in range(30):
            A = sum(U.T @ (U / (U @ p)[:, None] ** 2) for U in ops) / WVAR
            b = sum(U.T @ (di / (U @ p) ** 2) for U, di in zip(ops, d)) / WVAR
            p = np.linalg.solve(A, b)
        np.testing.assert_allclose(np.asarray(est), p, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(unc), np.sqrt(np.diag(np.linalg.inv(A))),
                                   rtol=1e-6)

    @needs_hydra
    def test_sampler_memory(self, long_white):
        mm, ops, d, kw = long_white
        s, peak = _peak_bytes(lambda: mm.posterior_samples(20, seed=0, **kw))
        assert peak < self.BUDGET, f"peak {peak / 1e6:.0f} MB"
        assert s.params.shape == (20, K_LONG) and np.all(np.isfinite(s.params))
