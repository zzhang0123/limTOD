"""End-to-end oracle for HPW_mapmaking: the class output must equal the
regularized normal-equations solution built independently in the test from
the class's own system operator,

    x = solve(A^T N^-1 A + S^-1 + reg*I,  A^T N^-1 d + S^-1 mu).

This is both a correctness statement and an exact-behavior pin protecting
the internal decomposition refactors of ``__call__``."""

import healpy as hp
import numpy as np
import pytest

from limTOD.HPW_filter import HPW_mapmaking

NSIDE = 4
LAT = -30.713
REG = 1e-8

# The comparison only makes sense on a WELL-POSED system: at nside=4 the
# sphere has 192 pixels, so the pixel-selection threshold must be high
# enough (0.7 -> ~160 selected) and the time axis long enough (220 samples)
# that A^T A is full rank; otherwise both the class and the oracle solve a
# regularization-dominated null space and agree only by accident.
N1, N2 = 120, 100


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(11)
    beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
    lst_g = [np.linspace(0.0, 60.0, N1), np.linspace(100.0, 160.0, N2)]
    az_g = [np.linspace(-30.0, 30.0, N1), np.zeros(N2)]
    el_g = [np.full(N1, 55.0), np.full(N2, 90.0)]

    mm = HPW_mapmaking(
        beam_map=beam_map,
        LST_deg_list_group=lst_g,
        lat_deg=LAT,
        azimuth_deg_list_group=az_g,
        elevation_deg_list_group=el_g,
        threshold=0.7,
        nside_target=NSIDE,
    )
    a_full = np.concatenate(mm.Tsys_operators, axis=0)
    assert np.linalg.matrix_rank(a_full) == a_full.shape[1], "test system must be well-posed"
    truth = rng.random(a_full.shape[1]) * 10.0
    sigma2 = 1e-4
    tod_full = a_full @ truth + np.sqrt(sigma2) * rng.standard_normal(a_full.shape[0])
    tod_group = [tod_full[:N1], tod_full[N1:]]
    return mm, a_full, tod_full, tod_group, sigma2


def _normal_eq_solution(a, d, sigma2, s_inv_diag=None, mu=None, reg=REG):
    n_par = a.shape[1]
    s_inv = np.diag(s_inv_diag) if s_inv_diag is not None else np.zeros((n_par, n_par))
    mu = np.zeros(n_par) if mu is None else mu
    lhs = a.T @ a / sigma2 + s_inv + reg * np.eye(n_par)
    rhs = a.T @ d / sigma2 + s_inv @ mu
    return np.linalg.solve(lhs, rhs)


class TestHPWMapmakingOracle:
    def test_unfiltered_solution_matches_normal_equations(self, setup):
        mm, a_full, tod_full, tod_group, sigma2 = setup
        est, unc = mm(
            TOD_group=tod_group, dtime=2.0, noise_variance=sigma2,
            regularization=REG,
        )
        expected = _normal_eq_solution(a_full, tod_full, sigma2)
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-8)
        assert np.all(np.isfinite(unc)) and unc.shape == est.shape

    def test_prior_pulls_solution_toward_mean(self, setup):
        mm, a_full, tod_full, tod_group, sigma2 = setup
        n_par = a_full.shape[1]
        mu = np.full(n_par, 5.0)
        s_inv_diag = np.full(n_par, 1e3)
        est, _ = mm(
            TOD_group=tod_group, dtime=2.0, noise_variance=sigma2,
            regularization=REG,
            Tsky_prior_mean=mu, Tsky_prior_inv_cov_diag=s_inv_diag,
        )
        expected = _normal_eq_solution(a_full, tod_full, sigma2, s_inv_diag, mu)
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-8)

    def test_per_tod_noise_variance_list(self, setup):
        mm, a_full, tod_full, tod_group, sigma2 = setup
        n1 = len(tod_group[0])
        v1, v2 = 1e-4, 4e-4
        est, _ = mm(
            TOD_group=tod_group, dtime=2.0,
            noise_variance=[v1, np.full(len(tod_group[1]), v2)],
            regularization=REG,
        )
        n_inv = np.diag(1.0 / np.concatenate([np.full(n1, v1), np.full(len(tod_group[1]), v2)]))
        n_par = a_full.shape[1]
        lhs = a_full.T @ n_inv @ a_full + REG * np.eye(n_par)
        rhs = a_full.T @ n_inv @ tod_full
        expected = np.linalg.solve(lhs, rhs)
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-8)

    def test_high_pass_filter_applied_consistently(self, setup):
        """With the high-pass on, the class must solve the FILTERED system;
        rebuild it from the class's own recorded filter matrices."""
        mm, a_full, tod_full, tod_group, sigma2 = setup
        est, _ = mm(
            TOD_group=tod_group, dtime=2.0, noise_variance=sigma2,
            cutoff_freq_group=[1e-3, 1e-3], use_high_pass=True,
            regularization=REG,
        )
        blocks = mm.HP_exact
        f_a = np.concatenate(
            [blocks[i] @ mm.Tsys_operators[i] for i in range(2)], axis=0
        )
        f_d = np.concatenate([blocks[i] @ tod_group[i] for i in range(2)])
        expected = _normal_eq_solution(f_a, f_d, sigma2)
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-8)

    def test_return_full_cov_appends_posterior_covariance(self, setup):
        """return_full_cov=True used to crash __call__ with a 2-name
        unpacking of wiener_filter_map's 3-tuple."""
        mm, a_full, tod_full, tod_group, sigma2 = setup
        est, unc, cov = mm(
            TOD_group=tod_group, dtime=2.0, noise_variance=sigma2,
            regularization=REG, return_full_cov=True,
        )
        n_par = a_full.shape[1]
        assert cov.shape == (n_par, n_par)
        np.testing.assert_allclose(np.asarray(unc), np.sqrt(np.diag(cov)), rtol=1e-10)
        expected_cov = np.linalg.inv(a_full.T @ a_full / sigma2 + REG * np.eye(n_par))
        np.testing.assert_allclose(cov, expected_cov, rtol=1e-6, atol=1e-12)

    def test_gain_and_known_injection(self, setup):
        mm, a_full, tod_full, tod_group, sigma2 = setup
        gains = [2.0, 4.0]
        inj = [np.full(len(t), 1.5) for t in tod_group]
        raw = [g * t for g, t in zip(gains, tod_group)]
        est, _ = mm(
            TOD_group=[gains[i] * (tod_group[i] + 0.0) for i in range(2)],
            dtime=2.0, noise_variance=sigma2, regularization=REG,
            gain_group=gains,
            known_injection_group=inj,
        )
        d = np.concatenate([tod_group[i] - inj[i] for i in range(2)])
        expected = _normal_eq_solution(a_full, d, sigma2)
        np.testing.assert_allclose(np.asarray(est), expected, rtol=1e-8)


class TestAnnotationPassBugFixes:
    """Regressions for the four bugs surfaced during the typing pass."""

    def test_sim_noise_accepts_plain_list(self):
        from limTOD.flicker_model import sim_noise

        np.random.seed(3)
        out = sim_noise(1.335e-5, 1.099e-3, 2, [0.0, 2.0, 4.0, 6.0])
        assert out.shape == (1, 4) and np.all(np.isfinite(out))

    def test_wiener_full_cov_singular_raises_clearly(self):
        from limTOD.HPW_filter import wiener_filter_map

        tod = np.zeros(4)
        operator = np.zeros((4, 3))  # singular normal equations, reg = 0
        with pytest.raises(np.linalg.LinAlgError, match="posterior covariance"):
            wiener_filter_map(
                tod, operator, noise_variance=1.0, regularization=0.0,
                return_full_cov=True,
            )

    @pytest.fixture()
    def single_tod_geometry(self):
        rng = np.random.default_rng(4)
        n = 150
        beam_map = rng.random(hp.nside2npix(NSIDE)) + 0.2
        kw = dict(
            beam_map=beam_map,
            LST_deg_list_group=np.linspace(0.0, 90.0, n),
            lat_deg=LAT,
            azimuth_deg_list_group=np.linspace(-20.0, 20.0, n),
            elevation_deg_list_group=np.full(n, 60.0),
            threshold=0.7,
            nside_target=NSIDE,
        )
        return n, kw, rng

    def test_single_tod_tsys_others_array_and_list_agree(self, single_tod_geometry):
        """The docstring's 'an array or a list of arrays' promise: a bare 2D
        array used to crash; both forms must now build the same operator."""
        n, kw, rng = single_tod_geometry
        other = rng.random((n, 2))
        mm_arr = HPW_mapmaking(Tsys_others_operator_group=other, **kw)
        mm_list = HPW_mapmaking(Tsys_others_operator_group=[other], **kw)
        np.testing.assert_array_equal(
            np.asarray(mm_arr.Tsys_operators), np.asarray(mm_list.Tsys_operators)
        )
        assert mm_arr.n_params_others == 2

    def test_tsys_others_group_length_mismatch_raises(self, single_tod_geometry):
        n, kw, rng = single_tod_geometry
        others = [rng.random((n, 2)), rng.random((n, 2))]
        with pytest.raises(ValueError, match="entries but there are"):
            HPW_mapmaking(Tsys_others_operator_group=others, **kw)
