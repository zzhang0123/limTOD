"""The Gaussian prior is the only regularisation of the map-making solve.

A ``+ lambda*I`` on the normal equations is a zero-mean Gaussian prior of
precision ``lambda`` -- and, with no matching ``lambda*mu`` on the
right-hand side, one that *contradicts* the declared prior mean. It does
nothing where the data are strong and everything where they are absent, so
its whole effect is to answer "zero" for the directions the scan never
measured, and to report ``1/sqrt(lambda)`` as the uncertainty on that
answer.

These tests pin the two consequences of having removed it:

1. a parameter the data do not touch comes back at *exactly* its prior
   mean, with *exactly* its prior standard deviation;
2. a parameter no prior touches either makes the solve fail loudly rather
   than silently returning a ridge-dominated number.

Test 1 is the sharp one: with the old ``regularization=1e-12`` default and
a 100 K prior sigma the estimate was low by a relative 1e-8, which the
tolerances below reject.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from limTOD.HPW_filter import HPW_mapmaking, simple_wiener_map, wiener_filter_map
from limTOD.gls_mapmaking import GLS_mapmaking

N_TIME = 20
PRIOR_MEAN = 5.0
PRIOR_SIGMA = 100.0          # deliberately LOOSE: a ridge competes hardest here
UNSEEN = 2                   # index of the column the "scan" never sees


def _blind_system(seed=0):
    """A 3-parameter system whose third column is identically zero."""
    rng = np.random.default_rng(seed)
    operator = np.zeros((N_TIME, 3))
    operator[:, :2] = rng.random((N_TIME, 2)) + 0.5
    truth = np.array([3.0, -1.0, 42.0])   # the third value is unobservable
    tod = operator @ truth
    return operator, tod


def _prior():
    mean = np.array([0.0, 0.0, PRIOR_MEAN])
    inv_cov = np.full(3, PRIOR_SIGMA**-2.0)
    return mean, inv_cov


class TestUnconstrainedDirectionIsThePriorExactly:
    def test_wiener_filter_map_returns_the_prior_mean(self):
        operator, tod = _blind_system()
        mean, inv_cov = _prior()

        est, unc, cov = wiener_filter_map(
            tod, operator, noise_variance=1e-2,
            prior_inv_cov=inv_cov, guess=mean, return_full_cov=True,
        )

        # The posterior of an untouched direction IS its prior -- not the
        # prior shrunk toward zero by a stabiliser.
        assert est[UNSEEN] == pytest.approx(PRIOR_MEAN, rel=1e-12)
        assert unc[UNSEEN] == pytest.approx(PRIOR_SIGMA, rel=1e-12)
        assert cov[UNSEEN, UNSEEN] == pytest.approx(PRIOR_SIGMA**2, rel=1e-12)
        # and the measured directions are still recovered. rtol 1e-5, not
        # tighter: the loose zero-mean prior on those two legitimately pulls
        # them by ~1e-6 relative. That pull is declared; a ridge's is not.
        np.testing.assert_allclose(est[:2], [3.0, -1.0], rtol=1e-5)

    def test_hpw_mapmaking_returns_the_prior_mean(self):
        operator, tod = _blind_system(seed=1)
        mean, inv_cov = _prior()

        mm = HPW_mapmaking.__new__(HPW_mapmaking)
        mm.num_tods, mm.npol = 1, 1
        mm.num_pixels = mm.nsky_params = 3
        mm.Tsys_others = False
        mm.Tsys_operators = operator

        est, unc = mm(
            TOD_group=tod, dtime=1.0, noise_variance=1e-2,
            Tsky_prior_mean=mean, Tsky_prior_inv_cov_diag=inv_cov,
        )

        assert est[UNSEEN] == pytest.approx(PRIOR_MEAN, rel=1e-12)
        assert unc[UNSEEN] == pytest.approx(PRIOR_SIGMA, rel=1e-12)

    def test_gls_mapmaking_returns_the_prior_mean(self):
        operator, tod = _blind_system(seed=2)
        mean, inv_cov = _prior()

        mm = GLS_mapmaking.__new__(GLS_mapmaking)
        mm.num_tods, mm.npol = 1, 1
        mm.num_pixels = mm.nsky_params = 3
        mm.Tsys_others = False
        mm.Tsys_operators = operator

        est, unc = mm(
            TOD_group=tod, noise_inv_cov_group=[np.eye(N_TIME) * 1e2],
            noise_model="additive",
            Tsky_prior_mean=mean, Tsky_prior_inv_cov_diag=inv_cov,
        )

        assert est[UNSEEN] == pytest.approx(PRIOR_MEAN, rel=1e-12)
        assert unc[UNSEEN] == pytest.approx(PRIOR_SIGMA, rel=1e-12)


class TestMissingInformationIsReported:
    def test_wiener_filter_map_raises_without_a_prior(self):
        operator, tod = _blind_system(seed=3)
        with pytest.raises(np.linalg.LinAlgError, match="neither the data nor"):
            wiener_filter_map(tod, operator, noise_variance=1e-2)

    def test_gls_mapmaking_raises_without_a_prior(self):
        operator, tod = _blind_system(seed=4)

        mm = GLS_mapmaking.__new__(GLS_mapmaking)
        mm.num_tods, mm.npol = 1, 1
        mm.num_pixels = mm.nsky_params = 3
        mm.Tsys_others = False
        mm.Tsys_operators = operator

        with pytest.raises(np.linalg.LinAlgError, match="neither the data nor"):
            mm(
                TOD_group=tod, noise_inv_cov_group=[np.eye(N_TIME)],
                noise_model="additive",
            )

    def test_a_prior_on_the_blind_direction_alone_is_enough(self):
        """The fix the error message points at actually works."""
        operator, tod = _blind_system(seed=5)
        inv_cov = np.array([0.0, 0.0, PRIOR_SIGMA**-2.0])
        est, _ = wiener_filter_map(
            tod, operator, noise_variance=1e-2,
            prior_inv_cov=inv_cov, guess=np.array([0.0, 0.0, PRIOR_MEAN]),
        )
        np.testing.assert_allclose(est[:2], [3.0, -1.0], rtol=1e-6)
        assert est[UNSEEN] == pytest.approx(PRIOR_MEAN, rel=1e-12)


class TestSimpleWienerMapHasNoHiddenRidge:
    def test_full_rank_solution_is_plain_least_squares(self):
        rng = np.random.default_rng(6)
        operator = rng.random((N_TIME, 3)) + 0.5
        truth = np.array([3.0, -1.0, 2.0])
        est = simple_wiener_map(operator @ truth, operator)
        np.testing.assert_allclose(est, truth, rtol=1e-10)

    def test_rank_deficient_gives_the_minimum_norm_solution(self):
        """No prior, so no ridge to invent one: lstsq's minimum-norm answer
        is the documented choice, and the blind direction stays at zero
        because minimum-norm says so -- not because a stabiliser said so."""
        operator, tod = _blind_system(seed=7)
        est = simple_wiener_map(tod, operator)
        np.testing.assert_allclose(est[:2], [3.0, -1.0], rtol=1e-6)
        assert est[UNSEEN] == 0.0
