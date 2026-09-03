"""What a *wrong* noise covariance costs ``GLS_mapmaking``.

Every test in ``test_gls_mapmaking.py`` generates the TOD and solves it
with the same noise parameters, so nothing there pins the behaviour when
the assumed covariance is wrong — which is the normal situation on real
data: ``GLS_mapmaking.__call__`` defaults ``gain_noise_params`` to
``DEFAULT_GAIN_NOISE_PARAMS`` (the simulator's own truth) and estimates
no noise parameters of its own.

Two facts, established by Monte Carlo over the drift-scan geometry of
``test_gls_mapmaking.py`` under the additive model, where the estimator
is linear and both facts have closed forms. With ``W`` the assumed
weight, ``M = (U' W U)^-1`` and ``N`` the covariance the data are really
drawn from:

1. **The point estimate stays unbiased for any assumed N.**
   ``p_hat = M U' W d`` is linear in ``d`` and ``E[d] = U p``, so
   ``E[p_hat] = M U' W U p = p`` — exactly, for every invertible ``W``.
   A wrong ``N`` costs *efficiency*, not accuracy.

2. **The reported uncertainties do not survive.** The map-maker reports
   ``sqrt(diag(M))``, but the true scatter is the sandwich
   ``Cov[p_hat] = M U' W N W U M``. The two agree only at
   ``W = N^-1``. Both directions occur, and the dangerous one —
   *overconfidence* — is what an underestimated 1/f knee produces.

The full-size run these thresholds are loosened from (400 samples, 173
pixels, 2000 realisations) measured, for median over pixels of
reported sigma / true scatter, with [min, max]::

    true N (matched)      1.00 [0.96, 1.04]    ||U(p_hat-p)|| = 0.131
    white, 1/f ignored    1.34 [0.70, 1.77]                     0.182
    knee 10x too high     0.96 [0.43, 1.44]                     0.196
    knee 10x too low      0.43 [0.40, 0.46]  <- overconfident    0.250
    alpha 2 -> 1.5        3.57 [3.01, 3.96]                     0.142

The committed run below shrinks that to nside=2 / 150 samples /
42 pixels / 800 realisations (a couple of seconds) and measures::

    true N (matched)      1.00 [0.95, 1.04]    ||U(p_hat-p)|| = 0.059
    white, 1/f ignored    0.78 [0.60, 1.03]                     0.081
    knee 10x too high     0.73 [0.56, 0.96]                     0.083
    knee 10x too low      0.39 [0.37, 0.54]  <- overconfident    0.137
    alpha 2 -> 1.5        3.24 [2.98, 3.61]                     0.063

Note that the *sign* of the milder deviations is geometry-dependent (the
white and knee-too-high models land on the other side of 1 here); what is
robust, and all the assertions below rely on, is that the matched model
reports honest error bars, that an underestimated knee is badly
overconfident, and that a wrong spectral index is badly underconfident.
The thresholds are loosened to cover both runs.
"""

import healpy as hp
import numpy as np
import pytest

from limTOD.gls_mapmaking import GLS_mapmaking, flicker_noise_cov

NSIDE = 2
LAT = -30.713
N_TIME = 150
DTIME = 2.0
WVAR = 2.5e-6
N_REAL = 800
SEED = 42

MATCHED = "true N (matched)"
KNEE_LOW = "knee 10x too low"
KNEE_HIGH = "knee 10x too high"
WHITE = "white (1/f ignored)"
ALPHA = "alpha 2 -> 1.5"
MISSPECIFIED = (WHITE, KNEE_HIGH, KNEE_LOW, ALPHA)

# max over pixels of |bias| in Monte-Carlo sigmas. 42 pixels of pure
# noise peak at ~3; measured here is 1.8. A real bias grows as
# sqrt(N_REAL): injecting one of 0.2 sigma takes z_max to 7.5.
Z_MAX = 5.0


def _flicker_at_knee(cycles_per_chunk, alpha=2):
    """limTOD flicker parameters with the knee at N cycles per chunk.

    Placing the knee inside the chunk is what makes the 1/f visible to
    the estimator at all: red noise whose period exceeds the observation
    is degenerate with the sky's overall scale.
    """
    fc = 2.0 * np.pi * cycles_per_chunk / (N_TIME * DTIME)
    return (np.sqrt(1e-4 * np.pi * fc), fc, alpha)


TRUE_FLICKER = _flicker_at_knee(8.0)


@pytest.fixture(scope="module")
def misspec():
    """Monte Carlo of ``GLS_mapmaking`` run under five assumed noise models.

    The TOD is always drawn from the same ``N_true``; only the covariance
    handed to the map-maker changes. Returns the per-model estimate
    ensemble, the sigmas the map-maker reported, and the truth.
    """
    rng = np.random.default_rng(SEED)
    mm = GLS_mapmaking(
        beam_map=rng.random(hp.nside2npix(NSIDE)) + 0.2,
        LST_deg_list_group=np.linspace(0.0, 120.0, N_TIME),
        lat_deg=LAT,
        azimuth_deg_list_group=np.linspace(-25.0, 25.0, N_TIME),
        elevation_deg_list_group=np.full(N_TIME, 55.0),
        threshold=0.7,
        nside_target=NSIDE,
    )
    U = np.asarray(mm.Tsys_operators)
    assert np.linalg.matrix_rank(U) == U.shape[1], "test system must be well-posed"
    p_true = rng.random(U.shape[1]) * 10.0 + 5.0

    t = np.arange(N_TIME) * DTIME
    N_true = flicker_noise_cov(t, TRUE_FLICKER, WVAR)
    assumed = {
        MATCHED: N_true,
        # "I saw a variance and called it white."
        WHITE: flicker_noise_cov(t, None, N_true[0, 0]),
        KNEE_HIGH: flicker_noise_cov(t, _flicker_at_knee(80.0), WVAR),
        KNEE_LOW: flicker_noise_cov(t, _flicker_at_knee(0.8), WVAR),
        ALPHA: flicker_noise_cov(t, _flicker_at_knee(8.0, alpha=1.5), WVAR),
    }

    L = np.linalg.cholesky(N_true)
    data = [U @ p_true + L @ rng.standard_normal(N_TIME) for _ in range(N_REAL)]

    estimates, reported = {}, {}
    for name, N_assumed in assumed.items():
        W = np.linalg.inv(N_assumed)
        ens = np.empty((N_REAL, U.shape[1]))
        sigma = None
        for i, d in enumerate(data):
            est, unc = mm(
                TOD_group=d, noise_inv_cov_group=[W], noise_model="additive"
            )
            ens[i] = est
            sigma = unc  # W and U fixed => the reported sigma is too
        estimates[name] = ens
        reported[name] = np.asarray(sigma)
    return dict(U=U, p_true=p_true, N_true=N_true, assumed=assumed,
                estimates=estimates, reported=reported)


def _sigma_ratio(misspec, name):
    """Reported sigma / true (Monte-Carlo) scatter, per pixel."""
    return misspec["reported"][name] / misspec["estimates"][name].std(axis=0)


def _projected_error(misspec, name):
    """Median ||U (p_hat - p_true)||: the metric the scan actually measures.

    Pixel-space norms are dominated by the geometry's weakest modes; the
    TOD projection weights each mode by how well the scan constrains it.
    """
    resid = misspec["estimates"][name] - misspec["p_true"]
    return float(np.median(np.linalg.norm(resid @ misspec["U"].T, axis=1)))


# ---------------------------------------------------------------------- #
# 1. The point estimate stays unbiased for any assumed N                 #
# ---------------------------------------------------------------------- #
class TestUnbiasedUnderAnyAssumedNoise:
    @pytest.mark.parametrize(
        "model", [MATCHED, WHITE, KNEE_HIGH, KNEE_LOW, ALPHA]
    )
    def test_no_bias(self, misspec, model):
        """E[p_hat] = p for every weight matrix, right or wrong."""
        ens = misspec["estimates"][model]
        bias = ens.mean(axis=0) - misspec["p_true"]
        mc_sigma = ens.std(axis=0) / np.sqrt(N_REAL)
        z = np.abs(bias) / mc_sigma
        assert z.max() < Z_MAX, (model, z.max(), np.argmax(z))

    def test_matched_covariance_is_the_most_efficient(self, misspec):
        """What a wrong N costs is variance: the matched weights are BLUE.

        Same realisations for every model, so this is a paired comparison
        and the ordering is stable at this ensemble size.
        """
        best = _projected_error(misspec, MATCHED)
        for model in MISSPECIFIED:
            assert best < _projected_error(misspec, model), model

    def test_underestimated_knee_costs_the_most_efficiency(self, misspec):
        """Measured 0.250 vs 0.131 at full size — demand at least 1.5x."""
        ratio = _projected_error(misspec, KNEE_LOW) / _projected_error(
            misspec, MATCHED
        )
        assert ratio > 1.5, ratio


# ---------------------------------------------------------------------- #
# 2. The reported uncertainties do not survive                           #
# ---------------------------------------------------------------------- #
class TestReportedUncertaintiesUnderMisspecification:
    def test_matched_model_reports_honest_error_bars(self, misspec):
        """The control: with the true N, sqrt(diag(M)) IS the scatter."""
        ratio = _sigma_ratio(misspec, MATCHED)
        assert 0.9 < np.median(ratio) < 1.1, np.median(ratio)
        assert ratio.min() > 0.8 and ratio.max() < 1.25, (ratio.min(), ratio.max())

    def test_some_misspecified_model_breaks_the_error_bars(self, misspec):
        """The headline: a wrong N buys error bars that are simply wrong.

        Deliberately weak — at least one of the four must be off by more
        than 1.5x in either direction — so the test discriminates without
        pinning the exact ensemble.
        """
        offenders = {
            m: float(np.median(_sigma_ratio(misspec, m)))
            for m in MISSPECIFIED
            if not (1 / 1.5 < np.median(_sigma_ratio(misspec, m)) < 1.5)
        }
        assert offenders, {
            m: float(np.median(_sigma_ratio(misspec, m))) for m in MISSPECIFIED
        }

    def test_underestimated_knee_is_overconfident(self, misspec):
        """The dangerous direction: too-red an assumed N under-reports.

        Measured 0.43 at full size (2.3x overconfident), 0.39 here — a
        map published with these error bars claims a precision it does
        not have.
        """
        assert np.median(_sigma_ratio(misspec, KNEE_LOW)) < 1 / 1.5

    def test_wrong_alpha_is_underconfident(self, misspec):
        """The other direction: alpha 2 -> 1.5 over-reports by ~3x."""
        assert np.median(_sigma_ratio(misspec, ALPHA)) > 1.5

    @pytest.mark.parametrize(
        "model", [MATCHED, WHITE, KNEE_HIGH, KNEE_LOW, ALPHA]
    )
    def test_true_scatter_follows_the_sandwich_covariance(self, misspec, model):
        """Why the error bars break, in closed form.

        The map-maker reports ``M = (U' W U)^-1``; the estimator's actual
        covariance is ``M U' W N W U M``. Pinning the sandwich against
        the Monte Carlo shows the discrepancies above are the weighting,
        not an accident of the ensemble.
        """
        U = misspec["U"]
        W = np.linalg.inv(misspec["assumed"][model])
        M = np.linalg.inv(U.T @ W @ U)
        UtW = U.T @ W
        sandwich = M @ UtW @ misspec["N_true"] @ UtW.T @ M
        predicted = np.sqrt(np.diag(sandwich))
        observed = misspec["estimates"][model].std(axis=0)
        # 800 realisations => ~2.5% relative error on a sample sigma.
        np.testing.assert_allclose(observed, predicted, rtol=0.15)

    def test_reported_sigma_is_the_inverse_normal_equations(self, misspec):
        """And what the map-maker reports is M, unaware of N_true."""
        U = misspec["U"]
        for model in [MATCHED, KNEE_LOW]:
            W = np.linalg.inv(misspec["assumed"][model])
            M = np.linalg.inv(U.T @ W @ U)
            np.testing.assert_allclose(
                misspec["reported"][model], np.sqrt(np.diag(M)), rtol=1e-6
            )
