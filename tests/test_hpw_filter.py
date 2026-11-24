import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from limTOD.HPW_filter import HPW_mapmaking, wiener_filter_map


def test_old_positional_ridge_cannot_be_interpreted_as_return_full_cov():
    data, operator = np.ones(3), np.ones((3, 1))
    with pytest.raises(TypeError):
        wiener_filter_map(data, operator, 1.0, None, None, 1e-12)
    _, _, covariance = wiener_filter_map(data, operator, 1.0, None, None, return_full_cov=True)
    np.testing.assert_allclose(covariance, [[1.0 / 3.0]])


def _identity_mapmaker(n_samples):
    mapper = HPW_mapmaking.__new__(HPW_mapmaking)
    mapper.num_tods = 1
    mapper.npol = 1
    mapper.num_pixels = n_samples
    mapper.nsky_params = n_samples
    mapper.Tsys_others = False
    mapper.Tsys_operators = np.eye(n_samples)
    return mapper


def test_mapmaker_defaults_to_no_high_pass_without_cutoff():
    tod = np.array([1.0, 2.0, 3.0])
    mapper = _identity_mapmaker(len(tod))

    sky_est, sky_unc = mapper(
        TOD_group=tod,
        dtime=1.0,
        cutoff_freq_group=None,
        noise_variance=1.0,
    )

    np.testing.assert_allclose(sky_est, tod)
    np.testing.assert_allclose(sky_unc, np.ones_like(tod))


def test_mapmaker_requires_cutoff_when_high_pass_enabled():
    tod = np.array([1.0, 2.0, 3.0])
    mapper = _identity_mapmaker(len(tod))

    with pytest.raises(ValueError, match="cutoff_freq_group must be provided"):
        mapper(
            TOD_group=tod,
            dtime=1.0,
            cutoff_freq_group=None,
            use_high_pass=True,
            noise_variance=1.0,
        )


@pytest.mark.parametrize("variance", [1.7, np.array([0.5, np.inf, 2.0, 1.5])])
def test_independent_complex_samples_give_scalar_gaussian_posterior(variance):
    # Completing the square for one unknown amplitude gives a scalar precision
    # and score. Each conjugated response weights its own datum, including a
    # zero contribution from a sample with infinite variance.
    response = np.array([1 + 0.5j, -2j, -0.5 + 3j, 0.7 - 1j], dtype=np.complex64)
    data = np.array([2 - 3j, 30 + 4j, 1j, -1 + 2j], dtype=np.complex64)
    mean, prior_precision = 1.0 - 0.4j, 0.3
    weights = np.broadcast_to(1.0 / variance, response.shape)
    a = response.astype(np.complex128)
    precision = prior_precision + sum(weight * abs(ai) ** 2 for weight, ai in zip(weights, a))
    score = prior_precision * mean + sum(
        weight * ai.conjugate() * di for weight, ai, di in zip(weights, a, data)
    )
    estimate, std, cov = wiener_filter_map(
        data, response[:, None], noise_variance=variance, prior_inv_cov=prior_precision,
        guess=np.array([mean]), return_full_cov=True,
    )
    np.testing.assert_allclose(estimate, [score / precision], rtol=1e-13)
    np.testing.assert_allclose(std, [1.0 / np.sqrt(precision)], rtol=1e-13)
    np.testing.assert_allclose(cov, [[1.0 / precision]], rtol=1e-13)


@pytest.mark.parametrize("variance", [0.5, np.linspace(0.5, 2.0, 32)])
def test_independent_noise_avoids_time_square(monkeypatch, variance):
    n_time = 32
    eye, diag = np.eye, np.diag

    def reject_time_eye(n, *args, **kwargs):
        assert n != n_time, "allocated dense time-domain identity"
        return eye(n, *args, **kwargs)

    def reject_time_diagonal(v, *args, **kwargs):
        assert np.shape(v) != (n_time,), "allocated dense time-domain diagonal"
        return diag(v, *args, **kwargs)

    monkeypatch.setattr(np, "eye", reject_time_eye)
    monkeypatch.setattr(np, "diag", reject_time_diagonal)
    wiener_filter_map(np.ones(n_time), np.ones((n_time, 1)), noise_variance=variance)


@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("phase", [1.0, 1j, np.exp(0.4j)])
def test_residual_variance_for_a_constant_signal(rolling, phase):
    # The least-squares projection onto a constant removes the mean exactly
    # in real arithmetic. These residuals have constant squared magnitude,
    # including in every reflected window used by the rolling estimator.
    n_time = 24
    operator = np.ones((n_time, 1))
    data = 3.0 + phase * np.tile([-0.5, 0.5], n_time // 2)
    estimate, std = wiener_filter_map(
        data, operator, prior_inv_cov=2.0, guess=np.array([1.0]), rolling_variance=rolling,
    )
    precision = n_time / 0.25 + 2.0
    np.testing.assert_allclose(estimate, [(n_time * 3.0 / 0.25 + 2.0) / precision])
    np.testing.assert_allclose(std, [1.0 / np.sqrt(precision)])
