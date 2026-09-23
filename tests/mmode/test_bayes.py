import numpy as np
import pytest

bayesmith = pytest.importorskip("bayesmith")

from limTOD.mmode import LSTWindow, solve, synthesise  # noqa: E402
from limTOD.mmode.bayes import exact_posterior, posterior_samples  # noqa: E402


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(2)
    mmax = 6
    amp = 100.0 * np.exp(-np.arange(mmax + 1) / 1.5) + 1e-2
    d = amp * np.exp(2j * np.pi * rng.random(mmax + 1))
    d[0] = abs(d[0])
    w = LSTWindow.uniform_arc(0.0, 360.0, cadence_s=1200.0)      # 72 samples, closed
    y = synthesise(w, d)[:, 0]
    return w, d, y, mmax


def test_wide_prior_reproduces_least_squares(small):
    w, d, y, mmax = small
    env = np.full(mmax + 1, 1e4)                                  # effectively flat
    res = exact_posterior(y, w, mmax, env, sigma=0.5)
    ls = solve(y, w, mmax, sigma=0.5)
    assert np.allclose(res.modes[0], ls.modes[0], atol=1e-6)
    assert res.std is None and res.samples is None


def test_tight_prior_pulls_towards_its_mean(small):
    w, d, y, mmax = small
    res = exact_posterior(y + 0 * y, w, mmax, np.full(mmax + 1, 1e-6), sigma=0.5)
    assert np.abs(res.modes[0]).max() < 1e-3


def test_samples_scatter_like_the_analytic_error(small):
    w, d, y, mmax = small
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0, 0.5, y.size)
    res = posterior_samples(noisy, w, mmax, np.full(mmax + 1, 1e4), sigma=0.5, n_draws=120, seed=1)
    ls = solve(noisy, w, mmax, sigma=0.5)
    assert np.isclose(res.std[0, 0], ls.std_d0()[0], rtol=0.25)
    lo, hi = res.d0_interval(0.68)
    assert lo[0] < d[0].real + 0.5 * 3 and hi[0] > d[0].real - 0.5 * 3
