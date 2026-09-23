import numpy as np
import pytest

from limTOD.mmode import LSTWindow, boxcar_kernel, dpss_kernel, identity_kernel
from limTOD.mmode.kernel import Kernel


@pytest.mark.parametrize("m", [0, 1, 3, 8, 20, 40])
@pytest.mark.parametrize("make", [lambda: dpss_kernel(31, 2.0), lambda: dpss_kernel(181, 4.0),
                                  lambda: boxcar_kernel(21), lambda: Kernel(np.array([1, 2, 5, 2, 1.0]))])
def test_fourier_modes_are_eigenfunctions_on_the_valid_region(m, make, arc300):
    """The identity the whole scheme rests on: k * e^{imt} = K(m) e^{imt}."""
    k = make()
    khat = k.transfer(arc300, 40)
    ph_v = k.valid(arc300).phase
    for f in (np.cos, np.sin):
        assert np.allclose(k.smooth(f(m * arc300.phase), arc300), khat[m] * f(m * ph_v), atol=1e-12)


def test_closed_scan_is_periodic_and_keeps_every_sample(closed):
    k = dpss_kernel(61, 3.0)
    assert k.valid(closed).n == closed.n
    for m in (3, 40):
        y = np.cos(m * closed.phase)
        assert np.allclose(k.smooth(y, closed), k.transfer(closed, m)[m] * y, atol=1e-12)


def test_open_arc_discards_half_a_kernel_at_each_end(arc300):
    k = dpss_kernel(61, 3.0)
    assert k.valid(arc300).n == arc300.n - 60
    assert k.smooth(np.ones(arc300.n), arc300).size == arc300.n - 60


def test_smoothing_operator_matches_convolution(arc300, closed):
    k = boxcar_kernel(9)
    y = np.random.default_rng(3).normal(size=arc300.n)
    assert np.allclose(k.smoothing_operator(arc300) @ y, np.convolve(y, k.taps, mode="valid"))
    yc = np.random.default_rng(4).normal(size=closed.n)
    periodic = np.convolve(np.r_[yc[-4:], yc, yc[:4]], k.taps, mode="valid")
    assert np.allclose(k.smoothing_operator(closed) @ yc, periodic)


def test_identity_kernel_changes_nothing(arc7h):
    k = identity_kernel()
    y = np.random.default_rng(0).normal(size=(arc7h.n, 3))
    assert np.array_equal(k.smooth(y, arc7h), y)
    assert k.valid(arc7h) is arc7h
    assert np.allclose(k.transfer(arc7h, 5), 1.0)


def test_kernel_validation():
    with pytest.raises(ValueError, match="odd"):
        Kernel(np.ones(4))
    with pytest.raises(ValueError, match="symmetric"):
        Kernel(np.array([1.0, 2.0, 0.5]))
    with pytest.raises(ValueError, match="sums to zero"):
        Kernel(np.array([1.0, -2.0, 1.0]))
    with pytest.raises(ValueError, match="NW"):
        dpss_kernel(31, 20.0)


def test_smoothing_refuses_non_uniform_unordered_and_too_short_windows():
    with pytest.raises(ValueError, match="uniformly"):
        dpss_kernel(3, 1.0).smooth(np.zeros(5), LSTWindow(np.array([0.0, 1.0, 3.0, 6.0, 10.0])))
    with pytest.raises(ValueError, match="time order"):
        dpss_kernel(3, 1.0).smooth(np.zeros(4), LSTWindow(np.array([0.0, 2.0, 1.0, 3.0])))
    with pytest.raises(ValueError, match="at least"):
        dpss_kernel(31, 2.0).smooth(np.zeros(5), LSTWindow.uniform_arc(0.0, 2.0))
    with pytest.raises(ValueError, match="at least"):
        dpss_kernel(181, 4.0).valid(LSTWindow.uniform_arc(0.0, 360.0, cadence_s=1200.0))


def test_noise_covariance_matches_monte_carlo(arc300):
    k = boxcar_kernel(21)
    cov = k.noise_covariance(arc300, sigma=2.0)
    rng = np.random.default_rng(1)
    draws = np.stack([k.smooth(rng.normal(0, 2.0, arc300.n), arc300) for _ in range(4000)])
    emp = np.cov(draws.T)
    assert np.allclose(np.diag(cov), np.diag(emp), rtol=0.08)
    assert np.allclose(cov[0, 10], emp[0, 10], rtol=0.12)
    assert np.allclose(cov[0, 30], 0.0, atol=1e-12)          # beyond one kernel length


def test_noise_covariance_with_per_sample_sigma_is_the_explicit_construction(arc300):
    k = dpss_kernel(21, 2.0)
    w = np.where(np.arange(arc300.n) < 150, 1.0, 10.0)
    s = k.smoothing_operator(arc300)
    explicit = s @ np.diag(1 / w) @ s.T
    assert np.allclose(k.noise_covariance(arc300, 1 / np.sqrt(w)), explicit)


def test_noise_covariance_is_periodic_on_a_closed_scan(closed):
    cov = dpss_kernel(31, 2.0).noise_covariance(closed)
    assert cov.shape == (closed.n, closed.n)
    assert np.allclose(cov[0, -1], cov[0, 1])
