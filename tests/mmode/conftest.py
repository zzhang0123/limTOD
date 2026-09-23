import numpy as np
import pytest

from limTOD.mmode import LSTWindow

MMAX = 40


@pytest.fixture(scope="session")
def modes():
    """A plausible d_m(nu): steeply falling in m, complex, real d_0; 7 channels."""
    rng = np.random.default_rng(5)
    amp = 1600.0 * np.exp(-np.arange(MMAX + 1) / 2.5) + 1e-3
    d = amp[None, :] * np.exp(2j * np.pi * rng.random((7, MMAX + 1)))
    d[:, 0] = np.abs(d[:, 0])
    return d * (1 + 0.1 * np.arange(7))[:, None]


@pytest.fixture(scope="session")
def closed():
    return LSTWindow.uniform_arc(0.0, 360.0, cadence_s=120.0)


@pytest.fixture(scope="session")
def arc7h():
    return LSTWindow.uniform_arc(338.816, 105.288, cadence_s=120.0)


@pytest.fixture(scope="session")
def arc300():
    return LSTWindow.uniform_arc(0.0, 300.0, cadence_s=120.0)
