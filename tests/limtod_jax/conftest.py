"""Shared fixtures for the limtod_jax test suite.

The whole directory is skipped when jax / s2fft / healpy are unavailable
(healpy is the correctness oracle throughout — it never enters the JAX path).

x64 must be enabled before any jax array is created: Wigner recursions and
the 1e-6 oracle-equivalence tolerances are float64 statements (see the port
contract, hard requirement 4).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("s2fft")
hp = pytest.importorskip("healpy")

# Process-global: enabling x64 here affects every jax-using test collected in
# the same pytest session (today only this directory touches jax).
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def quad_alm():
    """Quadrature alms: S̃_lm = Σ_p s(p)·Y*_lm(p) = (npix/4π)·map2alm(s, iter=0).

    With these sky alms, the harmonic beam-weighted dot equals limTOD's
    pixel-space ``np.sum(B_rot * s)`` EXACTLY (B_rot is bandlimited), which
    is what makes the 1e-6 oracle tests achievable.
    """

    def _quad(m, lmax):
        m = np.asarray(m, dtype=np.float64)
        return (m.shape[-1] / (4.0 * np.pi)) * hp.map2alm(
            m, lmax=lmax, iter=0, use_weights=False
        )

    return _quad


@pytest.fixture(scope="session")
def beam_alm_iter3():
    """Beam alms exactly as ``generate_TOD_sky`` computes them internally."""

    def _beam(m, lmax):
        return hp.map2alm(np.asarray(m, dtype=np.float64), lmax=lmax)

    return _beam


@pytest.fixture(scope="session")
def oracle_tod():
    """numpy limTOD oracle with truncation disabled (the linear chain).

    The native port is the linear forward model; ``_truncate_map`` (default
    ``truncate_frac_thres=1e-10``) is a nonlinear cleanup outside the port's
    scope, so equivalence is defined against ``truncate_frac_thres=0.0``.
    """
    sim = pytest.importorskip("limTOD.simulator")

    def _oracle(beam_map, sky_map, lst, lat, az, el, selfrot, normalize_beam=False):
        return sim.generate_TOD_sky(
            np.asarray(beam_map),
            np.asarray(sky_map),
            np.asarray(lst, dtype=np.float64),
            float(lat),
            np.asarray(az, dtype=np.float64),
            np.asarray(el, dtype=np.float64),
            np.asarray(selfrot, dtype=np.float64),
            normalize_beam=normalize_beam,
            truncate_frac_thres=0.0,
        )

    return _oracle
