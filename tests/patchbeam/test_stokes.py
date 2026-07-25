"""Unit tests for limTOD.patchbeam.stokes."""

from __future__ import annotations

import numpy as np
import pytest

from limTOD.patchbeam import stokes


@pytest.mark.unit
def test_integrate_stokes_I_uniform_returns_sky_temperature():
    """For a uniform sky T0 and a beam whose discrete integral equals Omega_b,
    the antenna temperature is T0."""
    n_freq = 3
    n_pix = 100
    rng = np.random.default_rng(0)
    beam = rng.uniform(0.0, 1.0, size=(n_freq, n_pix))
    sky = np.full(n_pix, 5.0)
    d_omega_pix = 0.01
    omega_b = beam.sum(axis=-1) * d_omega_pix
    out = stokes.integrate_stokes_I(
        beam_disc=beam, sky_disc=sky, omega_b=omega_b, d_omega_pix=d_omega_pix
    )
    np.testing.assert_allclose(out, 5.0)


@pytest.mark.unit
def test_integrate_stokes_I_zero_omega_b_raises():
    with pytest.raises(ValueError, match="omega_b"):
        stokes.integrate_stokes_I(
            beam_disc=np.ones((1, 10)),
            sky_disc=np.ones(10),
            omega_b=np.array([0.0]),
            d_omega_pix=0.01,
        )


@pytest.mark.unit
def test_integrate_stokes_full_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        stokes.integrate_stokes_full(
            beam_disc={"I": np.ones((1, 5))},
            sky_disc={"I": np.ones((1, 5))},
            parallactic_angle_rad=np.zeros(5),
            omega_b={"I": np.array([1.0])},
            d_omega_pix=0.01,
        )
