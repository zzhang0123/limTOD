"""Error paths: the static shape/argument checks raise clear ValueErrors."""

import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.alm import lmax_of_nalm, nalm_of_lmax
from limtod_jax.core import generate_tod_sky, generate_tod_sky_adjoint
from limtod_jax.wigner import rotate_flm_2d

LMAX = 3
N_ALM = nalm_of_lmax(LMAX)
ANGLES = jnp.zeros((4, 3))
ALM = jnp.zeros(N_ALM, dtype=jnp.complex128)


@pytest.mark.parametrize("bad", [-5, 0, 7])
def test_lmax_of_nalm_rejects_invalid(bad):
    with pytest.raises(ValueError, match="not a valid packed-alm length"):
        lmax_of_nalm(bad)


@pytest.mark.parametrize(
    "bad_angles",
    [jnp.zeros((4,)), jnp.zeros((4, 2)), jnp.zeros((2, 4, 3))],
)
def test_generate_tod_sky_rejects_bad_angle_shapes(bad_angles):
    with pytest.raises(ValueError, match="zyz_angles"):
        generate_tod_sky(ALM, ALM, bad_angles, lmax=LMAX)


def test_generate_tod_sky_rejects_alm_length_mismatch():
    with pytest.raises(ValueError, match="does not match lmax"):
        generate_tod_sky(jnp.zeros(N_ALM + 1, dtype=jnp.complex128), ALM, ANGLES, lmax=LMAX)


@pytest.mark.parametrize(
    "bad_tod",
    [jnp.zeros(3), jnp.zeros((2, 4)), jnp.zeros((4, 4))],
)
def test_adjoint_rejects_bad_tod_shapes(bad_tod):
    with pytest.raises(ValueError, match="tod must be 1D of length n_time"):
        generate_tod_sky_adjoint(bad_tod, ALM, ANGLES, lmax=LMAX)


def test_rotate_flm_2d_accepts_one_leading_axis_only():
    """ONE leading axis is the Stokes stack (see limtod_jax.stokes); two is
    still jax.vmap's job, and a wrong trailing shape is always an error."""
    L = LMAX + 1
    good = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    zero = jnp.asarray(0.0)
    out = rotate_flm_2d(good, L, zero, zero, zero)
    np.testing.assert_allclose(np.asarray(out), np.asarray(good))

    stacked = rotate_flm_2d(
        jnp.zeros((3, L, 2 * L - 1), dtype=jnp.complex128), L, zero, zero, zero
    )
    assert stacked.shape == (3, L, 2 * L - 1)

    with pytest.raises(ValueError, match="batch beyond one leading axis"):
        rotate_flm_2d(
            jnp.zeros((2, 3, L, 2 * L - 1), dtype=jnp.complex128), L, zero, zero, zero
        )
    with pytest.raises(ValueError, match="batch beyond one leading axis"):
        rotate_flm_2d(jnp.zeros((L, 2 * L), dtype=jnp.complex128), L, zero, zero, zero)
