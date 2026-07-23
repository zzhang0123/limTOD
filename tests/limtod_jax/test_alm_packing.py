"""Packed-healpy <-> dense 2D flm converters and the weighted inner product.

Oracles: healpy's index convention (``hp.Alm.getidx``) and s2fft's numpy
reference converters (``flm_hp_to_2d`` / ``flm_2d_to_hp``).
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from s2fft.sampling import s2_samples

from limtod_jax import alm as alm_mod


def _random_packed_alm(rng, nside, lmax):
    """A valid real-field packed alm (from a random map)."""
    m = rng.standard_normal(hp.nside2npix(nside))
    return hp.map2alm(m, lmax=lmax)


@pytest.mark.parametrize("lmax", [3, 23])
def test_index_arrays_match_healpy(lmax):
    ls, ms = alm_mod.packed_lm_arrays(lmax)
    assert len(ls) == alm_mod.nalm_of_lmax(lmax)
    for i, (l, m) in enumerate(zip(ls, ms)):
        assert i == hp.Alm.getidx(lmax, int(l), int(m))


def test_nalm_lmax_roundtrip():
    for lmax in [0, 1, 5, 23, 47]:
        assert alm_mod.lmax_of_nalm(alm_mod.nalm_of_lmax(lmax)) == lmax
    with pytest.raises(ValueError):
        alm_mod.lmax_of_nalm(7)  # not a triangular count


@pytest.mark.parametrize("lmax", [3, 23])
def test_packed_to_2d_matches_s2fft(lmax, rng):
    packed = _random_packed_alm(rng, 16, lmax)
    ours = np.asarray(alm_mod.packed_to_2d(jnp.asarray(packed), lmax))
    ref = s2_samples.flm_hp_to_2d(packed, lmax + 1)
    np.testing.assert_allclose(ours, ref, atol=1e-15)


@pytest.mark.parametrize("lmax", [3, 23])
def test_packed_from_2d_matches_s2fft_and_roundtrips(lmax, rng):
    packed = _random_packed_alm(rng, 16, lmax)
    flm = s2_samples.flm_hp_to_2d(packed, lmax + 1)
    ours = np.asarray(alm_mod.packed_from_2d(jnp.asarray(flm), lmax))
    ref = s2_samples.flm_2d_to_hp(flm, lmax + 1)
    np.testing.assert_allclose(ours, ref, atol=1e-15)
    # full round trip is exact
    back = alm_mod.packed_from_2d(alm_mod.packed_to_2d(jnp.asarray(packed), lmax), lmax)
    np.testing.assert_allclose(np.asarray(back), packed, atol=1e-15)


def test_alm_dot_equals_full_2d_sum(rng):
    lmax = 23
    a = jnp.asarray(_random_packed_alm(rng, 16, lmax))
    b = jnp.asarray(_random_packed_alm(rng, 16, lmax))
    ours = alm_mod.alm_dot(a, b)
    a2, b2 = alm_mod.packed_to_2d(a, lmax), alm_mod.packed_to_2d(b, lmax)
    full = jnp.real(jnp.sum(jnp.conj(a2) * b2))
    np.testing.assert_allclose(float(ours), float(full), rtol=1e-14)
    # symmetric and real
    np.testing.assert_allclose(float(alm_mod.alm_dot(b, a)), float(ours), rtol=1e-14)
    assert jnp.isrealobj(ours)


def test_shape_validation():
    with pytest.raises(ValueError):
        alm_mod.packed_to_2d(jnp.zeros(7, dtype=jnp.complex128), lmax=3)
    with pytest.raises(ValueError):
        alm_mod.packed_from_2d(jnp.zeros((4, 6), dtype=jnp.complex128), lmax=3)
    with pytest.raises(ValueError):
        alm_mod.alm_dot(
            jnp.zeros(6, dtype=jnp.complex128), jnp.zeros(10, dtype=jnp.complex128)
        )


def test_converters_jit_and_vmap(rng):
    lmax = 7
    batch = jnp.stack(
        [jnp.asarray(_random_packed_alm(rng, 8, lmax)) for _ in range(3)]
    )

    to2d = jax.jit(alm_mod.packed_to_2d, static_argnames="lmax")
    one = to2d(batch[0], lmax=lmax)
    _ = to2d(batch[1], lmax=lmax)  # second call, same shapes: must not error

    batched = jax.vmap(lambda x: alm_mod.packed_to_2d(x, lmax))(batch)
    assert batched.shape == (3,) + one.shape
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(one), atol=0)

    back = jax.vmap(lambda f: alm_mod.packed_from_2d(f, lmax))(batched)
    np.testing.assert_allclose(np.asarray(back), np.asarray(batch), atol=1e-15)


def test_dtype_follows_input(rng):
    lmax = 7
    packed64 = jnp.asarray(_random_packed_alm(rng, 8, lmax), dtype=jnp.complex64)
    assert alm_mod.packed_to_2d(packed64, lmax).dtype == jnp.complex64
    packed128 = jnp.asarray(_random_packed_alm(rng, 8, lmax), dtype=jnp.complex128)
    assert alm_mod.packed_to_2d(packed128, lmax).dtype == jnp.complex128
    assert alm_mod.alm_dot(packed128, packed128).dtype == jnp.float64
