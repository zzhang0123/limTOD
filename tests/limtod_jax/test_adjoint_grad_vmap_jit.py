"""Acceptance tests 2-5: adjoint dot-test, grad vs finite differences,
jit no-retrace, and vmap consistency."""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.alm import alm_dot, nalm_of_lmax
from limtod_jax.angles import zyz_of_pointing
from limtod_jax.core import (
    beam_weighted_sum,
    generate_tod_sky,
    generate_tod_sky_adjoint,
    rotate_alm,
)

NSIDE, LMAX = 8, 23
N_TIME = 5


@pytest.fixture(scope="module")
def setup():
    npix = hp.nside2npix(NSIDE)
    beam_map = np.random.default_rng(1).random(npix)
    sky_map = np.random.default_rng(2).random(npix)
    beam_alm = jnp.asarray(hp.map2alm(beam_map, lmax=LMAX))
    sky_alm = jnp.asarray(
        (npix / (4 * np.pi)) * hp.map2alm(sky_map, lmax=LMAX, iter=0)
    )
    ones_alm = jnp.asarray(
        (npix / (4 * np.pi)) * hp.map2alm(np.ones(npix), lmax=LMAX, iter=0)
    )
    lst = jnp.linspace(0.0, 300.0, N_TIME)
    psi, theta, phi = zyz_of_pointing(
        lst, 53.24, jnp.linspace(-60.0, 60.0, N_TIME), jnp.full(N_TIME, 41.0),
        jnp.zeros(N_TIME),
    )
    angles = jnp.stack([psi, theta, phi], axis=-1)
    return beam_alm, sky_alm, ones_alm, angles


# ---------------------------------------------------------------- adjoint --
@pytest.mark.parametrize("normalize", [False, True])
def test_adjoint_dot_identity(setup, normalize, rng):
    """<forward(x), y>_R == <x, adjoint(y)>_w  (acceptance test 2)."""
    beam_alm, sky_alm, ones_alm, angles = setup
    y = jnp.asarray(rng.standard_normal(N_TIME))
    fwd = generate_tod_sky(
        beam_alm, sky_alm, angles, lmax=LMAX, normalize=normalize, ones_alm=ones_alm
    )
    adj = generate_tod_sky_adjoint(
        y, beam_alm, angles, lmax=LMAX, normalize=normalize, ones_alm=ones_alm
    )
    lhs = float(jnp.sum(fwd * y))
    rhs = float(alm_dot(adj, sky_alm))
    assert abs(lhs - rhs) / abs(lhs) < 1e-6
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)  # expected: roundoff-exact


def test_adjoint_shape_dtype(setup):
    beam_alm, _, _, angles = setup
    adj = generate_tod_sky_adjoint(jnp.ones(N_TIME), beam_alm, angles, lmax=LMAX)
    assert adj.shape == (nalm_of_lmax(LMAX),)
    assert adj.dtype == jnp.complex128


# ------------------------------------------------------------------- grad --
def _fd_grad(f, x0, h=1e-6):
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


@pytest.mark.parametrize("k,directions", [(2, ("re",)), (40, ("re", "im"))])
def test_grad_wrt_beam_matches_fd(setup, k, directions):
    """k=2 is an m=0 index (real for real fields — only d/dRe is meaningful:
    the imag direction's true gradient is ~1e-8 of the TOD scale, below the
    float64 finite-difference noise floor on an O(1e3) function); k=40 is
    m>0, where both real and imag parts carry signal."""
    beam_alm, sky_alm, _, angles = setup

    def f_re(re):
        b = beam_alm.at[k].set(re + 1j * jnp.imag(beam_alm[k]))
        return jnp.sum(generate_tod_sky(b, sky_alm, angles, lmax=LMAX))

    def f_im(im):
        b = beam_alm.at[k].set(jnp.real(beam_alm[k]) + 1j * im)
        return jnp.sum(generate_tod_sky(b, sky_alm, angles, lmax=LMAX))

    funcs = {"re": (f_re, jnp.real(beam_alm[k])), "im": (f_im, jnp.imag(beam_alm[k]))}
    for d in directions:
        f, x0 = funcs[d]
        g = float(jax.grad(f)(x0))
        fd = float(_fd_grad(f, x0))
        assert np.isfinite(g) and abs(g) > 1e-12
        assert abs(g - fd) / abs(fd) < 1e-4, (d, g, fd)


@pytest.mark.parametrize("k", [2, 40])
def test_grad_wrt_sky_matches_fd(setup, k):
    beam_alm, sky_alm, _, angles = setup

    def f_re(re):
        s = sky_alm.at[k].set(re + 1j * jnp.imag(sky_alm[k]))
        return jnp.sum(generate_tod_sky(beam_alm, s, angles, lmax=LMAX))

    re0 = jnp.real(sky_alm[k])
    g = float(jax.grad(f_re)(re0))
    fd = float(_fd_grad(f_re, re0))
    assert np.isfinite(g) and abs(g) > 1e-12
    assert abs(g - fd) / abs(fd) < 1e-4, (g, fd)


def test_grad_linear_in_sky(setup):
    """Forward is linear in sky: grad w.r.t. sky must be input-independent."""
    beam_alm, sky_alm, _, angles = setup

    def f(s):
        return jnp.sum(generate_tod_sky(beam_alm, s, angles, lmax=LMAX))

    g1 = jax.grad(f, holomorphic=False)(sky_alm)
    g2 = jax.grad(f, holomorphic=False)(2.0 * sky_alm + 1.0)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), rtol=1e-12)


# -------------------------------------------------------------------- jit --
def test_jit_no_retrace_on_second_call(setup, rng):
    beam_alm, sky_alm, _, angles = setup
    n_traces = 0

    def counted(b, s, ang):
        nonlocal n_traces
        n_traces += 1
        return generate_tod_sky(b, s, ang, lmax=LMAX)

    f = jax.jit(counted)
    out1 = f(beam_alm, sky_alm, angles)
    out2 = f(2.0 * beam_alm, sky_alm + 1.0, angles + 0.01)  # same shapes
    assert n_traces == 1, f"retraced: {n_traces} traces"
    assert out1.shape == out2.shape == (N_TIME,)
    assert not np.allclose(np.asarray(out1), np.asarray(out2))
    if hasattr(f, "_cache_size"):
        assert f._cache_size() == 1


# ------------------------------------------------------------------- vmap --
def test_vmap_over_frequency_matches_loop(setup, rng):
    """Acceptance test 5: vmapped-over-frequency == Python loop."""
    beam_alm, sky_alm, _, angles = setup
    n_freq = 3
    scales_b = jnp.asarray(rng.random(n_freq)) + 0.5
    scales_s = jnp.asarray(rng.random(n_freq)) + 0.5
    beams = scales_b[:, None] * beam_alm[None, :]
    skies = scales_s[:, None] * sky_alm[None, :]

    vm = jax.vmap(lambda b, s: generate_tod_sky(b, s, angles, lmax=LMAX))(beams, skies)
    loop = jnp.stack(
        [generate_tod_sky(beams[f], skies[f], angles, lmax=LMAX) for f in range(n_freq)]
    )
    assert vm.shape == (n_freq, N_TIME)
    np.testing.assert_allclose(np.asarray(vm), np.asarray(loop), rtol=1e-12)


def test_rotate_alm_vmap_over_angles_matches_loop(setup):
    beam_alm, _, _, angles = setup
    vm = jax.vmap(
        lambda ang: rotate_alm(beam_alm, ang[0], ang[1], ang[2], lmax=LMAX)
    )(angles)
    loop = jnp.stack(
        [
            rotate_alm(beam_alm, angles[t, 0], angles[t, 1], angles[t, 2], lmax=LMAX)
            for t in range(N_TIME)
        ]
    )
    np.testing.assert_allclose(np.asarray(vm), np.asarray(loop), rtol=1e-12)


# ------------------------------------------------------- beam_weighted_sum --
def test_beam_weighted_sum_matches_alm_dot(setup):
    beam_alm, sky_alm, ones_alm, _ = setup
    plain = beam_weighted_sum(beam_alm, sky_alm)
    np.testing.assert_allclose(float(plain), float(alm_dot(beam_alm, sky_alm)))
    normed = beam_weighted_sum(beam_alm, sky_alm, normalize=True, ones_alm=ones_alm)
    np.testing.assert_allclose(
        float(normed), float(plain) / float(alm_dot(beam_alm, ones_alm))
    )


def test_beam_weighted_sum_normalize_requires_ones():
    a = jnp.zeros(nalm_of_lmax(3), dtype=jnp.complex128)
    with pytest.raises(ValueError, match="ones_alm"):
        beam_weighted_sum(a, a, normalize=True)
    with pytest.raises(ValueError, match="ones_alm"):
        generate_tod_sky(a, a, jnp.zeros((2, 3)), lmax=3, normalize=True)
