"""s2fft healpix-sampling wrappers vs healpy, and JAX-path purity.

``alm2map`` must equal ``hp.alm2map`` to roundoff (both are exact synthesis
of a bandlimited function at pixel centers). ``map2alm_quad`` must equal
``(npix/4π)·hp.map2alm(m, iter=0, use_weights=False)`` — the plain uniform
quadrature that makes the harmonic dot equal the pixel dot exactly.
"""

import subprocess
import sys

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from limtod_jax.alm import alm_dot
from limtod_jax.core import rotate_alm
from limtod_jax.hpx import alm2map, map2alm_quad, ones_quadrature_alm


@pytest.mark.parametrize("nside", [8, 16])
def test_alm2map_matches_healpy(nside, rng):
    lmax = 3 * nside - 1
    alm = hp.map2alm(rng.standard_normal(hp.nside2npix(nside)), lmax=lmax)
    ours = np.asarray(alm2map(jnp.asarray(alm), nside=nside, lmax=lmax))
    ref = hp.alm2map(alm.copy(), nside)
    assert ours.shape == (hp.nside2npix(nside),)
    np.testing.assert_allclose(ours, ref, atol=1e-11 * np.max(np.abs(ref)))


@pytest.mark.parametrize("nside", [8, 16])
def test_map2alm_quad_matches_healpy_iter0(nside, rng):
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)
    m = rng.random(npix)
    ours = np.asarray(map2alm_quad(jnp.asarray(m), nside=nside, lmax=lmax))
    ref = (npix / (4 * np.pi)) * hp.map2alm(m, lmax=lmax, iter=0, use_weights=False)
    np.testing.assert_allclose(ours, ref, atol=1e-11 * np.max(np.abs(ref)))


def test_ones_alm_reproduces_pixel_sum(rng):
    """⟨R b, ones_quad⟩ == Σ_p alm2map(R b)[p] — the normalize denominator."""
    nside, lmax = 8, 23
    beam_alm = jnp.asarray(
        hp.map2alm(rng.random(hp.nside2npix(nside)), lmax=lmax)
    )
    rot = rotate_alm(
        beam_alm, jnp.asarray(0.7), jnp.asarray(1.2), jnp.asarray(0.3), lmax=lmax
    )
    ones_alm = ones_quadrature_alm(nside=nside, lmax=lmax)
    harmonic = float(alm_dot(rot, ones_alm))
    pixel = float(np.sum(hp.alm2map(np.asarray(rot).copy(), nside)))
    np.testing.assert_allclose(harmonic, pixel, rtol=1e-12)


def test_quad_dot_reproduces_pixel_dot(rng):
    """⟨b, map2alm_quad(s)⟩ == Σ_p alm2map(b)·s — the forward-sample identity."""
    nside, lmax = 8, 23
    npix = hp.nside2npix(nside)
    beam_alm = jnp.asarray(hp.map2alm(rng.random(npix), lmax=lmax))
    sky = rng.random(npix)
    harmonic = float(
        alm_dot(beam_alm, map2alm_quad(jnp.asarray(sky), nside=nside, lmax=lmax))
    )
    pixel = float(np.dot(hp.alm2map(np.asarray(beam_alm).copy(), nside), sky))
    np.testing.assert_allclose(harmonic, pixel, rtol=1e-12)


def test_jit_vmap_grad(rng):
    nside, lmax = 8, 23
    npix = hp.nside2npix(nside)
    maps = jnp.asarray(rng.random((3, npix)))

    f = jax.jit(lambda m: map2alm_quad(m, nside=nside, lmax=lmax))
    a0 = f(maps[0])
    _ = f(maps[1])
    batched = jax.vmap(lambda m: map2alm_quad(m, nside=nside, lmax=lmax))(maps)
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(a0), atol=1e-13)

    # gradients flow through synthesis (linear map)
    alm = map2alm_quad(maps[0], nside=nside, lmax=lmax)
    g = jax.grad(lambda a: jnp.sum(alm2map(a, nside=nside, lmax=lmax) ** 2))(alm)
    assert bool(jnp.all(jnp.isfinite(g.real))) and bool(jnp.all(jnp.isfinite(g.imag)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_dtypes_follow_x64(rng):
    nside, lmax = 8, 23
    m = jnp.asarray(rng.random(hp.nside2npix(nside)))
    alm = map2alm_quad(m, nside=nside, lmax=lmax)
    assert alm.dtype == jnp.complex128
    back = alm2map(alm, nside=nside, lmax=lmax)
    assert back.dtype == jnp.float64


def test_jax_path_never_imports_healpy():
    """The package must be importable and usable without healpy present."""
    code = (
        "import sys; import limtod_jax; "
        "import limtod_jax.hpx, limtod_jax.core, limtod_jax.projection; "
        "assert 'healpy' not in sys.modules, 'healpy leaked into the JAX path'; "
        "print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "clean" in out.stdout
