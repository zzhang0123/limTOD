"""HEALPix (RING) map <-> alm inside JAX, via s2fft — healpy never enters.

Two operations, both linear and jit/vmap/grad-safe:

* :func:`alm2map` — exact synthesis of a bandlimited function at pixel
  centers (matches ``hp.alm2map`` to roundoff).
* :func:`map2alm_quad` — QUADRATURE analysis
  ``S̃_lm = Σ_p m(p)·Y*_lm(p) = (npix/4π)·hp.map2alm(m, iter=0)``.
  This is deliberately NOT an approximation of the true alms: paired with
  :func:`limtod_jax.alm.alm_dot` it turns HEALPix pixel dots into exact
  harmonic identities (see :mod:`limtod_jax.core`), which is what the
  TOD forward model and its adjoint are built on.

``method="jax"`` equivalents only — s2fft's ``jax_healpy`` backend routes
through healpy and is banned from this package.

PRECISION: s2fft's healpix transforms use the Price-McEwen on-the-fly
recursion, which needs float64 — in a float32 session (``jax_enable_x64``
off) they carry O(10%) errors even at L ~ 12 (s2fft warns about exactly
this at import). Enable x64 wherever these functions feed real analysis.
The Wigner rotation path (:mod:`limtod_jax.wigner`, Risbo recursion) does
NOT share this problem — it stays at roundoff accuracy in float32.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import s2fft

from limtod_jax.alm import packed_from_2d, packed_to_2d


def alm2map(alm: jnp.ndarray, *, nside: int, lmax: int) -> jnp.ndarray:
    """Packed real-field alms -> HEALPix RING map ``(12·nside²,)``."""
    flm = packed_to_2d(alm, lmax)
    # s2fft returns a complex array with ~0 imaginary part even for
    # reality=True; the synthesis of a real field is real by construction.
    return jnp.real(
        s2fft.inverse_jax(
            flm, lmax + 1, spin=0, nside=nside, sampling="healpix", reality=True
        )
    )


def map2alm_quad(m: jnp.ndarray, *, nside: int, lmax: int) -> jnp.ndarray:
    """HEALPix RING map -> packed quadrature alms ``Σ_p m(p)·Y*_lm(p)``."""
    flm = s2fft.forward_jax(
        m, lmax + 1, spin=0, nside=nside, sampling="healpix", reality=True
    )
    npix = 12 * nside * nside
    return (npix / (4.0 * np.pi)) * packed_from_2d(flm, lmax)


def map2alm_iter(
    m: jnp.ndarray, *, nside: int, lmax: int, iterations: int = 3
) -> jnp.ndarray:
    """healpy-style iterative analysis: matches ``hp.map2alm(m, lmax, iter=k)``.

    Jacobi refinement of the quadrature estimate: starting from
    ``a = (4π/npix)·Σ_p m(p)·Y*_lm(p)``, repeat ``iterations`` times

        ``a ← a + (4π/npix)·analysis(m − synthesis(a))``

    which is exactly healpy's ``iter`` scheme (oracle-locked in
    ``tests/limtod_jax/test_driftscan.py``). Unlike :func:`map2alm_quad`
    this RETURNS TRUE alms of the (non-bandlimited) map's best bandlimited
    representation — use it when a map-space operation (e.g. a horizon
    mask) must be carried back to harmonic space the same way numpy
    limTOD's internal ``hp.map2alm(..., iter=3)`` would.

    ``iterations`` is static; the whole function is linear in ``m`` and
    jit/vmap/grad-safe.
    """
    npix = 12 * nside * nside
    scale = (4.0 * np.pi) / npix
    alm = scale * map2alm_quad(m, nside=nside, lmax=lmax)
    for _ in range(iterations):
        residual = m - alm2map(alm, nside=nside, lmax=lmax)
        alm = alm + scale * map2alm_quad(residual, nside=nside, lmax=lmax)
    return alm


def ones_quadrature_alm(*, nside: int, lmax: int) -> jnp.ndarray:
    """Quadrature alms of the ones map — the exact pixel-sum functional.

    ``alm_dot(x, ones_quadrature_alm(...)) == Σ_p alm2map(x)[p]`` for any
    bandlimited ``x``; this is the ``normalize`` denominator in
    :func:`limtod_jax.core.generate_tod_sky` (numpy limTOD's
    ``normalize_beam`` divides by the rotated beam's pixel sum).
    """
    return map2alm_quad(jnp.ones(12 * nside * nside), nside=nside, lmax=lmax)
