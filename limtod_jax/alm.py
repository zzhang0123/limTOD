"""Packed healpy alm layout <-> dense 2D flm, and the real-field inner product.

healpy stores the alms of a real field as a 1D complex array holding only
m >= 0, with index ``m·(2·lmax+1−m)/2 + l``.  The Wigner rotation kernel
works on a dense 2D array ``flm[l, lmax+m]`` covering all m in
[-lmax, lmax]; for a real field the negative-m entries are fixed by
``flm[l, −m] = (−1)^m · conj(flm[l, m])``.

Domain: VALID real-field alms — in particular the m = 0 coefficients (the
first ``lmax+1`` packed entries) must be real, as ``hp.map2alm`` produces
them. Inputs with imaginary m = 0 parts lie outside the representable
space: the forward dot silently drops that component while the
packed->2D->packed round trip re-symmetrizes it, so forward/adjoint pairs
are transposes of each other only on the valid subspace.

``lmax`` is always a static Python int; index arrays are built with numpy at
trace time (cached per lmax) so every function is jit/vmap/grad-safe.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


def nalm_of_lmax(lmax: int) -> int:
    """Number of packed healpy alm coefficients for a given lmax."""
    return (lmax + 1) * (lmax + 2) // 2


def lmax_of_nalm(n_alm: int) -> int:
    """Inverse of :func:`nalm_of_lmax`; raises if ``n_alm`` is not triangular."""
    lmax = int(round((np.sqrt(8 * n_alm + 1) - 3) / 2))
    if lmax < 0 or nalm_of_lmax(lmax) != n_alm:
        raise ValueError(f"n_alm={n_alm} is not a valid packed-alm length")
    return lmax


@lru_cache(maxsize=None)
def packed_lm_arrays(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """(l, m) of every packed index, as static numpy int arrays.

    Iterating m-major with l in [m, lmax] visits healpy's packed indices
    ``m·(2·lmax+1−m)/2 + l`` in exactly sequential order.
    """
    ls, ms = [], []
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            ls.append(l)
            ms.append(m)
    return np.asarray(ls, dtype=np.int64), np.asarray(ms, dtype=np.int64)


def _validate_packed(alm, lmax: int) -> None:
    if alm.shape[-1] != nalm_of_lmax(lmax):
        raise ValueError(
            f"packed alm length {alm.shape[-1]} does not match lmax={lmax} "
            f"(expected {nalm_of_lmax(lmax)})"
        )


def packed_to_2d(alm: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """Packed healpy alm (real field) -> dense ``(lmax+1, 2·lmax+1)`` flm.

    Column ``lmax + m`` holds order m; negative m filled via the real-field
    symmetry. Leading batch dimensions pass through.
    """
    _validate_packed(alm, lmax)
    L = lmax + 1
    ls, ms = packed_lm_arrays(lmax)
    dtype = jnp.result_type(alm.dtype, np.complex64)
    flm = jnp.zeros(alm.shape[:-1] + (L, 2 * L - 1), dtype=dtype)
    flm = flm.at[..., ls, L - 1 + ms].set(alm)
    pos = ms > 0
    signs = jnp.asarray((-1.0) ** ms[pos], dtype=dtype)
    flm = flm.at[..., ls[pos], L - 1 - ms[pos]].set(signs * jnp.conj(alm[..., pos]))
    return flm


def packed_from_2d(flm_2d: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """Dense 2D flm -> packed healpy alm (keeps m >= 0 entries only)."""
    L = lmax + 1
    if flm_2d.shape[-2:] != (L, 2 * L - 1):
        raise ValueError(
            f"flm shape {flm_2d.shape[-2:]} does not match lmax={lmax} "
            f"(expected {(L, 2 * L - 1)})"
        )
    ls, ms = packed_lm_arrays(lmax)
    return flm_2d[..., ls, L - 1 + ms]


@lru_cache(maxsize=None)
def alm_weights(lmax: int) -> np.ndarray:
    """Real-field inner-product weights: 1 for m = 0, 2 for m > 0."""
    _, ms = packed_lm_arrays(lmax)
    return np.where(ms > 0, 2.0, 1.0)


def alm_dot(a: jnp.ndarray, b: jnp.ndarray, lmax: int | None = None) -> jnp.ndarray:
    """Weighted real inner product of two packed real-field alms.

    ``⟨a, b⟩ = Σ_l Re[conj(a_l0)·b_l0] + 2·Σ_{l,m>0} Re[conj(a_lm)·b_lm]``

    For real fields this equals the full-sphere harmonic sum
    ``Re Σ_{l,m∈[−l,l]} conj(a_lm)·b_lm``. When ``b`` holds quadrature alms
    ``(npix/4π)·map2alm(s, iter=0)`` and ``a`` is bandlimited, it equals the
    HEALPix pixel dot ``Σ_p A(p)·s(p)`` exactly — the identity the oracle
    tests rely on. Reduces over the last axis; batch dims pass through.
    """
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"alm length mismatch: {a.shape[-1]} vs {b.shape[-1]}")
    if lmax is None:
        lmax = lmax_of_nalm(a.shape[-1])
    else:
        _validate_packed(a, lmax)
    prod = jnp.real(jnp.conj(a) * b)
    w = jnp.asarray(alm_weights(lmax), dtype=prod.dtype)
    return jnp.sum(w * prod, axis=-1)
