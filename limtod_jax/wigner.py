"""Traced Wigner-D application on dense 2D flm arrays.

s2fft's public rotation API (``s2fft.utils.rotation.rotate_flms`` /
``generate_rotate_dls``) declares the rotation angles as STATIC jit
arguments, so it cannot be fed per-pointing traced angles. These are
re-implementations of those two ~30-line wrappers with the angles traced
(and without the float64 hardcodes), calling the same underlying pure-jnp
Wigner-d kernel ``s2fft.recursions.risbo_jax.compute_full`` (Risbo
recursion; beta is a traced argument there).

Semantics (identical to s2fft): applying ``(alpha, beta, gamma)`` performs

    f'_{lm} = e^{-i m alpha} · Σ_n d^l_{mn}(beta) · e^{-i n gamma} · f_{ln}

i.e. the active field rotation ``R = Rz(alpha)·Ry(beta)·Rz(gamma)``.

``HEALPY_CONVENTION`` is the single frozen mapping from limTOD's
``(psi, theta, phi)`` rotation-angle triple to the ``(alpha, beta, gamma)``
fed here, locked NUMERICALLY against ``healpy.rotate_alm`` (see
``tests/limtod_jax/test_rotation_convention.py``) — per the port contract,
sign/order conventions are never trusted on paper.
"""

from __future__ import annotations

import jax.numpy as jnp
from s2fft.recursions.risbo_jax import compute_full

# limTOD's `_rotate_healpix_map(alm, psi, theta, phi, ...)` calls
# `hp.rotate_alm(alm, phi, theta, psi)` (its phi in healpy's first slot).
# healpy applies the field rotation Rz(psi_lim)·Ry(theta)·Rz(phi_lim) to
# that call, which in the s2fft D-matrix convention above is exactly
# (alpha, beta, gamma) = (psi_lim, theta, phi_lim) — the identity mapping.
# Locked by test_rotation_convention.py (unique winner among 8 candidates,
# atol 1e-11, angle grid including theta ~ 0 and theta ~ pi).
HEALPY_CONVENTION = "identity"  # (alpha, beta, gamma) = (psi, theta, phi)


def angles_to_alpha_beta_gamma(
    psi: jnp.ndarray, theta: jnp.ndarray, phi: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frozen limTOD ``(psi, theta, phi)`` -> s2fft ``(alpha, beta, gamma)``."""
    return psi, theta, phi


def generate_rotate_dls(L: int, beta: jnp.ndarray) -> jnp.ndarray:
    """Wigner-d plane ``d^l_{mn}(beta)`` for all l < L; beta is TRACED.

    Returns shape ``(L, 2L-1, 2L-1)`` in beta's real dtype. Traced-angle
    port of ``s2fft.utils.rotation.generate_rotate_dls``.
    """
    beta = jnp.asarray(beta)
    dtype = jnp.result_type(beta.dtype, jnp.zeros(0).dtype)
    dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=dtype)
    dls = []
    for el in range(L):
        dl_iter = compute_full(dl_iter, beta, L, el)
        dls.append(dl_iter)
    return jnp.stack(dls)


def rotate_flm_2d(
    flm: jnp.ndarray,
    L: int,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    gamma: jnp.ndarray,
    dl_array: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Rotate dense 2D flm ``(L, 2L-1)`` by traced Euler angles.

    Traced-angle port of ``s2fft.utils.rotation.rotate_flms``. ``L`` is
    static; ``alpha``/``beta``/``gamma`` are traced scalars. ``dl_array``
    optionally supplies a precomputed ``(L, 2L-1, 2L-1)`` Wigner-d plane
    (from :func:`generate_rotate_dls`), in which case ``beta`` is unused.
    """
    if flm.shape[-2:] != (L, 2 * L - 1):
        raise ValueError(f"flm shape {flm.shape} does not match L={L}")

    m_grid = jnp.arange(-L + 1, L)
    alpha_phases = jnp.exp(-1j * m_grid * jnp.asarray(alpha))
    gamma_phases = jnp.exp(-1j * m_grid * jnp.asarray(gamma))

    flm_rotated = jnp.zeros_like(flm)
    dl_iter = None
    if dl_array is None:
        dtype = jnp.result_type(jnp.asarray(beta).dtype, jnp.zeros(0).dtype)
        dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=dtype)

    for el in range(L):
        if dl_array is None:
            dl_iter = compute_full(dl_iter, beta, L, el)
            dl_el = dl_iter
        else:
            dl_el = dl_array[el]
        m = jnp.arange(-el, el + 1)
        block = dl_el[m + L - 1][:, m + L - 1]
        flm_rotated = flm_rotated.at[el, L - 1 + m].add(
            jnp.einsum(
                "mn,m,n,n->m",
                block,
                alpha_phases[m + L - 1],
                gamma_phases[m + L - 1],
                flm[el, L - 1 + m],
                optimize=True,
            )
        )
    return flm_rotated
