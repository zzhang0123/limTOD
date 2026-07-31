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


def _dl_dtype(beta: jnp.ndarray):
    """Real dtype of the Wigner-d plane: the caller's, floored at float32.

    The floor exists so an integer or float16 ``beta`` still gives a sane
    plane. It used to be ``jnp.zeros(0).dtype`` — the session default — which
    in an x64 session is float64, silently OVERRIDING a deliberate float32
    request and doubling the largest array in the rotation (215 MB at
    lmax=191). The Risbo recursion is float32-stable (~2e-6 relative), so the
    caller's choice is the one that should win.
    """
    return jnp.result_type(jnp.asarray(beta).dtype, jnp.float32)


def generate_rotate_dls(L: int, beta: jnp.ndarray) -> jnp.ndarray:
    """Wigner-d plane ``d^l_{mn}(beta)`` for all l < L; beta is TRACED.

    Returns shape ``(L, 2L-1, 2L-1)`` in beta's real dtype. Traced-angle
    port of ``s2fft.utils.rotation.generate_rotate_dls``.
    """
    beta = jnp.asarray(beta)
    dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=_dl_dtype(beta))
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
    """Rotate dense flm ``(L, 2L-1)`` — or a stack ``(n_row, L, 2L-1)`` — by
    traced Euler angles.

    Traced-angle port of ``s2fft.utils.rotation.rotate_flms``. ``L`` is
    static; ``alpha``/``beta``/``gamma`` are traced scalars. ``dl_array``
    optionally supplies a precomputed ``(L, 2L-1, 2L-1)`` Wigner-d plane
    (from :func:`generate_rotate_dls`), in which case ``beta`` is unused.

    THE STACKED FORM EXISTS FOR POLARISATION, and it is not the same thing as
    ``jax.vmap`` over rows. Every row shares one rotation, so it shares the
    Risbo recursion: vmapping would repeat the only O(lmax^3) step ``n_row``
    times, while this shares ``dl_iter`` across the rows and pays ``n_row``
    times the cheap per-l einsum only. Hoisting the FULL plane into
    ``dl_array`` would do the same but costs O(lmax^3) MEMORY (541 MB at
    lmax=256) where the recursion carries only O(lmax^2); that trade is the
    caller's to make, this one is free.

    Exactly ONE leading axis is allowed — deeper batching is still
    ``jax.vmap``'s job. The 2-D path keeps its original contraction string, so
    every unpolarised result is unchanged BIT FOR BIT; a 1-row STACK is the
    same arithmetic in a different association order and matches only to
    roundoff (~1 ulp, data-dependent). Pinned in
    ``tests/limtod_jax/test_polarisation.py``.
    """
    stacked = flm.ndim == 3
    if flm.shape[-2:] != (L, 2 * L - 1) or flm.ndim not in (2, 3):
        raise ValueError(
            f"flm must have shape (L, 2L-1)={(L, 2 * L - 1)} or "
            f"(n_row, L, 2L-1), got {flm.shape}; batch beyond one leading "
            "axis with jax.vmap"
        )
    if dl_array is not None:
        expected = (L, 2 * L - 1, 2 * L - 1)
        if dl_array.shape != expected:
            raise ValueError(
                f"dl_array must have shape (L, 2L-1, 2L-1)={expected} for "
                f"L={L}, got {dl_array.shape}; build it with "
                f"generate_rotate_dls(L={L}, beta) or "
                f"dl_plane_for_pointing(..., lmax={L - 1}) — note the public "
                f"APIs take lmax = L-1 while this one takes L. JAX CLAMPS "
                f"out-of-bounds indices instead of raising, so a mismatched "
                f"plane would otherwise be accepted silently and rotate by "
                f"the wrong sub-block (measured: ~100% error, no warning)."
            )

    m_grid = jnp.arange(-L + 1, L)
    alpha_phases = jnp.exp(-1j * m_grid * jnp.asarray(alpha))
    gamma_phases = jnp.exp(-1j * m_grid * jnp.asarray(gamma))

    flm_rotated = jnp.zeros_like(flm)
    dl_iter = None
    if dl_array is None:
        dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1), dtype=_dl_dtype(beta))

    for el in range(L):
        if dl_array is None:
            dl_iter = compute_full(dl_iter, beta, L, el)
            dl_el = dl_iter
        else:
            dl_el = dl_array[el]
        m = jnp.arange(-el, el + 1)
        block = dl_el[m + L - 1][:, m + L - 1]
        flm_rotated = flm_rotated.at[..., el, L - 1 + m].add(
            jnp.einsum(
                "mn,m,n,pn->pm" if stacked else "mn,m,n,n->m",
                block,
                alpha_phases[m + L - 1],
                gamma_phases[m + L - 1],
                flm[..., el, L - 1 + m],
                optimize=True,
            )
        )
    return flm_rotated
