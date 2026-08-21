"""Native sky -> Tsys projection-matrix builder (port-contract phase 2).

JAX port of the linear part of ``limTOD.simulator.generate_sky2sys_projection``
(Stokes I): each row is the beam rotated to one pointing, synthesized on the
HEALPix grid, and sampled at the selected pixels. The output feeds a
downstream fixed-pointing differentiable projector, where the matrix IS the
model and no beam rotation happens per sample.

Pixel SELECTION stays in numpy limTOD (``truncate_stacked_beam``) — it is a
discrete, non-differentiable choice made offline; this module consumes the
resulting ``pixel_indices``. Truncation/masks are out of scope as in
:mod:`limtod_jax.core` (oracle equivalence holds at ``truncate_frac_thres=0``).

STOKES I ONLY, unlike the rest of the linear chain, and for a cost reason
rather than a correctness one. A projection ROW is the beam evaluated in
PIXEL space, so the polarised version needs the spin-2 synthesis of
:func:`limtod_jax.hpx.eb_to_qu` on EVERY pointing — where the horizon mask
pays it once — and that synthesis carries an O(nside·lmax^2) precompute
kernel (see the warning in :mod:`limtod_jax.hpx`). The TOD and m-mode paths
never leave harmonic space at all, which is why they are polarised for free.

A Stokes stack is rejected here rather than accepted and synthesized row-wise
as if every row were spin-0: that would run without error and be wrong in Q
and U.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from limtod_jax.alm import nalm_of_lmax, packed_from_2d, packed_to_2d
from limtod_jax.core import _full_sphere_dot, _require_ones, _validate_angles
from limtod_jax.hpx import alm2map
from limtod_jax.wigner import angles_to_alpha_beta_gamma, rotate_flm_2d


def generate_projection_rows(
    beam_alm: jnp.ndarray,
    zyz_angles: jnp.ndarray,
    pixel_indices: jnp.ndarray,
    *,
    lmax: int,
    nside: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Projection matrix rows: rotated beam at selected pixels.

    Args:
        beam_alm: packed beam alms ``(n_alm,)``.
        zyz_angles: ``(n_time, 3)`` rows of ``(psi, theta, phi)`` [radians].
        pixel_indices: selected HEALPix RING pixel indices ``(n_sel,)``
            (e.g. from ``limTOD.simulator.truncate_stacked_beam``).
        lmax, nside: static grid configuration.
        normalize: static; divide each row by the rotated beam's pixel sum
            (numpy limTOD's ``normalize_beam``), computed exactly via
            ``ones_alm`` (required iff normalize; see
            :func:`limtod_jax.hpx.ones_quadrature_alm`).

    Returns
    -------
        ``(n_time, n_sel)`` real matrix, matching
        ``limTOD.simulator.generate_sky2sys_projection`` with
        ``truncate_frac_thres=0.0``. Rows
        are produced with ``lax.map`` (sequential over pointings); the
        function is vmappable over a leading frequency axis of ``beam_alm``.
    """
    _validate_angles(zyz_angles)
    _require_ones(normalize, ones_alm)
    if beam_alm.shape[-1] != nalm_of_lmax(lmax):
        raise ValueError(
            f"beam_alm length {beam_alm.shape[-1]} does not match lmax={lmax} "
            f"(expected {nalm_of_lmax(lmax)})"
        )
    if beam_alm.ndim != 1:
        raise ValueError(
            f"beam_alm must be 1D (n_alm,) — this builder is Stokes I only "
            f"(module docstring: a projection row lives in pixel space and so "
            f"would need a spin-2 synthesis) — got shape {beam_alm.shape}. "
            f"Batch frequencies with jax.vmap."
        )
    L = lmax + 1
    beam_flm = packed_to_2d(beam_alm, lmax)
    ones_flm = None
    if normalize:
        assert ones_alm is not None  # _require_ones guarantees; narrows the type
        ones_flm = packed_to_2d(ones_alm, lmax)

    def row(angles: jnp.ndarray) -> jnp.ndarray:
        a, b, g = angles_to_alpha_beta_gamma(angles[0], angles[1], angles[2])
        rot = rotate_flm_2d(beam_flm, L, a, b, g)
        full = alm2map(packed_from_2d(rot, lmax), nside=nside, lmax=lmax)
        vals = full[..., pixel_indices]
        if not normalize:
            return vals
        assert ones_flm is not None  # set above whenever normalize is True
        return vals / _full_sphere_dot(rot, ones_flm)

    return jax.lax.map(row, zyz_angles)
