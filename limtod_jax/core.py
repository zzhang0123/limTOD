"""Core sky -> TOD chain: alm rotation, beam-weighted dot, TOD generation.

Pure-JAX port of the linear part of ``limTOD.simulator.generate_TOD_sky``
(Stokes I). Everything here is jit/vmap/grad-safe: ``lmax`` and
``normalize`` are static; alms, angles, and TODs are traced.

Exactness contract (the reason this matches numpy limTOD to float64
roundoff): the rotated beam is exactly bandlimited, so limTOD's pixel-space
sample ``np.sum(B_rot * s)`` equals the weighted harmonic dot
``⟨R b, s̃⟩`` EXACTLY when ``s̃`` holds QUADRATURE alms
``(npix/4π)·map2alm(s, iter=0)`` (see :func:`limtod_jax.hpx.map2alm_quad`).
Likewise ``normalize`` divides by the rotated beam's PIXEL SUM, computed
exactly as ``⟨R b, ones_alm⟩`` with ``ones_alm`` the quadrature alms of the
ones map — identical semantics to numpy limTOD's ``normalize_beam``.

Out of scope (nonlinear cleanups, not part of the linear chain):
``_truncate_map`` (``truncate_frac_thres``) and horizontal masks — the
numpy oracle reproduces this module with ``truncate_frac_thres=0.0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from limtod_jax.alm import alm_dot, nalm_of_lmax, packed_from_2d, packed_to_2d
from limtod_jax.wigner import angles_to_alpha_beta_gamma, rotate_flm_2d


def _full_sphere_dot(a2d: jnp.ndarray, b2d: jnp.ndarray) -> jnp.ndarray:
    """Re Σ_{l,m∈[−l,l]} conj(a)·b on dense flm — equals the weighted packed
    inner product :func:`limtod_jax.alm.alm_dot` for real fields."""
    return jnp.real(jnp.sum(jnp.conj(a2d) * b2d, axis=(-2, -1)))


def _validate_angles(zyz_angles: jnp.ndarray) -> None:
    if zyz_angles.ndim != 2 or zyz_angles.shape[-1] != 3:
        raise ValueError(
            f"zyz_angles must have shape (n_time, 3) [(psi, theta, phi) rows, "
            f"radians]; got {zyz_angles.shape}"
        )


def _require_ones(normalize: bool, ones_alm) -> None:
    if normalize and ones_alm is None:
        raise ValueError(
            "normalize=True requires ones_alm — the quadrature alms of the "
            "ones map ((npix/4π)·map2alm(1, iter=0); see "
            "limtod_jax.hpx.ones_quadrature_alm) — so that the normalizer is "
            "the rotated beam's exact pixel sum, as in numpy limTOD."
        )


def rotate_alm(
    alm: jnp.ndarray,
    psi: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    lmax: int,
) -> jnp.ndarray:
    """Wigner rotation of packed real-field alms; angles traced, radians.

    Reproduces the alm operation of ``limTOD.simulator._rotate_healpix_map``
    for the same ``(psi, theta, phi)`` — which is
    ``hp.rotate_alm(alm, phi, theta, psi)``: limTOD passes its phi into
    healpy's first slot (convention locked numerically, see
    ``tests/limtod_jax/test_rotation_convention.py``).
    """
    flm = packed_to_2d(alm, lmax)
    a, b, g = angles_to_alpha_beta_gamma(psi, theta, phi)
    return packed_from_2d(rotate_flm_2d(flm, lmax + 1, a, b, g), lmax)


def beam_weighted_sum(
    beam_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    *,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Harmonic-space beam-weighted sum -> scalar antenna temperature.

    Equals limTOD's pixel-space ``_beam_weighted_sum(B, s)`` exactly when
    ``sky_alm`` (and ``ones_alm``) hold quadrature alms (module docstring).
    ``normalize`` (static) divides by the beam's pixel sum ``⟨beam, ones⟩``.
    """
    _require_ones(normalize, ones_alm)
    num = alm_dot(beam_alm, sky_alm)
    if not normalize:
        return num
    assert ones_alm is not None  # _require_ones guarantees; narrows the type
    return num / alm_dot(beam_alm, ones_alm)


def generate_tod_sky(
    beam_alm: jnp.ndarray,
    sky_alm: jnp.ndarray,
    zyz_angles: jnp.ndarray,
    *,
    lmax: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sky TOD: rotate the beam to each pointing, dot with the sky.

    Args:
        beam_alm: packed beam alms ``(n_alm,)`` (as ``hp.map2alm(beam_map)``
            computes them in numpy limTOD).
        sky_alm: packed QUADRATURE sky alms ``(n_alm,)`` (module docstring).
        zyz_angles: ``(n_time, 3)`` rows of ``(psi, theta, phi)`` [radians],
            i.e. stacked :func:`limtod_jax.angles.zyz_of_pointing` output.
        lmax: static band-limit matching the alm lengths.
        normalize: static; divide each sample by the rotated beam's pixel
            sum (numpy limTOD's ``normalize_beam`` semantics).
        ones_alm: quadrature alms of the ones map; required iff normalize.

    Returns:
        ``(n_time,)`` real TOD. Pointings are iterated with ``lax.map``
        (sequential — per-step Wigner memory is O(lmax^3), so batching the
        time axis would multiply that by n_time); the whole function is
        vmappable over a leading frequency axis of ``beam_alm``/``sky_alm``.
    """
    _validate_angles(zyz_angles)
    _require_ones(normalize, ones_alm)
    if beam_alm.shape[-1] != nalm_of_lmax(lmax) or sky_alm.shape[-1] != nalm_of_lmax(lmax):
        raise ValueError(
            f"alm lengths ({beam_alm.shape[-1]}, {sky_alm.shape[-1]}) do not "
            f"match lmax={lmax} (expected {nalm_of_lmax(lmax)})"
        )
    L = lmax + 1
    beam_flm = packed_to_2d(beam_alm, lmax)
    sky_flm = packed_to_2d(sky_alm, lmax)
    ones_flm = None
    if normalize:
        assert ones_alm is not None  # _require_ones guarantees; narrows the type
        ones_flm = packed_to_2d(ones_alm, lmax)

    def sample(angles: jnp.ndarray) -> jnp.ndarray:
        a, b, g = angles_to_alpha_beta_gamma(angles[0], angles[1], angles[2])
        rot = rotate_flm_2d(beam_flm, L, a, b, g)
        num = _full_sphere_dot(rot, sky_flm)
        if not normalize:
            return num
        assert ones_flm is not None  # set above whenever normalize is True
        return num / _full_sphere_dot(rot, ones_flm)

    return jax.lax.map(sample, zyz_angles)


def generate_tod_sky_adjoint(
    tod: jnp.ndarray,
    beam_alm: jnp.ndarray,
    zyz_angles: jnp.ndarray,
    *,
    lmax: int,
    normalize: bool = False,
    ones_alm: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Exact transpose of :func:`generate_tod_sky` in the sky slot.

    With the weighted real inner product ``⟨·,·⟩`` of
    :func:`limtod_jax.alm.alm_dot` on the sky side and the plain dot on the
    TOD side, the forward map ``s̃ ↦ (⟨R_t b, s̃⟩)_t`` (optionally divided by
    ``d_t = ⟨R_t b, ones⟩``) has adjoint

        ``y ↦ Σ_t (y_t / d_t) · (R_t b)``

    — an accumulation of rotated beam alms, no synthesis/analysis involved.
    Satisfies ``⟨forward(x), y⟩_R == ⟨x, adjoint(y)⟩_w`` to roundoff
    (dot-test in the suite); this is what map-making normal equations use.
    """
    _validate_angles(zyz_angles)
    _require_ones(normalize, ones_alm)
    if tod.ndim != 1 or tod.shape[0] != zyz_angles.shape[0]:
        raise ValueError(
            f"tod must be 1D of length n_time={zyz_angles.shape[0]} "
            f"(batch with jax.vmap), got shape {tod.shape}"
        )
    L = lmax + 1
    beam_flm = packed_to_2d(beam_alm, lmax)
    ones_flm = None
    if normalize:
        assert ones_alm is not None  # _require_ones guarantees; narrows the type
        ones_flm = packed_to_2d(ones_alm, lmax)
    acc_dtype = jnp.result_type(beam_flm.dtype, tod.dtype)

    def step(accum, inputs):
        angles, y = inputs
        a, b, g = angles_to_alpha_beta_gamma(angles[0], angles[1], angles[2])
        rot = rotate_flm_2d(beam_flm, L, a, b, g)
        if normalize:
            y = y / _full_sphere_dot(rot, ones_flm)
        return accum + y * rot, None

    accum0 = jnp.zeros(beam_flm.shape, dtype=acc_dtype)
    accum, _ = jax.lax.scan(step, accum0, (zyz_angles, tod))
    return packed_from_2d(accum, lmax)
